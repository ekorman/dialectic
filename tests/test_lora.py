import copy

import pytest
import torch
import torch.nn as nn

from dialectic.llm.lora import (
    apply_lora,
    freeze_base_params,
    merge_lora,
    merged_state_dict,
    resolve_lora_targets,
)


class _Block(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.gate_proj = nn.Linear(d, 2 * d, bias=False)
        self.other = nn.Linear(d, d)


class _Tiny(nn.Module):
    def __init__(self, d: int = 8):
        super().__init__()
        self.blocks = nn.ModuleList([_Block(d) for _ in range(2)])


def test_merged_state_dict_passthrough_without_lora():
    net = _Tiny()
    sd = merged_state_dict(net)
    assert set(sd.keys()) == set(net.state_dict().keys())
    for k, v in sd.items():
        assert torch.equal(v, net.state_dict()[k])


def test_merged_state_dict_zero_init_equals_base():
    net = _Tiny()
    plain_keys = set(net.state_dict().keys())
    apply_lora(net, rank=4, alpha=16.0)
    sd = merged_state_dict(net)
    assert set(sd.keys()) == plain_keys
    # lora_B is zero-initialized, so merged weights equal the base weights
    assert torch.equal(
        sd["blocks.0.q_proj.weight"],
        net.state_dict()["blocks.0.q_proj.base.weight"],
    )
    assert torch.equal(
        sd["blocks.0.q_proj.bias"],
        net.state_dict()["blocks.0.q_proj.base.bias"],
    )


def test_merged_state_dict_matches_merge_lora():
    torch.manual_seed(0)
    net = _Tiny()
    plain_keys = set(net.state_dict().keys())
    apply_lora(net, rank=4, alpha=16.0)
    with torch.no_grad():
        for name, param in net.named_parameters():
            if "lora_B" in name:
                param.normal_()

    sd = merged_state_dict(net)
    ref_sd = merge_lora(copy.deepcopy(net)).state_dict()
    assert set(sd.keys()) == plain_keys == set(ref_sd.keys())
    for k in plain_keys:
        assert torch.equal(sd[k], ref_sd[k]), k

    # the net itself is not mutated
    assert any("lora_A" in k for k in net.state_dict())


def test_freeze_base_params_leaves_only_adapters_trainable():
    net = _Tiny()
    apply_lora(net, rank=4, alpha=16.0)
    freeze_base_params(net)
    trainable = {n for n, p in net.named_parameters() if p.requires_grad}
    assert trainable
    assert all("lora_" in n for n in trainable)


def test_resolve_lora_targets():
    assert resolve_lora_targets("attn") == ("q_proj", "k_proj", "v_proj", "o_proj")
    assert resolve_lora_targets("mlp") == ("gate_proj", "up_proj", "down_proj")
    assert "q_proj" in resolve_lora_targets("all")
    with pytest.raises(ValueError):
        resolve_lora_targets("bogus")
