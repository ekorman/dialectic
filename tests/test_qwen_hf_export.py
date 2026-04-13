"""Fast, network-free checks for ``export_qwen3_to_hf_dir``.

The heavy HF round-trip lives in ``test_vllm_matches_pytorch.py`` behind
``TEST_LLM_AGAINST_HF``. These tests are cheap: a tiny dialectic-Qwen net,
an in-memory programmatic tokenizer, and asserts on the synthesized
``config.json`` and directory layout.
"""

import json
from pathlib import Path

import safetensors.torch as safetensors_torch
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from dialectic.llm.qwen import create_qwen
from dialectic.llm.qwen_hf_export import (
    _qwen3_config_from_net,
    export_qwen3_to_hf_dir,
)


def _tiny_qwen(tie_weights: bool = True):
    return create_qwen(
        d=32,
        vocab_size=128,
        n_decoder_layers=2,
        attn_head_d=16,
        attn_num_heads=4,
        attn_num_kv_heads=2,
        mlp_hidden_d=64,
        tie_weights=tie_weights,
        rope_base_value=1000000,
    )


def _tiny_tokenizer() -> Tokenizer:
    return Tokenizer(WordLevel({"[UNK]": 0, "a": 1, "b": 2}, unk_token="[UNK]"))


def test_qwen3_config_from_net_matches_hyperparameters():
    net = _tiny_qwen()
    cfg = _qwen3_config_from_net(net, eos_token_id=151645, pad_token_id=151643)

    assert cfg["architectures"] == ["Qwen3ForCausalLM"]
    assert cfg["model_type"] == "qwen3"
    assert cfg["hidden_size"] == 32
    assert cfg["num_hidden_layers"] == 2
    assert cfg["num_attention_heads"] == 4
    assert cfg["num_key_value_heads"] == 2
    assert cfg["head_dim"] == 16
    assert cfg["intermediate_size"] == 64
    assert cfg["vocab_size"] == 128
    assert cfg["rope_theta"] == 1000000.0
    assert cfg["hidden_act"] == "silu"
    assert cfg["tie_word_embeddings"] is True
    assert cfg["eos_token_id"] == 151645
    assert cfg["pad_token_id"] == 151643
    assert cfg["attention_bias"] is False


def test_qwen3_config_detects_untied_embeddings():
    net = _tiny_qwen(tie_weights=False)
    cfg = _qwen3_config_from_net(net, eos_token_id=0, pad_token_id=0)
    assert cfg["tie_word_embeddings"] is False


def test_export_qwen3_to_hf_dir_writes_expected_files(tmp_path: Path):
    net = _tiny_qwen()
    tokenizer = _tiny_tokenizer()

    out = export_qwen3_to_hf_dir(
        net,
        tokenizer=tokenizer,
        out_dir=tmp_path,
        eos_token_id=151645,
        pad_token_id=151643,
    )
    assert out == tmp_path

    files = {p.name for p in tmp_path.iterdir()}
    assert files == {"config.json", "model.safetensors", "tokenizer.json"}

    config = json.loads((tmp_path / "config.json").read_text())
    assert config["model_type"] == "qwen3"
    assert config["hidden_size"] == 32

    reloaded_tok = Tokenizer.from_file(str(tmp_path / "tokenizer.json"))
    assert reloaded_tok.token_to_id("a") == 1

    state = safetensors_torch.load_file(str(tmp_path / "model.safetensors"))
    assert "model.embed_tokens.weight" in state
    assert "lm_head.weight" not in state
    assert "model.lm_head.weight" not in state
    assert "model.layers.0.self_attn.q_proj.weight" in state

    torch.testing.assert_close(
        state["model.embed_tokens.weight"],
        net.embed_tokens.weight.detach().cpu(),
    )


def test_export_qwen3_to_hf_dir_drops_tied_lm_head(tmp_path: Path):
    net = _tiny_qwen(tie_weights=True)
    tokenizer = _tiny_tokenizer()

    export_qwen3_to_hf_dir(
        net,
        tokenizer=tokenizer,
        out_dir=tmp_path,
        eos_token_id=0,
        pad_token_id=0,
    )
    state = safetensors_torch.load_file(str(tmp_path / "model.safetensors"))
    assert "model.lm_head.weight" not in state
    assert "lm_head.weight" not in state


def test_export_strips_caller_padding_mutation(tmp_path: Path):
    """Regression: ``generate_from_text`` flips the caller's tokenizer into
    left-padding mode. The exporter must not persist that runtime state to
    disk, since vLLM would reload it and tokenize prompts differently than
    the pure-PyTorch path it is supposed to match.
    """
    net = _tiny_qwen()
    tokenizer = _tiny_tokenizer()
    tokenizer.enable_padding(pad_id=0, pad_token="[UNK]", direction="left")

    export_qwen3_to_hf_dir(
        net,
        tokenizer=tokenizer,
        out_dir=tmp_path,
        eos_token_id=0,
        pad_token_id=0,
    )

    reloaded = Tokenizer.from_file(str(tmp_path / "tokenizer.json"))
    assert reloaded.padding is None
    assert reloaded.truncation is None

    assert tokenizer.padding is not None
