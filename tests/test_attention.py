import torch
import torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3RotaryEmbedding,
)

from dialectic.llm.components.attention import MHSA, attention


def test_mhsa_not_causal_no_rope():
    l, b, d = 4, 6, 20

    num_heads = 2

    x = torch.rand(b, l, d)

    mhsa = MHSA(d, d // num_heads, num_heads=num_heads)

    torch_mhsa = nn.MultiheadAttention(d, num_heads=num_heads, bias=False)
    torch_mhsa.in_proj_weight = nn.Parameter(
        torch.cat([mhsa.q_proj.weight, mhsa.k_proj.weight, mhsa.v_proj.weight], dim=0)
    )
    torch_mhsa.out_proj.weight = nn.Parameter(mhsa.o_proj.weight)

    y1 = mhsa(x)

    xT = x.transpose(1, 0)
    y2 = torch_mhsa(key=xT, value=xT, query=xT)

    assert y1.shape == y2[0].transpose(1, 0).shape

    torch.testing.assert_close(y1, y2[0].transpose(1, 0))


def test_attention_no_gqa():
    l, b, d, num_heads = 4, 6, 20, 4

    q, k, v = [torch.rand(b, num_heads, l, d) for _ in range(3)]

    # test no causal
    a1 = attention(q, k, v, causal=False)
    a2 = scaled_dot_product_attention(q, k, v, is_causal=False)
    torch.testing.assert_close(a1, a2)

    # test causal
    a1 = attention(q, k, v, causal=True)
    a2 = scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(a1, a2)


def test_attention_with_gqa():
    l, b, d, num_heads, num_kv_heads = 4, 6, 20, 4, 2
    k, v = [torch.rand(b, num_kv_heads, l, d) for _ in range(2)]
    q = torch.rand(b, num_heads, l, d)

    a1 = attention(q, k, v, causal=False)
    a2 = scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=True)

    torch.testing.assert_close(a1, a2)


def test_attention_vs_hf_qwen():
    l, b, d, head_d, num_heads, num_kv_heads = 4, 6, 20, 16, 8, 2

    conf = Qwen3Config()
    conf.head_dim = head_d
    conf.num_key_value_heads = num_kv_heads
    conf.num_attention_heads = num_heads
    conf.hidden_size = d
    conf._attn_implementation = "sdpa"

    hf_att = Qwen3Attention(conf, 0)

    our_att = MHSA(
        d,
        head_d,
        num_heads,
        num_kv_heads,
        causal=True,
        apply_rms_norm=True,
        rope_base_value=10000,
        rms_norm_eps=1e-6,
    )
    our_att.load_state_dict(hf_att.state_dict())

    x = torch.rand(b, l, d)
    rot_emb = Qwen3RotaryEmbedding(conf)
    position_ids = torch.stack([torch.arange(0, l) for _ in range(b)])

    att1 = our_att(x)
    att2, _ = hf_att(
        x, position_embeddings=rot_emb(x, position_ids), attention_mask=None
    )

    torch.testing.assert_close(att1, att2)
