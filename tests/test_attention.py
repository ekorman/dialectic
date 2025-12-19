import torch
import torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention
from transformers.models.qwen3.modeling_qwen3 import (
    apply_rotary_pos_emb,
    Qwen3Attention,
    Qwen3RotaryEmbedding,
    Qwen3MLP,
)
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


from tllm.attention import (
    GatedMLP,
    RopeBufferParams,
    apply_rope,
    attention,
    MHSA,
    create_rope_sine_cosine_tensors,
)


def test_mhsa_not_causal_no_rope():
    l, b, d = 4, 6, 20

    num_heads = 2

    x = torch.rand(b, l, d)

    mhsa = MHSA(d, d // num_heads, num_heads=num_heads, rope_params=None)

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


def test_rope_cosine_sine_against_hf():
    b, num_heads, l, d = 4, 1, 2, 32
    x = torch.rand(b, num_heads, l, d)

    conf = Qwen3Config()
    conf.head_dim = d

    rot_emb = Qwen3RotaryEmbedding(conf)
    position_ids = torch.stack([torch.arange(0, l) for _ in range(b)])
    hf_cos, hf_sin = rot_emb(x, position_ids)

    sin, cos = create_rope_sine_cosine_tensors(d, RopeBufferParams(context_length=l))

    dec_point_tol = 6

    # our version is interweaved versus huggingface's two-halves approach
    for ours, hfs in [(sin, hf_sin), (cos, hf_cos)]:
        assert ours.shape == torch.Size((1, l, d))
        assert hfs.shape == torch.Size((b, l, d))  # hf's is duplciated across batch

        # sanity check hf duplicates
        for i in range(b):
            torch.testing.assert_close(hfs[0], hfs[i])

        assert set([round(x, dec_point_tol) for x in ours.flatten().tolist()]) == set(
            [round(x, dec_point_tol) for x in hfs.flatten().tolist()]
        )

        torch.testing.assert_close(ours[0], hfs[0])

    hf_x_with_pe, _ = apply_rotary_pos_emb(x, x, cos=hf_cos, sin=hf_sin)
    our_x_with_pe = apply_rope(x, sin=sin, cos=cos)

    assert our_x_with_pe.shape == hf_x_with_pe.shape == torch.Size((b, num_heads, l, d))
    assert set(
        [round(x, dec_point_tol) for x in our_x_with_pe.flatten().tolist()]
    ) == set([round(x, dec_point_tol) for x in hf_x_with_pe.flatten().tolist()])

    torch.testing.assert_close(our_x_with_pe, hf_x_with_pe)


def convert_hf_att_weights_to_att_weights(
    sd: dict[str, torch.Tensor],  # permute_qkv: bool
) -> dict[str, torch.Tensor]:
    key_mapper = {
        "q_proj.weight": "Q.weight",
        "k_proj.weight": "K.weight",
        "v_proj.weight": "V.weight",
        "o_proj.weight": "out_proj.weight",
        "q_norm.weight": "q_norm.weight",
        "k_norm.weight": "k_norm.weight",
    }
    return {key_mapper[k]: v for k, v in sd.items()}


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
        rope_params=RopeBufferParams(l),
    )
    our_att.load_state_dict(hf_att.state_dict(), strict=False)

    x = torch.rand(b, l, d)
    rot_emb = Qwen3RotaryEmbedding(conf)
    position_ids = torch.stack([torch.arange(0, l) for _ in range(b)])

    att1 = our_att(x)
    att2, _ = hf_att(
        x, position_embeddings=rot_emb(x, position_ids), attention_mask=None
    )

    torch.testing.assert_close(att1, att2)


def test_gated_mlp():
    b, l, d, hidden_d = 4, 6, 20, 32
    x = torch.rand(b, l, d)

    conf = Qwen3Config()
    conf.hidden_size = d
    conf.intermediate_size = hidden_d

    mlp1 = GatedMLP(d, hidden_d)
    mlp2 = Qwen3MLP(conf)

    mlp1.load_state_dict(mlp2.state_dict())

    torch.testing.assert_close(mlp1(x), mlp2(x))
