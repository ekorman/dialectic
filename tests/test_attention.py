import torch
import torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention
from transformers.models.qwen3.modeling_qwen3 import (
    apply_rotary_pos_emb,
    Qwen3Attention,
    Qwen3RotaryEmbedding,
)
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


from tllm.attention import (
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

    mhsa = MHSA(d, num_heads=num_heads, rope_params=None)

    torch_mhsa = nn.MultiheadAttention(d, num_heads=num_heads, bias=False)
    torch_mhsa.in_proj_weight = nn.Parameter(
        torch.cat([mhsa.Q.weight, mhsa.K.weight, mhsa.V.weight], dim=0)
    )
    torch_mhsa.out_proj.weight = nn.Parameter(mhsa.out_proj.weight)

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

    sin, cos = create_rope_sine_cosine_tensors(d, b, RopeBufferParams(context_length=l))

    dec_point_tol = 6

    # our version is interweaved versus huggingface's two-halves approach
    for ours, hfs in [(sin, hf_sin), (cos, hf_cos)]:
        assert ours.shape == hfs.shape == torch.Size((b, l, d))
        assert set([round(x, dec_point_tol) for x in ours.flatten().tolist()]) == set(
            [round(x, dec_point_tol) for x in hfs.flatten().tolist()]
        )

        torch.testing.assert_close(
            ours[:, :, torch.arange(0, d, 2)], hfs[:, :, : d // 2]
        )
        torch.testing.assert_close(
            ours[:, :, torch.arange(1, d, 2)], hfs[:, :, d // 2 :]
        )

    def transform_us_to_hf(y: torch.Tensor):
        """permutes the components of the tensor by moving the odd indices to the second half and the
        even indices to the first half
        """
        return torch.cat(
            [y[:, :, :, torch.arange(0, d, 2)], y[:, :, :, torch.arange(1, d, 2)]], -1
        )

    hf_x_with_pe, _ = apply_rotary_pos_emb(
        transform_us_to_hf(x), transform_us_to_hf(x), cos=hf_cos, sin=hf_sin
    )
    our_x_with_pe = apply_rope(x, sin=sin, cos=cos)

    assert our_x_with_pe.shape == hf_x_with_pe.shape == torch.Size((b, num_heads, l, d))
    assert set(
        [round(x, dec_point_tol) for x in our_x_with_pe.flatten().tolist()]
    ) == set([round(x, dec_point_tol) for x in hf_x_with_pe.flatten().tolist()])

    torch.testing.assert_close(transform_us_to_hf(our_x_with_pe), hf_x_with_pe)
