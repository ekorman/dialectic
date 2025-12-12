import torch
import torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention

from tllm.attention import attention, MHSA


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


def test_attention():
    l, b, d, num_heads = 4, 6, 20, 2

    q, k, v = [torch.rand(b, num_heads, l, d) for _ in range(3)]

    # test no causal
    a1 = attention(q, k, v, causal=False)
    a2 = scaled_dot_product_attention(q, k, v, is_causal=False)
    torch.testing.assert_close(a1, a2)

    # test causal
    a1 = attention(q, k, v, causal=True)
    a2 = scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(a1, a2)
