import torch
import torch.nn as nn

from tllm.attention import MHSA


def test_mhsa():
    l, b, d = 4, 6, 20

    num_heads = 2

    x = torch.rand(b, l, d)

    mhsa = MHSA(d, num_heads=num_heads)

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
