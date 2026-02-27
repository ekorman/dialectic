import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
)

from dialectic.llm.components import apply_rope, create_rope_sine_cosine_tensors


def test_rope_cosine_sine_against_hf():
    b, num_heads, l, d = 4, 1, 2, 32
    x = torch.rand(b, num_heads, l, d)

    conf = Qwen3Config()
    conf.head_dim = d

    rot_emb = Qwen3RotaryEmbedding(conf)
    position_ids = torch.stack([torch.arange(0, l) for _ in range(b)])
    hf_cos, hf_sin = rot_emb(x, position_ids)

    sin, cos = create_rope_sine_cosine_tensors(d, base_value=10000, context_length=l)

    dec_point_tol = 5

    for ours, hfs in [(sin, hf_sin), (cos, hf_cos)]:
        assert ours.shape == torch.Size((1, l, d))
        assert hfs.shape == torch.Size((b, l, d))  # hf's is duplicated across batch

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
