import os

import pytest
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from torch.nn.functional import scaled_dot_product_attention
from transformers import AutoModelForCausalLM
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    Qwen3MLP,
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
)

from dialectic.qwen import (
    MHSA,
    GatedMLP,
    Qwen,
    QwenDecoderLayer,
    apply_rope,
    attention,
    create_rope_sine_cosine_tensors,
    generate_from_chat,
    generate_from_tokens,
    load_qwen_06b,
)
from dialectic.tokenizer import Message

torch.manual_seed(18)


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


def test_rope_cosine_sine_against_hf():
    b, num_heads, l, d = 4, 1, 2, 32
    x = torch.rand(b, num_heads, l, d)

    conf = Qwen3Config()
    conf.head_dim = d

    rot_emb = Qwen3RotaryEmbedding(conf)
    position_ids = torch.stack([torch.arange(0, l) for _ in range(b)])
    hf_cos, hf_sin = rot_emb(x, position_ids)

    sin, cos = create_rope_sine_cosine_tensors(d, base_value=10000, context_length=l)

    dec_point_tol = 6

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


def test_qwen_decoder_layer():
    l, b, d, head_d, num_heads, num_kv_heads, mlp_hidden_d = 4, 6, 20, 16, 8, 2, 32
    x = torch.rand(b, l, d)

    conf = Qwen3Config()
    conf.hidden_size = d
    conf.intermediate_size = mlp_hidden_d
    conf.head_dim = head_d
    conf.num_key_value_heads = num_kv_heads
    conf.num_attention_heads = num_heads
    conf._attn_implementation = "sdpa"

    rot_emb = Qwen3RotaryEmbedding(conf)

    d1 = QwenDecoderLayer(
        d,
        attn_head_d=head_d,
        attn_num_heads=num_heads,
        attn_num_kv_heads=num_kv_heads,
        mlp_hidden_d=mlp_hidden_d,
        rope_base_value=10000,
    )
    d2 = Qwen3DecoderLayer(conf, 0)

    d1.load_state_dict(d2.state_dict())

    position_ids = torch.stack([torch.arange(0, l) for _ in range(b)])
    torch.testing.assert_close(
        d1(x),
        d2(x, position_embeddings=rot_emb(x, position_ids=position_ids)),
    )


def test_qwen():
    l, b, d, head_d, num_heads, num_kv_heads, mlp_hidden_d = 4, 6, 20, 16, 8, 2, 32
    vocab_size = 500
    n_decoder_layers = 3
    rope_base_value = 10000
    x = torch.randint(0, vocab_size, size=(b, l))

    model = Qwen(
        d=d,
        vocab_size=vocab_size,
        n_decoder_layers=n_decoder_layers,
        attn_head_d=head_d,
        attn_num_heads=num_heads,
        attn_num_kv_heads=num_kv_heads,
        mlp_hidden_d=mlp_hidden_d,
        rope_base_value=rope_base_value,
    ).eval()

    assert model(x).shape == torch.Size((b, 1, vocab_size))

    # check against a huggingface defined net
    conf = Qwen3Config()
    conf.head_dim = head_d
    conf.num_key_value_heads = num_kv_heads
    conf.num_attention_heads = num_heads
    conf.hidden_size = d
    conf.num_hidden_layers = n_decoder_layers
    conf.vocab_size = vocab_size
    conf.intermediate_size = mlp_hidden_d
    conf._attn_implementation = "sdpa"

    hf_model = Qwen3ForCausalLM(conf).eval()

    def map_key(k: str):
        if not k.startswith("lm_head"):
            return "model." + k

        return k

    hf_model.load_state_dict({map_key(k): v for k, v in model.state_dict().items()})

    with torch.inference_mode():
        out1 = model(x)
        out2 = hf_model(x)

    torch.testing.assert_close(out1, out2.logits[:, -1:])


def test_qwen_generate():
    l, b, d, head_d, num_heads, num_kv_heads, mlp_hidden_d = 4, 6, 20, 16, 8, 2, 32
    vocab_size = 500
    n_decoder_layers = 3
    rope_base_value = 10000
    x = torch.randint(0, vocab_size, size=(b, l))

    model = Qwen(
        d=d,
        vocab_size=vocab_size,
        n_decoder_layers=n_decoder_layers,
        attn_head_d=head_d,
        attn_num_heads=num_heads,
        attn_num_kv_heads=num_kv_heads,
        mlp_hidden_d=mlp_hidden_d,
        rope_base_value=rope_base_value,
    ).eval()

    out_no_cache = generate_from_tokens(
        net=model,
        token_ids=x,
        eos_token_id=-1,
        max_tokens_generated=24,
        use_kv_cache=False,
    )

    out_with_cache = generate_from_tokens(
        net=model,
        token_ids=x,
        eos_token_id=-1,
        max_tokens_generated=24,
        use_kv_cache=False,
    )

    torch.testing.assert_close(out_no_cache, out_with_cache)
    assert out_with_cache.shape == torch.Size((b, 24 + l))


@pytest.mark.skipif(
    os.getenv("TEST_LLM_AGAINST_HF") is None,
    reason="skipping `test_load_qwen_06b` since env variable `TEST_LLM_AGAINST_HF` not set",
)
def test_load_qwen_06b():
    model = load_qwen_06b()

    hf_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", dtype=torch.float32
    ).eval()

    def map_key(k: str):
        if k.startswith("model"):
            return k[6:]
        return k

    model.load_state_dict({map_key(k): v for k, v in hf_model.state_dict().items()})
    model.eval()

    x = torch.randint(0, hf_model.config.vocab_size, size=(1, 10))

    with torch.inference_mode():
        torch.testing.assert_close(
            model(x), hf_model(x).logits[:, -1:], atol=1e-4, rtol=1e-4
        )

        tokenizer: Tokenizer = Tokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        messages = [Message(role="user", content="Hello who are you?")]
        resp = generate_from_chat(model, tokenizer, messages)
        assert (
            resp[0]
            == """user
Hello who are you?
assistant
<think>
Okay, the user asked, "Hello who are you?" I need to respond appropriately. First, I should acknowledge their greeting. Then, I should explain my role as a language model. I should mention that I can assist with various tasks like answering questions, providing information, or helping with specific needs. It's important to keep the response friendly and open-ended to encourage further interaction. I should also make sure the tone is helpful and not too technical. Let me put that together in a natural way.
</think>

Hello! I'm a language model designed to assist with a wide range of tasks, from answering questions to providing information. How can I help you today?
"""
        )
