import os

import pytest
import torch
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    Qwen3RotaryEmbedding,
)

from dialectic.llm.generate import generate_from_tokens, qwen_generate_from_chat
from dialectic.llm.qwen import create_qwen, create_qwen_decoder_layer, load_qwen_06b
from dialectic.llm.templates import Message

torch.manual_seed(18)


class MockGenerateModel(torch.nn.Module):
    def __init__(self, token_schedule: list[torch.Tensor], vocab_size: int):
        super().__init__()
        self.token_schedule = token_schedule
        self.vocab_size = vocab_size
        self.step = 0
        self.attn_num_kv_heads = 1
        self.attn_head_d = 1
        self.layers = torch.nn.ModuleList([torch.nn.Identity()])
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, kv_caches=None, attention_mask=None):
        tokens = self.token_schedule[self.step].to(input_ids.device)
        self.step += 1
        batch_size = input_ids.shape[0]
        assert tokens.shape[0] == batch_size
        logits = torch.full(
            (batch_size, 1, self.vocab_size),
            fill_value=-1e4,
            device=input_ids.device,
        )
        logits[torch.arange(batch_size), 0, tokens] = 0.0
        return logits


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

    d1 = create_qwen_decoder_layer(
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

    model = create_qwen(
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
    assert model(x, return_all_logits=True).shape == torch.Size((b, l, vocab_size))

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

    model = create_qwen(
        d=d,
        vocab_size=vocab_size,
        n_decoder_layers=n_decoder_layers,
        attn_head_d=head_d,
        attn_num_heads=num_heads,
        attn_num_kv_heads=num_kv_heads,
        mlp_hidden_d=mlp_hidden_d,
        rope_base_value=rope_base_value,
    ).eval()

    # check we get the same thing if we cache or not
    out_no_cache = generate_from_tokens(
        net=model,
        token_ids=x,
        eos_token_id=-1,
        max_tokens_generated=24,
        use_kv_cache=False,
        sampling_strategy="greedy",
    )

    out_with_cache = generate_from_tokens(
        net=model,
        token_ids=x,
        eos_token_id=-1,
        max_tokens_generated=24,
        use_kv_cache=True,
        sampling_strategy="greedy",
    )

    torch.testing.assert_close(out_no_cache, out_with_cache)
    assert out_with_cache.shape == torch.Size((b, 24 + l))

    # test we get the same thing for a batch or not
    out_singleton = generate_from_tokens(
        net=model,
        token_ids=x[:1],
        eos_token_id=-1,
        max_tokens_generated=24,
        use_kv_cache=False,
        sampling_strategy="greedy",
    )
    assert out_singleton.shape == torch.Size((1, 24 + l))
    torch.testing.assert_close(out_singleton, out_with_cache[:1])


def test_qwen_generate_attention_mask():
    d, head_d, num_heads, num_kv_heads, mlp_hidden_d = 20, 16, 8, 2, 32
    vocab_size = 500
    n_decoder_layers = 3
    rope_base_value = 10000

    model = create_qwen(
        d=d,
        vocab_size=vocab_size,
        n_decoder_layers=n_decoder_layers,
        attn_head_d=head_d,
        attn_num_heads=num_heads,
        attn_num_kv_heads=num_kv_heads,
        mlp_hidden_d=mlp_hidden_d,
        rope_base_value=rope_base_value,
    ).eval()

    x1 = torch.randint(0, vocab_size, size=(1, 4))
    x2 = torch.randint(0, vocab_size, size=(1, 7))

    x_batched = torch.randint(0, vocab_size, size=(2, 7))  # replace with rando...
    # left pad
    x_batched[0, 3:] = x1
    x_batched[1] = x2

    attention_mask = torch.zeros((2, 7), dtype=torch.bool)
    attention_mask[0, :3] = True

    out1 = generate_from_tokens(
        net=model,
        token_ids=x1,
        eos_token_id=-1,
        max_tokens_generated=24,
        use_kv_cache=False,
        sampling_strategy="greedy",
        attention_mask=None,
    )
    out2 = generate_from_tokens(
        net=model,
        token_ids=x2,
        eos_token_id=-1,
        max_tokens_generated=24,
        use_kv_cache=False,
        sampling_strategy="greedy",
        attention_mask=None,
    )
    out_batched = generate_from_tokens(
        net=model,
        token_ids=x_batched,
        eos_token_id=-1,
        max_tokens_generated=24,
        use_kv_cache=False,
        sampling_strategy="greedy",
        attention_mask=attention_mask,
    )

    assert (out1 == out_batched[:1, 3:]).all()
    assert (out2 == out_batched[1:]).all()


def test_qwen_generate_attention_mask_with_kv_cache():
    d, head_d, num_heads, num_kv_heads, mlp_hidden_d = 20, 16, 8, 2, 32
    vocab_size = 500
    n_decoder_layers = 3
    rope_base_value = 10000

    model = create_qwen(
        d=d,
        vocab_size=vocab_size,
        n_decoder_layers=n_decoder_layers,
        attn_head_d=head_d,
        attn_num_heads=num_heads,
        attn_num_kv_heads=num_kv_heads,
        mlp_hidden_d=mlp_hidden_d,
        rope_base_value=rope_base_value,
    ).eval()

    x1 = torch.randint(0, vocab_size, size=(1, 4))
    x2 = torch.randint(0, vocab_size, size=(1, 7))

    x_batched = torch.randint(0, vocab_size, size=(2, 7))  # replace with rando...
    # left pad
    x_batched[0, 3:] = x1
    x_batched[1] = x2

    attention_mask = torch.zeros((2, 7), dtype=torch.bool)
    attention_mask[0, :3] = True

    out1 = generate_from_tokens(
        net=model,
        token_ids=x1,
        eos_token_id=-1,
        max_tokens_generated=24,
        use_kv_cache=True,
        sampling_strategy="greedy",
        attention_mask=None,
    )
    out2 = generate_from_tokens(
        net=model,
        token_ids=x2,
        eos_token_id=-1,
        max_tokens_generated=24,
        use_kv_cache=True,
        sampling_strategy="greedy",
        attention_mask=None,
    )
    out_batched = generate_from_tokens(
        net=model,
        token_ids=x_batched,
        eos_token_id=-1,
        max_tokens_generated=24,
        use_kv_cache=True,
        sampling_strategy="greedy",
        attention_mask=attention_mask,
    )

    assert (out1 == out_batched[:1, 3:]).all()
    assert (out2 == out_batched[1:]).all()


def test_generate_from_tokens_stopping_condition_partial_batch():
    pad_token_id = 0
    eos_token_id = 2
    vocab_size = 12
    token_ids = torch.tensor([[4, 5], [6, 7], [8, 9]])
    token_schedule = [
        torch.tensor([3, 4, eos_token_id]),
        torch.tensor([5, 6, 7]),
        torch.tensor([6, 7, 8]),
        torch.tensor([7, 8, 9]),
    ]

    model = MockGenerateModel(
        token_schedule=token_schedule, vocab_size=vocab_size
    ).eval()

    max_tokens_generated = 4
    output = generate_from_tokens(
        net=model,
        token_ids=token_ids,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        sampling_strategy="greedy",
        max_tokens_generated=max_tokens_generated,
        use_kv_cache=True,
    )

    assert output.shape == torch.Size((3, 2 + max_tokens_generated))
    generated_tokens = output[:, 2:]
    assert (generated_tokens[2] == pad_token_id).all()
    assert (generated_tokens[:2] != pad_token_id).all()


def test_generate_from_tokens_stopping_condition_full_batch():
    pad_token_id = 0
    eos_token_id = 2
    vocab_size = 12
    token_ids = torch.tensor([[4, 5], [6, 7], [8, 9]])
    token_schedule = [
        torch.tensor([eos_token_id, 5, 6]),
        torch.tensor([7, eos_token_id, 8]),
        torch.tensor([9, 10, eos_token_id]),
    ]

    model = MockGenerateModel(
        token_schedule=token_schedule, vocab_size=vocab_size
    ).eval()

    max_tokens_generated = 5
    output = generate_from_tokens(
        net=model,
        token_ids=token_ids,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        sampling_strategy="greedy",
        max_tokens_generated=max_tokens_generated,
        use_kv_cache=True,
    )

    assert output.shape[1] < token_ids.shape[1] + max_tokens_generated
    assert output.shape == torch.Size((3, 4))
    generated_tokens = output[:, 2:]
    assert (generated_tokens[0] == torch.tensor([pad_token_id, pad_token_id])).all()
    assert (generated_tokens[1] == torch.tensor([5, pad_token_id])).all()
    assert (generated_tokens[2] == torch.tensor([6, 8])).all()


def test_generate_from_tokens_stopping_condition_partial_batch_soft():
    pad_token_id = 9
    eos_token_id = 2
    vocab_size = 12
    token_ids = torch.tensor([[4, 5], [6, 7], [8, 9]])
    token_schedule = [
        torch.tensor([3, 4, eos_token_id]),
        torch.tensor([5, 6, 7]),
        torch.tensor([6, 7, 8]),
        torch.tensor([7, 8, 9]),
    ]

    model = MockGenerateModel(
        token_schedule=token_schedule, vocab_size=vocab_size
    ).eval()

    max_tokens_generated = 4
    output = generate_from_tokens(
        net=model,
        token_ids=token_ids,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        sampling_strategy="greedy",
        max_tokens_generated=max_tokens_generated,
        use_kv_cache=True,
        soft_tokens=True,
    )

    assert output.shape == torch.Size((3, 2 + max_tokens_generated, vocab_size))
    generated_tokens = output[:, 2:]

    pad_token_one_hot = torch.zeros(vocab_size)
    pad_token_one_hot[pad_token_id] = 1

    # last element of batch should terminated immediately and just have pad token distribution
    assert (generated_tokens[2] == pad_token_one_hot).all()

    # others should always have no pad token component
    assert (generated_tokens[:2, :, pad_token_id] == 0).all()


def test_generate_from_tokens_stopping_condition_full_batch_soft():
    pad_token_id = 8
    eos_token_id = 2
    vocab_size = 12
    token_ids = torch.tensor([[4, 5], [6, 7], [8, 9]])
    token_schedule = [
        torch.tensor([eos_token_id, 5, 6]),
        torch.tensor([7, eos_token_id, 8]),
        torch.tensor([9, 10, eos_token_id]),
    ]

    model = MockGenerateModel(
        token_schedule=token_schedule, vocab_size=vocab_size
    ).eval()

    max_tokens_generated = 5
    output = generate_from_tokens(
        net=model,
        token_ids=token_ids,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        sampling_strategy="greedy",
        max_tokens_generated=max_tokens_generated,
        use_kv_cache=True,
        soft_tokens=True,
    )

    assert output.shape[1] < token_ids.shape[1] + max_tokens_generated
    assert output.shape == torch.Size((3, 4, vocab_size))
    generated_tokens = output[:, 2:]

    def _create_one_hot(token_id):
        ret = torch.zeros(vocab_size)
        ret[token_id] = 1
        return ret

    pad_token_one_hot = _create_one_hot(pad_token_id)

    assert (
        generated_tokens[0] == torch.stack([pad_token_one_hot, pad_token_one_hot])
    ).all()
    assert (
        generated_tokens[1] == torch.stack([_create_one_hot(5), pad_token_one_hot])
    ).all()
    assert (
        generated_tokens[2] == torch.stack([_create_one_hot(6), _create_one_hot(8)])
    ).all()


def test_qwen_generate_temperature():
    d, head_d, num_heads, num_kv_heads, mlp_hidden_d = 20, 16, 8, 2, 32
    vocab_size = 500
    n_decoder_layers = 3
    rope_base_value = 10000

    model = create_qwen(
        d=d,
        vocab_size=vocab_size,
        n_decoder_layers=n_decoder_layers,
        attn_head_d=head_d,
        attn_num_heads=num_heads,
        attn_num_kv_heads=num_kv_heads,
        mlp_hidden_d=mlp_hidden_d,
        rope_base_value=rope_base_value,
    ).eval()

    x = torch.randint(0, vocab_size, size=(2, 4))

    out_greedy = generate_from_tokens(
        net=model,
        token_ids=x,
        eos_token_id=-1,
        max_tokens_generated=10,
        sampling_strategy="greedy",
    )

    # very low temperature should approximate greedy
    out_low_temp = generate_from_tokens(
        net=model,
        token_ids=x,
        eos_token_id=-1,
        max_tokens_generated=10,
        sampling_strategy="sample",
        temperature=0.001,
    )

    torch.testing.assert_close(out_greedy, out_low_temp)


@pytest.mark.skipif(
    os.getenv("TEST_LLM_AGAINST_HF") is None,
    reason="skipping `test_model_generation_against_qwen_06b` since env variable `TEST_LLM_AGAINST_HF` not set",
)
def test_model_generation_against_qwen_06b():
    """Test model generation against HuggingFace. the expected output was obtained with the code
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    messages = [
        {"role": "user", "content": "Hello who are you?"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)


    outputs = model.generate(**inputs, do_sample=False, max_new_tokens=500)
    print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :]))
    """
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
        messages1 = [Message(role="user", content="Hello who are you?")]
        messages2 = [
            Message(role="user", content="What is the capital of France? /nothink")
        ]
        resp = qwen_generate_from_chat(
            model,
            tokenizer,
            [messages1, messages2],
            sampling_strategy="greedy",
            enable_thinking=True,
        )
        assert (
            resp[0]
            == """user
Hello who are you?
assistant
<think>
Okay, the user asked, "Hello who are you?" I need to respond appropriately. First, I should acknowledge their greeting. Then, I should explain my role as a language model. I should mention that I can assist with various tasks like answering questions, providing information, or helping with specific needs. It's important to keep the response friendly and open-ended to encourage further interaction. I should also make sure the tone is helpful and not too technical. Let me put that together in a natural way.
</think>

Hello! I'm a language model designed to assist with a wide range of tasks, from answering questions to providing information. How can I help you today?"""
        )

        assert (
            resp[1]
            == """user
What is the capital of France? /nothink
assistant
<think>
Okay, the user is asking for the capital of France. I need to make sure I recall the correct answer. France's capital is Paris. Let me think... Yes, Paris is the capital city. I should confirm that there isn't any other city that's considered the capital. For example, maybe some other city has a similar name, but I don't think so. Also, checking my memory, the capital is indeed Paris. I should state that clearly and maybe add a brief note if needed, like mentioning that it's the largest city in France. That should cover the user's question.
</think>

The capital of France is **Paris**."""
        )
