import pytest
import torch

from dialectic.llm.base import BaseTransformer
from dialectic.llm.generate import (
    PreFill,
    check_and_apply_prefill,
    generate_hard_tokens,
    generate_soft_tokens,
)


def test_generate_from_tokens_preserves_eos(MockGenerateModel):
    """EOS token should be present in the generated output, not replaced with pad."""
    vocab_size = 10
    eos_token_id = 2
    pad_token_id = 0

    token_schedule = [
        torch.tensor([5]),
        torch.tensor([6]),
        torch.tensor([eos_token_id]),
    ]

    net = MockGenerateModel(token_schedule=token_schedule, vocab_size=vocab_size).eval()
    result = generate_hard_tokens(
        net=net,
        token_ids=torch.tensor([[1, 3]]),
        sampling_strategy="greedy",
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        use_kv_cache=False,
    )

    assert result.tokens[0].tolist() == [1, 3, 5, 6, eos_token_id]


def test_generate_from_tokens_preserves_eos_batch(MockGenerateModel):
    """In a batch where sequences finish at different times, EOS should be
    preserved for each sequence and padding should only appear after EOS."""
    vocab_size = 10
    eos_token_id = 2
    pad_token_id = 0

    token_schedule = [
        torch.tensor([5, 5, 7]),
        torch.tensor([eos_token_id, 6, pad_token_id]),
        torch.tensor([9, eos_token_id, pad_token_id]),
        torch.tensor([-1, -1, pad_token_id]),
        torch.tensor([-1, -1, eos_token_id]),
    ]

    net = MockGenerateModel(token_schedule=token_schedule, vocab_size=vocab_size)
    result = generate_hard_tokens(
        net=net,
        token_ids=torch.tensor([[1, 3], [pad_token_id, 3], [5, 7]]),
        sampling_strategy="greedy",
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        use_kv_cache=False,
    )

    assert result.tokens[0].tolist() == [
        1,
        3,
        5,
        eos_token_id,
        pad_token_id,
        pad_token_id,
        pad_token_id,
    ]
    assert result.tokens[1].tolist() == [
        pad_token_id,
        3,
        5,
        6,
        eos_token_id,
        pad_token_id,
        pad_token_id,
    ]
    assert result.tokens[2].tolist() == [
        5,
        7,
        7,
        pad_token_id,
        pad_token_id,
        pad_token_id,
        eos_token_id,
    ]


@torch.inference_mode()
def test_soft_tokens_forward_pass(tiny_model: BaseTransformer):
    """Tests that the forward pass gives the same thing if we pass an integer tensor
    of token ids, the corresponding one-hot float tensor, or pre-computed embeddings
    """
    b, l = 4, 6
    token_ids = torch.randint(0, tiny_model.vocab_size, (b, l))
    one_hot_tokens = torch.nn.functional.one_hot(
        token_ids, tiny_model.vocab_size
    ).float()
    assert one_hot_tokens.shape == torch.Size((b, l, tiny_model.vocab_size))

    embeddings = tiny_model.embed_tokens(token_ids)

    out1 = tiny_model(token_ids)
    out2 = tiny_model(one_hot_tokens)
    out3 = tiny_model(embeddings)

    torch.testing.assert_close(out1, out2)
    torch.testing.assert_close(out1, out3)


@torch.inference_mode()
def test_soft_tokens_generation(tiny_model: BaseTransformer):
    b, l = 4, 6

    input_token_ids = torch.randint(0, tiny_model.vocab_size, (b, l))
    output = generate_soft_tokens(
        net=tiny_model,
        token_ids=input_token_ids,
        eos_token_id=-1,
        pad_token_id=2,
        max_tokens_generated=24,
        use_kv_cache=True,
    )

    assert output.embeddings.shape == torch.Size((b, l + 24, tiny_model.d))
    assert output.shadow_ids.shape == torch.Size((b, l + 24))


def test_generate_from_tokens_stopping_condition_partial_batch_soft(MockGenerateModel):
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
    output = generate_soft_tokens(
        net=model,
        token_ids=token_ids,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        max_tokens_generated=max_tokens_generated,
        use_kv_cache=True,
    )

    assert output.shadow_ids.shape == torch.Size((3, 2 + max_tokens_generated))
    generated_ids = output.shadow_ids[:, 2:]

    assert generated_ids[2, 0] == eos_token_id
    assert (generated_ids[2, 1:] == pad_token_id).all()

    assert (generated_ids[:2] != pad_token_id).all()


def test_generate_from_tokens_stopping_condition_full_batch_soft(MockGenerateModel):
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
    output = generate_soft_tokens(
        net=model,
        token_ids=token_ids,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        max_tokens_generated=max_tokens_generated,
        use_kv_cache=True,
    )

    assert output.shadow_ids.shape[1] < token_ids.shape[1] + max_tokens_generated
    assert output.shadow_ids.shape == torch.Size((3, 5))
    generated_ids = output.shadow_ids[:, 2:]

    assert generated_ids[0].tolist() == [eos_token_id, pad_token_id, pad_token_id]
    assert generated_ids[1].tolist() == [5, eos_token_id, pad_token_id]
    assert generated_ids[2].tolist() == [6, 8, eos_token_id]


def test_prefill_pos():
    prefill = PreFill(
        condition=torch.tensor([3, 2, 4]), filling=torch.tensor([30, 100])
    )

    token_ids = torch.tensor(
        [
            [1, 2, 3, 5],
            [1, 3, 2, 4],
            [0, 2, 4, 7],
        ]
    )

    prefilled, attention_mask = check_and_apply_prefill(
        token_ids=token_ids,
        prefill=prefill,
        pad_token_id=-1,
        attention_mask=torch.ones_like(token_ids, dtype=torch.bool),
    )

    assert (
        prefilled
        == torch.tensor(
            [
                [1, 2, 3, 5, -1, -1],
                [1, 3, 2, 4, 30, 100],
                [0, 2, 4, 7, -1, -1],
            ]
        )
    ).all()

    assert (
        attention_mask
        == torch.tensor(
            [[True] * 4 + [False, False], [True] * 6, [True] * 4 + [False, False]]
        )
    ).all()


def test_prefill_no_op():
    prefill = PreFill(condition=torch.tensor([2, 10]), filling=torch.tensor([30, 100]))

    token_ids = torch.tensor(
        [
            [1, 2, 3, 5],
            [1, 3, 2, 4],
            [0, 2, 4, 7],
        ]
    )

    prefilled, _ = check_and_apply_prefill(
        token_ids=token_ids, prefill=prefill, pad_token_id=-1, attention_mask=None
    )

    assert (prefilled == token_ids).all()


def test_generate_with_prefill_and_attention_mask(tiny_model: BaseTransformer):
    """When using prefill with an attention mask (left-padded batch), prefill
    should fire for the element that hits the condition, inserting fill tokens
    that differ from the natural continuation. The other batch element should
    be unaffected.
    """

    torch.manual_seed(42)
    model = tiny_model.eval()
    vocab_size = model.vocab_size
    pad_token_id = 0
    eos_token_id = -1

    x1 = torch.randint(1, vocab_size, size=(1, 4))
    x2 = torch.randint(1, vocab_size, size=(1, 7))

    out1_natural = generate_hard_tokens(
        net=model,
        token_ids=x1,
        sampling_strategy="greedy",
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        max_tokens_generated=20,
        use_kv_cache=True,
    ).tokens

    generated1 = out1_natural[0, x1.shape[1] :]
    trigger = generated1[:3]
    natural_fill = generated1[3:5]
    fill = (natural_fill + 1) % vocab_size
    fill = torch.where(fill == pad_token_id, (fill + 1) % vocab_size, fill)

    prefill = PreFill(condition=trigger, filling=fill)

    out1 = generate_hard_tokens(
        net=model,
        token_ids=x1,
        sampling_strategy="greedy",
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        max_tokens_generated=20,
        use_kv_cache=True,
        prefill=prefill,
    ).tokens

    assert (out1[0, x1.shape[1] + 3 : x1.shape[1] + 5] == fill).all(), (
        "Prefill did not fire: fill tokens not found after trigger"
    )

    x_batched = torch.full((2, 7), pad_token_id, dtype=torch.long)
    x_batched[0, 3:] = x1
    x_batched[1] = x2

    attention_mask = torch.ones(2, 7, dtype=torch.bool)
    attention_mask[0, :3] = False

    out_batched = generate_hard_tokens(
        net=model,
        token_ids=x_batched,
        sampling_strategy="greedy",
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        max_tokens_generated=20,
        use_kv_cache=True,
        attention_mask=attention_mask,
        prefill=prefill,
    ).tokens

    assert (out1[0] == out_batched[0, 3:]).all(), (
        "Batch element 0 (with prefill) doesn't match individual generation"
    )


def test_generate_with_prefill_without_kv_cache(tiny_model: BaseTransformer):
    """Without KV cache, the full sequence is passed each time. This tests that
    prefill correctly inserts fill tokens that differ from the natural continuation.
    """

    torch.manual_seed(42)
    model = tiny_model.eval()
    vocab_size = model.vocab_size
    pad_token_id = 0
    eos_token_id = -1

    x = torch.randint(1, vocab_size, size=(1, 4))

    out_natural = generate_hard_tokens(
        net=model,
        token_ids=x,
        sampling_strategy="greedy",
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        max_tokens_generated=20,
        use_kv_cache=False,
    ).tokens

    generated = out_natural[0, x.shape[1] :]
    trigger = generated[:3]
    natural_fill = generated[3:5]
    fill = (natural_fill + 1) % vocab_size
    fill = torch.where(fill == pad_token_id, (fill + 1) % vocab_size, fill)

    prefill = PreFill(condition=trigger, filling=fill)

    out_prefill = generate_hard_tokens(
        net=model,
        token_ids=x,
        sampling_strategy="greedy",
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        max_tokens_generated=20,
        use_kv_cache=False,
        prefill=prefill,
    ).tokens

    assert (
        out_prefill[0, : x.shape[1] + 3] == out_natural[0, : x.shape[1] + 3]
    ).all(), "Output before trigger should match natural generation"
    assert (out_prefill[0, x.shape[1] + 3 : x.shape[1] + 5] == fill).all(), (
        "Prefill did not fire: fill tokens not found after trigger"
    )


def test_soft_generator_switch_to_hard_tokens(MockGenerateModel):
    """Once the switch condition fires, ALL subsequent tokens must be hard
    even when the shadow sequence no longer ends with the condition (latching).

    Uses a mock with distinct scheduled tokens so the condition provably cannot
    re-trigger by coincidence — any post-switch hard token proves latching works.
    """
    vocab_size = 20
    schedule = [
        torch.tensor([5]),
        torch.tensor([3]),
        torch.tensor([7]),
        torch.tensor([12]),
        torch.tensor([9]),
        torch.tensor([1]),
        torch.tensor([15]),
        torch.tensor([4]),
        torch.tensor([8]),
        torch.tensor([6]),
    ]

    net = MockGenerateModel(token_schedule=list(schedule), vocab_size=vocab_size).eval()
    x = torch.tensor([[10, 11, 13, 14]])
    switch_condition = torch.tensor([5, 3, 7])

    out = generate_soft_tokens(
        net=net,
        token_ids=x,
        eos_token_id=-1,
        pad_token_id=0,
        max_tokens_generated=10,
        use_kv_cache=True,
        switch_to_hard_tokens_condition=switch_condition,
    )

    shadow = out.shadow_ids[0, x.shape[1] :]

    for i in range(1, len(shadow) - len(switch_condition) + 1):
        assert not (shadow[i : i + len(switch_condition)] == switch_condition).all(), (
            f"Condition re-triggers at position {i} — test cannot verify latching"
        )

    prompt_len = x.shape[1]
    mask = out.hard_tokens_mask[0]
    assert mask[:prompt_len].all(), "Prompt tokens should be hard"
    assert not mask[prompt_len : prompt_len + 3].any(), (
        "First 3 generated tokens should be soft (before condition met)"
    )
    assert mask[prompt_len + 3 :].all(), (
        "All tokens after switch should be hard (latching)"
    )

    W = net.embed_tokens.weight
    for i in range(3, 10):
        expected_emb = W[shadow[i]].float()
        actual_emb = out.embeddings[0, prompt_len + i].float()
        torch.testing.assert_close(
            actual_emb,
            expected_emb,
            msg=f"Token {i} embedding should match W[shadow_id] (hard token)",
        )


def test_soft_generator_switch_to_hard_tokens_batch(tiny_model: BaseTransformer):
    """In a batch, different elements may hit the switch condition at different
    times. Each element should switch independently.
    """

    torch.manual_seed(42)
    model = tiny_model.eval()
    vocab_size = model.vocab_size

    x1 = torch.randint(1, vocab_size, size=(1, 4))
    x2 = torch.randint(1, vocab_size, size=(1, 4))

    out1 = generate_soft_tokens(
        net=model,
        token_ids=x1,
        eos_token_id=-1,
        pad_token_id=0,
        max_tokens_generated=10,
        use_kv_cache=True,
    )
    shadow1 = out1.shadow_ids[0, x1.shape[1] :]

    out2 = generate_soft_tokens(
        net=model,
        token_ids=x2,
        eos_token_id=-1,
        pad_token_id=0,
        max_tokens_generated=10,
        use_kv_cache=True,
    )
    shadow2 = out2.shadow_ids[0, x2.shape[1] :]

    switch_condition = shadow1[:2]

    switch_step_1 = 2
    switch_step_2 = None
    for i in range(10 - len(switch_condition)):
        if (shadow2[i : i + len(switch_condition)] == switch_condition).all():
            switch_step_2 = i + len(switch_condition)
            break

    x_batched = torch.cat([x1, x2], dim=0)

    out_batched = generate_soft_tokens(
        net=model,
        token_ids=x_batched,
        eos_token_id=-1,
        pad_token_id=0,
        max_tokens_generated=10,
        use_kv_cache=True,
        switch_to_hard_tokens_condition=switch_condition,
    )

    mask_batched = out_batched.hard_tokens_mask[:, x1.shape[1] :]

    for i in range(switch_step_1):
        assert not mask_batched[0, i], (
            f"Batch 0, token {i}: should be soft (before switch)"
        )
    for i in range(switch_step_1, 10):
        assert mask_batched[0, i], f"Batch 0, token {i}: should be hard (after switch)"

    if switch_step_2 is not None:
        for i in range(switch_step_2):
            assert not mask_batched[1, i], (
                f"Batch 1, token {i}: should be soft (before switch)"
            )
        for i in range(switch_step_2, 10):
            assert mask_batched[1, i], (
                f"Batch 1, token {i}: should be hard (after switch)"
            )
    else:
        for i in range(10):
            assert not mask_batched[1, i], (
                f"Batch 1, token {i}: should be soft (never switched)"
            )


def test_soft_generator_no_switch_without_condition(tiny_model: BaseTransformer):
    """Without a switch condition, all generated tokens should remain soft
    (hard_tokens_mask should be False for all generated positions).
    """

    torch.manual_seed(42)
    model = tiny_model.eval()
    vocab_size = model.vocab_size

    x = torch.randint(1, vocab_size, size=(1, 4))

    out = generate_soft_tokens(
        net=model,
        token_ids=x,
        eos_token_id=-1,
        pad_token_id=0,
        max_tokens_generated=10,
        use_kv_cache=True,
    )

    mask = out.hard_tokens_mask[0, x.shape[1] :]

    for i in range(10):
        assert not mask[i], f"Token {i} should be soft (no switch condition set)"


def test_soft_token_argmax_matches_hard_greedy(tiny_model: BaseTransformer):
    """The shadow_ids of soft tokens should equal greedy hard token generation.

    Both process identical logits on the first step (same prompt embeddings).
    From step 2 onward, soft feeds back a mixture embedding while hard feeds
    back an exact embedding, so they may diverge — but the first generated
    token must always match.
    """
    torch.manual_seed(42)
    model = tiny_model.eval()
    V = model.vocab_size

    x = torch.randint(1, V, size=(1, 4))

    hard_out = generate_hard_tokens(
        net=model,
        token_ids=x.clone(),
        sampling_strategy="greedy",
        eos_token_id=-1,
        pad_token_id=0,
        max_tokens_generated=5,
        use_kv_cache=True,
        temperature=1.0,
    )

    soft_out = generate_soft_tokens(
        net=model,
        token_ids=x.clone(),
        eos_token_id=-1,
        pad_token_id=0,
        max_tokens_generated=5,
        use_kv_cache=True,
        temperature=1.0,
    )

    hard_generated = hard_out.tokens[0, x.shape[1] :]
    soft_shadow = soft_out.shadow_ids[0, x.shape[1] :]

    assert hard_generated[0] == soft_shadow[0], (
        f"First generated token should match: hard={hard_generated[0].item()}, "
        f"soft shadow={soft_shadow[0].item()}"
    )


def test_soft_generator_hard_tokens_mask_without_switch_condition(
    tiny_model: BaseTransformer,
):
    """Without a switch condition, prompt tokens should be marked hard and
    generated tokens should be marked soft in hard_tokens_mask."""

    torch.manual_seed(42)
    model = tiny_model.eval()
    vocab_size = model.vocab_size
    prompt_len = 4

    x = torch.randint(1, vocab_size, size=(1, prompt_len))
    n_gen = 10

    out = generate_soft_tokens(
        net=model,
        token_ids=x,
        eos_token_id=-1,
        pad_token_id=0,
        max_tokens_generated=n_gen,
        use_kv_cache=True,
    )

    assert out.hard_tokens_mask[0, :prompt_len].all(), "Prompt positions should be hard"
    assert not out.hard_tokens_mask[0, prompt_len:].any(), (
        "Generated positions should be soft when no switch condition is set"
    )


def test_soft_generate_without_kv_cache_raises(tiny_model: BaseTransformer):
    """Soft generation without KV cache is unsupported due to noise accumulation."""
    model = tiny_model.eval()
    x = torch.randint(1, model.vocab_size, size=(1, 4))

    with pytest.raises(ValueError, match="use_kv_cache=True"):
        generate_soft_tokens(
            net=model,
            token_ids=x,
            eos_token_id=-1,
            pad_token_id=0,
            max_tokens_generated=5,
            use_kv_cache=False,
        )


def test_soft_generate_with_prefill_and_attention_mask(tiny_model: BaseTransformer):
    """Soft generator prefill with a left-padded batch should fire for the
    element that hits the condition, and the batched output for that element
    should match its individual generation.
    """
    torch.manual_seed(42)
    model = tiny_model.eval()
    vocab_size = model.vocab_size
    pad_token_id = 0
    eos_token_id = -1

    x1 = torch.randint(1, vocab_size, size=(1, 4))
    x2 = torch.randint(1, vocab_size, size=(1, 7))

    out1_natural = generate_soft_tokens(
        net=model,
        token_ids=x1,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        max_tokens_generated=20,
        use_kv_cache=True,
    )

    shadow_tokens = out1_natural.shadow_ids[0, x1.shape[1] :]
    trigger = shadow_tokens[:3]
    natural_fill = shadow_tokens[3:5]
    fill = (natural_fill + 1) % vocab_size
    fill = torch.where(fill == pad_token_id, (fill + 1) % vocab_size, fill)

    prefill = PreFill(condition=trigger, filling=fill)

    out1 = generate_soft_tokens(
        net=model,
        token_ids=x1,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        max_tokens_generated=20,
        use_kv_cache=True,
        prefill=prefill,
    )

    assert (out1.shadow_ids[0, x1.shape[1] + 3 : x1.shape[1] + 5] == fill).all(), (
        "Prefill did not fire: fill tokens not found after trigger"
    )

    x_batched = torch.full((2, 7), pad_token_id, dtype=torch.long)
    x_batched[0, 3:] = x1
    x_batched[1] = x2

    attention_mask = torch.ones(2, 7, dtype=torch.bool)
    attention_mask[0, :3] = False

    out_batched = generate_soft_tokens(
        net=model,
        token_ids=x_batched,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        max_tokens_generated=20,
        use_kv_cache=True,
        attention_mask=attention_mask,
        prefill=prefill,
    )

    torch.testing.assert_close(
        out1.embeddings[0],
        out_batched.embeddings[0, 3:],
        msg="Batch element 0 (with prefill) doesn't match individual generation",
    )


def test_soft_generator_noise_shape(tiny_model: BaseTransformer):
    """Verify that embeddings include noise: generated-position embeddings
    should differ from the noiseless W[shadow_id] embeddings.
    """
    torch.manual_seed(42)
    model = tiny_model.eval()
    vocab_size = model.vocab_size
    b, prompt_len = 2, 4
    n_gen = 10

    x = torch.randint(1, vocab_size, size=(b, prompt_len))
    out = generate_soft_tokens(
        net=model,
        token_ids=x,
        eos_token_id=-1,
        pad_token_id=0,
        max_tokens_generated=n_gen,
        use_kv_cache=True,
        soft_token_noise_std=0.1,
    )

    assert out.embeddings.shape == (b, prompt_len + n_gen, model.d)
    assert out.shadow_ids.shape == (b, prompt_len + n_gen)

    with torch.no_grad():
        W = model.embed_tokens.weight
        noiseless = W[out.shadow_ids[:, prompt_len:]].float()
        actual = out.embeddings[:, prompt_len:].float()
    assert not torch.allclose(actual, noiseless, atol=1e-6), (
        "Generated embeddings should differ from noiseless W[shadow_id] when noise is applied"
    )


def test_soft_generator_prompt_noise_is_zero(tiny_model: BaseTransformer):
    """Prompt is processed as 2D int tokens during generation, so noise is not
    applied to prompt embeddings. The returned embeddings at prompt positions
    should match W[shadow_ids] exactly (no noise).
    """
    torch.manual_seed(42)
    model = tiny_model.eval()
    vocab_size = model.vocab_size
    b, prompt_len = 2, 6
    n_gen = 10

    x = torch.randint(1, vocab_size, size=(b, prompt_len))
    out = generate_soft_tokens(
        net=model,
        token_ids=x,
        eos_token_id=-1,
        pad_token_id=0,
        max_tokens_generated=n_gen,
        use_kv_cache=True,
        soft_token_noise_std=0.5,
    )

    with torch.no_grad():
        W = model.embed_tokens.weight
        prompt_embeddings = out.embeddings[:, :prompt_len].float()
        expected_prompt_embeddings = W[out.shadow_ids[:, :prompt_len]].float()
        torch.testing.assert_close(
            prompt_embeddings,
            expected_prompt_embeddings,
            msg="Prompt embeddings should equal W[shadow_ids] (no noise applied)",
        )

        gen_embeddings = out.embeddings[:, prompt_len:].float()
        expected_gen_embeddings = W[out.shadow_ids[:, prompt_len:]].float()
    assert not torch.allclose(gen_embeddings, expected_gen_embeddings, atol=1e-6), (
        "Generated embeddings should differ from W[shadow_ids] (noise applied)"
    )


def test_soft_generator_logits_consistent_with_zero_prompt_noise(
    tiny_model: BaseTransformer,
):
    """Verify that a forward pass with the prompt embeddings from generation
    output produces the same logits as a forward pass with 2D int prompt
    tokens. This confirms: the D-dim recomputation path matches what happened
    during generation.
    """
    torch.manual_seed(42)
    model = tiny_model.eval()
    vocab_size = model.vocab_size
    b, prompt_len = 1, 6
    n_gen = 5

    x = torch.randint(1, vocab_size, size=(b, prompt_len))
    out = generate_soft_tokens(
        net=model,
        token_ids=x,
        eos_token_id=-1,
        pad_token_id=0,
        max_tokens_generated=n_gen,
        use_kv_cache=True,
        soft_token_noise_std=0.3,
    )

    with torch.no_grad():
        logits_2d = model(x, return_all_logits=True)
        first_completion_logit_2d = logits_2d[:, -1:]

        prompt_embeddings = out.embeddings[:, :prompt_len]
        logits_d = model(
            prompt_embeddings,
            return_all_logits=True,
        )
        first_completion_logit_d = logits_d[:, -1:]

    torch.testing.assert_close(
        first_completion_logit_2d,
        first_completion_logit_d,
        msg="Logits at first completion position should match between 2D int "
        "prompt (generation path) and D-dim embeddings (recomputation path)",
    )


def test_soft_generate_with_switch_condition_and_attention_mask(
    tiny_model: BaseTransformer,
):
    torch.manual_seed(20)
    model = tiny_model.eval()
    vocab_size = model.vocab_size
    pad_token_id = 0
    eos_token_id = vocab_size - 1

    x1 = torch.randint(1, vocab_size, size=(1, 4))
    x2 = torch.randint(1, vocab_size, size=(1, 7))

    x_batched = torch.full((2, 7), pad_token_id, dtype=torch.long)
    x_batched[0, 3:] = x1
    x_batched[1] = x2

    attention_mask = torch.ones(2, 7, dtype=torch.bool)
    attention_mask[0, :3] = False

    switch_condition = torch.tensor([3, 5, 7])
    max_tokens_prefill = torch.tensor([3, 5, 7])

    out = generate_soft_tokens(
        net=model,
        token_ids=x_batched,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        max_tokens_generated=50,
        use_kv_cache=True,
        attention_mask=attention_mask,
        temperature=0.5,
        use_bf16=False,
        switch_to_hard_tokens_condition=switch_condition,
        max_tokens_prefill=max_tokens_prefill,
        max_tokens_prefill_steps_before_end=10,
        soft_token_noise_std=0.1,
        min_soft_steps=0,
    )

    assert out.embeddings.ndim == 3
    assert out.embeddings.shape[0] == 2
    assert out.embeddings.shape[2] == model.d


def test_soft_generator_no_noise_without_std(tiny_model: BaseTransformer):
    """Without noise_std, prompt embeddings should equal W[shadow_ids] and
    generated soft embeddings should be mixture embeddings (softmax @ W),
    which differ from W[argmax].
    """
    torch.manual_seed(42)
    model = tiny_model.eval()
    vocab_size = model.vocab_size
    b, prompt_len = 2, 4

    x = torch.randint(1, vocab_size, size=(b, prompt_len))
    out = generate_soft_tokens(
        net=model,
        token_ids=x,
        eos_token_id=-1,
        pad_token_id=0,
        max_tokens_generated=5,
        use_kv_cache=True,
    )

    with torch.no_grad():
        W = model.embed_tokens.weight
        prompt_expected = W[out.shadow_ids[:, :prompt_len]].float()
        torch.testing.assert_close(
            out.embeddings[:, :prompt_len].float(),
            prompt_expected,
            msg="Prompt embeddings should equal W[shadow_ids]",
        )

        gen_expected = W[out.shadow_ids[:, prompt_len:]].float()
        gen_actual = out.embeddings[:, prompt_len:].float()
    assert not torch.allclose(gen_actual, gen_expected, atol=1e-6), (
        "Soft generated embeddings are mixture embeddings (softmax @ W), "
        "should differ from W[argmax]"
    )
