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
    of token ids or the corresponding one-hot float tensor
    """
    b, l = 4, 6
    token_ids = torch.randint(0, tiny_model.vocab_size, (b, l))
    one_hot_tokens = torch.nn.functional.one_hot(
        token_ids, tiny_model.vocab_size
    ).float()
    assert one_hot_tokens.shape == torch.Size((b, l, tiny_model.vocab_size))

    out1 = tiny_model(token_ids)
    out2 = tiny_model(one_hot_tokens)

    torch.testing.assert_close(out1, out2)


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

    assert output.tokens.shape == torch.Size((b, l + 24, tiny_model.vocab_size))


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

    assert output.tokens.shape == torch.Size((3, 2 + max_tokens_generated, vocab_size))
    generated_tokens = output.tokens[:, 2:]

    pad_token_one_hot = torch.nn.functional.one_hot(
        torch.tensor(pad_token_id), vocab_size
    )
    eos_token_one_hot = torch.nn.functional.one_hot(
        torch.tensor(eos_token_id), vocab_size
    )

    # last element of batch should terminated immediately and just have pad token distribution
    assert (generated_tokens[2, 0] == eos_token_one_hot).all()
    assert (generated_tokens[2, 1:] == pad_token_one_hot).all()

    # others should always have no pad token component
    assert (generated_tokens[:2, :, pad_token_id] == 0).all()


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

    assert output.tokens.shape[1] < token_ids.shape[1] + max_tokens_generated
    assert output.tokens.shape == torch.Size((3, 5, vocab_size))
    generated_tokens = output.tokens[:, 2:]

    def _create_one_hot(token_id):
        return torch.nn.functional.one_hot(torch.tensor(token_id), vocab_size)

    pad_token_one_hot = _create_one_hot(pad_token_id)
    eos_token_one_hot = _create_one_hot(eos_token_id)

    assert (
        generated_tokens[0]
        == torch.stack([eos_token_one_hot, pad_token_one_hot, pad_token_one_hot])
    ).all()
    assert (
        generated_tokens[1]
        == torch.stack([_create_one_hot(5), eos_token_one_hot, pad_token_one_hot])
    ).all()
    assert (
        generated_tokens[2]
        == torch.stack([_create_one_hot(6), _create_one_hot(8), eos_token_one_hot])
    ).all()


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


def _is_one_hot(tensor: torch.Tensor, tol: float = 1e-6) -> bool:
    """Check if a tensor is approximately one-hot encoded."""
    max_val = tensor.max()
    num_ones = (tensor > 1 - tol).sum()
    num_zeros = (tensor < tol).sum()
    return abs(max_val - 1.0) < tol and num_ones == 1 and num_zeros == len(tensor) - 1


def test_soft_generator_switch_to_hard_tokens(tiny_model: BaseTransformer):
    """After the switch condition is met, generated tokens should be one-hot
    (hard) instead of soft probability distributions.
    """

    torch.manual_seed(42)
    model = tiny_model.eval()
    vocab_size = model.vocab_size

    x = torch.randint(1, vocab_size, size=(1, 4))

    out_no_switch = generate_soft_tokens(
        net=model,
        token_ids=x,
        eos_token_id=-1,
        pad_token_id=0,
        max_tokens_generated=10,
        use_kv_cache=True,
    ).tokens

    generated_no_switch = out_no_switch[0, x.shape[1] :]
    shadow_tokens = generated_no_switch.argmax(-1)
    switch_condition = shadow_tokens[:3]

    out_with_switch = generate_soft_tokens(
        net=model,
        token_ids=x,
        eos_token_id=-1,
        pad_token_id=0,
        max_tokens_generated=10,
        use_kv_cache=True,
        switch_to_hard_tokens_condition=switch_condition,
    ).tokens

    generated_with_switch = out_with_switch[0, x.shape[1] :]

    for i in range(3):
        token_dist = generated_with_switch[i]
        assert not _is_one_hot(token_dist), (
            f"Token {i} should be soft (before switch condition met)"
        )

    for i in range(3, 10):
        token_dist = generated_with_switch[i]
        assert _is_one_hot(token_dist), (
            f"Token {i} should be one-hot (after switch condition met)"
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
    ).tokens
    shadow1 = out1[0, x1.shape[1] :].argmax(-1)

    out2 = generate_soft_tokens(
        net=model,
        token_ids=x2,
        eos_token_id=-1,
        pad_token_id=0,
        max_tokens_generated=10,
        use_kv_cache=True,
    ).tokens
    shadow2 = out2[0, x2.shape[1] :].argmax(-1)

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
    ).tokens

    generated_batched = out_batched[:, x1.shape[1] :]

    for i in range(switch_step_1):
        assert not _is_one_hot(generated_batched[0, i]), (
            f"Batch 0, token {i}: should be soft (before switch)"
        )
    for i in range(switch_step_1, 10):
        assert _is_one_hot(generated_batched[0, i]), (
            f"Batch 0, token {i}: should be hard (after switch)"
        )

    if switch_step_2 is not None:
        for i in range(switch_step_2):
            assert not _is_one_hot(generated_batched[1, i]), (
                f"Batch 1, token {i}: should be soft (before switch)"
            )
        for i in range(switch_step_2, 10):
            assert _is_one_hot(generated_batched[1, i]), (
                f"Batch 1, token {i}: should be hard (after switch)"
            )
    else:
        for i in range(10):
            assert not _is_one_hot(generated_batched[1, i]), (
                f"Batch 1, token {i}: should be soft (never switched)"
            )


def test_soft_generator_no_switch_without_condition(tiny_model: BaseTransformer):
    """Without a switch condition, all generated tokens should remain soft
    (not one-hot).
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
    ).tokens

    generated = out[0, x.shape[1] :]

    for i in range(10):
        token_dist = generated[i]
        assert not _is_one_hot(token_dist), (
            f"Token {i} should be soft (no switch condition set)"
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


def test_soft_generate_with_prefill_without_kv_cache(tiny_model: BaseTransformer):
    """Soft generator prefill should insert one-hot encoded fill tokens into the
    output at the position where the shadow sequence matches the trigger.
    """
    torch.manual_seed(42)
    model = tiny_model.eval()
    vocab_size = model.vocab_size
    pad_token_id = 0
    eos_token_id = -1

    x = torch.randint(1, vocab_size, size=(1, 4))

    out_natural = generate_soft_tokens(
        net=model,
        token_ids=x,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        max_tokens_generated=20,
        use_kv_cache=False,
    ).tokens

    shadow_tokens = out_natural[0, x.shape[1] :].argmax(-1)
    trigger = shadow_tokens[:3]
    natural_fill = shadow_tokens[3:5]
    fill = (natural_fill + 1) % vocab_size
    fill = torch.where(fill == pad_token_id, (fill + 1) % vocab_size, fill)

    prefill = PreFill(condition=trigger, filling=fill)

    out_prefill = generate_soft_tokens(
        net=model,
        token_ids=x,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        max_tokens_generated=20,
        use_kv_cache=False,
        prefill=prefill,
    ).tokens

    torch.testing.assert_close(
        out_prefill[0, : x.shape[1] + 3],
        out_natural[0, : x.shape[1] + 3],
        msg="Output before trigger should match natural generation",
    )

    fill_one_hot = torch.nn.functional.one_hot(fill, vocab_size).float()
    assert (out_prefill[0, x.shape[1] + 3 : x.shape[1] + 5] == fill_one_hot).all(), (
        "Prefill did not fire: one-hot fill tokens not found after trigger"
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
    ).tokens

    shadow_tokens = out1_natural[0, x1.shape[1] :].argmax(-1)
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
    ).tokens

    fill_one_hot = torch.nn.functional.one_hot(fill, vocab_size).float()
    assert (out1[0, x1.shape[1] + 3 : x1.shape[1] + 5] == fill_one_hot).all(), (
        "Prefill did not fire: one-hot fill tokens not found after trigger"
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
    ).tokens

    torch.testing.assert_close(
        out1[0],
        out_batched[0, 3:],
        msg="Batch element 0 (with prefill) doesn't match individual generation",
    )
