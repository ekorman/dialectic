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
