import torch

from dialectic.llm.generate import generate_hard_tokens


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
