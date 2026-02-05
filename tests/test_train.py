import torch
import torch.nn as nn

from dialectic.llm.generate import generate_from_tokens
from dialectic.rl.train import compute_log_probs, compute_logits_of_group, stack_and_pad


def test_stack_and_pad():
    pad_token_id = -1
    t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])  # [2, 3]
    t2 = torch.tensor([[7, 8], [9, 10]])  # [2, 2]

    result = stack_and_pad([t1, t2], pad_token_id)

    assert result.shape == (2, 2, 3)  # [batch, group, max_len]
    assert result[0, 0].tolist() == [1, 2, 3]
    assert result[0, 1].tolist() == [7, 8, -1]
    assert result[1, 0].tolist() == [4, 5, 6]
    assert result[1, 1].tolist() == [9, 10, -1]


def test_compute_logits_of_group():
    vocab_size = 10
    d = 16

    class MockNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d)
            self.linear = nn.Linear(d, vocab_size)

        def forward(self, x, return_all_logits=False, attention_mask=None):
            emb = self.embed(x)
            logits = self.linear(emb)
            if not return_all_logits:
                logits = logits[:, -1:]
            return logits

    net = MockNet()

    batch_size = 2
    group_size = 3
    seq_len = 5

    input_ids = torch.randint(0, vocab_size, (batch_size, group_size, seq_len))

    expected_logits = []
    for g in range(group_size):
        group_input = input_ids[:, g, :]
        group_logits = net(group_input, return_all_logits=True)
        expected_logits.append(group_logits)
    expected = torch.stack(expected_logits, dim=1)

    actual = compute_logits_of_group(net=net, input_ids=input_ids, attention_mask=None)

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected)


def test_compute_log_probs():
    vocab_size = 10
    batch_size = 2
    group_size = 3
    prompt_len = 4
    completion_len = 5
    total_len = prompt_len + completion_len

    class MockNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = nn.Parameter(
                torch.randn(batch_size * group_size, total_len, vocab_size)
            )

        def forward(self, x, return_all_logits=False, attention_mask=None):
            if return_all_logits:
                return self.logits
            return self.logits[:, -1:]

    net = MockNet()
    pad_token_id = 0

    completion_token_ids = [
        torch.randint(1, vocab_size, (batch_size, total_len)) for _ in range(group_size)
    ]
    attention_mask = torch.ones(batch_size, prompt_len, dtype=torch.long)

    result, completion_mask = compute_log_probs(
        net=net,
        attention_mask=attention_mask,
        completion_token_ids=completion_token_ids,
        pad_token_id=pad_token_id,
    )

    assert result.shape == (batch_size, group_size, completion_len)
    assert completion_mask.shape == (batch_size, group_size, completion_len)

    stacked = stack_and_pad(completion_token_ids, pad_token_id)
    all_logits = net.logits.view(batch_size, group_size, total_len, vocab_size)
    all_logits = all_logits[:, :, :-1]
    all_log_probs = all_logits.log_softmax(-1)

    expected = torch.zeros(batch_size, group_size, completion_len)
    for b in range(batch_size):
        for g in range(group_size):
            for t in range(completion_len):
                logit_pos = prompt_len - 1 + t
                token_pos = prompt_len + t
                token_id = stacked[b, g, token_pos]
                expected[b, g, t] += all_log_probs[b, g, logit_pos, token_id]

    torch.testing.assert_close(result, expected)

    expected_mask = stacked[:, :, prompt_len:] != pad_token_id
    torch.testing.assert_close(completion_mask, expected_mask)


def test_generate_from_tokens_preserves_eos():
    """EOS token should be present in the generated output, not replaced with pad."""
    vocab_size = 10
    eos_token_id = 2
    pad_token_id = 0

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(1))

        def forward(self, x, kv_caches=None, attention_mask=None):
            prompt_len = 2
            step = x.shape[1] - prompt_len
            tokens = [5, 6, eos_token_id]
            logits = torch.full((x.shape[0], 1, vocab_size), -100.0)
            idx = min(step, len(tokens) - 1)
            logits[:, :, tokens[idx]] = 100.0
            return logits

    net = MockModel()
    result = generate_from_tokens(
        net=net,
        token_ids=torch.tensor([[1, 3]]),
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        sampling_strategy="greedy",
        use_kv_cache=False,
    )

    assert result[0].tolist() == [1, 3, 5, 6, eos_token_id]


def test_generate_from_tokens_preserves_eos_batch():
    """In a batch where sequences finish at different times, EOS should be
    preserved for each sequence and padding should only appear after EOS."""
    vocab_size = 10
    eos_token_id = 2
    pad_token_id = 0

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(1))

        def forward(self, x, kv_caches=None, attention_mask=None):
            prompt_len = 2
            step = x.shape[1] - prompt_len
            seq0_tokens = [5, eos_token_id, 9]
            seq1_tokens = [5, 6, eos_token_id]
            seq3_tokens = [7, pad_token_id, pad_token_id, pad_token_id, eos_token_id]
            logits = torch.full((x.shape[0], 1, vocab_size), -100.0)
            for b, tokens in enumerate([seq0_tokens, seq1_tokens, seq3_tokens]):
                idx = min(step, len(tokens) - 1)
                logits[b, :, tokens[idx]] = 100.0
            return logits

    net = MockModel()
    result = generate_from_tokens(
        net=net,
        token_ids=torch.tensor([[1, 3], [pad_token_id, 3], [5, 7]]),
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        sampling_strategy="greedy",
        use_kv_cache=False,
    )

    # seq 0: generates [5, EOS], then waits for seq 1 → [1, 3, 5, EOS, pad]
    # seq 1: generates [5, 6, EOS] → [1, 3, 5, 6, EOS]
    assert result[0].tolist() == [
        1,
        3,
        5,
        eos_token_id,
        pad_token_id,
        pad_token_id,
        pad_token_id,
    ]
    assert result[1].tolist() == [
        pad_token_id,
        3,
        5,
        6,
        eos_token_id,
        pad_token_id,
        pad_token_id,
    ]
    assert result[2].tolist() == [
        5,
        7,
        7,
        pad_token_id,
        pad_token_id,
        pad_token_id,
        eos_token_id,
    ]


def test_compute_logits_of_group_attention_mask_extended():
    """Test that attention mask is correctly extended from prompt length to full sequence length."""
    vocab_size = 10
    d = 16
    batch_size = 2
    group_size = 3
    prompt_len = 4
    total_len = 7

    class MockNetWithMaskCheck(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d)
            self.linear = nn.Linear(d, vocab_size)
            self.received_mask_shape = None

        def forward(self, x, return_all_logits=False, attention_mask=None):
            self.received_mask_shape = (
                attention_mask.shape if attention_mask is not None else None
            )
            emb = self.embed(x)
            logits = self.linear(emb)
            if not return_all_logits:
                logits = logits[:, -1:]
            return logits

    net = MockNetWithMaskCheck()
    input_ids = torch.randint(0, vocab_size, (batch_size, group_size, total_len))
    attention_mask = torch.ones(batch_size, prompt_len, dtype=torch.bool)
    attention_mask[0, 0] = False

    compute_logits_of_group(net=net, input_ids=input_ids, attention_mask=attention_mask)

    expected_mask_shape = (batch_size * group_size, total_len)
    assert net.received_mask_shape == expected_mask_shape, (
        f"Expected mask shape {expected_mask_shape}, got {net.received_mask_shape}"
    )
