import pytest
import torch
import torch.nn as nn

from dialectic.rl.reinforce import (
    build_policy_net,
    compute_log_probs,
    compute_logits_of_group,
    log_prob_act,
    rewards_to_go,
    stack_and_pad,
)


@pytest.fixture
def dim_state():
    return 3


@pytest.fixture
def n_actions():
    return 2


@pytest.fixture
def policy_net(dim_state: int, n_actions: int) -> nn.Module:
    return build_policy_net([8], dim_state, n_actions)


@torch.no_grad()
def test_policy_methods(policy_net: nn.Module, dim_state: int, n_actions: int):
    n_batch = 4
    states = torch.rand((n_batch, dim_state))

    actions: torch.Tensor = policy_net(states)
    assert actions.shape == torch.Size([n_batch, n_actions])

    log_probs = log_prob_act(policy_net, states, torch.Tensor([1, 0, 0, 1]))
    assert log_probs.shape == torch.Size([n_batch])
    assert (torch.exp(log_probs) >= 0).all()
    assert (torch.exp(log_probs) <= 1).all()


def test_reward_to_go():
    rewards = torch.Tensor([[1.5, 6.1, 7.8]])
    discount_factor = 0.6

    rtg = rewards_to_go(
        batch_rewards=rewards,
        discount_factor=discount_factor,
        batch_states=None,
        batch_actions=None,
    )
    assert len(rtg) == 1
    rtg = rtg[0]
    expected = torch.Tensor(
        [
            1.5 + 6.1 * discount_factor + 7.8 * discount_factor**2,
            6.1 + 7.8 * discount_factor,
            7.8,
        ]
    )

    torch.testing.assert_close(rtg, expected)


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

    result = compute_log_probs(
        net=net,
        attention_mask=attention_mask,
        completion_token_ids=completion_token_ids,
        pad_token_id=pad_token_id,
    )

    assert result.shape == (batch_size, group_size)

    stacked = stack_and_pad(completion_token_ids, pad_token_id)
    all_logits = net.logits.view(batch_size, group_size, total_len, vocab_size)
    all_logits = all_logits[:, :, :-1]
    all_log_probs = all_logits.log_softmax(-1)

    expected = torch.zeros(batch_size, group_size)
    for b in range(batch_size):
        for g in range(group_size):
            for t in range(completion_len):
                logit_pos = prompt_len - 1 + t
                token_pos = prompt_len + t
                token_id = stacked[b, g, token_pos]
                expected[b, g] += all_log_probs[b, g, logit_pos, token_id]

    torch.testing.assert_close(result, expected)
