import pytest
import torch
import torch.nn as nn

from dialectic.rl.reinforce import build_policy_net, log_prob_act, rewards_to_go


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
