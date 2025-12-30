from typing import Protocol

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from dialectic.rl.env import Env


class Phi(Protocol):
    """This is a protocol for the factor in the policy gradient that is
    multiplied by the gradient of the logprobs. it is a function of a trajectory and
    e.g. in basic policy gradient this would be the total (discounted) reward, repeated
    for each state
    """

    def __call__(
        self,
        *,
        batch_states: list[torch.Tensor],  # [b, n],
        batch_rewards: list[torch.Tensor],
        batch_actions: list[torch.Tensor],
        discount_factor: float,
    ) -> list[torch.Tensor]: ...


def build_policy_net(
    hidden_layer_dims: list[int], dim_state: int, n_actions: int
) -> nn.Module:
    first_layer = nn.Sequential(
        *[nn.Linear(dim_state, hidden_layer_dims[0]), nn.Tanh()]
    )
    middle_layers = []
    for i in range(1, len(hidden_layer_dims)):
        middle_layers.extend(
            [nn.Linear(hidden_layer_dims[i - 1], hidden_layer_dims[i]), nn.Tanh()]
        )
    last_layer = nn.Linear(hidden_layer_dims[-1], n_actions)
    return nn.Sequential(first_layer, *middle_layers, last_layer)


def sample_action(policy_net: nn.Module, state: torch.tensor) -> int:
    dist = Categorical(logits=policy_net(state))
    return dist.sample().item()


def log_prob_act(policy_net: nn.Module, states: torch.Tensor, actions: torch.Tensor):
    dist = Categorical(logits=policy_net(states))
    return dist.log_prob(actions)


def rewards_to_go(
    *,
    batch_states: list[torch.Tensor],  # [b, n],
    batch_rewards: list[torch.Tensor],
    batch_actions: list[torch.Tensor],
    discount_factor: float,
) -> list[torch.Tensor]:  # length of list is b, and each tensor has variable length
    ret = []
    for rewards in batch_rewards:
        d = torch.Tensor([discount_factor**i for i in range(len(rewards))])
        ret.append(((rewards * d).flip(0)).cumsum(0).flip(0) / d)

    return ret


def rloo(
    *,
    batch_states: list[torch.Tensor],  # [b, n],
    batch_rewards: list[torch.Tensor],
    batch_actions: list[torch.Tensor],
    discount_factor: float,
) -> list[torch.Tensor]:
    pass


def grad_ascend_policy(
    *,
    policy_net: nn.Module,
    batch_states: list[torch.Tensor],  # list of tensors of shape [N, dim_state]
    batch_rewards: list[torch.Tensor],
    batch_actions: list[torch.Tensor],
    opt: torch.optim.Optimizer,
    discount_factor: float,
    phi: Phi,
) -> None:
    batch_states_tensor = torch.cat(batch_states, 0)
    batch_action_tensor = torch.cat(batch_actions, 0)
    log_probs = log_prob_act(
        policy_net=policy_net, states=batch_states_tensor, actions=batch_action_tensor
    )

    phis = phi(
        batch_actions=batch_actions,
        batch_rewards=batch_rewards,
        batch_states=batch_states,
        discount_factor=discount_factor,
    )

    flat_phis = torch.cat(phis, 0)
    pg = -(log_probs * flat_phis).mean()

    # an alternative with slightly different weighting would be
    # traj_lengths = [len(states) for states in batch_states]
    # log_probs = torch.split(log_probs, traj_lengths, dim=0)
    # pg = -sum([(lp * p).mean() for lp, p in zip(log_probs, phis)])
    opt.zero_grad()
    pg.backward()
    opt.step()

    return -pg.item()


def mean(a: list):
    return sum(a) / len(a)


def reinforce_loop(
    *,
    policy_net: nn.Module,
    opt: torch.optim.Optimizer,
    env: Env,
    discount_factor: float,
    max_episodes: int,
    batch_size: int,  # need to have ability for batch to come from same initial state just different samples
    phi: Phi,
):
    n_episodes = 0
    state = env.reset()
    batch_actions, batch_rewards, batch_states = [], [], []
    actions, rewards, states = [], [], []

    while n_episodes < max_episodes:
        state = torch.from_numpy(state)
        states.append(state)
        action = sample_action(policy_net, state)
        actions.append(action)
        state, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)

        if terminated or truncated:
            batch_states.append(torch.stack(states))
            batch_actions.append(torch.tensor(actions))
            batch_rewards.append(torch.tensor(rewards))

            n_episodes += 1

            state, _ = env.reset()
            actions, rewards, states = [], [], []

            if (n_episodes % batch_size == 0) or (n_episodes == max_episodes):
                pg = grad_ascend_policy(
                    policy_net=policy_net,
                    batch_states=batch_states,
                    batch_actions=batch_actions,
                    batch_rewards=batch_rewards,
                    discount_factor=discount_factor,
                    phi=phi,
                    opt=opt,
                )
                pg
                # run.log(
                #     {
                #         "ave_steps_in_batch": mean(
                #             [len(states) for states in batch_states]
                #         ),
                #         "ave_non_discounted_reward": mean(
                #             [sum(rewards) for rewards in batch_rewards]
                #         ),
                #         "discounted_reward": mean(
                #             [
                #                 sum(
                #                     [
                #                         r * discount_factor**i
                #                         for i, r in enumerate(rewards)
                #                     ]
                #                 )
                #                 for rewards in batch_rewards
                #             ]
                #         ),
                #         "policy_gradient": pg,
                #     }
                # )

                batch_actions, batch_rewards, batch_states = [], [], []


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--env", type=str, required=True)
#     parser.add_argument("--lr", type=float, required=True)
#     parser.add_argument("--max_episodes", type=int, required=True)
#     parser.add_argument("--discount_factor", type=float, required=True)
#     parser.add_argument("--hidden_dims", type=json.loads, required=True)
#     parser.add_argument("--batch_size", type=int, required=True)

#     args = parser.parse_args()

#     env = gym.make(args.env)

#     policy_net = build_policy_net(
#         args.hidden_dims, env.observation_space.shape[0], int(env.action_space.n)
#     )
#     opt = torch.optim.Adam(policy_net.parameters(), lr=args.lr)

#     run = wandb.init(project="rl", config=vars(args))
#     reinforce_loop(
#         policy_net=policy_net,
#         opt=opt,
#         env=env,
#         discount_factor=args.discount_factor,
#         max_episodes=args.max_episodes,
#         batch_size=args.batch_size,
#         phi=rewards_to_go,
#     )
