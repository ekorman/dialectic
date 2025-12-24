import argparse
import json

import gymnasium as gym
import torch
import torch.nn as nn
import wandb
from torch.distributions.categorical import Categorical


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


def log_prob_act(
    policy_net: nn.Module, states: torch.Tensor, actions: torch.Tensor
) -> int:
    dist = Categorical(logits=policy_net(states))
    return dist.log_prob(actions)


def rewards_to_go(rewards: torch.Tensor, discount_factor: float) -> torch.Tensor:
    d = torch.Tensor([discount_factor**i for i in range(len(rewards))])
    return ((rewards * d).flip(0)).cumsum(0).flip(0) / d


def grad_ascend_policy(
    policy_net: nn.Module,
    states: torch.Tensor,  # [n, d_s]
    actions: torch.Tensor,  # [n]
    opt: torch.optim.Optimizer,
    rewards: torch.Tensor,  # [n]
    discount_factor: float,
    opt_step: bool,
) -> None:
    log_probs = log_prob_act(policy_net=policy_net, states=states, actions=actions)
    rtg = rewards_to_go(rewards, discount_factor)
    # negative since optimizer will do grad descent not ascent
    pg = -(log_probs * rtg).mean()
    pg.backward()
    if opt_step:
        opt.step()
        opt.zero_grad()
    return -pg.item()


def reinforce_loop(
    *,
    policy_net: nn.Module,
    opt: torch.optim.Optimizer,
    env: gym.Env,
    discount_factor: float,
    max_episodes: int,
    batch_size: int,
):
    n_episodes = 0
    state, _ = env.reset()
    actions, rewards, states = [], [], []
    opt.zero_grad()  # safeguard; shouldn't be necessary

    while n_episodes < max_episodes:
        state = torch.from_numpy(state)
        states.append(state)
        action = sample_action(policy_net, state)
        actions.append(action)
        state, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)

        if terminated or truncated:
            pg = grad_ascend_policy(
                policy_net=policy_net,
                states=torch.stack(states),
                actions=torch.tensor(actions),
                rewards=torch.tensor(rewards),
                discount_factor=discount_factor,
                opt=opt,
                opt_step=((n_episodes + 1) % batch_size == 0)
                or (n_episodes == max_episodes - 1),
            )
            n_episodes += 1
            run.log(
                {
                    "steps": len(states),
                    "non_discounted_reward": sum(rewards),
                    "discounted_reward": sum(
                        [r * discount_factor**i for i, r in enumerate(rewards)]
                    ),
                    "policy_gradient": pg,
                }
            )
            print(
                f"Episode {n_episodes}: {len(states)} steps, total reward = {sum(rewards)}"
            )
            state, _ = env.reset()
            actions, rewards, states = [], [], []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--max_episodes", type=int, required=True)
    parser.add_argument("--discount_factor", type=float, required=True)
    parser.add_argument("--hidden_dims", type=json.loads, required=True)
    parser.add_argument("--batch_size", type=int, required=True)

    args = parser.parse_args()

    env = gym.make(args.env)

    policy_net = build_policy_net(
        args.hidden_dims, env.observation_space.shape[0], int(env.action_space.n)
    )
    opt = torch.optim.Adam(policy_net.parameters(), lr=args.lr)

    run = wandb.init(project="rl", config=vars(args))
    reinforce_loop(
        policy_net=policy_net,
        opt=opt,
        env=env,
        discount_factor=args.discount_factor,
        max_episodes=args.max_episodes,
        batch_size=args.batch_size,
    )
