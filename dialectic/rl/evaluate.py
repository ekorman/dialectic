from dataclasses import dataclass
from typing import Callable

from dialectic.rl.env import Env
from dialectic.rl.reward import RewardFn
from dialectic.rl.types import E, T


@dataclass
class EvaluationStats:
    n_episodes: int
    average_reward: float


def evaluate(
    env: Env[T, E],
    model_generation: Callable[[T], str],
    extractor: Callable[[str], E],
    reward_fn: RewardFn[T, E],  # TODO: maybe support multiple reward functions?
    max_episodes: int | float = float("inf"),
):
    episodes_rewards_sum = 0
    n_episodes = 0
    while n_episodes < max_episodes:
        episode_reward = 0
        s = env.reset()

        if s is None:
            break

        while True:
            model_out = model_generation(s.data)
            extracted = extractor(model_out)
            result = reward_fn(
                env_response=s,
                raw_model_output=model_out,
                extracted_model_output=extracted,
            )
            episode_reward += result.total
            if s.is_done:
                break

            # will always be using extracted for both reward and action?
            s = env.step(extracted)

        episodes_rewards_sum += episode_reward
        n_episodes += 1
    return EvaluationStats(
        n_episodes=n_episodes, average_reward=episodes_rewards_sum / n_episodes
    )
