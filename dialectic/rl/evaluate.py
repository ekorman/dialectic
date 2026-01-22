from dataclasses import dataclass
from typing import Callable

import extty
import torch
from tokenizers import Tokenizer

from dialectic.llm.qwen import Qwen
from dialectic.rl.env import Env
from dialectic.rl.reward import RewardFn
from dialectic.rl.train import generate_rollout_batch
from dialectic.rl.types import A, E, T


@dataclass
class EvaluationResult:
    n_episodes: int
    reward_mean: float
    reward_std: float
    component_means: dict[str, float]
    all_rewards: list[float]


@torch.no_grad()
def evaluate(
    *,
    net: Qwen,
    env: Env[T, A],
    reward_fn: RewardFn[T, E],
    state_to_str: Callable[[T], str],
    tokenizer: Tokenizer,
    eos_token_id: int,
    pad_token_id: int,
    extractor: Callable[[str], E],
    max_tokens_generated: int,
    max_episodes: int,
    batch_size: int = 1,
    group_size: int = 1,
    temperature: float = 0.0,
    n_examples_to_log: int = 5,
    use_bf16: bool = False,
) -> EvaluationResult:
    """Evaluate a model against an environment and reward function.

    Parameters
    ----------
    net
        The language model to evaluate.
    env
        The environment to sample episodes from.
    reward_fn
        The reward function to evaluate completions.
    state_to_str
        Function to convert environment state to prompt string.
    tokenizer
        Tokenizer for encoding/decoding.
    eos_token_id
        End of sequence token ID.
    pad_token_id
        Padding token ID.
    extractor
        Function to extract structured output from model completion.
    max_tokens_generated
        Maximum tokens to generate per completion.
    max_episodes
        Maximum number of episodes to evaluate.
    batch_size
        Number of episodes per batch.
    group_size
        Number of completions per prompt.
    temperature
        Sampling temperature (0.0 for greedy).
    n_examples_to_log
        Number of examples to log to extty.
    use_bf16
        Whether to use bfloat16 for generation.

    Returns
    -------
    EvaluationResult
        Aggregated evaluation statistics.
    """
    all_rewards: list[float] = []
    all_reward_results: list[list[dict[str, float]]] = []
    all_prompts: list[str] = []
    all_output_strs_nested: list[list[str]] = []

    n_episodes = 0
    while n_episodes < max_episodes:
        current_batch_size = min(batch_size, max_episodes - n_episodes)

        rollout = generate_rollout_batch(
            net=net,
            env=env,
            reward_fn=reward_fn,
            state_to_str=state_to_str,
            tokenizer=tokenizer,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            extractor=extractor,
            batch_size=current_batch_size,
            group_size=group_size,
            temperature=temperature,
            max_tokens_generated=max_tokens_generated,
            use_bf16=use_bf16,
        )

        for g in range(group_size):
            for b in range(current_batch_size):
                all_rewards.append(rollout.reward_results[g][b].total)

        for b in range(current_batch_size):
            all_prompts.append(rollout.prompts[b])
            all_output_strs_nested.append(
                [rollout.output_strs[g][b] for g in range(group_size)]
            )
            all_reward_results.append(
                [rollout.reward_results[g][b].components for g in range(group_size)]
            )

        n_episodes += current_batch_size

    reward_tensor = torch.tensor(all_rewards)
    reward_mean = reward_tensor.mean().item()
    reward_std = reward_tensor.std().item()

    flat_reward_results = [r for row in all_reward_results for r in row]
    component_means: dict[str, float] = {}
    if flat_reward_results:
        all_components: dict[str, list[float]] = {}
        for r in flat_reward_results:
            for name, value in r.items():
                all_components.setdefault(name, []).append(value)
        component_means = {
            name: sum(vals) / len(vals) for name, vals in all_components.items()
        }

    if extty._active_run is not None:
        examples = extty.BatchExample(
            prompts=all_prompts[:n_examples_to_log],
            responses=all_output_strs_nested[:n_examples_to_log],
            rewards=all_reward_results[:n_examples_to_log],  # type: ignore[arg-type]
        )

        extty.log(
            {
                "eval/reward_mean": reward_mean,
                "eval/reward_std": reward_std,
                "eval/examples": examples,
                **{
                    f"eval/reward/{name}": mean
                    for name, mean in component_means.items()
                },
            },
            step=0,
        )

    return EvaluationResult(
        n_episodes=n_episodes,
        reward_mean=reward_mean,
        reward_std=reward_std,
        component_means=component_means,
        all_rewards=all_rewards,
    )
