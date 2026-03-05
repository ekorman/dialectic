from dataclasses import dataclass
from typing import Callable

import extty
import torch
from tokenizers import Tokenizer

from dialectic.llm.base import BaseTransformer
from dialectic.llm.generate import generate_with_soft_prefill
from dialectic.rl.env import Env
from dialectic.rl.reward import RewardFn
from dialectic.rl.rollout import (
    generate_internal_reasoning_rollout_batch,
    generate_rollout_batch,
)
from dialectic.rl.types import A, E, T


@dataclass
class EvaluationResult:
    n_episodes: int
    reward_mean: float
    reward_std: float
    component_means: dict[str, float]


@torch.no_grad()
def evaluate(
    *,
    net: BaseTransformer,
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
    n_examples: int = 10,
    use_bf16: bool = False,
) -> tuple[EvaluationResult, list[extty.Example]]:
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
    n_examples
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

    sample_idxs = range(min(n_examples, len(all_prompts)))

    examples = [
        extty.Example(
            prompt=all_prompts[i],
            responses=all_output_strs_nested[i],
            rewards=all_reward_results[i],
        )
        for i in sample_idxs
    ]

    return EvaluationResult(
        n_episodes=n_episodes,
        reward_mean=reward_mean,
        reward_std=reward_std,
        component_means=component_means,
    ), examples


@torch.no_grad()
def evaluate_internal_reasoning(
    *,
    net: BaseTransformer,
    env: Env[T, A],
    reward_fn: RewardFn[T, E],
    state_to_str: Callable[[T], str],
    tokenizer: Tokenizer,
    eos_token_id: int,
    pad_token_id: int,
    extractor: Callable[[list[str]], E],
    move_id_to_name: dict[int, str],
    valid_hard_token_ids: list[int],
    soft_block_size: int,
    max_cycles: int,
    max_episodes: int,
    batch_size: int = 1,
    group_size: int = 1,
    temperature: float = 0.0,
    n_examples: int = 10,
    use_bf16: bool = False,
    think_token_id: int | None = None,
) -> tuple[EvaluationResult, list[extty.Example]]:
    all_rewards: list[float] = []
    all_reward_results: list[list[dict[str, float]]] = []
    all_prompts: list[str] = []
    all_output_strs_nested: list[list[str]] = []

    n_episodes = 0
    while n_episodes < max_episodes:
        current_batch_size = min(batch_size, max_episodes - n_episodes)

        rollout = generate_internal_reasoning_rollout_batch(
            net=net,
            env=env,
            reward_fn=reward_fn,
            state_to_str=state_to_str,
            tokenizer=tokenizer,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            extractor=extractor,
            move_id_to_name=move_id_to_name,
            valid_hard_token_ids=valid_hard_token_ids,
            batch_size=current_batch_size,
            group_size=group_size,
            temperature=temperature,
            soft_block_size=soft_block_size,
            max_cycles=max_cycles,
            use_bf16=use_bf16,
            think_token_id=think_token_id,
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

    sample_idxs = range(min(n_examples, len(all_prompts)))
    examples = [
        extty.Example(
            prompt=all_prompts[i],
            responses=all_output_strs_nested[i],
            rewards=all_reward_results[i],
        )
        for i in sample_idxs
    ]

    return EvaluationResult(
        n_episodes=n_episodes,
        reward_mean=reward_mean,
        reward_std=reward_std,
        component_means=component_means,
    ), examples


@torch.no_grad()
def evaluate_soft_prefill(
    *,
    net: BaseTransformer,
    env: Env[T, A],
    state_to_str: Callable[[T], str],
    answer_extractor: Callable[[T], str],
    tokenizer: Tokenizer,
    eos_token_id: int,
    pad_token_id: int,
    soft_block_size: int,
    max_new_tokens: int = 32,
    max_episodes: int,
    batch_size: int = 1,
    temperature: float = 0.0,
    use_bf16: bool = False,
    n_examples: int = 10,
    num_soft_layers: int | None = None,
) -> tuple[EvaluationResult, list[extty.Example]]:
    """Evaluate soft-prefill generation via exact-match against ground truth.

    Parameters
    ----------
    net
        The language model to evaluate.
    env
        The environment to sample episodes from.
    state_to_str
        Function to convert environment state to prompt string.
    answer_extractor
        Function to extract the ground-truth answer string from state.
    tokenizer
        Tokenizer for encoding/decoding.
    eos_token_id
        End of sequence token ID.
    pad_token_id
        Padding token ID.
    soft_block_size
        Number of hidden-state passes in the soft block.
    max_new_tokens
        Maximum tokens to generate after the soft block.
    max_episodes
        Maximum number of episodes to evaluate.
    batch_size
        Number of episodes per batch.
    temperature
        Sampling temperature (0.0 for greedy).
    use_bf16
        Whether to use bfloat16 for generation.
    n_examples
        Number of examples to log to extty.

    Returns
    -------
    tuple[EvaluationResult, list[extty.Example]]
        Aggregated evaluation statistics and logged examples.
    """
    all_rewards: list[float] = []
    all_prompts: list[str] = []
    all_outputs: list[str] = []
    all_reward_components: list[dict[str, float]] = []

    device = next(net.parameters()).device
    n_episodes = 0

    while n_episodes < max_episodes:
        current_batch_size = min(batch_size, max_episodes - n_episodes)

        env_responses = []
        for _ in range(current_batch_size):
            env_responses.append(env.reset())

        prompts = [state_to_str(resp.data) for resp in env_responses]
        ground_truths = [answer_extractor(resp.data) for resp in env_responses]

        tokenizer.enable_padding(direction="left")
        tokens = tokenizer.encode_batch(prompts)
        attention_mask = torch.tensor(
            [t.attention_mask for t in tokens], dtype=torch.bool, device=device
        )
        token_ids = torch.tensor([t.ids for t in tokens], device=device)

        out = generate_with_soft_prefill(
            net=net,
            token_ids=token_ids,
            attention_mask=attention_mask,
            soft_block_size=soft_block_size,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            temperature=temperature,
            use_bf16=use_bf16,
            num_soft_layers=num_soft_layers,
        )

        for b in range(current_batch_size):
            length = out.lengths[b].item()
            gen_ids = out.token_ids[b, :length].tolist()
            gen_ids = [t for t in gen_ids if t != eos_token_id and t != pad_token_id]
            gen_text = tokenizer.decode(gen_ids).strip()

            correct = gen_text == ground_truths[b].strip()
            reward = 1.0 if correct else 0.0
            all_rewards.append(reward)
            all_prompts.append(prompts[b])
            all_outputs.append(gen_text)
            all_reward_components.append({"correct": reward})

        n_episodes += current_batch_size

    reward_tensor = torch.tensor(all_rewards)
    reward_mean = reward_tensor.mean().item()
    reward_std = reward_tensor.std().item()

    component_means: dict[str, float] = {}
    if all_reward_components:
        all_components: dict[str, list[float]] = {}
        for r in all_reward_components:
            for name, value in r.items():
                all_components.setdefault(name, []).append(value)
        component_means = {
            name: sum(vals) / len(vals) for name, vals in all_components.items()
        }

    sample_idxs = range(min(n_examples, len(all_prompts)))
    examples = [
        extty.Example(
            prompt=all_prompts[i],
            responses=[all_outputs[i]],
            rewards=[all_reward_components[i]],
        )
        for i in sample_idxs
    ]

    return EvaluationResult(
        n_episodes=n_episodes,
        reward_mean=reward_mean,
        reward_std=reward_std,
        component_means=component_means,
    ), examples
