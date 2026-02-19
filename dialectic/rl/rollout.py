import time
import warnings
from dataclasses import dataclass
from typing import Callable, Generic

import torch
from jaxtyping import Bool, Float, Integer
from tokenizers import Tokenizer

from dialectic.llm.base import BaseTransformer
from dialectic.llm.generate import PreFill, generate_hard_tokens, generate_soft_tokens
from dialectic.rl.env import Env
from dialectic.rl.reward import RewardFn
from dialectic.rl.types import A, E, EnvResponse, RewardResult, T


def _embedding_rms_norm(net: BaseTransformer) -> float:
    W = net.embed_tokens.weight
    return W.pow(2).mean().sqrt().item()


def get_batch(env: Env, batch_size: int) -> list[EnvResponse]:
    env_responses = []
    for _ in range(batch_size):
        env_response = env.reset()
        if env_response is not None:
            env_responses.append(env_response)
            if not env_response.is_done:
                raise RuntimeError("Only single step environments supported for now")
        else:
            warnings.warn("Got None response from `env.reset`")
    return env_responses


@dataclass
class RolloutBatch(Generic[T]):
    env_responses: list[EnvResponse[T]]
    prompts: list[str]
    output_strs: list[list[str]]  # [G][B]
    reward_results: list[list[RewardResult]]  # [G][B]
    rewards: Float[torch.Tensor, "G B"]
    completion_token_ids: list[Integer[torch.Tensor, "B L"]]  # len G
    attention_mask: Bool[torch.Tensor, "B L_prompt"]
    t_generation: float


@dataclass
class SoftRolloutBatch(Generic[T]):
    env_responses: list[EnvResponse[T]]
    prompts: list[str]
    output_strs: list[list[str]]  # [G][B]
    reward_results: list[list[RewardResult]]  # [G][B]
    rewards: Float[torch.Tensor, "G B"]
    completion_tokens: list[
        Float[torch.Tensor, "B L V"]
    ]  # len G, full seq (prompt+gen)
    completion_action_embeddings: list[
        Float[torch.Tensor, "B L D"]
    ]  # len G, full seq (prompt+gen), from behavior net
    completion_noise: list[Float[torch.Tensor, "B L D"]]  # len G
    hard_tokens_mask: list[Bool[torch.Tensor, "B L"]]  # len G
    noise_std: float
    temperature: float
    attention_mask: Bool[torch.Tensor, "B L_prompt"]
    t_generation: float


@torch.no_grad()
def generate_rollout_batch(
    *,
    net: BaseTransformer,
    env: Env[T, A],
    reward_fn: RewardFn[T, E],
    state_to_str: Callable[[T], str],
    tokenizer: Tokenizer,
    eos_token_id: int,
    pad_token_id: int,
    extractor: Callable[[str], E],
    batch_size: int,
    group_size: int,
    temperature: float,
    max_tokens_generated: int,
    use_bf16: bool = False,
) -> RolloutBatch[T]:
    """Generate completions and compute rewards for a batch from the environment."""
    env_responses = get_batch(env, batch_size)
    prompts = [state_to_str(resp.data) for resp in env_responses]
    device = next(net.parameters()).device

    tokenizer.enable_padding(direction="left")
    tokens = tokenizer.encode_batch(prompts)
    attention_mask = torch.tensor(
        [t.attention_mask for t in tokens], dtype=torch.bool, device=device
    )
    token_ids = torch.tensor([t.ids for t in tokens], device=device)

    expanded_token_ids = token_ids.repeat_interleave(group_size, dim=0)
    expanded_attention_mask = attention_mask.repeat_interleave(group_size, dim=0)

    was_training = net.training
    net.eval()
    t_gen_start = time.perf_counter()

    all_completions = generate_hard_tokens(
        net=net,
        token_ids=expanded_token_ids,
        sampling_strategy="sample" if temperature > 0 else "greedy",
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        max_tokens_generated=max_tokens_generated,
        use_kv_cache=True,
        attention_mask=expanded_attention_mask,
        temperature=temperature if temperature > 0 else 1.0,
        use_bf16=use_bf16,
    ).tokens
    t_generation = time.perf_counter() - t_gen_start
    if was_training:
        net.train()

    all_completions = all_completions.view(batch_size, group_size, -1)
    all_completions = all_completions.permute(1, 0, 2)
    completion_token_ids: list[Integer[torch.Tensor, "B L"]] = list(
        all_completions.unbind(0)
    )

    prompt_len = token_ids.shape[1]

    output_strs: list[list[str]] = [
        tokenizer.decode_batch(c[:, prompt_len:].tolist()) for c in completion_token_ids
    ]
    reward_results: list[list[RewardResult]] = [
        [
            reward_fn(
                env_response=env_response,
                raw_model_output=s,
                extracted_model_output=extractor(s),
            )
            for s, env_response in zip(group_batch, env_responses)
        ]
        for group_batch in output_strs
    ]

    rewards: Float[torch.Tensor, "G B"] = torch.tensor(
        [[r.total for r in row] for row in reward_results],
        device=device,
    )

    return RolloutBatch(
        env_responses=env_responses,
        prompts=prompts,
        output_strs=output_strs,
        reward_results=reward_results,
        rewards=rewards,
        completion_token_ids=completion_token_ids,
        attention_mask=attention_mask,
        t_generation=t_generation,
    )


@torch.no_grad()
def generate_soft_rollout_batch(
    *,
    net: BaseTransformer,
    env: Env[T, A],
    reward_fn: RewardFn[T, E],
    state_to_str: Callable[[T], str],
    tokenizer: Tokenizer,
    eos_token_id: int,
    pad_token_id: int,
    extractor: Callable[[str], E],
    batch_size: int,
    group_size: int,
    temperature: float,
    max_tokens_generated: int,
    noise_std: float,
    switch_to_hard_tokens_condition: torch.Tensor | None = None,
    max_tokens_prefill: torch.Tensor | None = None,
    max_tokens_prefill_steps_before_end: int = 0,
    min_soft_steps: int = 0,
    prefill: PreFill | None = None,
    use_bf16: bool = False,
) -> SoftRolloutBatch[T]:
    actual_noise_std = noise_std * _embedding_rms_norm(net)

    env_responses = get_batch(env, batch_size)
    prompts = [state_to_str(resp.data) for resp in env_responses]
    device = next(net.parameters()).device

    tokenizer.enable_padding(direction="left")
    tokens = tokenizer.encode_batch(prompts)
    attention_mask = torch.tensor(
        [t.attention_mask for t in tokens], dtype=torch.bool, device=device
    )
    token_ids = torch.tensor([t.ids for t in tokens], device=device)

    expanded_token_ids = token_ids.repeat_interleave(group_size, dim=0)
    expanded_attention_mask = attention_mask.repeat_interleave(group_size, dim=0)

    was_training = net.training
    net.eval()
    t_gen_start = time.perf_counter()

    gen_output = generate_soft_tokens(
        net=net,
        token_ids=expanded_token_ids,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        max_tokens_generated=max_tokens_generated,
        use_kv_cache=True,
        attention_mask=expanded_attention_mask,
        temperature=temperature if temperature > 0 else 1.0,
        use_bf16=use_bf16,
        switch_to_hard_tokens_condition=switch_to_hard_tokens_condition,
        max_tokens_prefill=max_tokens_prefill,
        max_tokens_prefill_steps_before_end=max_tokens_prefill_steps_before_end,
        prefill=prefill,
        soft_token_noise_std=actual_noise_std,
        min_soft_steps=min_soft_steps,
    )
    t_generation = time.perf_counter() - t_gen_start
    if was_training:
        net.train()

    all_soft_tokens = gen_output.tokens.view(
        batch_size, group_size, *gen_output.tokens.shape[1:]
    )
    all_soft_tokens = all_soft_tokens.permute(1, 0, 2, 3)
    completion_tokens: list[Float[torch.Tensor, "B L V"]] = list(
        all_soft_tokens.unbind(0)
    )

    noise = gen_output.noise
    if noise is None:
        raise RuntimeError("soft token generation did not produce noise")
    all_noise = noise.view(batch_size, group_size, *noise.shape[1:])
    all_noise = all_noise.permute(1, 0, 2, 3)
    completion_noise: list[Float[torch.Tensor, "B L D"]] = list(all_noise.unbind(0))

    # Action embeddings computed with behavior policy's embedding matrix.
    behavior_W = net.embed_tokens.weight
    completion_action_embeddings: list[Float[torch.Tensor, "B L D"]] = [
        (c.float() @ behavior_W.float()) + n.float()
        for c, n in zip(completion_tokens, completion_noise)
    ]

    all_hard_masks = gen_output.hard_tokens_mask.view(
        batch_size, group_size, gen_output.hard_tokens_mask.shape[1]
    )
    all_hard_masks = all_hard_masks.permute(1, 0, 2)
    hard_tokens_mask_list: list[Bool[torch.Tensor, "B L"]] = list(
        all_hard_masks.unbind(0)
    )

    prompt_len = token_ids.shape[1]
    output_strs: list[list[str]] = [
        tokenizer.decode_batch(c[:, prompt_len:].argmax(-1).tolist())
        for c in completion_tokens
    ]

    reward_results: list[list[RewardResult]] = [
        [
            reward_fn(
                env_response=env_response,
                raw_model_output=s,
                extracted_model_output=extractor(s),
            )
            for s, env_response in zip(group_batch, env_responses)
        ]
        for group_batch in output_strs
    ]

    rewards: Float[torch.Tensor, "G B"] = torch.tensor(
        [[r.total for r in row] for row in reward_results],
        device=device,
    )

    return SoftRolloutBatch(
        env_responses=env_responses,
        prompts=prompts,
        output_strs=output_strs,
        reward_results=reward_results,
        rewards=rewards,
        completion_tokens=completion_tokens,
        completion_action_embeddings=completion_action_embeddings,
        completion_noise=completion_noise,
        hard_tokens_mask=hard_tokens_mask_list,
        noise_std=actual_noise_std,
        temperature=temperature if temperature > 0 else 1.0,
        attention_mask=attention_mask,
        t_generation=t_generation,
    )
