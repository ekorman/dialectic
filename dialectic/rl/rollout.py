import time
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Generic

import torch
from jaxtyping import Bool, Float, Integer
from tokenizers import Tokenizer

from dialectic.distributed import unwrap_model
from dialectic.llm.base import BaseTransformer
from dialectic.llm.generate import PreFill, generate_hard_tokens, generate_soft_tokens
from dialectic.rl.env import Env
from dialectic.rl.reward import RewardFn
from dialectic.rl.types import A, E, EnvResponse, RewardResult, T

if TYPE_CHECKING:
    from vllm import LLM


def _embedding_rms_norm(net: BaseTransformer) -> float:
    W = unwrap_model(net).embed_tokens.weight
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
    completion_embeddings: list[Float[torch.Tensor, "B L D"]]  # len G
    completion_shadow_ids: list[Integer[torch.Tensor, "B L"]]  # len G
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
def generate_rollout_batch_vllm(
    *,
    llm: "LLM",
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
    device: torch.device,
    seed: int | None = None,
) -> RolloutBatch[T]:
    """vLLM-backed rollout generation for GRPO training.

    Same contract as :func:`generate_rollout_batch` — produces a
    ``RolloutBatch`` whose ``attention_mask`` and ``completion_token_ids``
    are shaped for downstream :func:`compute_log_probs` without any
    per-caller adaptation. The only difference is that sampling is done
    through a preconstructed ``vllm.LLM`` engine; the caller is responsible
    for keeping that engine's weights in sync with the current training
    policy via :func:`sync_weights_to_vllm`.

    Parameters
    ----------
    llm
        Engine built by :func:`build_vllm_for_training`. Sampling uses
        ``n=group_size`` so each prompt gets ``group_size`` independent
        samples in one call (vLLM shares the prefill KV across samples).
    device
        Device to place the returned prompt mask and completion token
        tensors on. Typically ``next(net.parameters()).device``.
    seed
        Optional sampler seed. ``None`` lets vLLM use its configured default.
    """
    from vllm import SamplingParams, TokensPrompt

    env_responses = get_batch(env, batch_size)
    prompts = [state_to_str(resp.data) for resp in env_responses]

    # Tokenize each prompt without padding (vLLM handles variable lengths
    # natively via continuous batching). We left-pad afterwards only to build
    # the downstream `attention_mask` / `completion_token_ids` tensors that
    # `compute_log_probs` expects.
    tokenizer.no_padding()
    tokenizer.no_truncation()
    encs = tokenizer.encode_batch(prompts)
    raw_prompt_ids: list[list[int]] = [list(e.ids) for e in encs]

    l_prompt = max(len(ids) for ids in raw_prompt_ids)
    padded_prompt_ids = torch.full(
        (batch_size, l_prompt), pad_token_id, dtype=torch.long, device=device
    )
    attention_mask = torch.zeros(
        (batch_size, l_prompt), dtype=torch.bool, device=device
    )
    for b, ids in enumerate(raw_prompt_ids):
        n = len(ids)
        padded_prompt_ids[b, l_prompt - n :] = torch.tensor(
            ids, dtype=torch.long, device=device
        )
        attention_mask[b, l_prompt - n :] = True

    sampling_params = SamplingParams(
        n=group_size,
        temperature=temperature if temperature > 0 else 0.0,
        max_tokens=max_tokens_generated,
        stop_token_ids=[eos_token_id],
        seed=seed,
    )

    t_gen_start = time.perf_counter()
    vllm_outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=ids) for ids in raw_prompt_ids],
        sampling_params,
        use_tqdm=False,
    )
    t_generation = time.perf_counter() - t_gen_start

    # Build per-group completion_token_ids: each is [B, L_prompt + L_gen_g]
    # where rows are left-padded prompts followed by right-padded generations.
    completion_token_ids: list[Integer[torch.Tensor, "B L"]] = []
    output_strs: list[list[str]] = []
    for g in range(group_size):
        gen_ids_per_prompt = [list(out.outputs[g].token_ids) for out in vllm_outputs]
        max_gen = max((len(ids) for ids in gen_ids_per_prompt), default=0)
        group_tensor = torch.full(
            (batch_size, l_prompt + max_gen),
            pad_token_id,
            dtype=torch.long,
            device=device,
        )
        group_tensor[:, :l_prompt] = padded_prompt_ids
        for b, ids in enumerate(gen_ids_per_prompt):
            if ids:
                group_tensor[b, l_prompt : l_prompt + len(ids)] = torch.tensor(
                    ids, dtype=torch.long, device=device
                )
        completion_token_ids.append(group_tensor)
        output_strs.append([out.outputs[g].text for out in vllm_outputs])

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

    all_embeddings = gen_output.embeddings.view(
        batch_size, group_size, *gen_output.embeddings.shape[1:]
    )
    all_embeddings = all_embeddings.permute(1, 0, 2, 3)
    completion_embeddings: list[Float[torch.Tensor, "B L D"]] = list(
        all_embeddings.unbind(0)
    )

    all_shadow_ids = gen_output.shadow_ids.view(
        batch_size, group_size, gen_output.shadow_ids.shape[1]
    )
    all_shadow_ids = all_shadow_ids.permute(1, 0, 2)
    completion_shadow_ids: list[Integer[torch.Tensor, "B L"]] = list(
        all_shadow_ids.unbind(0)
    )

    all_hard_masks = gen_output.hard_tokens_mask.view(
        batch_size, group_size, gen_output.hard_tokens_mask.shape[1]
    )
    all_hard_masks = all_hard_masks.permute(1, 0, 2)
    hard_tokens_mask_list: list[Bool[torch.Tensor, "B L"]] = list(
        all_hard_masks.unbind(0)
    )

    prompt_len = token_ids.shape[1]
    output_strs: list[list[str]] = [
        tokenizer.decode_batch(s[:, prompt_len:].tolist())
        for s in completion_shadow_ids
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
        completion_embeddings=completion_embeddings,
        completion_shadow_ids=completion_shadow_ids,
        hard_tokens_mask=hard_tokens_mask_list,
        noise_std=actual_noise_std,
        temperature=temperature if temperature > 0 else 1.0,
        attention_mask=attention_mask,
        t_generation=t_generation,
    )
