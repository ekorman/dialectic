import time
import warnings
from dataclasses import dataclass
from typing import Callable, Generic

import torch
from jaxtyping import Bool, Float, Int, Integer
from tokenizers import Tokenizer

from dialectic.llm.base import BaseTransformer
from dialectic.llm.generate import (
    PreFill,
    VariableLengthInternalReasoningGeneratorOutput,
    generate_hard_tokens,
    generate_internal_reasoning_tokens,
    generate_noise_reasoning_tokens,
    generate_soft_cycling_tokens,
    generate_soft_tokens,
    generate_variable_length_internal_reasoning_tokens,
)
from dialectic.rl.env import Env
from dialectic.rl.reward import RewardFn
from dialectic.rl.types import A, E, EnvResponse, RewardResult, T


def _embedding_rms_norm(net: BaseTransformer) -> float:
    W = net.embed_tokens.weight
    return W.pow(2).mean().sqrt().item()


def decode_variable_length_gen_output(
    gen_output: VariableLengthInternalReasoningGeneratorOutput,
    batch_size: int,
    pad_token_id: int,
    eos_token_id: int,
    tokenizer: Tokenizer,
) -> list[str]:
    output_strs: list[str] = []
    for b in range(batch_size):
        all_ids: list[int] = []
        nc = gen_output.n_cycles[b].item()
        for c in range(nc):
            tlen = gen_output.hard_token_lengths[b, c].item()
            for t in range(tlen):
                tid = gen_output.hard_token_ids[b, c, t].item()
                if tid != pad_token_id and tid != eos_token_id:
                    all_ids.append(tid)
        output_strs.append(tokenizer.decode(all_ids) if all_ids else "")
    return output_strs


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


@dataclass
class InternalReasoningRolloutBatch(Generic[T]):
    env_responses: list[EnvResponse[T]]
    prompts: list[str]
    output_strs: list[list[str]]  # [G][B]
    reward_results: list[list[RewardResult]]  # [G][B]
    rewards: Float[torch.Tensor, "G B"]
    hard_token_ids: list[Int[torch.Tensor, "B C"]]  # len G
    n_cycles: list[Int[torch.Tensor, " B"]]  # len G
    prompt_token_ids: Int[torch.Tensor, "B L_prompt"]
    attention_mask: Bool[torch.Tensor, "B L_prompt"]
    t_generation: float


@torch.no_grad()
def generate_internal_reasoning_rollout_batch(
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
    batch_size: int,
    group_size: int,
    temperature: float,
    soft_block_size: int = 4,
    max_cycles: int = 30,
    use_bf16: bool = False,
    think_token_id: int | None = None,
) -> InternalReasoningRolloutBatch[T]:
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

    gen_output = generate_internal_reasoning_tokens(
        net=net,
        token_ids=expanded_token_ids,
        soft_block_size=soft_block_size,
        max_cycles=max_cycles,
        valid_hard_token_ids=valid_hard_token_ids,
        done_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        temperature=temperature if temperature > 0 else 1.0,
        attention_mask=expanded_attention_mask,
        use_bf16=use_bf16,
        think_token_id=think_token_id,
    )
    t_generation = time.perf_counter() - t_gen_start
    if was_training:
        net.train()

    all_hard_ids = gen_output.hard_token_ids.view(batch_size, group_size, -1).permute(
        1, 0, 2
    )
    hard_token_ids_list: list[Int[torch.Tensor, "B C"]] = list(all_hard_ids.unbind(0))

    all_n_cycles = gen_output.n_cycles.view(batch_size, group_size).permute(1, 0)
    n_cycles_list: list[Int[torch.Tensor, " B"]] = list(all_n_cycles.unbind(0))

    def _decode_moves(
        hard_ids: Int[torch.Tensor, "B C"], n_cyc: Int[torch.Tensor, " B"]
    ) -> list[str]:
        results = []
        for b in range(hard_ids.shape[0]):
            moves = []
            for c in range(n_cyc[b].item()):
                tid = hard_ids[b, c].item()
                if tid in move_id_to_name:
                    moves.append(move_id_to_name[tid])
            results.append(", ".join(moves))
        return results

    output_strs: list[list[str]] = [
        _decode_moves(hids, ncyc)
        for hids, ncyc in zip(hard_token_ids_list, n_cycles_list)
    ]

    def _extract_moves(hard_ids: Int[torch.Tensor, " C"], n_cyc: int) -> list[str]:
        return [
            move_id_to_name[hard_ids[c].item()]
            for c in range(n_cyc)
            if hard_ids[c].item() in move_id_to_name
        ]

    reward_results: list[list[RewardResult]] = [
        [
            reward_fn(
                env_response=env_response,
                raw_model_output=out_str,
                extracted_model_output=extractor(
                    _extract_moves(hids[b], ncyc[b].item())
                ),
            )
            for b, (env_response, out_str) in enumerate(
                zip(env_responses, group_out_strs)
            )
        ]
        for hids, ncyc, group_out_strs in zip(
            hard_token_ids_list, n_cycles_list, output_strs
        )
    ]

    rewards: Float[torch.Tensor, "G B"] = torch.tensor(
        [[r.total for r in row] for row in reward_results],
        device=device,
    )

    return InternalReasoningRolloutBatch(
        env_responses=env_responses,
        prompts=prompts,
        output_strs=output_strs,
        reward_results=reward_results,
        rewards=rewards,
        hard_token_ids=hard_token_ids_list,
        n_cycles=n_cycles_list,
        prompt_token_ids=token_ids,
        attention_mask=attention_mask,
        t_generation=t_generation,
    )


@dataclass
class VariableLengthInternalReasoningRolloutBatch(Generic[T]):
    env_responses: list[EnvResponse[T]]
    prompts: list[str]
    output_strs: list[str]  # [B] - decoded text per sample
    reward_results: list[RewardResult]  # [B]
    rewards: Float[torch.Tensor, " B"]
    hard_token_ids: Int[torch.Tensor, "B C T_max"]
    hard_token_lengths: Int[torch.Tensor, "B C"]
    n_cycles: Int[torch.Tensor, " B"]
    prompt_token_ids: Int[torch.Tensor, "B L_prompt"]
    attention_mask: Bool[torch.Tensor, "B L_prompt"]
    t_generation: float


@torch.no_grad()
def generate_variable_length_internal_reasoning_rollout_batch(
    *,
    net: BaseTransformer,
    env: Env[T, A],
    reward_fn: RewardFn[T, E],
    state_to_str: Callable[[T], str],
    tokenizer: Tokenizer,
    eos_token_id: int,
    pad_token_id: int,
    separator_token_id: int,
    batch_size: int,
    temperature: float,
    soft_block_size: int = 4,
    max_cycles: int = 10,
    max_tokens_per_cycle: int = 20,
    use_bf16: bool = False,
    think_token_id: int | None = None,
) -> VariableLengthInternalReasoningRolloutBatch[T]:
    env_responses = get_batch(env, batch_size)
    prompts = [state_to_str(resp.data) for resp in env_responses]
    device = next(net.parameters()).device

    tokenizer.enable_padding(direction="left")
    tokens = tokenizer.encode_batch(prompts)
    attention_mask = torch.tensor(
        [t.attention_mask for t in tokens], dtype=torch.bool, device=device
    )
    token_ids = torch.tensor([t.ids for t in tokens], device=device)

    was_training = net.training
    net.eval()
    t_gen_start = time.perf_counter()

    gen_output = generate_variable_length_internal_reasoning_tokens(
        net=net,
        token_ids=token_ids,
        soft_block_size=soft_block_size,
        max_cycles=max_cycles,
        max_tokens_per_cycle=max_tokens_per_cycle,
        separator_token_id=separator_token_id,
        done_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        temperature=temperature,
        attention_mask=attention_mask,
        use_bf16=use_bf16,
        think_token_id=think_token_id,
    )
    t_generation = time.perf_counter() - t_gen_start
    if was_training:
        net.train()

    output_strs = decode_variable_length_gen_output(
        gen_output, batch_size, pad_token_id, eos_token_id, tokenizer
    )

    reward_results: list[RewardResult] = [
        reward_fn(
            env_response=er,
            raw_model_output=out_str,
            extracted_model_output=out_str,
        )
        for er, out_str in zip(env_responses, output_strs)
    ]

    rewards = torch.tensor([r.total for r in reward_results], device=device)

    return VariableLengthInternalReasoningRolloutBatch(
        env_responses=env_responses,
        prompts=prompts,
        output_strs=output_strs,
        reward_results=reward_results,
        rewards=rewards,
        hard_token_ids=gen_output.hard_token_ids,
        hard_token_lengths=gen_output.hard_token_lengths,
        n_cycles=gen_output.n_cycles,
        prompt_token_ids=token_ids,
        attention_mask=attention_mask,
        t_generation=t_generation,
    )


@dataclass
class GroupedVariableLengthInternalReasoningRolloutBatch(Generic[T]):
    env_responses: list[EnvResponse[T]]
    prompts: list[str]
    output_strs: list[list[str]]  # [G][B]
    reward_results: list[list[RewardResult]]  # [G][B]
    rewards: Float[torch.Tensor, "G B"]
    hard_token_ids: list[Int[torch.Tensor, "B C T_max"]]  # len G
    hard_token_lengths: list[Int[torch.Tensor, "B C"]]  # len G
    n_cycles: list[Int[torch.Tensor, " B"]]  # len G
    prompt_token_ids: Int[torch.Tensor, "B L_prompt"]
    attention_mask: Bool[torch.Tensor, "B L_prompt"]
    t_generation: float


@torch.no_grad()
def generate_grouped_variable_length_internal_reasoning_rollout_batch(
    *,
    net: BaseTransformer,
    env: Env[T, A],
    reward_fn: RewardFn[T, E],
    state_to_str: Callable[[T], str],
    tokenizer: Tokenizer,
    eos_token_id: int,
    pad_token_id: int,
    separator_token_id: int,
    batch_size: int,
    group_size: int,
    temperature: float,
    soft_block_size: int = 4,
    max_cycles: int = 10,
    max_tokens_per_cycle: int = 20,
    use_bf16: bool = False,
    think_token_id: int | None = None,
) -> GroupedVariableLengthInternalReasoningRolloutBatch[T]:
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

    gen_output = generate_variable_length_internal_reasoning_tokens(
        net=net,
        token_ids=expanded_token_ids,
        soft_block_size=soft_block_size,
        max_cycles=max_cycles,
        max_tokens_per_cycle=max_tokens_per_cycle,
        separator_token_id=separator_token_id,
        done_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        temperature=temperature,
        attention_mask=expanded_attention_mask,
        use_bf16=use_bf16,
        think_token_id=think_token_id,
    )
    t_generation = time.perf_counter() - t_gen_start
    if was_training:
        net.train()

    all_hard_ids = gen_output.hard_token_ids.view(
        batch_size, group_size, *gen_output.hard_token_ids.shape[1:]
    ).permute(1, 0, 2, 3)
    hard_token_ids_list: list[Int[torch.Tensor, "B C T_max"]] = list(
        all_hard_ids.unbind(0)
    )

    all_lengths = gen_output.hard_token_lengths.view(
        batch_size, group_size, gen_output.hard_token_lengths.shape[1]
    ).permute(1, 0, 2)
    hard_token_lengths_list: list[Int[torch.Tensor, "B C"]] = list(
        all_lengths.unbind(0)
    )

    all_n_cycles = gen_output.n_cycles.view(batch_size, group_size).permute(1, 0)
    n_cycles_list: list[Int[torch.Tensor, " B"]] = list(all_n_cycles.unbind(0))

    output_strs: list[list[str]] = []
    for g in range(group_size):
        group_strs = decode_variable_length_gen_output(
            VariableLengthInternalReasoningGeneratorOutput(
                hard_token_ids=hard_token_ids_list[g],
                hard_token_lengths=hard_token_lengths_list[g],
                n_cycles=n_cycles_list[g],
            ),
            batch_size,
            pad_token_id,
            eos_token_id,
            tokenizer,
        )
        output_strs.append(group_strs)

    reward_results: list[list[RewardResult]] = [
        [
            reward_fn(
                env_response=env_response,
                raw_model_output=out_str,
                extracted_model_output=out_str,
            )
            for env_response, out_str in zip(env_responses, group_out_strs)
        ]
        for group_out_strs in output_strs
    ]

    rewards: Float[torch.Tensor, "G B"] = torch.tensor(
        [[r.total for r in row] for row in reward_results],
        device=device,
    )

    return GroupedVariableLengthInternalReasoningRolloutBatch(
        env_responses=env_responses,
        prompts=prompts,
        output_strs=output_strs,
        reward_results=reward_results,
        rewards=rewards,
        hard_token_ids=hard_token_ids_list,
        hard_token_lengths=hard_token_lengths_list,
        n_cycles=n_cycles_list,
        prompt_token_ids=token_ids,
        attention_mask=attention_mask,
        t_generation=t_generation,
    )


@dataclass
class SoftCyclingRolloutBatch(Generic[T]):
    env_responses: list[EnvResponse[T]]
    prompts: list[str]
    output_strs: list[list[str]]
    reward_results: list[list[RewardResult]]
    rewards: Float[torch.Tensor, "G B"]
    hard_token_ids: list[Int[torch.Tensor, "B C T_max"]]
    hard_token_lengths: list[Int[torch.Tensor, "B C"]]
    n_cycles: list[Int[torch.Tensor, " B"]]
    soft_lengths: list[Int[torch.Tensor, "B C"]]
    prompt_token_ids: Int[torch.Tensor, "B L_prompt"]
    attention_mask: Bool[torch.Tensor, "B L_prompt"]
    t_generation: float


@torch.no_grad()
def generate_soft_cycling_rollout_batch(
    *,
    net: BaseTransformer,
    env: Env[T, A],
    reward_fn: RewardFn[T, E],
    state_to_str: Callable[[T], str],
    tokenizer: Tokenizer,
    extractor: Callable[[str], E],
    eos_token_id: int,
    pad_token_id: int,
    grammar_specs: list,
    batch_size: int,
    group_size: int,
    temperature: float,
    max_tokens_generated: int = 512,
    max_cycles: int = 20,
    max_tokens_per_cycle: int = 30,
    use_bf16: bool = False,
    use_gumbel: bool = False,
    soft_token_noise_std: float | None = None,
    min_soft_steps: int = 0,
    max_soft_steps_per_cycle: int | None = None,
    min_cycles: int = 0,
    evict_soft_kv: bool = False,
    n_expert_trajectories: int = 0,
) -> SoftCyclingRolloutBatch[T]:
    env_responses = get_batch(env, batch_size)
    prompts = [state_to_str(resp.data) for resp in env_responses]
    device = next(net.parameters()).device

    tokenizer.enable_padding(direction="left")
    tokens = tokenizer.encode_batch(prompts)
    attention_mask = torch.tensor(
        [t.attention_mask for t in tokens], dtype=torch.bool, device=device
    )
    token_ids = torch.tensor([t.ids for t in tokens], device=device)

    actual_noise_std = (
        soft_token_noise_std * _embedding_rms_norm(net)
        if soft_token_noise_std is not None
        else None
    )

    hard_token_ids_list: list[Int[torch.Tensor, "B C T_max"]] = []
    hard_token_lengths_list: list[Int[torch.Tensor, "B C"]] = []
    n_cycles_list: list[Int[torch.Tensor, " B"]] = []
    soft_lengths_list: list[Int[torch.Tensor, "B C"]] = []
    all_shadow_ids: list[Int[torch.Tensor, "B L"]] = []
    cycle_is_terminal_list: list[Bool[torch.Tensor, "B C"]] = []

    was_training = net.training
    net.eval()
    t_gen_start = time.perf_counter()

    for _ in range(group_size):
        gen_output = generate_soft_cycling_tokens(
            net=net,
            token_ids=token_ids,
            grammar_specs=grammar_specs,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            max_tokens_generated=max_tokens_generated,
            max_cycles=max_cycles,
            max_tokens_per_cycle=max_tokens_per_cycle,
            temperature=temperature if temperature > 0 else 1.0,
            attention_mask=attention_mask,
            use_bf16=use_bf16,
            use_gumbel=use_gumbel,
            soft_token_noise_std=actual_noise_std,
            min_soft_steps=min_soft_steps,
            max_soft_steps_per_cycle=max_soft_steps_per_cycle,
            min_cycles=min_cycles,
            evict_soft_kv=evict_soft_kv,
        )
        hard_token_ids_list.append(gen_output.hard_token_ids)
        hard_token_lengths_list.append(gen_output.hard_token_lengths)
        n_cycles_list.append(gen_output.n_cycles)
        soft_lengths_list.append(gen_output.soft_lengths)
        all_shadow_ids.append(gen_output.shadow_ids)
        cycle_is_terminal_list.append(gen_output.cycle_is_terminal)

    if n_expert_trajectories > 0:
        from dialectic.rl.grammar import countdown_solution_to_hard_tokens

        forced_ids_batch = []
        forced_lens_batch = []
        forced_nc_batch = []
        for resp in env_responses:
            if resp.data.solution is not None:
                f_ids, f_lens, f_nc = countdown_solution_to_hard_tokens(
                    solution=resp.data.solution,
                    numbers=resp.data.numbers,
                    target=resp.data.target,
                    tokenizer=tokenizer,
                    max_cycles=max_cycles,
                    max_tokens_per_cycle=max_tokens_per_cycle,
                    pad_token_id=pad_token_id,
                )
            else:
                f_ids = torch.full(
                    (max_cycles, max_tokens_per_cycle),
                    pad_token_id,
                    dtype=torch.long,
                )
                f_lens = torch.zeros(max_cycles, dtype=torch.long)
                f_nc = 1
            forced_ids_batch.append(f_ids)
            forced_lens_batch.append(f_lens)
            forced_nc_batch.append(f_nc)

        forced_ids_t = torch.stack(forced_ids_batch).to(device)
        forced_lens_t = torch.stack(forced_lens_batch).to(device)
        forced_nc_t = torch.tensor(forced_nc_batch, dtype=torch.long, device=device)

        for _ in range(n_expert_trajectories):
            gen_output = generate_soft_cycling_tokens(
                net=net,
                token_ids=token_ids,
                grammar_specs=grammar_specs,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                max_tokens_generated=max_tokens_generated,
                max_cycles=max_cycles,
                max_tokens_per_cycle=max_tokens_per_cycle,
                temperature=temperature if temperature > 0 else 1.0,
                attention_mask=attention_mask,
                use_bf16=use_bf16,
                use_gumbel=use_gumbel,
                soft_token_noise_std=actual_noise_std,
                min_soft_steps=min_soft_steps,
                max_soft_steps_per_cycle=max_soft_steps_per_cycle,
                forced_hard_token_ids=forced_ids_t,
                forced_hard_token_lengths=forced_lens_t,
                forced_n_cycles=forced_nc_t,
            )
            hard_token_ids_list.append(gen_output.hard_token_ids)
            hard_token_lengths_list.append(gen_output.hard_token_lengths)
            n_cycles_list.append(gen_output.n_cycles)
            soft_lengths_list.append(gen_output.soft_lengths)
            all_shadow_ids.append(gen_output.shadow_ids)
            cycle_is_terminal_list.append(gen_output.cycle_is_terminal)

    t_generation = time.perf_counter() - t_gen_start
    if was_training:
        net.train()

    output_strs: list[list[str]] = []
    for g_idx in range(len(hard_token_ids_list)):
        group_strs: list[str] = []
        for b in range(batch_size):
            nc = n_cycles_list[g_idx][b].item()
            parts: list[str] = []
            for c in range(nc):
                hl = hard_token_lengths_list[g_idx][b, c].item()
                if hl == 0:
                    continue
                hard_toks = hard_token_ids_list[g_idx][b, c, :hl].tolist()
                decoded_hard = tokenizer.decode(hard_toks)
                is_term = cycle_is_terminal_list[g_idx][b, c].item()
                tag = "<answer>" if is_term else "<SCRATCH>"
                parts.append(tag + decoded_hard)
            group_strs.append("\n".join(parts))
        output_strs.append(group_strs)

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

    return SoftCyclingRolloutBatch(
        env_responses=env_responses,
        prompts=prompts,
        output_strs=output_strs,
        reward_results=reward_results,
        rewards=rewards,
        hard_token_ids=hard_token_ids_list,
        hard_token_lengths=hard_token_lengths_list,
        n_cycles=n_cycles_list,
        soft_lengths=soft_lengths_list,
        prompt_token_ids=token_ids,
        attention_mask=attention_mask,
        t_generation=t_generation,
    )


@dataclass
class NoiseReasoningRolloutBatch(Generic[T]):
    env_responses: list[EnvResponse[T]]
    prompts: list[str]
    output_strs: list[list[str]]
    reward_results: list[list[RewardResult]]
    rewards: Float[torch.Tensor, "G B"]
    hard_token_ids: list[Int[torch.Tensor, "B C T_max"]]
    hard_token_lengths: list[Int[torch.Tensor, "B C"]]
    n_cycles: list[Int[torch.Tensor, " B"]]
    noise_vectors: list[Float[torch.Tensor, "B C k D"]]
    cycle_is_terminal: list[Bool[torch.Tensor, "B C"]]
    prompt_token_ids: Int[torch.Tensor, "B L_prompt"]
    attention_mask: Bool[torch.Tensor, "B L_prompt"]
    t_generation: float


@torch.no_grad()
def generate_noise_reasoning_rollout_batch(
    *,
    net: BaseTransformer,
    env: Env[T, A],
    reward_fn: RewardFn[T, E],
    state_to_str: Callable[[T], str],
    tokenizer: Tokenizer,
    extractor: Callable[[str], E],
    eos_token_id: int,
    pad_token_id: int,
    grammar_factory: Callable,
    batch_size: int,
    group_size: int,
    n_noise_per_cycle: int,
    noise_std: float,
    temperature: float,
    max_cycles: int = 10,
    min_cycles: int = 0,
    max_tokens_per_cycle: int = 30,
    use_bf16: bool = False,
) -> NoiseReasoningRolloutBatch[T]:
    env_responses = get_batch(env, batch_size)
    prompts = [state_to_str(resp.data) for resp in env_responses]
    device = next(net.parameters()).device

    tokenizer.enable_padding(direction="left")
    tokens = tokenizer.encode_batch(prompts)
    attention_mask = torch.tensor(
        [t.attention_mask for t in tokens], dtype=torch.bool, device=device
    )
    token_ids = torch.tensor([t.ids for t in tokens], device=device)

    hard_token_ids_list: list[Int[torch.Tensor, "B C T_max"]] = []
    hard_token_lengths_list: list[Int[torch.Tensor, "B C"]] = []
    n_cycles_list: list[Int[torch.Tensor, " B"]] = []
    noise_vectors_list: list[Float[torch.Tensor, "B C k D"]] = []
    cycle_is_terminal_list: list[Bool[torch.Tensor, "B C"]] = []

    was_training = net.training
    net.eval()
    t_gen_start = time.perf_counter()

    for _ in range(group_size):
        gen_output = generate_noise_reasoning_tokens(
            net=net,
            token_ids=token_ids,
            grammar_factory=grammar_factory,
            n_noise_per_cycle=n_noise_per_cycle,
            noise_std=noise_std,
            max_cycles=max_cycles,
            min_cycles=min_cycles,
            max_tokens_per_cycle=max_tokens_per_cycle,
            pad_token_id=pad_token_id,
            temperature=temperature if temperature > 0 else 1.0,
            attention_mask=attention_mask,
            use_bf16=use_bf16,
        )
        hard_token_ids_list.append(gen_output.hard_token_ids)
        hard_token_lengths_list.append(gen_output.hard_token_lengths)
        n_cycles_list.append(gen_output.n_cycles)
        noise_vectors_list.append(gen_output.noise_vectors)
        cycle_is_terminal_list.append(gen_output.cycle_is_terminal)

    t_generation = time.perf_counter() - t_gen_start
    if was_training:
        net.train()

    output_strs: list[list[str]] = []
    for g_idx in range(group_size):
        group_strs: list[str] = []
        for b in range(batch_size):
            nc = n_cycles_list[g_idx][b].item()
            parts: list[str] = []
            for c in range(nc):
                hl = hard_token_lengths_list[g_idx][b, c].item()
                if hl > 0:
                    toks = hard_token_ids_list[g_idx][b, c, :hl].tolist()
                    parts.append(tokenizer.decode(toks))
            group_strs.append("\n".join(parts))
        output_strs.append(group_strs)

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

    return NoiseReasoningRolloutBatch(
        env_responses=env_responses,
        prompts=prompts,
        output_strs=output_strs,
        reward_results=reward_results,
        rewards=rewards,
        hard_token_ids=hard_token_ids_list,
        hard_token_lengths=hard_token_lengths_list,
        n_cycles=n_cycles_list,
        noise_vectors=noise_vectors_list,
        cycle_is_terminal=cycle_is_terminal_list,
        prompt_token_ids=token_ids,
        attention_mask=attention_mask,
        t_generation=t_generation,
    )
