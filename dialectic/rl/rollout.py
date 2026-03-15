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
    generate_soft_tokens,
    generate_variable_length_internal_reasoning_tokens,
)
from dialectic.rl.env import Env
from dialectic.rl.reward import RewardFn
from dialectic.rl.types import A, E, EnvResponse, RewardResult, T


def _embedding_rms_norm(net: BaseTransformer) -> float:
    W = net.embed_tokens.weight
    return W.pow(2).mean().sqrt().item()


def decode_variable_length_gen_output_per_cycle(
    gen_output: VariableLengthInternalReasoningGeneratorOutput,
    batch_size: int,
    pad_token_id: int,
    eos_token_id: int,
    separator_token_id: int,
    tokenizer: Tokenizer,
) -> list[list[str]]:
    result: list[list[str]] = []
    for b in range(batch_size):
        cycles: list[str] = []
        nc = gen_output.n_cycles[b].item()
        for c in range(nc):
            cycle_ids: list[int] = []
            tlen = gen_output.hard_token_lengths[b, c].item()
            for t in range(tlen):
                tid = gen_output.hard_token_ids[b, c, t].item()
                if tid not in (pad_token_id, eos_token_id, separator_token_id):
                    cycle_ids.append(tid)
            cycles.append(tokenizer.decode(cycle_ids) if cycle_ids else "")
        result.append(cycles)
    return result


def decode_variable_length_gen_output(
    gen_output: VariableLengthInternalReasoningGeneratorOutput,
    batch_size: int,
    pad_token_id: int,
    eos_token_id: int,
    tokenizer: Tokenizer,
    separator_token_id: int | None = None,
) -> list[str]:
    if separator_token_id is not None:
        per_cycle = decode_variable_length_gen_output_per_cycle(
            gen_output,
            batch_size,
            pad_token_id,
            eos_token_id,
            separator_token_id,
            tokenizer,
        )
        return [" | ".join(cycles) for cycles in per_cycle]
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
    valid_hard_token_ids: list[int] | None = None,
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
        valid_hard_token_ids=valid_hard_token_ids,
    )
    t_generation = time.perf_counter() - t_gen_start
    if was_training:
        net.train()

    per_cycle_strs = decode_variable_length_gen_output_per_cycle(
        gen_output,
        batch_size,
        pad_token_id,
        eos_token_id,
        separator_token_id,
        tokenizer,
    )
    output_strs = [" | ".join(cycles) for cycles in per_cycle_strs]

    reward_results: list[RewardResult] = [
        reward_fn(
            env_response=er,
            raw_model_output=out_str,
            extracted_model_output=cycle_strs,
        )
        for er, out_str, cycle_strs in zip(env_responses, output_strs, per_cycle_strs)
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
    valid_hard_token_ids: list[int] | None = None,
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
        valid_hard_token_ids=valid_hard_token_ids,
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
    per_cycle_strs_all: list[list[list[str]]] = []
    for g in range(group_size):
        group_gen = VariableLengthInternalReasoningGeneratorOutput(
            hard_token_ids=hard_token_ids_list[g],
            hard_token_lengths=hard_token_lengths_list[g],
            n_cycles=n_cycles_list[g],
        )
        group_per_cycle = decode_variable_length_gen_output_per_cycle(
            group_gen,
            batch_size,
            pad_token_id,
            eos_token_id,
            separator_token_id,
            tokenizer,
        )
        per_cycle_strs_all.append(group_per_cycle)
        output_strs.append([" | ".join(cycles) for cycles in group_per_cycle])

    reward_results: list[list[RewardResult]] = [
        [
            reward_fn(
                env_response=env_response,
                raw_model_output=out_str,
                extracted_model_output=cycle_strs,
            )
            for env_response, out_str, cycle_strs in zip(
                env_responses, group_out_strs, group_per_cycle
            )
        ]
        for group_out_strs, group_per_cycle in zip(output_strs, per_cycle_strs_all)
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
