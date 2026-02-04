import sys
import time
import warnings
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Generic

import extty
import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Integer
from tokenizers import Tokenizer

from dialectic.llm.base import BaseTransformer
from dialectic.llm.generate import generate_from_tokens
from dialectic.rl.env import Env
from dialectic.rl.reward import RewardFn
from dialectic.rl.types import A, E, EnvResponse, RewardResult, T


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


def aggregate_reward_components(
    results: list[list[RewardResult]],
) -> dict[str, float]:
    """Compute mean of each component across all results."""
    all_components: dict[str, list[float]] = {}
    for row in results:
        for r in row:
            for name, value in r.components.items():
                all_components.setdefault(name, []).append(value)
    return {name: sum(vals) / len(vals) for name, vals in all_components.items()}


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


def compute_logits_of_group(
    *,
    net: nn.Module,
    input_ids: Integer[torch.Tensor, "B G L"],
    attention_mask: Bool[torch.Tensor, "B L_prompt"] | None,
) -> Float[torch.Tensor, "B G L VC"]:
    batch_size, group_size, seq_len = input_ids.shape[:3]
    input_ids = input_ids.view(batch_size * group_size, seq_len)

    if attention_mask is not None:
        prompt_len = attention_mask.shape[1]
        full_mask = torch.ones(
            batch_size,
            seq_len,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        full_mask[:, :prompt_len] = attention_mask
        full_mask = full_mask.unsqueeze(1).expand(-1, group_size, -1)
        full_mask = full_mask.reshape(batch_size * group_size, seq_len)
    else:
        full_mask = None

    out = net(input_ids, return_all_logits=True, attention_mask=full_mask)
    out = out.view(batch_size, group_size, seq_len, -1)
    return out


def compute_log_probs(
    *,
    net: BaseTransformer,
    attention_mask: Bool[torch.Tensor, "B L_prompt"],
    completion_token_ids: list[Integer[torch.Tensor, "B L_completion"]],
    pad_token_id: int,
    chunk_size: int = 0,
) -> tuple[Float[torch.Tensor, "B G L_new"], Bool[torch.Tensor, "B G L_new"]]:
    """Compute log probabilities for completions.

    Parameters
    ----------
    net
        The language model.
    attention_mask
        Attention mask for the prompt.
    completion_token_ids
        List of completion token tensors, one per group member.
    pad_token_id
        Token ID used for padding.
    chunk_size
        If > 0, compute log probs in chunks to reduce memory usage.
        Recommended: 64-128 for large vocab models.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        (log_probs, completion_mask) both of shape [B, G, L_completion]
    """
    l_prompt = attention_mask.shape[1]
    stacked = stack_and_pad(tensors=completion_token_ids, pad_token_id=pad_token_id)

    if chunk_size > 0:
        log_probs = _compute_log_probs_chunked(
            net=net,
            input_ids=stacked,
            attention_mask=attention_mask,
            l_prompt=l_prompt,
            chunk_size=chunk_size,
        )
    else:
        logits: Float[torch.Tensor, "B G L_completion VC"] = compute_logits_of_group(
            net=net,
            input_ids=stacked,
            attention_mask=attention_mask,
        )
        logits = logits[:, :, l_prompt - 1 : -1]
        only_completion = stacked[:, :, l_prompt:]
        B, G, L, V = logits.shape
        log_probs = -torch.nn.functional.cross_entropy(
            logits.reshape(B * G * L, V),
            only_completion.reshape(B * G * L),
            reduction="none",
        ).reshape(B, G, L)

    only_completion = stacked[:, :, l_prompt:]
    completion_mask = only_completion != pad_token_id
    return log_probs, completion_mask


def _compute_log_probs_chunked(
    *,
    net: nn.Module,
    input_ids: Integer[torch.Tensor, "B G L"],
    attention_mask: Bool[torch.Tensor, "B L_prompt"],
    l_prompt: int,
    chunk_size: int,
) -> Float[torch.Tensor, "B G L_completion"]:
    """Compute log probs in chunks to avoid OOM from large logits tensor."""

    batch_size, group_size, seq_len = input_ids.shape
    flat_input_ids = input_ids.view(batch_size * group_size, seq_len)

    if attention_mask is not None:
        full_mask = torch.ones(
            batch_size,
            seq_len,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        full_mask[:, :l_prompt] = attention_mask
        full_mask = full_mask.unsqueeze(1).expand(-1, group_size, -1)
        full_mask = full_mask.reshape(batch_size * group_size, seq_len)
    else:
        full_mask = None

    hidden_states: Float[torch.Tensor, "BG L D"] = net(
        flat_input_ids,
        attention_mask=full_mask,
        return_hidden_states=True,
    )

    completion_len = seq_len - l_prompt
    hidden_for_completion = hidden_states[:, l_prompt - 1 : -1]
    target_tokens = flat_input_ids[:, l_prompt:]

    log_probs_list = []
    for chunk_start in range(0, completion_len, chunk_size):
        chunk_end = min(chunk_start + chunk_size, completion_len)

        chunk_hidden = hidden_for_completion[:, chunk_start:chunk_end]
        chunk_targets = target_tokens[:, chunk_start:chunk_end]

        chunk_logits = net.lm_head(chunk_hidden)

        BG, L_chunk, V = chunk_logits.shape
        chunk_log_probs = -torch.nn.functional.cross_entropy(
            chunk_logits.reshape(BG * L_chunk, V),
            chunk_targets.reshape(BG * L_chunk),
            reduction="none",
        ).reshape(BG, L_chunk)

        log_probs_list.append(chunk_log_probs)
        del chunk_logits

    log_probs_flat = torch.cat(log_probs_list, dim=1)
    return log_probs_flat.view(batch_size, group_size, completion_len)


def stack_and_pad(
    tensors: list[Integer[torch.Tensor, "B G *"]], pad_token_id: int
) -> Integer[torch.Tensor, "B G L"]:
    max_len = max([t.shape[1] for t in tensors])
    batch_size = tensors[0].shape[0]
    group_size = len(tensors)
    ret = torch.full(
        (batch_size, group_size, max_len),
        pad_token_id,
        dtype=torch.long,
        device=tensors[0].device,
    )
    for g, t in enumerate(tensors):
        ret[:, g, : t.shape[1]] = t
    return ret


def compute_advantages(
    rewards: Float[torch.Tensor, "G B"],
    normalize: bool = True,
    eps: float = 1e-8,
) -> Float[torch.Tensor, "G B"]:
    mean = rewards.mean(0, keepdim=True)
    if normalize:
        if rewards.shape[0] <= 1:
            return rewards - mean
        return (rewards - mean) / (rewards.std(0, keepdim=True) + eps)
    return rewards - mean


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

    all_completions = generate_from_tokens(
        net=net,
        token_ids=expanded_token_ids,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        sampling_strategy="sample" if temperature > 0 else "greedy",
        temperature=temperature if temperature > 0 else 1.0,
        attention_mask=expanded_attention_mask,
        use_kv_cache=True,
        max_tokens_generated=max_tokens_generated,
        use_bf16=use_bf16,
    )
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


def compute_grpo_loss(
    *,
    log_probs: Float[torch.Tensor, "B G L_new"],
    old_log_probs: Float[torch.Tensor, "B G L_new"],
    ref_log_probs: Float[torch.Tensor, "B G L_new"],
    completion_mask: Bool[torch.Tensor, "B G L_new"],
    advs: Float[torch.Tensor, "B G 1"],
    beta: float,
    eps: float,
    clip_ratio_c: float = 3.0,
) -> tuple[torch.Tensor, float, float]:
    """Compute GRPO loss without backward pass or optimizer step.

    Parameters
    ----------
    log_probs
        Current policy log probabilities.
    old_log_probs
        Old policy log probabilities (for importance sampling ratio).
    ref_log_probs
        Reference policy log probabilities (for KL penalty).
    completion_mask
        Mask for valid completion tokens.
    advs
        Pre-computed advantages, shape [B, G, 1].
    beta
        KL penalty coefficient.
    eps
        PPO clipping epsilon.
    clip_ratio_c
        Dual-clip ratio for negative advantages.

    Returns
    -------
    tuple[torch.Tensor, float, float]
        (loss tensor, ppo_loss scalar, kl_loss scalar)
    """
    ratio = (log_probs - old_log_probs).exp()
    unclipped = ratio * advs
    clipped = torch.clip(ratio, 1 - eps, 1 + eps) * advs

    ppo_obj = torch.min(unclipped, clipped)
    dual_clip_obj = clip_ratio_c * advs
    ppo_obj = torch.where(advs < 0, torch.max(ppo_obj, dual_clip_obj), ppo_obj)
    kl_diff = ref_log_probs - log_probs
    kl_diff = torch.clamp(kl_diff, min=-20, max=20)
    kl_loss = torch.exp(kl_diff) - kl_diff - 1
    kl_loss = torch.clamp(kl_loss, min=-10, max=10)

    mask_sum = completion_mask.sum()
    ppo_loss_scalar = -(ppo_obj * completion_mask).sum() / mask_sum
    kl_loss_scalar = (kl_loss * completion_mask).sum() / mask_sum

    loss = ppo_loss_scalar + beta * kl_loss_scalar
    return loss, ppo_loss_scalar.item(), kl_loss_scalar.item()


def grpo_step(
    *,
    opt: torch.optim.Optimizer,
    log_probs: Float[torch.Tensor, "B G L_new"],
    old_log_probs: Float[torch.Tensor, "B G L_new"] | None,
    ref_log_probs: Float[torch.Tensor, "B G L_new"],
    completion_mask: Bool[torch.Tensor, "B G L_new"],
    beta: float,
    eps: float,
    rewards: Float[torch.Tensor, "G B"],
    normalize_advantages: bool = True,
    clip_ratio_c: float = 3.0,
    max_grad_norm: float = 1.0,
) -> tuple[float, float, float]:
    batch_size, g = log_probs.shape[:2]
    if old_log_probs is None:
        old_log_probs = log_probs.detach()

    advs: Float[torch.Tensor, "B G 1"] = compute_advantages(
        rewards, normalize=normalize_advantages
    ).T.view(batch_size, g, 1)

    loss, ppo_loss, kl_loss = compute_grpo_loss(
        log_probs=log_probs,
        old_log_probs=old_log_probs,
        ref_log_probs=ref_log_probs,
        completion_mask=completion_mask,
        advs=advs,
        beta=beta,
        eps=eps,
        clip_ratio_c=clip_ratio_c,
    )

    opt.zero_grad()
    loss.backward()
    if max_grad_norm > 0:
        params = [p for group in opt.param_groups for p in group["params"]]
        torch.nn.utils.clip_grad_norm_(params, max_norm=max_grad_norm)
    opt.step()

    return loss.item(), ppo_loss, kl_loss


@torch.no_grad()
def collect_micro_batch(
    *,
    net: BaseTransformer,
    ref_net: BaseTransformer,
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
    logprob_chunk_size: int = 0,
    use_bf16: bool = False,
) -> dict:
    """Collect a single micro-batch of data for gradient accumulation."""
    rollout = generate_rollout_batch(
        net=net,
        env=env,
        reward_fn=reward_fn,
        state_to_str=state_to_str,
        tokenizer=tokenizer,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        extractor=extractor,
        batch_size=batch_size,
        group_size=group_size,
        temperature=temperature,
        max_tokens_generated=max_tokens_generated,
        use_bf16=use_bf16,
    )

    device = next(net.parameters()).device
    t_logprobs_start = time.perf_counter()
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
    ):
        ref_log_probs, completion_mask = compute_log_probs(
            net=ref_net,
            attention_mask=rollout.attention_mask,
            completion_token_ids=rollout.completion_token_ids,
            pad_token_id=pad_token_id,
            chunk_size=logprob_chunk_size,
        )
        old_log_probs, _ = compute_log_probs(
            net=net,
            attention_mask=rollout.attention_mask,
            completion_token_ids=rollout.completion_token_ids,
            pad_token_id=pad_token_id,
            chunk_size=logprob_chunk_size,
        )
    t_logprobs = time.perf_counter() - t_logprobs_start

    return {
        "prompts": rollout.prompts,
        "env_responses": rollout.env_responses,
        "attention_mask": rollout.attention_mask,
        "completion_token_ids": rollout.completion_token_ids,
        "output_strs": rollout.output_strs,
        "reward_results": rollout.reward_results,
        "rewards": rollout.rewards,
        "ref_log_probs": ref_log_probs,
        "old_log_probs": old_log_probs,
        "completion_mask": completion_mask,
        "t_gen": rollout.t_generation,
        "t_logprobs": t_logprobs,
    }


def train_grpo(
    *,
    net: BaseTransformer,
    opt: torch.optim.Optimizer,
    env: Env[T, A],  # assume single step
    reward_fn: RewardFn[T, E],
    state_to_str: Callable[[T], str],
    tokenizer: Tokenizer,
    eos_token_id: int,
    pad_token_id: int,
    extractor: Callable[[str], E],
    beta: float,
    eps: float,
    mu: int,  # number of optimization passes per accumulated batch
    max_tokens_generated: int,
    max_episodes: int,
    update_ref_net_batch_cadence: int,
    batch_size: int,
    group_size: int,
    temperature: float,
    normalize_advantages: bool = True,
    accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    logprob_chunk_size: int = 64,
    use_bf16: bool = True,
    save_ckpt_freq: int = sys.maxsize,
) -> None:
    device = next(net.parameters()).device
    if use_bf16:
        net = net.to(dtype=torch.bfloat16)

    n_episodes = 0
    step = 0

    while n_episodes < max_episodes:
        if step % update_ref_net_batch_cadence == 0:
            ref_net = deepcopy(net)

        t_gen_total = 0.0
        t_logprobs_total = 0.0
        micro_batches: list[dict] = []

        for _ in range(accumulation_steps):
            micro_batch = collect_micro_batch(
                net=net,
                ref_net=ref_net,
                env=env,
                reward_fn=reward_fn,
                state_to_str=state_to_str,
                tokenizer=tokenizer,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                extractor=extractor,
                batch_size=batch_size,
                group_size=group_size,
                temperature=temperature,
                max_tokens_generated=max_tokens_generated,
                logprob_chunk_size=logprob_chunk_size,
                use_bf16=use_bf16,
            )
            micro_batches.append(micro_batch)
            t_gen_total += micro_batch["t_gen"]
            t_logprobs_total += micro_batch["t_logprobs"]

        all_rewards = torch.cat(
            [mb["rewards"] for mb in micro_batches], dim=1
        )  # [G, B*accum]
        global_advs: Float[torch.Tensor, "G B*accum"] = compute_advantages(
            all_rewards, normalize=normalize_advantages
        )

        t_opt_start = time.perf_counter()
        total_loss = 0.0
        total_ppo_loss = 0.0
        total_kl_loss = 0.0

        for _ in range(mu):
            opt.zero_grad()

            for accum_idx, micro_batch in enumerate(micro_batches):
                advs_slice = global_advs[
                    :, accum_idx * batch_size : (accum_idx + 1) * batch_size
                ]
                advs_for_loss: Float[torch.Tensor, "B G 1"] = advs_slice.T.unsqueeze(-1)

                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
                ):
                    log_probs, _ = compute_log_probs(
                        net=net,
                        attention_mask=micro_batch["attention_mask"],
                        completion_token_ids=micro_batch["completion_token_ids"],
                        pad_token_id=pad_token_id,
                        chunk_size=logprob_chunk_size,
                    )

                loss, ppo_loss, kl_loss = compute_grpo_loss(
                    log_probs=log_probs,
                    old_log_probs=micro_batch["old_log_probs"],
                    ref_log_probs=micro_batch["ref_log_probs"],
                    completion_mask=micro_batch["completion_mask"],
                    advs=advs_for_loss,
                    beta=beta,
                    eps=eps,
                )

                scaled_loss = loss / accumulation_steps
                scaled_loss.backward()

                total_loss += loss.item() / accumulation_steps
                total_ppo_loss += ppo_loss / accumulation_steps
                total_kl_loss += kl_loss / accumulation_steps

            if max_grad_norm > 0:
                params = [p for group in opt.param_groups for p in group["params"]]
                torch.nn.utils.clip_grad_norm_(params, max_norm=max_grad_norm)
            opt.step()

        t_opt = time.perf_counter() - t_opt_start

        step += 1
        n_episodes += batch_size * accumulation_steps

        all_output_strs = [
            s
            for g in range(group_size)
            for mb in micro_batches
            for s in mb["output_strs"][g]
        ]
        all_output_strs_nested: list[list[str]] = [
            [mb["output_strs"][g][b] for g in range(group_size)]
            for mb in micro_batches
            for b in range(batch_size)
        ]
        completion_len_mean = sum(len(s) for s in all_output_strs) / len(
            all_output_strs
        )

        all_prompts = [p for mb in micro_batches for p in mb["prompts"]]
        all_reward_results = [
            [mb["reward_results"][g][b] for g in range(group_size)]
            for mb in micro_batches
            for b in range(batch_size)
        ]

        examples = extty.BatchExample(
            prompts=all_prompts,
            responses=all_output_strs_nested,
            rewards=[[r.components for r in row] for row in all_reward_results],
        )

        flat_reward_results = [
            mb["reward_results"][g][b]
            for mb in micro_batches
            for g in range(group_size)
            for b in range(batch_size)
        ]
        component_means = aggregate_reward_components(
            [[r] for r in flat_reward_results]
        )

        if extty._active_run is not None:
            extty.log(
                {
                    "train/loss": total_loss / mu,
                    "train/ppo_loss": total_ppo_loss / mu,
                    "train/kl_loss": total_kl_loss / mu,
                    "train/reward_mean": all_rewards.mean().item(),
                    "train/reward_std": all_rewards.std().item(),
                    "train/completion_len_mean": completion_len_mean,
                    "train/example": examples,
                    "train/generation_time": t_gen_total,
                    "train/logprobs_time": t_logprobs_total,
                    "train/optimization_time": t_opt,
                    **{
                        f"train/reward/{name}": mean
                        for name, mean in component_means.items()
                    },
                },
                step=step,
            )

            if step % save_ckpt_freq:
                extty.save_checkpoint(
                    step=step,
                    state_dict=net.state_dict(),
                    optimizer_state_dict=opt.state_dict(),
                )
