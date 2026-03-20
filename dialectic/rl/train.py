import sys
import time
import warnings
from copy import deepcopy
from typing import Any, Callable, Sequence

import extty
import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Integer
from tokenizers import Tokenizer

from dialectic.llm.base import BaseTransformer
from dialectic.llm.components import GradSafeKVCache, KVCache
from dialectic.llm.generate import PreFill
from dialectic.rl.env import Env, build_countdown_equation
from dialectic.rl.evaluate import (
    evaluate,
    evaluate_internal_reasoning,
    evaluate_soft_prefill,
)
from dialectic.rl.extractors import extract_from_answer_tags
from dialectic.rl.reward import RewardFn
from dialectic.rl.rollout import (
    generate_grouped_variable_length_internal_reasoning_rollout_batch,
    generate_internal_reasoning_rollout_batch,
    generate_rollout_batch,
    generate_soft_rollout_batch,
)
from dialectic.rl.types import A, E, RewardResult, T
from dialectic.training import StepFunctionReturn, train_loop


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
    full_mask = _expand_attention_mask(attention_mask, batch_size, group_size, seq_len)

    out = net(input_ids, return_all_logits=True, attention_mask=full_mask)
    out = out.view(batch_size, group_size, seq_len, -1)
    return out


def get_completion_mask(
    *,
    attention_mask: Bool[torch.Tensor, "B L_prompt"],
    completion_token_ids: list[Integer[torch.Tensor, "B L_completion"]]
    | list[Float[torch.Tensor, "B L_completion V"]],
    pad_token_id: int,
) -> Bool[torch.Tensor, "B G L_new"]:
    stacked = stack_and_pad(tensors=completion_token_ids, pad_token_id=pad_token_id)
    l_prompt = attention_mask.shape[1]
    only_completion = stacked[:, :, l_prompt:]
    completion_mask = only_completion != pad_token_id
    return completion_mask


def get_completion_mask_soft(
    *,
    attention_mask: Bool[torch.Tensor, "B L_prompt"],
    completion_embeddings: list[Float[torch.Tensor, "B L D"]],
    completion_shadow_ids: list[Integer[torch.Tensor, "B L"]],
    hard_tokens_mask: list[Bool[torch.Tensor, "B L"]],
    pad_token_id: int,
) -> Bool[torch.Tensor, "B G L_new"]:
    l_prompt = attention_mask.shape[1]
    _, stacked_shadow_ids, _, non_pad_mask = stack_and_pad_soft(
        embeddings=completion_embeddings,
        shadow_ids=completion_shadow_ids,
        hard_masks=hard_tokens_mask,
        pad_token_id=pad_token_id,
    )
    pad_mask = stacked_shadow_ids != pad_token_id
    completion_mask = (non_pad_mask & pad_mask)[:, :, l_prompt:]
    return completion_mask


def compute_log_probs(
    *,
    net: BaseTransformer,
    attention_mask: Bool[torch.Tensor, "B L_prompt"],
    completion_token_ids: list[Integer[torch.Tensor, "B L_completion"]]
    | list[Float[torch.Tensor, "B L_completion V"]],
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
        Size of chunks for log prob computation. 0 means no chunking
        (process full completion at once). Recommended: 64-128 for large
        vocab models to reduce memory usage.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        (log_probs, completion_mask) both of shape [B, G, L_completion]
    """
    l_prompt = attention_mask.shape[1]
    stacked = stack_and_pad(tensors=completion_token_ids, pad_token_id=pad_token_id)

    log_probs = _compute_log_probs_chunked(
        net=net,
        input_ids=stacked,
        attention_mask=attention_mask,
        l_prompt=l_prompt,
        chunk_size=chunk_size,
    )
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
    full_mask = _expand_attention_mask(attention_mask, batch_size, group_size, seq_len)

    hidden_states: Float[torch.Tensor, "BG L D"] = net(
        flat_input_ids,
        attention_mask=full_mask,
        return_hidden_states=True,
    )

    completion_len = seq_len - l_prompt
    hidden_for_completion = hidden_states[:, l_prompt - 1 : -1]
    target_tokens = flat_input_ids[:, l_prompt:]

    if chunk_size == 0:
        chunk_size = completion_len

    log_probs_list = []
    for chunk_start in range(0, completion_len, chunk_size):
        chunk_end = min(chunk_start + chunk_size, completion_len)

        chunk_hidden = hidden_for_completion[:, chunk_start:chunk_end]
        chunk_targets = target_tokens[:, chunk_start:chunk_end]

        chunk_logits = net.lm_head(chunk_hidden)

        BG, L_chunk, V = chunk_logits.shape
        chunk_log_probs = -torch.nn.functional.cross_entropy(
            chunk_logits.reshape(BG * L_chunk, V).float(),
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


def grpo_advantage(
    rewards: Float[torch.Tensor, "G B"],
    normalize: bool = True,
    eps: float = 1e-8,
) -> Float[torch.Tensor, "G B"]:
    mean = rewards.mean(0, keepdim=True)
    if normalize:
        if rewards.shape[0] <= 1:
            return rewards - mean
        # TODO: think there's some clipping option for std
        return (rewards - mean) / (rewards.std(0, keepdim=True) + eps)
    return rewards - mean


def rloo_advantage(rewards: Float[torch.Tensor, "G B"]) -> Float[torch.Tensor, "G B"]:
    G = rewards.shape[0]
    if G == 1:
        raise RuntimeError("Group size must be > 1 to use RLOO")
    return G / (G - 1) * rewards - 1 / (G - 1) * rewards.sum(0, keepdim=True)


def compute_grpo_loss(
    *,
    log_probs: Float[torch.Tensor, "B G L_new"],
    old_log_probs: Float[torch.Tensor, "B G L_new"] | None,
    ref_log_probs: Float[torch.Tensor, "B G L_new"] | None,
    completion_mask: Bool[torch.Tensor, "B G L_new"],
    advs: Float[torch.Tensor, "B G 1"],
    beta: float,
    eps: float | None,
    normalize_by_sequence_length: bool,
    clip_ratio_c: float = 3.0,
) -> tuple[torch.Tensor, float, float | None]:
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
        PPO clipping epsilon. (if None then no clipping)
    clip_ratio_c
        Dual-clip ratio for negative advantages.

    Returns
    -------
    tuple[torch.Tensor, float, float]
        (loss tensor, ppo_loss scalar, kl_loss scalar)
    """
    if (beta == 0) != (ref_log_probs is None):
        raise RuntimeError("`ref_log_probs` should be None if and only if `beta` is 0")

    if (eps is None) != (old_log_probs is None):
        raise RuntimeError(
            "`eps` should be None if and only if `old_log_probs` is None"
        )

    compute_kl_loss = beta != 0

    if eps is not None:
        # PPO style
        ratio = (log_probs - old_log_probs).exp()
        unclipped = ratio * advs
        clipped = torch.clip(ratio, 1 - eps, 1 + eps) * advs
        main_obj = torch.min(unclipped, clipped)
        dual_clip_obj = clip_ratio_c * advs
        main_obj = torch.where(advs < 0, torch.max(main_obj, dual_clip_obj), main_obj)
    else:
        # REINFORCE
        main_obj = log_probs * advs

    if compute_kl_loss:
        kl_diff = ref_log_probs - log_probs
        kl_diff = torch.clamp(kl_diff, min=-20, max=20)
        kl_loss = torch.exp(kl_diff) - kl_diff - 1
        kl_loss = torch.clamp(kl_loss, min=-10, max=10)
    else:
        kl_loss = None

    # Normalize per-sequence to avoid length bias: longer sequences should not
    # contribute more to the loss just because they have more tokens.
    # Each sequence's contribution is its mean (over tokens), then we average
    # across all sequences.
    if normalize_by_sequence_length:
        sequence_lengths = completion_mask.sum(dim=-1, keepdim=True).clamp(
            min=1
        )  # [B, G, 1]
        main_obj_per_seq = (main_obj * completion_mask).sum(
            dim=-1, keepdim=True
        ) / sequence_lengths
        kl_loss_per_seq = (
            (kl_loss * completion_mask).sum(dim=-1, keepdim=True) / sequence_lengths
            if compute_kl_loss
            else None
        )
    else:
        main_obj_per_seq = (main_obj * completion_mask).sum(dim=-1, keepdim=True)
        kl_loss_per_seq = (
            (kl_loss * completion_mask).sum(dim=-1, keepdim=True)
            if compute_kl_loss
            else None
        )

    main_loss_scalar = -main_obj_per_seq.mean()
    kl_loss_scalar = kl_loss_per_seq.mean() if compute_kl_loss else 0

    loss = main_loss_scalar + beta * kl_loss_scalar
    return (
        loss,
        main_loss_scalar.item(),
        kl_loss_scalar.item() if compute_kl_loss else None,
    )


@torch.no_grad()
def collect_micro_batch(
    *,
    net: BaseTransformer,
    ref_net: BaseTransformer | None,
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
    collect_old_log_probs: bool,
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
        if ref_net is not None:
            ref_log_probs, _ = compute_log_probs(
                net=ref_net,
                attention_mask=rollout.attention_mask,
                completion_token_ids=rollout.completion_token_ids,
                pad_token_id=pad_token_id,
                chunk_size=logprob_chunk_size,
            )
        else:
            ref_log_probs = None
        if collect_old_log_probs:
            old_log_probs, completion_mask = compute_log_probs(
                net=net,
                attention_mask=rollout.attention_mask,
                completion_token_ids=rollout.completion_token_ids,
                pad_token_id=pad_token_id,
                chunk_size=logprob_chunk_size,
            )
        else:
            old_log_probs = None
            completion_mask = get_completion_mask(
                attention_mask=rollout.attention_mask,
                completion_token_ids=rollout.completion_token_ids,
                pad_token_id=pad_token_id,
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


def create_grpo_step_fn(
    *,
    net: BaseTransformer,
    opt: torch.optim.Optimizer,
    collect_fn: Callable[[BaseTransformer, BaseTransformer | None], dict],
    recompute_log_probs_fn: Callable[
        [BaseTransformer, dict], tuple[torch.Tensor, torch.Tensor]
    ],
    beta: float,
    eps: float | None,
    mu: int,
    update_ref_net_batch_cadence: int | None,
    batch_size: int,
    group_size: int,
    normalize_by_sequence_length: bool,
    accumulation_steps: int,
    max_grad_norm: float,
    use_bf16: bool,
    device: torch.device,
    advantage_fn: Callable[[Float[torch.Tensor, "G B"]], Float[torch.Tensor, "G B"]],
    warmup_steps: int = 0,
):
    if mu > 1 and eps is None:
        raise RuntimeError(
            "Should not have `mu` > 1 when not doing PPO style training."
        )

    scheduler = None
    if warmup_steps > 0:
        scheduler = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps
        )

    ref_net = None

    def _step(step: int):
        nonlocal ref_net, net
        if beta != 0 and (
            (update_ref_net_batch_cadence is None and step == 0)
            or (
                update_ref_net_batch_cadence is not None
                and (step % update_ref_net_batch_cadence == 0)
            )
        ):
            ref_net = deepcopy(net)

        t_gen_total = 0.0
        t_logprobs_total = 0.0
        micro_batches: list[dict] = []

        for _ in range(accumulation_steps):
            micro_batch = collect_fn(net, ref_net)
            micro_batches.append(micro_batch)
            t_gen_total += micro_batch["t_gen"]
            t_logprobs_total += micro_batch["t_logprobs"]

        all_rewards = torch.cat(
            [mb["rewards"] for mb in micro_batches], dim=1
        )  # [G, B*accum]
        global_advs: Float[torch.Tensor, "G B*accum"] = advantage_fn(all_rewards)

        t_opt_start = time.perf_counter()
        total_loss = 0.0
        total_main_loss = 0.0
        total_kl_loss = 0.0 if beta != 0 else None
        grad_norm_val: float | None = None

        logprob_recompute_max_diffs: list[float] = []
        for mu_idx in range(mu):
            opt.zero_grad()

            for accum_idx, micro_batch in enumerate(micro_batches):
                advs_slice = global_advs[
                    :, accum_idx * batch_size : (accum_idx + 1) * batch_size
                ]
                advs_for_loss: Float[torch.Tensor, "B G 1"] = advs_slice.T.unsqueeze(-1)

                micro_batch["_advs_for_loss"] = advs_for_loss
                micro_batch["_loss_scale"] = 1.0 / accumulation_steps

                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
                ):
                    log_probs, _ = recompute_log_probs_fn(net, micro_batch)

                if mu_idx == 0 and "diag_log_probs" in micro_batch:
                    mask = micro_batch["completion_mask"]
                    if mask.any():
                        lp_diff = (
                            log_probs.detach() - micro_batch["diag_log_probs"]
                        ).abs()
                        logprob_recompute_max_diffs.append(lp_diff[mask].max().item())

                loss, main_loss, kl_loss = compute_grpo_loss(
                    log_probs=log_probs,
                    old_log_probs=micro_batch["old_log_probs"],
                    ref_log_probs=micro_batch["ref_log_probs"],
                    completion_mask=micro_batch["completion_mask"],
                    advs=advs_for_loss,
                    beta=beta,
                    eps=eps,
                    normalize_by_sequence_length=normalize_by_sequence_length,
                )

                scaled_loss = loss / accumulation_steps
                scaled_loss.backward()

                total_loss += loss.item() / accumulation_steps
                total_main_loss += main_loss / accumulation_steps
                if kl_loss is not None:
                    total_kl_loss += kl_loss / accumulation_steps

            grad_norm_val = None
            if max_grad_norm > 0:
                params = [p for group in opt.param_groups for p in group["params"]]
                grad_norm_val = torch.nn.utils.clip_grad_norm_(
                    params, max_norm=max_grad_norm
                ).item()
            opt.step()
            if scheduler is not None:
                scheduler.step()

        t_opt = time.perf_counter() - t_opt_start
        current_lr = opt.param_groups[0]["lr"]

        all_output_strs_nested: list[list[str]] = [
            [mb["output_strs"][g][b] for g in range(group_size)]
            for mb in micro_batches
            for b in range(batch_size)
        ]
        total_completion_tokens = sum(
            mb["completion_mask"].sum().item() for mb in micro_batches
        )
        total_completions = sum(
            mb["completion_mask"].shape[0] * mb["completion_mask"].shape[1]
            for mb in micro_batches
        )
        completion_token_len_mean = total_completion_tokens / total_completions

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
        metrics = {
            "train/loss": total_loss / mu,
            "train/main_loss": total_main_loss / mu,
            "train/reward_mean": all_rewards.mean().item(),
            "train/reward_std": all_rewards.std().item(),
            "train/completion_token_len_mean": completion_token_len_mean,
            "train/example": examples,
            "train/generation_time": t_gen_total,
            "train/logprobs_time": t_logprobs_total,
            "train/optimization_time": t_opt,
            "train/lr": current_lr,
            **(
                {
                    "train/hard_completion_ratio": sum(
                        mb["hard_completion_ratio"] for mb in micro_batches
                    )
                    / len(micro_batches)
                }
                if "hard_completion_ratio" in micro_batches[0]
                else {}
            ),
            **(
                {
                    "train/hard_lp_mean": sum(
                        mb["hard_lp_mean"] for mb in micro_batches
                    )
                    / len(micro_batches),
                    "train/hard_lp_std": sum(mb["hard_lp_std"] for mb in micro_batches)
                    / len(micro_batches),
                    "train/soft_lp_mean": sum(
                        mb["soft_lp_mean"] for mb in micro_batches
                    )
                    / len(micro_batches),
                    "train/soft_lp_std": sum(mb["soft_lp_std"] for mb in micro_batches)
                    / len(micro_batches),
                    "train/gaussian_dist_mean": sum(
                        mb["gaussian_dist_mean"] for mb in micro_batches
                    )
                    / len(micro_batches),
                    "train/soft_tokens_per_seq": sum(
                        mb["soft_tokens_per_seq"] for mb in micro_batches
                    )
                    / len(micro_batches),
                    "train/hard_tokens_per_seq": sum(
                        mb["hard_tokens_per_seq"] for mb in micro_batches
                    )
                    / len(micro_batches),
                }
                if "hard_lp_mean" in micro_batches[0]
                else {}
            ),
            **{f"train/reward/{name}": mean for name, mean in component_means.items()},
        }
        if total_kl_loss is not None:
            metrics["train/kl_loss"] = total_kl_loss / mu
        if grad_norm_val is not None:
            metrics["train/grad_norm"] = grad_norm_val
        if logprob_recompute_max_diffs:
            metrics["train/logprob_recompute_max_diff"] = max(
                logprob_recompute_max_diffs
            )
        if net.soft_projection_alpha is not None:
            metrics["train/soft_projection_alpha"] = net.soft_projection_alpha.item()

        return StepFunctionReturn(
            n_episodes_processed=all_rewards.shape[1], metrics=metrics
        )

    return _step


def create_grpo_val_fn(
    *,
    net: BaseTransformer,
    state_to_str: Callable,
    tokenizer: Tokenizer,
    pad_token_id: int,
    eos_token_id: int,
    use_bf16: bool,
    val_episodes: int,
    val_batch_size: int,
    reward_fn: RewardFn,
    max_tokens_generated: int,
):
    def _val(env: Env):
        return evaluate(
            net=net,
            env=env,
            reward_fn=reward_fn,
            state_to_str=state_to_str,
            tokenizer=tokenizer,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            extractor=extract_from_answer_tags,
            max_tokens_generated=max_tokens_generated,
            max_episodes=val_episodes,
            batch_size=val_batch_size,
            group_size=1,
            temperature=0.0,
            use_bf16=use_bf16,
            n_examples=val_episodes,
        )

    return _val


def _grpo_train_loop(
    *,
    net: BaseTransformer,
    opt: torch.optim.Optimizer,
    collect_fn: Callable[[BaseTransformer, BaseTransformer], dict],
    recompute_log_probs_fn: Callable[
        [BaseTransformer, dict], tuple[torch.Tensor, torch.Tensor]
    ],
    beta: float,
    eps: float | None,
    mu: int,
    max_episodes: int,
    update_ref_net_batch_cadence: int | None,
    batch_size: int,
    group_size: int,
    normalize_by_sequence_length: bool,
    accumulation_steps: int,
    max_grad_norm: float,
    use_bf16: bool,
    save_ckpt_freq: int,
    advantage_fn: Callable[[Float[torch.Tensor, "G B"]], Float[torch.Tensor, "G B"]],
    val_fn,
    val_envs: list[Env],
    val_freq: int = 0,
    warmup_steps: int = 0,
) -> None:
    device = next(net.parameters()).device

    train_step = create_grpo_step_fn(
        net=net,
        opt=opt,
        collect_fn=collect_fn,
        recompute_log_probs_fn=recompute_log_probs_fn,
        beta=beta,
        eps=eps,
        mu=mu,
        update_ref_net_batch_cadence=update_ref_net_batch_cadence,
        batch_size=batch_size,
        group_size=group_size,
        normalize_by_sequence_length=normalize_by_sequence_length,
        accumulation_steps=accumulation_steps,
        max_grad_norm=max_grad_norm,
        use_bf16=use_bf16,
        device=device,
        advantage_fn=advantage_fn,
        warmup_steps=warmup_steps,
    )

    train_loop(
        max_episodes=max_episodes,
        save_ckpt_freq=save_ckpt_freq,
        val_freq=val_freq,
        net=net,
        opt=opt,
        train_step=train_step,
        val_envs=val_envs,
        val_fn=val_fn,
    )


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
    eps: float | None,
    mu: int,  # number of optimization passes per accumulated batch
    max_tokens_generated: int,
    max_episodes: int,
    update_ref_net_batch_cadence: int | None,
    batch_size: int,
    group_size: int,
    temperature: float,
    advantage_fn: Callable[[Float[torch.Tensor, "G B"]], Float[torch.Tensor, "G B"]],
    normalize_by_sequence_length: bool,
    accumulation_steps: int,
    max_grad_norm: float,
    logprob_chunk_size: int,
    use_bf16: bool,
    save_ckpt_freq: int = sys.maxsize,
    val_episodes: int,
    val_envs: list[Env],
    val_batch_size: int,
    val_freq: int = 0,
    warmup_steps: int = 0,
) -> None:
    if use_bf16:
        net = net.to(dtype=torch.bfloat16)

    collect_old_log_probs = eps is not None

    def collect_fn(net: BaseTransformer, ref_net: BaseTransformer | None) -> dict:
        return collect_micro_batch(
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
            collect_old_log_probs=collect_old_log_probs,
        )

    def recompute_fn(
        net: BaseTransformer, mb: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return compute_log_probs(
            net=net,
            attention_mask=mb["attention_mask"],
            completion_token_ids=mb["completion_token_ids"],
            pad_token_id=pad_token_id,
            chunk_size=logprob_chunk_size,
        )

    val_fn = create_grpo_val_fn(
        net=net,
        state_to_str=state_to_str,
        tokenizer=tokenizer,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        use_bf16=use_bf16,
        val_episodes=val_episodes,
        val_batch_size=val_batch_size,
        reward_fn=reward_fn,
        max_tokens_generated=max_tokens_generated,
    )

    _grpo_train_loop(
        net=net,
        opt=opt,
        collect_fn=collect_fn,
        recompute_log_probs_fn=recompute_fn,
        beta=beta,
        eps=eps,
        mu=mu,
        max_episodes=max_episodes,
        update_ref_net_batch_cadence=update_ref_net_batch_cadence,
        batch_size=batch_size,
        group_size=group_size,
        advantage_fn=advantage_fn,
        accumulation_steps=accumulation_steps,
        max_grad_norm=max_grad_norm,
        use_bf16=use_bf16,
        save_ckpt_freq=save_ckpt_freq,
        normalize_by_sequence_length=normalize_by_sequence_length,
        val_freq=val_freq,
        val_envs=val_envs,
        val_fn=val_fn,
        warmup_steps=warmup_steps,
    )


def stack_and_pad_soft(
    *,
    embeddings: list[Float[torch.Tensor, "B L D"]],
    shadow_ids: list[Integer[torch.Tensor, "B L"]],
    hard_masks: list[Bool[torch.Tensor, "B L"]],
    pad_token_id: int,
) -> tuple[
    Float[torch.Tensor, "B G L D"],
    Integer[torch.Tensor, "B G L"],
    Bool[torch.Tensor, "B G L"],
    Bool[torch.Tensor, "B G L"],
]:
    max_len = max(e.shape[1] for e in embeddings)
    batch_size = embeddings[0].shape[0]
    group_size = len(embeddings)
    D = embeddings[0].shape[2]
    device = embeddings[0].device

    stacked_embeddings = torch.zeros(
        batch_size, group_size, max_len, D, device=device, dtype=embeddings[0].dtype
    )
    stacked_shadow_ids = torch.full(
        (batch_size, group_size, max_len),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    stacked_masks = torch.ones(
        batch_size, group_size, max_len, device=device, dtype=torch.bool
    )
    non_pad_mask = torch.zeros(
        batch_size, group_size, max_len, device=device, dtype=torch.bool
    )

    for g, (e, s, m) in enumerate(zip(embeddings, shadow_ids, hard_masks)):
        L = e.shape[1]
        stacked_embeddings[:, g, :L] = e
        stacked_shadow_ids[:, g, :L] = s
        stacked_masks[:, g, :L] = m
        non_pad_mask[:, g, :L] = True

    return stacked_embeddings, stacked_shadow_ids, stacked_masks, non_pad_mask


def compute_soft_log_probs(
    *,
    net: BaseTransformer,
    attention_mask: Bool[torch.Tensor, "B L_prompt"],
    completion_embeddings: list[Float[torch.Tensor, "B L D"]],
    completion_shadow_ids: list[Integer[torch.Tensor, "B L"]],
    hard_tokens_mask: list[Bool[torch.Tensor, "B L"]],
    noise_std: float,
    temperature: float,
    pad_token_id: int,
    normalize_soft_pdf_by_dim: bool,
    chunk_size: int = 0,
) -> tuple[Float[torch.Tensor, "B G L_c"], Bool[torch.Tensor, "B G L_c"]]:
    l_prompt = attention_mask.shape[1]
    stacked_embeddings, stacked_shadow_ids, stacked_masks, non_pad_mask = (
        stack_and_pad_soft(
            embeddings=completion_embeddings,
            shadow_ids=completion_shadow_ids,
            hard_masks=hard_tokens_mask,
            pad_token_id=pad_token_id,
        )
    )

    log_probs = _compute_soft_log_probs_chunked(
        net=net,
        stacked_embeddings=stacked_embeddings,
        stacked_shadow_ids=stacked_shadow_ids,
        stacked_masks=stacked_masks,
        attention_mask=attention_mask,
        l_prompt=l_prompt,
        noise_std=noise_std,
        temperature=temperature,
        chunk_size=chunk_size,
        normalize_soft_pdf_by_dim=normalize_soft_pdf_by_dim,
    )

    pad_mask = stacked_shadow_ids != pad_token_id
    completion_mask = (non_pad_mask & pad_mask)[:, :, l_prompt:]

    return log_probs, completion_mask


def _expand_attention_mask(
    attention_mask: Bool[torch.Tensor, "B L_prompt"] | None,
    batch_size: int,
    group_size: int,
    seq_len: int,
) -> Bool[torch.Tensor, "BG L"] | None:
    if attention_mask is None:
        return None
    l_prompt = attention_mask.shape[1]
    full_mask = torch.ones(
        batch_size,
        seq_len,
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    full_mask[:, :l_prompt] = attention_mask
    full_mask = full_mask.unsqueeze(1).expand(-1, group_size, -1)
    return full_mask.reshape(batch_size * group_size, seq_len)


def _compute_soft_log_probs_chunked(
    *,
    net: BaseTransformer,
    stacked_embeddings: Float[torch.Tensor, "B G L D"],
    stacked_shadow_ids: Integer[torch.Tensor, "B G L"],
    stacked_masks: Bool[torch.Tensor, "B G L"],
    attention_mask: Bool[torch.Tensor, "B L_prompt"],
    l_prompt: int,
    noise_std: float,
    temperature: float,
    chunk_size: int,
    normalize_soft_pdf_by_dim: bool,
) -> Float[torch.Tensor, "B G L_c"]:
    batch_size, group_size, seq_len, D = stacked_embeddings.shape
    BG = batch_size * group_size
    V = net.vocab_size

    flat_embeddings = stacked_embeddings.view(BG, seq_len, D)
    flat_shadow_ids = stacked_shadow_ids.view(BG, seq_len)

    full_mask = _expand_attention_mask(attention_mask, batch_size, group_size, seq_len)

    hidden_states = net(
        flat_embeddings,
        attention_mask=full_mask,
        return_hidden_states=True,
    )

    completion_len = seq_len - l_prompt
    hidden_for_completion = hidden_states[:, l_prompt - 1 : -1]
    comp_shadow_ids = flat_shadow_ids[:, l_prompt:]
    comp_embeddings = flat_embeddings[:, l_prompt:]
    comp_masks = stacked_masks.view(BG, seq_len)[:, l_prompt:]
    W = net.embed_tokens.weight

    if chunk_size == 0:
        chunk_size = completion_len

    log_probs_list = []
    for chunk_start in range(0, completion_len, chunk_size):
        chunk_end = min(chunk_start + chunk_size, completion_len)

        chunk_hidden = hidden_for_completion[:, chunk_start:chunk_end]
        chunk_logits = net.lm_head(chunk_hidden)
        chunk_shadow_ids = comp_shadow_ids[:, chunk_start:chunk_end]
        chunk_embeddings = comp_embeddings[:, chunk_start:chunk_end]
        chunk_masks = comp_masks[:, chunk_start:chunk_end]

        BG_c, L_chunk, _ = chunk_logits.shape

        hard_lp = -torch.nn.functional.cross_entropy(
            chunk_logits.reshape(BG_c * L_chunk, V).float(),
            chunk_shadow_ids.reshape(BG_c * L_chunk),
            reduction="none",
        ).reshape(BG_c, L_chunk)

        device_type = chunk_embeddings.device.type
        with torch.amp.autocast(device_type=device_type, enabled=False):
            e_action = chunk_embeddings.float()
            mu_new = (
                torch.softmax(chunk_logits.float() / temperature, dim=-1) @ W.float()
            )
            soft_lp = -0.5 * ((e_action - mu_new) ** 2) / (noise_std**2)
            if normalize_soft_pdf_by_dim:
                soft_lp = soft_lp.mean(-1)
            else:
                soft_lp = soft_lp.sum(-1)

        chunk_log_probs = torch.where(chunk_masks, hard_lp, soft_lp)
        log_probs_list.append(chunk_log_probs)

        del chunk_logits

    log_probs_flat = torch.cat(log_probs_list, dim=1).view(
        batch_size, group_size, completion_len
    )

    return log_probs_flat


@torch.no_grad()
def collect_soft_micro_batch(
    *,
    net: BaseTransformer,
    ref_net: BaseTransformer | None,
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
    collect_old_log_probs: bool,
    normalize_soft_pdf_by_dim: bool,
    switch_to_hard_tokens_condition: torch.Tensor | None = None,
    max_tokens_prefill: torch.Tensor | None = None,
    max_tokens_prefill_steps_before_end: int = 0,
    min_soft_steps: int = 0,
    prefill: PreFill | None = None,
    logprob_chunk_size: int = 0,
    use_bf16: bool = False,
) -> dict:
    rollout = generate_soft_rollout_batch(
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
        noise_std=noise_std,
        switch_to_hard_tokens_condition=switch_to_hard_tokens_condition,
        max_tokens_prefill=max_tokens_prefill,
        max_tokens_prefill_steps_before_end=max_tokens_prefill_steps_before_end,
        min_soft_steps=min_soft_steps,
        prefill=prefill,
        use_bf16=use_bf16,
    )

    device = next(net.parameters()).device
    t_logprobs_start = time.perf_counter()
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
    ):
        if ref_net is not None:
            ref_log_probs, _ = compute_soft_log_probs(
                net=ref_net,
                attention_mask=rollout.attention_mask,
                completion_embeddings=rollout.completion_embeddings,
                completion_shadow_ids=rollout.completion_shadow_ids,
                hard_tokens_mask=rollout.hard_tokens_mask,
                noise_std=rollout.noise_std,
                temperature=rollout.temperature,
                pad_token_id=pad_token_id,
                chunk_size=logprob_chunk_size,
                normalize_soft_pdf_by_dim=normalize_soft_pdf_by_dim,
            )
        else:
            ref_log_probs = None
        diag_log_probs, completion_mask = compute_soft_log_probs(
            net=net,
            attention_mask=rollout.attention_mask,
            completion_embeddings=rollout.completion_embeddings,
            completion_shadow_ids=rollout.completion_shadow_ids,
            hard_tokens_mask=rollout.hard_tokens_mask,
            noise_std=rollout.noise_std,
            temperature=rollout.temperature,
            pad_token_id=pad_token_id,
            chunk_size=logprob_chunk_size,
            normalize_soft_pdf_by_dim=normalize_soft_pdf_by_dim,
        )
        old_log_probs = diag_log_probs if collect_old_log_probs else None
    t_logprobs = time.perf_counter() - t_logprobs_start

    # Soft-token debug stats (cheap): fraction of completion tokens that are hard,
    # and a guard that pad tokens are not counted in completion_mask.
    l_prompt = rollout.attention_mask.shape[1]
    B = rollout.attention_mask.shape[0]
    G = len(rollout.hard_tokens_mask)
    Lc = completion_mask.shape[2]
    hard_completion_mask = torch.zeros((B, G, Lc), device=device, dtype=torch.bool)
    pad_ok_mask = torch.zeros((B, G, Lc), device=device, dtype=torch.bool)

    for g, hard_mask in enumerate(rollout.hard_tokens_mask):
        comp_hard = hard_mask[:, l_prompt:]
        comp_len = min(comp_hard.shape[1], Lc)
        hard_completion_mask[:, g, :comp_len] = comp_hard[:, :comp_len]

        comp_pad_ok = rollout.completion_shadow_ids[g][:, l_prompt:] != pad_token_id
        pad_len = min(comp_pad_ok.shape[1], Lc)
        pad_ok_mask[:, g, :pad_len] = comp_pad_ok[:, :pad_len]

    if __debug__:
        if torch.any(completion_mask & ~pad_ok_mask):
            raise RuntimeError("completion_mask includes padded tokens in soft rollout")

    hard_completion_ratio = (
        (hard_completion_mask & completion_mask).sum().float()
        / completion_mask.sum().float().clamp_min(1.0)
    ).item()

    hard_pos_mask = hard_completion_mask & completion_mask
    soft_pos_mask = ~hard_completion_mask & completion_mask

    if hard_pos_mask.any():
        hard_lp_vals = diag_log_probs[hard_pos_mask]
        hard_lp_mean = hard_lp_vals.mean().item()
        hard_lp_std = hard_lp_vals.std().item()
    else:
        hard_lp_mean = hard_lp_std = 0.0

    if soft_pos_mask.any():
        soft_lp_vals = diag_log_probs[soft_pos_mask]
        soft_lp_mean = soft_lp_vals.mean().item()
        soft_lp_std = soft_lp_vals.std().item()
        if normalize_soft_pdf_by_dim:
            gaussian_dist_mean = (-2.0 * soft_lp_vals).mean().item()
        else:
            gaussian_dist_mean = (-2.0 * soft_lp_vals / net.d).mean().item()
    else:
        soft_lp_mean = soft_lp_std = gaussian_dist_mean = 0.0

    soft_tokens_per_seq = soft_pos_mask.sum(dim=-1).float().mean().item()
    hard_tokens_per_seq = hard_pos_mask.sum(dim=-1).float().mean().item()

    return {
        "prompts": rollout.prompts,
        "env_responses": rollout.env_responses,
        "attention_mask": rollout.attention_mask,
        "completion_embeddings": rollout.completion_embeddings,
        "completion_shadow_ids": rollout.completion_shadow_ids,
        "hard_tokens_mask": rollout.hard_tokens_mask,
        "noise_std": rollout.noise_std,
        "temperature": rollout.temperature,
        "output_strs": rollout.output_strs,
        "reward_results": rollout.reward_results,
        "rewards": rollout.rewards,
        "ref_log_probs": ref_log_probs,
        "old_log_probs": old_log_probs,
        "diag_log_probs": diag_log_probs,
        "completion_mask": completion_mask,
        "hard_completion_ratio": hard_completion_ratio,
        "hard_lp_mean": hard_lp_mean,
        "hard_lp_std": hard_lp_std,
        "soft_lp_mean": soft_lp_mean,
        "soft_lp_std": soft_lp_std,
        "gaussian_dist_mean": gaussian_dist_mean,
        "soft_tokens_per_seq": soft_tokens_per_seq,
        "hard_tokens_per_seq": hard_tokens_per_seq,
        "t_gen": rollout.t_generation,
        "t_logprobs": t_logprobs,
    }


def train_soft_grpo(
    *,
    net: BaseTransformer,
    opt: torch.optim.Optimizer,
    env: Env[T, A],
    reward_fn: RewardFn[T, E],
    state_to_str: Callable[[T], str],
    tokenizer: Tokenizer,
    eos_token_id: int,
    pad_token_id: int,
    extractor: Callable[[str], E],
    beta: float,
    eps: float | None,
    mu: int,
    max_tokens_generated: int,
    max_episodes: int,
    update_ref_net_batch_cadence: int | None,
    batch_size: int,
    group_size: int,
    temperature: float,
    noise_std: float,
    normalize_by_sequence_length: bool,
    normalize_soft_pdf_by_dim: bool,
    advantage_fn: Callable[[Float[torch.Tensor, "G B"]], Float[torch.Tensor, "G B"]],
    switch_to_hard_tokens_condition: torch.Tensor | None = None,
    max_tokens_prefill: torch.Tensor | None = None,
    min_soft_steps: int = 0,
    prefill: PreFill | None = None,
    max_tokens_prefill_steps_before_end: int = 0,
    accumulation_steps: int,
    max_grad_norm: float,
    logprob_chunk_size: int = 64,
    use_bf16: bool = True,
    save_ckpt_freq: int = sys.maxsize,
    val_episodes: int,
    val_envs: list[Env],
    val_batch_size: int,
    val_freq: int = 0,
    warmup_steps: int = 0,
) -> None:
    if use_bf16:
        net = net.to(dtype=torch.bfloat16)

    collect_old_log_probs = eps is not None

    def collect_fn(net: BaseTransformer, ref_net: BaseTransformer) -> dict:
        return collect_soft_micro_batch(
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
            collect_old_log_probs=collect_old_log_probs,
            noise_std=noise_std,
            switch_to_hard_tokens_condition=switch_to_hard_tokens_condition,
            max_tokens_prefill=max_tokens_prefill,
            max_tokens_prefill_steps_before_end=max_tokens_prefill_steps_before_end,
            min_soft_steps=min_soft_steps,
            prefill=prefill,
            logprob_chunk_size=logprob_chunk_size,
            use_bf16=use_bf16,
            normalize_soft_pdf_by_dim=normalize_soft_pdf_by_dim,
        )

    def recompute_fn(
        net: BaseTransformer, mb: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return compute_soft_log_probs(
            net=net,
            attention_mask=mb["attention_mask"],
            completion_embeddings=mb["completion_embeddings"],
            completion_shadow_ids=mb["completion_shadow_ids"],
            hard_tokens_mask=mb["hard_tokens_mask"],
            noise_std=mb["noise_std"],
            temperature=mb["temperature"],
            pad_token_id=pad_token_id,
            chunk_size=logprob_chunk_size,
            normalize_soft_pdf_by_dim=normalize_soft_pdf_by_dim,
        )

    val_fn = create_grpo_val_fn(
        net=net,
        state_to_str=state_to_str,
        tokenizer=tokenizer,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        use_bf16=use_bf16,
        val_episodes=val_episodes,
        val_batch_size=val_batch_size,
        reward_fn=reward_fn,
        max_tokens_generated=max_tokens_generated,
    )

    _grpo_train_loop(
        net=net,
        opt=opt,
        collect_fn=collect_fn,
        recompute_log_probs_fn=recompute_fn,
        advantage_fn=advantage_fn,
        beta=beta,
        eps=eps,
        mu=mu,
        max_episodes=max_episodes,
        update_ref_net_batch_cadence=update_ref_net_batch_cadence,
        batch_size=batch_size,
        group_size=group_size,
        normalize_by_sequence_length=normalize_by_sequence_length,
        accumulation_steps=accumulation_steps,
        max_grad_norm=max_grad_norm,
        use_bf16=use_bf16,
        save_ckpt_freq=save_ckpt_freq,
        val_freq=val_freq,
        val_fn=val_fn,
        val_envs=val_envs,
        warmup_steps=warmup_steps,
    )


def stack_and_pad_internal_reasoning(
    hard_token_ids: list[Integer[torch.Tensor, "B C"]],
    n_cycles: list[Integer[torch.Tensor, " B"]],
    pad_token_id: int,
) -> tuple[Integer[torch.Tensor, "B G C"], Integer[torch.Tensor, "B G"]]:
    max_c = max(t.shape[1] for t in hard_token_ids)
    B = hard_token_ids[0].shape[0]
    G = len(hard_token_ids)
    device = hard_token_ids[0].device

    stacked_ids = torch.full(
        (B, G, max_c), pad_token_id, dtype=torch.long, device=device
    )
    stacked_n = torch.zeros(B, G, dtype=torch.long, device=device)

    for g, (ids, nc) in enumerate(zip(hard_token_ids, n_cycles)):
        C = ids.shape[1]
        stacked_ids[:, g, :C] = ids
        stacked_n[:, g] = nc

    return stacked_ids, stacked_n


def compute_internal_reasoning_log_probs(
    *,
    net: BaseTransformer,
    prompt_token_ids: Integer[torch.Tensor, "B L"],
    attention_mask: Bool[torch.Tensor, "B L"],
    hard_token_ids: Integer[torch.Tensor, "B G C"],
    n_cycles: Integer[torch.Tensor, "B G"],
    valid_hard_token_ids: list[int],
    soft_block_size: int,
    use_bf16: bool = False,
    soft_bptt_window: int | None = None,
    cycle_callback: Callable[
        [Float[torch.Tensor, "B G"], int], Float[torch.Tensor, "B G"]
    ]
    | None = None,
    think_token_id: int | None = None,
) -> tuple[Float[torch.Tensor, "B G C"], Bool[torch.Tensor, "B G C"]]:
    """Compute log probs for fixed-length (1 token per cycle) internal reasoning.

    Thin wrapper around ``compute_variable_length_internal_reasoning_log_probs``
    with T_max=1. See that function for full documentation.
    """
    B, G, C = hard_token_ids.shape
    hard_token_lengths = torch.ones(
        B, G, C, dtype=torch.long, device=prompt_token_ids.device
    )
    return compute_variable_length_internal_reasoning_log_probs(
        net=net,
        prompt_token_ids=prompt_token_ids,
        attention_mask=attention_mask,
        hard_token_ids=hard_token_ids.unsqueeze(-1),
        hard_token_lengths=hard_token_lengths,
        n_cycles=n_cycles,
        soft_block_size=soft_block_size,
        valid_hard_token_ids=valid_hard_token_ids,
        use_bf16=use_bf16,
        soft_bptt_window=soft_bptt_window,
        cycle_callback=cycle_callback,
        think_token_id=think_token_id,
    )


def compute_soft_prefill_log_probs(
    *,
    net: BaseTransformer,
    prompt_token_ids: Integer[torch.Tensor, "B L"],
    attention_mask: Bool[torch.Tensor, "B L"],
    answer_token_ids: Integer[torch.Tensor, "B A"],
    answer_lengths: Integer[torch.Tensor, " B"],
    soft_block_size: int,
    pad_token_id: int,
    use_bf16: bool = True,
    soft_bptt_window: int | None = None,
) -> tuple[Float[torch.Tensor, "B A"], Bool[torch.Tensor, "B A"]]:
    """Compute log probs for soft-prefill + teacher-forced answer.

    Parameters
    ----------
    net
        The transformer model.
    prompt_token_ids
        Prompt token IDs [B, L], left-padded.
    attention_mask
        Attention mask for prompt [B, L].
    answer_token_ids
        Answer token IDs [B, A], right-padded with pad_token_id.
    answer_lengths
        Actual answer length per sample [B].
    soft_block_size
        Number of hidden-state passes in the soft block.
    pad_token_id
        Padding token ID.
    use_bf16
        Whether to use bf16 autocast.
    soft_bptt_window
        Number of soft passes at the end to backprop through.
        None means full BPTT (all soft passes).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        (log_probs [B, A], completion_mask [B, A])
    """
    if soft_bptt_window is None:
        soft_bptt_window = soft_block_size
    n_no_grad_soft = soft_block_size - soft_bptt_window

    B, L = prompt_token_ids.shape
    A = answer_token_ids.shape[1]
    device = prompt_token_ids.device

    max_seq_len = L + soft_block_size + A
    std_kv_caches = [
        KVCache(
            max_seq_len=max_seq_len,
            num_heads=net.attn_num_kv_heads,
            head_dim=net.attn_head_d,
            device=device,
        )
        for _ in range(len(net.layers))
    ]

    with torch.no_grad():
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
        ):
            h = net(
                prompt_token_ids,
                kv_caches=std_kv_caches,
                attention_mask=attention_mask,
                return_hidden_states=True,
            )
    h = h[:, -1:]  # [B, 1, D]

    grad_kv_caches: list[GradSafeKVCache | KVCache] = [
        GradSafeKVCache.from_standard(c) for c in std_kv_caches
    ]

    attn_mask = attention_mask
    ones = torch.ones(B, 1, dtype=torch.bool, device=device)

    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
    ):
        if n_no_grad_soft > 0:
            with torch.no_grad():
                for _ in range(n_no_grad_soft):
                    attn_mask = torch.cat([attn_mask, ones], dim=1)
                    h = net(
                        h,
                        kv_caches=grad_kv_caches,
                        attention_mask=attn_mask,
                        return_hidden_states=True,
                    )
                    h = net.apply_soft_projection(h)
            h = h.detach()

        for _ in range(soft_bptt_window):
            attn_mask = torch.cat([attn_mask, ones], dim=1)
            h = net(
                h,
                kv_caches=grad_kv_caches,
                attention_mask=attn_mask,
                return_hidden_states=True,
            )
            h = net.apply_soft_projection(h)

        attn_mask = torch.cat([attn_mask, ones], dim=1)
        logits_0 = net(
            h,
            kv_caches=grad_kv_caches,
            attention_mask=attn_mask,
        )  # [B, 1, V]

        logits_list = [logits_0]
        for i in range(A - 1):
            attn_mask = torch.cat([attn_mask, ones], dim=1)
            logits_i = net(
                answer_token_ids[:, i : i + 1],
                kv_caches=grad_kv_caches,
                attention_mask=attn_mask,
            )  # [B, 1, V]
            logits_list.append(logits_i)

        all_logits = torch.cat(logits_list, dim=1)  # [B, A, V]

    V = all_logits.shape[-1]
    log_probs_all = -torch.nn.functional.cross_entropy(
        all_logits.reshape(B * A, V).float(),
        answer_token_ids.reshape(B * A),
        reduction="none",
    ).reshape(B, A)

    answer_indices = torch.arange(A, device=device).unsqueeze(0).expand(B, A)
    completion_mask = answer_indices < answer_lengths.unsqueeze(1)

    return log_probs_all, completion_mask


def make_per_cycle_backward_callback(
    *,
    B: int,
    G: int,
    advs: Float[torch.Tensor, "B G 1"],
    completion_mask: Bool[torch.Tensor, "B G C"],
    old_log_probs: Float[torch.Tensor, "B G C"] | None,
    ref_log_probs: Float[torch.Tensor, "B G C"] | None,
    beta: float,
    eps: float | None,
    normalize_by_sequence_length: bool,
    loss_scale: float,
    clip_ratio_c: float = 3.0,
) -> Callable[[Float[torch.Tensor, "B G"], int], Float[torch.Tensor, "B G"]]:
    """Build a cycle_callback that computes per-cycle GRPO loss and calls backward.

    This frees each cycle's autograd graph immediately, reducing peak memory
    from O(C) cycles to O(1).
    """
    advs_bg = advs.squeeze(-1)  # [B, G]
    if normalize_by_sequence_length:
        seq_lengths = completion_mask.sum(dim=-1).clamp(min=1).float()  # [B, G]
    else:
        seq_lengths = torch.ones(B, G, device=advs.device)

    def callback(
        lp: Float[torch.Tensor, "B G"], cycle_idx: int
    ) -> Float[torch.Tensor, "B G"]:
        lp_bg = lp.view(B, G)
        mask_c = completion_mask[:, :, cycle_idx]

        if eps is not None and old_log_probs is not None:
            ratio = (lp_bg - old_log_probs[:, :, cycle_idx]).exp()
            unclipped = ratio * advs_bg
            clipped = torch.clip(ratio, 1 - eps, 1 + eps) * advs_bg
            main_obj = torch.min(unclipped, clipped)
            dual_clip_obj = clip_ratio_c * advs_bg
            main_obj = torch.where(
                advs_bg < 0, torch.max(main_obj, dual_clip_obj), main_obj
            )
        else:
            main_obj = lp_bg * advs_bg

        cycle_loss = -(main_obj * mask_c / seq_lengths).mean()

        if beta != 0 and ref_log_probs is not None:
            kl_diff = torch.clamp(
                ref_log_probs[:, :, cycle_idx] - lp_bg, min=-20, max=20
            )
            kl = torch.clamp(torch.exp(kl_diff) - kl_diff - 1, min=-10, max=10)
            cycle_loss = cycle_loss + beta * (kl * mask_c / seq_lengths).mean()

        (cycle_loss * loss_scale).backward()
        return lp.detach()

    return callback


@torch.no_grad()
def collect_internal_reasoning_micro_batch(
    *,
    net: BaseTransformer,
    ref_net: BaseTransformer | None,
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
    soft_block_size: int,
    max_cycles: int,
    collect_old_log_probs: bool,
    use_bf16: bool = False,
    soft_bptt_window: int | None = None,
    think_token_id: int | None = None,
) -> dict:
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
        batch_size=batch_size,
        group_size=group_size,
        temperature=temperature,
        soft_block_size=soft_block_size,
        max_cycles=max_cycles,
        use_bf16=use_bf16,
        think_token_id=think_token_id,
    )

    stacked_hard_ids, stacked_n_cycles = stack_and_pad_internal_reasoning(
        rollout.hard_token_ids, rollout.n_cycles, pad_token_id
    )

    device = next(net.parameters()).device
    t_logprobs_start = time.perf_counter()

    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
    ):
        if ref_net is not None:
            ref_log_probs, _ = compute_internal_reasoning_log_probs(
                net=ref_net,
                prompt_token_ids=rollout.prompt_token_ids,
                attention_mask=rollout.attention_mask,
                hard_token_ids=stacked_hard_ids,
                n_cycles=stacked_n_cycles,
                valid_hard_token_ids=valid_hard_token_ids,
                soft_block_size=soft_block_size,
                use_bf16=use_bf16,
                think_token_id=think_token_id,
            )
        else:
            ref_log_probs = None

        if collect_old_log_probs:
            old_log_probs, completion_mask = compute_internal_reasoning_log_probs(
                net=net,
                prompt_token_ids=rollout.prompt_token_ids,
                attention_mask=rollout.attention_mask,
                hard_token_ids=stacked_hard_ids,
                n_cycles=stacked_n_cycles,
                valid_hard_token_ids=valid_hard_token_ids,
                soft_block_size=soft_block_size,
                use_bf16=use_bf16,
                soft_bptt_window=soft_bptt_window,
                think_token_id=think_token_id,
            )
        else:
            old_log_probs = None
            cycle_indices = torch.arange(stacked_hard_ids.shape[2], device=device)
            cycle_indices = (
                cycle_indices.unsqueeze(0).unsqueeze(0).expand_as(stacked_hard_ids)
            )
            completion_mask = cycle_indices < stacked_n_cycles.unsqueeze(-1)

    t_logprobs = time.perf_counter() - t_logprobs_start

    return {
        "prompts": rollout.prompts,
        "env_responses": rollout.env_responses,
        "prompt_token_ids": rollout.prompt_token_ids,
        "attention_mask": rollout.attention_mask,
        "hard_token_ids": stacked_hard_ids,
        "n_cycles": stacked_n_cycles,
        "output_strs": rollout.output_strs,
        "reward_results": rollout.reward_results,
        "rewards": rollout.rewards,
        "ref_log_probs": ref_log_probs,
        "old_log_probs": old_log_probs,
        "completion_mask": completion_mask,
        "t_gen": rollout.t_generation,
        "t_logprobs": t_logprobs,
    }


def _encode_prompts(
    env_responses, state_to_str, tokenizer, pad_token_id, batch_size, device
):
    prompts = [state_to_str(er.data) for er in env_responses]
    encoded = tokenizer.encode_batch(prompts)
    max_prompt_len = max(len(e.ids) for e in encoded)
    prompt_token_ids = torch.full(
        (batch_size, max_prompt_len),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros(
        batch_size, max_prompt_len, dtype=torch.bool, device=device
    )
    for b, enc in enumerate(encoded):
        ids = torch.tensor(enc.ids, device=device)
        prompt_token_ids[b, max_prompt_len - len(ids) :] = ids
        attention_mask[b, max_prompt_len - len(ids) :] = True

    return prompt_token_ids, attention_mask


# TODO: why no logprobs chunk size here?
def train_internal_reasoning_grpo(
    *,
    net: BaseTransformer,
    opt: torch.optim.Optimizer,
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
    soft_bptt_window: int,
    max_cycles: int,
    beta: float,
    eps: float | None,
    mu: int = 1,
    max_episodes: int,
    update_ref_net_batch_cadence: int | None,
    batch_size: int,
    group_size: int,
    temperature: float,
    advantage_fn: Callable[
        [Float[torch.Tensor, "G B"]], Float[torch.Tensor, "G B"]
    ] = grpo_advantage,
    normalize_by_sequence_length,
    accumulation_steps: int,
    max_grad_norm: float,
    use_bf16: bool,
    save_ckpt_freq: int = sys.maxsize,
    val_freq: int,
    val_episodes: int,
    val_envs: list[Env],
    val_batch_size: int,
    think_token_id: int | None,
    clip_ratio_c: float = 3.0,
    warmup_steps: int = 0,
) -> None:
    if use_bf16:
        net = net.to(dtype=torch.bfloat16)

    collect_old_log_probs = eps is not None

    def collect_fn(net: BaseTransformer, ref_net: BaseTransformer | None) -> dict:
        return collect_internal_reasoning_micro_batch(
            net=net,
            ref_net=ref_net,
            env=env,
            reward_fn=reward_fn,
            state_to_str=state_to_str,
            tokenizer=tokenizer,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            extractor=extractor,
            move_id_to_name=move_id_to_name,
            valid_hard_token_ids=valid_hard_token_ids,
            batch_size=batch_size,
            group_size=group_size,
            temperature=temperature,
            soft_block_size=soft_block_size,
            max_cycles=max_cycles,
            collect_old_log_probs=collect_old_log_probs,
            use_bf16=use_bf16,
            soft_bptt_window=soft_bptt_window,
            think_token_id=think_token_id,
        )

    def recompute_fn(
        net: BaseTransformer, mb: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # When a per-cycle callback is active, gradients are accumulated during
        # compute_internal_reasoning_log_probs (the callback calls .backward()
        # per cycle to free each cycle's graph immediately). The returned
        # log_probs are detached and given requires_grad_(True) so the outer
        # _grpo_train_loop's compute_grpo_loss + backward() still runs for loss
        # reporting, but is a no-op on model parameters since the log_probs
        # leaf has no connection to the model graph.
        callback = None
        advs = mb.get("_advs_for_loss")
        if advs is not None:
            callback = make_per_cycle_backward_callback(
                B=mb["hard_token_ids"].shape[0],
                G=mb["hard_token_ids"].shape[1],
                advs=advs,
                completion_mask=mb["completion_mask"],
                old_log_probs=mb["old_log_probs"],
                ref_log_probs=mb["ref_log_probs"],
                beta=beta,
                eps=eps,
                normalize_by_sequence_length=normalize_by_sequence_length,
                loss_scale=mb.get("_loss_scale", 1.0),
                clip_ratio_c=clip_ratio_c,
            )
        log_probs, mask = compute_internal_reasoning_log_probs(
            net=net,
            prompt_token_ids=mb["prompt_token_ids"],
            attention_mask=mb["attention_mask"],
            hard_token_ids=mb["hard_token_ids"],
            n_cycles=mb["n_cycles"],
            valid_hard_token_ids=valid_hard_token_ids,
            soft_block_size=soft_block_size,
            use_bf16=use_bf16,
            soft_bptt_window=soft_bptt_window,
            cycle_callback=callback,
            think_token_id=think_token_id,
        )
        if callback is not None:
            log_probs = log_probs.detach().requires_grad_(True)
        return log_probs, mask

    val_fn = create_internal_reasoning_sft_val_fn(
        net=net,
        state_to_str=state_to_str,
        tokenizer=tokenizer,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        soft_block_size=soft_block_size,
        use_bf16=use_bf16,
        val_episodes=val_episodes,
        val_batch_size=val_batch_size,
        val_reward_fn=reward_fn,
        think_token_id=think_token_id,
        move_id_to_name=move_id_to_name,
        valid_hard_token_ids=valid_hard_token_ids,
        max_cycles=max_cycles,
    )

    _grpo_train_loop(
        net=net,
        opt=opt,
        collect_fn=collect_fn,
        recompute_log_probs_fn=recompute_fn,
        beta=beta,
        eps=eps,
        mu=mu,
        max_episodes=max_episodes,
        update_ref_net_batch_cadence=update_ref_net_batch_cadence,
        batch_size=batch_size,
        group_size=group_size,
        advantage_fn=advantage_fn,
        normalize_by_sequence_length=normalize_by_sequence_length,
        accumulation_steps=accumulation_steps,
        max_grad_norm=max_grad_norm,
        use_bf16=use_bf16,
        save_ckpt_freq=save_ckpt_freq,
        val_envs=val_envs,
        val_freq=val_freq,
        val_fn=val_fn,
        warmup_steps=warmup_steps,
    )


def stack_and_pad_variable_length_internal_reasoning(
    hard_token_ids: list[Integer[torch.Tensor, "B C T"]],
    hard_token_lengths: list[Integer[torch.Tensor, "B C"]],
    n_cycles: list[Integer[torch.Tensor, " B"]],
    pad_token_id: int,
) -> tuple[
    Integer[torch.Tensor, "B G C T"],
    Integer[torch.Tensor, "B G C"],
    Integer[torch.Tensor, "B G"],
]:
    B = hard_token_ids[0].shape[0]
    G = len(hard_token_ids)
    device = hard_token_ids[0].device

    max_c = max(t.shape[1] for t in hard_token_ids)
    max_t = max(t.shape[2] for t in hard_token_ids)

    stacked_ids = torch.full(
        (B, G, max_c, max_t), pad_token_id, dtype=torch.long, device=device
    )
    stacked_lengths = torch.zeros(B, G, max_c, dtype=torch.long, device=device)
    stacked_n = torch.zeros(B, G, dtype=torch.long, device=device)

    for g, (ids, lens, nc) in enumerate(
        zip(hard_token_ids, hard_token_lengths, n_cycles)
    ):
        C_g, T_g = ids.shape[1], ids.shape[2]
        stacked_ids[:, g, :C_g, :T_g] = ids
        stacked_lengths[:, g, :C_g] = lens
        stacked_n[:, g] = nc

    return stacked_ids, stacked_lengths, stacked_n


def make_variable_length_per_cycle_backward_callback(
    *,
    B: int,
    G: int,
    advs: Float[torch.Tensor, "B G 1"],
    completion_mask: Bool[torch.Tensor, "B G C"],
    hard_token_lengths: Integer[torch.Tensor, "B G C"],
    n_cycles: Integer[torch.Tensor, "B G"],
    old_log_probs: Float[torch.Tensor, "B G C"] | None,
    ref_log_probs: Float[torch.Tensor, "B G C"] | None,
    beta: float,
    eps: float | None,
    normalize_by_sequence_length: bool,
    loss_scale: float,
    clip_ratio_c: float = 3.0,
) -> Callable[[Float[torch.Tensor, "B G"], int], Float[torch.Tensor, "B G"]]:
    """Build a cycle_callback for variable-length GRPO that calls backward per cycle.

    When normalize_by_sequence_length is True, normalizes by total hard tokens
    (sum of hard_token_lengths over valid cycles) instead of cycle count.
    """
    advs_bg = advs.squeeze(-1)  # [B, G]
    if normalize_by_sequence_length:
        C = hard_token_lengths.shape[2]
        cycle_mask = torch.arange(C, device=advs.device).unsqueeze(0).unsqueeze(
            0
        ) < n_cycles.unsqueeze(-1)
        seq_lengths = (hard_token_lengths * cycle_mask).sum(dim=-1).clamp(min=1).float()
    else:
        seq_lengths = torch.ones(B, G, device=advs.device)

    def callback(
        lp: Float[torch.Tensor, "B G"], cycle_idx: int
    ) -> Float[torch.Tensor, "B G"]:
        lp_bg = lp.view(B, G)
        mask_c = completion_mask[:, :, cycle_idx]

        if eps is not None and old_log_probs is not None:
            ratio = (lp_bg - old_log_probs[:, :, cycle_idx]).exp()
            unclipped = ratio * advs_bg
            clipped = torch.clip(ratio, 1 - eps, 1 + eps) * advs_bg
            main_obj = torch.min(unclipped, clipped)
            dual_clip_obj = clip_ratio_c * advs_bg
            main_obj = torch.where(
                advs_bg < 0, torch.max(main_obj, dual_clip_obj), main_obj
            )
        else:
            main_obj = lp_bg * advs_bg

        cycle_loss = -(main_obj * mask_c / seq_lengths).mean()

        if beta != 0 and ref_log_probs is not None:
            kl_diff = torch.clamp(
                ref_log_probs[:, :, cycle_idx] - lp_bg, min=-20, max=20
            )
            kl = torch.clamp(torch.exp(kl_diff) - kl_diff - 1, min=-10, max=10)
            cycle_loss = cycle_loss + beta * (kl * mask_c / seq_lengths).mean()

        (cycle_loss * loss_scale).backward()
        return lp.detach()

    return callback


@torch.no_grad()
def collect_variable_length_internal_reasoning_micro_batch(
    *,
    net: BaseTransformer,
    ref_net: BaseTransformer | None,
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
    soft_block_size: int,
    max_cycles: int,
    max_tokens_per_cycle: int,
    collect_old_log_probs: bool,
    use_bf16: bool = False,
    soft_bptt_window: int | None = None,
    think_token_id: int | None = None,
) -> dict:
    rollout = generate_grouped_variable_length_internal_reasoning_rollout_batch(
        net=net,
        env=env,
        reward_fn=reward_fn,
        state_to_str=state_to_str,
        tokenizer=tokenizer,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        separator_token_id=separator_token_id,
        batch_size=batch_size,
        group_size=group_size,
        temperature=temperature,
        soft_block_size=soft_block_size,
        max_cycles=max_cycles,
        max_tokens_per_cycle=max_tokens_per_cycle,
        use_bf16=use_bf16,
        think_token_id=think_token_id,
    )

    stacked_hard_ids, stacked_lengths, stacked_n_cycles = (
        stack_and_pad_variable_length_internal_reasoning(
            rollout.hard_token_ids,
            rollout.hard_token_lengths,
            rollout.n_cycles,
            pad_token_id,
        )
    )

    device = next(net.parameters()).device
    t_logprobs_start = time.perf_counter()

    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
    ):
        if ref_net is not None:
            ref_log_probs, _ = compute_variable_length_internal_reasoning_log_probs(
                net=ref_net,
                prompt_token_ids=rollout.prompt_token_ids,
                attention_mask=rollout.attention_mask,
                hard_token_ids=stacked_hard_ids,
                hard_token_lengths=stacked_lengths,
                n_cycles=stacked_n_cycles,
                soft_block_size=soft_block_size,
                use_bf16=use_bf16,
                think_token_id=think_token_id,
            )
        else:
            ref_log_probs = None

        if collect_old_log_probs:
            old_log_probs, completion_mask = (
                compute_variable_length_internal_reasoning_log_probs(
                    net=net,
                    prompt_token_ids=rollout.prompt_token_ids,
                    attention_mask=rollout.attention_mask,
                    hard_token_ids=stacked_hard_ids,
                    hard_token_lengths=stacked_lengths,
                    n_cycles=stacked_n_cycles,
                    soft_block_size=soft_block_size,
                    use_bf16=use_bf16,
                    soft_bptt_window=soft_bptt_window,
                    think_token_id=think_token_id,
                )
            )
        else:
            old_log_probs = None
            C = stacked_hard_ids.shape[2]
            cycle_indices = torch.arange(C, device=device)
            cycle_indices = (
                cycle_indices.unsqueeze(0)
                .unsqueeze(0)
                .expand(stacked_hard_ids.shape[0], stacked_hard_ids.shape[1], C)
            )
            completion_mask = cycle_indices < stacked_n_cycles.unsqueeze(-1)

    t_logprobs = time.perf_counter() - t_logprobs_start

    return {
        "prompts": rollout.prompts,
        "env_responses": rollout.env_responses,
        "prompt_token_ids": rollout.prompt_token_ids,
        "attention_mask": rollout.attention_mask,
        "hard_token_ids": stacked_hard_ids,
        "hard_token_lengths": stacked_lengths,
        "n_cycles": stacked_n_cycles,
        "output_strs": rollout.output_strs,
        "reward_results": rollout.reward_results,
        "rewards": rollout.rewards,
        "ref_log_probs": ref_log_probs,
        "old_log_probs": old_log_probs,
        "completion_mask": completion_mask,
        "t_gen": rollout.t_generation,
        "t_logprobs": t_logprobs,
    }


def train_variable_length_internal_reasoning_grpo(
    *,
    net: BaseTransformer,
    opt: torch.optim.Optimizer,
    env: Env[T, A],
    reward_fn: RewardFn[T, E],
    state_to_str: Callable[[T], str],
    tokenizer: Tokenizer,
    eos_token_id: int,
    pad_token_id: int,
    separator_token_id: int,
    soft_block_size: int,
    soft_bptt_window: int,
    max_cycles: int,
    max_tokens_per_cycle: int = 20,
    beta: float,
    eps: float | None,
    mu: int = 1,
    max_episodes: int,
    update_ref_net_batch_cadence: int | None,
    batch_size: int,
    group_size: int,
    temperature: float,
    advantage_fn: Callable[
        [Float[torch.Tensor, "G B"]], Float[torch.Tensor, "G B"]
    ] = grpo_advantage,
    normalize_by_sequence_length: bool,
    accumulation_steps: int,
    max_grad_norm: float,
    use_bf16: bool,
    save_ckpt_freq: int = sys.maxsize,
    val_freq: int,
    val_episodes: int,
    val_envs: Sequence[Env],
    val_batch_size: int,
    think_token_id: int | None,
    pass_at_k_samples: int = 0,
    pass_at_k_temperature: float = 0.7,
    clip_ratio_c: float = 3.0,
    warmup_steps: int = 0,
) -> None:
    if use_bf16:
        net = net.to(dtype=torch.bfloat16)

    collect_old_log_probs = eps is not None

    def collect_fn(net: BaseTransformer, ref_net: BaseTransformer | None) -> dict:
        return collect_variable_length_internal_reasoning_micro_batch(
            net=net,
            ref_net=ref_net,
            env=env,
            reward_fn=reward_fn,
            state_to_str=state_to_str,
            tokenizer=tokenizer,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            separator_token_id=separator_token_id,
            batch_size=batch_size,
            group_size=group_size,
            temperature=temperature,
            soft_block_size=soft_block_size,
            max_cycles=max_cycles,
            max_tokens_per_cycle=max_tokens_per_cycle,
            collect_old_log_probs=collect_old_log_probs,
            use_bf16=use_bf16,
            soft_bptt_window=soft_bptt_window,
            think_token_id=think_token_id,
        )

    def recompute_fn(
        net: BaseTransformer, mb: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        callback = None
        advs = mb.get("_advs_for_loss")
        if advs is not None:
            callback = make_variable_length_per_cycle_backward_callback(
                B=mb["hard_token_ids"].shape[0],
                G=mb["hard_token_ids"].shape[1],
                advs=advs,
                completion_mask=mb["completion_mask"],
                hard_token_lengths=mb["hard_token_lengths"],
                n_cycles=mb["n_cycles"],
                old_log_probs=mb["old_log_probs"],
                ref_log_probs=mb["ref_log_probs"],
                beta=beta,
                eps=eps,
                normalize_by_sequence_length=normalize_by_sequence_length,
                loss_scale=mb.get("_loss_scale", 1.0),
                clip_ratio_c=clip_ratio_c,
            )
        log_probs, mask = compute_variable_length_internal_reasoning_log_probs(
            net=net,
            prompt_token_ids=mb["prompt_token_ids"],
            attention_mask=mb["attention_mask"],
            hard_token_ids=mb["hard_token_ids"],
            hard_token_lengths=mb["hard_token_lengths"],
            n_cycles=mb["n_cycles"],
            soft_block_size=soft_block_size,
            use_bf16=use_bf16,
            soft_bptt_window=soft_bptt_window,
            cycle_callback=callback,
            think_token_id=think_token_id,
        )
        if callback is not None:
            log_probs = log_probs.detach().requires_grad_(True)
        return log_probs, mask

    val_fn = create_variable_length_internal_reasoning_sft_val_fn(
        net=net,
        state_to_str=state_to_str,
        tokenizer=tokenizer,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        separator_token_id=separator_token_id,
        soft_block_size=soft_block_size,
        max_cycles=max_cycles,
        max_tokens_per_cycle=max_tokens_per_cycle,
        use_bf16=use_bf16,
        val_episodes=val_episodes,
        val_batch_size=val_batch_size,
        val_reward_fn=reward_fn,
        think_token_id=think_token_id,
        pass_at_k_samples=pass_at_k_samples,
        pass_at_k_temperature=pass_at_k_temperature,
    )

    _grpo_train_loop(
        net=net,
        opt=opt,
        collect_fn=collect_fn,
        recompute_log_probs_fn=recompute_fn,
        beta=beta,
        eps=eps,
        mu=mu,
        max_episodes=max_episodes,
        update_ref_net_batch_cadence=update_ref_net_batch_cadence,
        batch_size=batch_size,
        group_size=group_size,
        advantage_fn=advantage_fn,
        normalize_by_sequence_length=normalize_by_sequence_length,
        accumulation_steps=accumulation_steps,
        max_grad_norm=max_grad_norm,
        use_bf16=use_bf16,
        save_ckpt_freq=save_ckpt_freq,
        val_envs=val_envs,
        val_freq=val_freq,
        val_fn=val_fn,
        warmup_steps=warmup_steps,
    )


def make_sft_per_cycle_backward_callback(
    *,
    B: int,
    completion_mask: Bool[torch.Tensor, "B 1 C"],
    normalize_by_sequence_length: bool,
    loss_scale: float,
) -> Callable[[Float[torch.Tensor, "B 1"], int], Float[torch.Tensor, "B 1"]]:
    if normalize_by_sequence_length:
        seq_lengths = completion_mask[:, 0, :].sum(dim=-1).clamp(min=1).float()  # [B]
    else:
        seq_lengths = torch.ones(B, device=completion_mask.device)

    def callback(
        lp: Float[torch.Tensor, "B 1"], cycle_idx: int
    ) -> Float[torch.Tensor, "B 1"]:
        lp_b = lp.view(B)
        mask_c = completion_mask[:, 0, cycle_idx]  # [B]
        cycle_loss = -(lp_b * mask_c / seq_lengths).mean()
        (cycle_loss * loss_scale).backward()
        return lp.detach()

    return callback


def create_internal_reasoning_sft_step_fn(
    *,
    net: BaseTransformer,
    opt: torch.optim.Optimizer,
    env: Env[T, A],
    state_to_str: Callable[[T], str],
    tokenizer: Tokenizer,
    pad_token_id: int,
    eos_token_id: int,
    move_name_to_id: dict[str, int],
    valid_hard_token_ids: list[int],
    soft_block_size: int,
    soft_bptt_window: int | None,
    max_cycles: int,
    batch_size: int,
    accumulation_steps: int,
    max_grad_norm: float,
    normalize_by_sequence_length: bool,
    use_bf16: bool,
    think_token_id: int | None,
):
    device = next(net.parameters()).device

    def _step(step: int):
        nonlocal net
        opt.zero_grad()
        step_nll_sum = 0.0
        step_solution_lengths: list[int] = []

        for _ in range(accumulation_steps):
            env_responses = [env.reset() for _ in range(batch_size)]

            solutions: list[list[int]] = []
            for er in env_responses:
                move_names = er.data.maze.solution
                move_ids = [move_name_to_id[m] for m in move_names]
                move_ids.append(eos_token_id)
                if len(move_ids) > max_cycles:
                    warnings.warn(
                        f"Solution length {len(move_ids)} exceeds max_cycles "
                        f"{max_cycles}, truncating"
                    )
                    move_ids = move_ids[:max_cycles]
                solutions.append(move_ids)
                step_solution_lengths.append(len(move_ids))

            max_c = max(len(s) for s in solutions)
            hard_token_ids = torch.full(
                (batch_size, 1, max_c), pad_token_id, dtype=torch.long, device=device
            )
            n_cycles = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
            for b, sol in enumerate(solutions):
                hard_token_ids[b, 0, : len(sol)] = torch.tensor(sol, device=device)
                n_cycles[b, 0] = len(sol)

            prompt_token_ids, attention_mask = _encode_prompts(
                env_responses, state_to_str, tokenizer, pad_token_id, batch_size, device
            )

            cycle_indices = (
                torch.arange(max_c, device=device)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(batch_size, 1, max_c)
            )
            completion_mask = cycle_indices < n_cycles.unsqueeze(-1)

            callback = make_sft_per_cycle_backward_callback(
                B=batch_size,
                completion_mask=completion_mask,
                normalize_by_sequence_length=normalize_by_sequence_length,
                loss_scale=1.0 / accumulation_steps,
            )

            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
            ):
                log_probs, _ = compute_internal_reasoning_log_probs(
                    net=net,
                    prompt_token_ids=prompt_token_ids,
                    attention_mask=attention_mask,
                    hard_token_ids=hard_token_ids,
                    n_cycles=n_cycles,
                    valid_hard_token_ids=valid_hard_token_ids,
                    soft_block_size=soft_block_size,
                    use_bf16=use_bf16,
                    soft_bptt_window=soft_bptt_window,
                    cycle_callback=callback,
                    think_token_id=think_token_id,
                )

            masked_lp = log_probs.detach() * completion_mask
            if normalize_by_sequence_length:
                seq_lens = completion_mask.sum(dim=-1).clamp(min=1).float()
                per_seq_nll = -(masked_lp.sum(dim=-1) / seq_lens)
            else:
                per_seq_nll = -masked_lp.sum(dim=-1)
            step_nll_sum += per_seq_nll.mean().item() / accumulation_steps

        grad_norm_val: float | None = None
        if max_grad_norm > 0:
            params = [p for group in opt.param_groups for p in group["params"]]
            grad_norm_val = torch.nn.utils.clip_grad_norm_(
                params, max_norm=max_grad_norm
            ).item()
        opt.step()

        metrics: dict[str, Any] = {
            "sft/nll_loss": step_nll_sum,
            "sft/mean_solution_length": (
                sum(step_solution_lengths) / len(step_solution_lengths)
            ),
        }
        if grad_norm_val is not None:
            metrics["sft/grad_norm"] = grad_norm_val
        if net.soft_projection_alpha is not None:
            metrics["sft/soft_projection_alpha"] = net.soft_projection_alpha.item()

        return StepFunctionReturn(
            n_episodes_processed=batch_size * accumulation_steps, metrics=metrics
        )

    return _step


def create_internal_reasoning_sft_val_fn(
    *,
    net: BaseTransformer,
    state_to_str: Callable,
    tokenizer: Tokenizer,
    pad_token_id: int,
    eos_token_id: int,
    soft_block_size: int,
    use_bf16: bool,
    val_episodes: int,
    val_batch_size: int,
    val_reward_fn: RewardFn,
    think_token_id: int | None,
    move_id_to_name: dict[int, str],
    valid_hard_token_ids: list[int],
    max_cycles: int,
):
    def _val(env: Env):
        return evaluate_internal_reasoning(
            net=net,
            env=env,
            reward_fn=val_reward_fn,
            state_to_str=state_to_str,
            tokenizer=tokenizer,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            extractor=lambda moves: moves if moves else None,
            move_id_to_name=move_id_to_name,
            valid_hard_token_ids=valid_hard_token_ids,
            soft_block_size=soft_block_size,
            max_cycles=max_cycles,
            max_episodes=val_episodes,
            batch_size=val_batch_size,
            group_size=1,
            temperature=0.0,
            use_bf16=use_bf16,
            n_examples=val_episodes,
            think_token_id=think_token_id,
        )

    return _val


# multi-step
def train_internal_reasoning_sft(
    *,
    net: BaseTransformer,
    opt: torch.optim.Optimizer,
    env: Env[T, A],
    state_to_str: Callable[[T], str],
    tokenizer: Tokenizer,
    pad_token_id: int,
    eos_token_id: int,
    move_name_to_id: dict[str, int],
    valid_hard_token_ids: list[int],
    soft_block_size: int,
    soft_bptt_window: int | None = None,
    max_cycles: int,
    max_episodes: int,
    batch_size: int,
    accumulation_steps: int,
    max_grad_norm: float,
    normalize_by_sequence_length: bool,
    use_bf16: bool,
    save_ckpt_freq: int = sys.maxsize,
    val_reward_fn: RewardFn,
    val_episodes: int,
    val_batch_size: int,
    val_freq: int = 0,
    val_envs: list[Env],
    think_token_id: int | None,
) -> None:
    if use_bf16:
        net = net.to(dtype=torch.bfloat16)

    move_id_to_name = {v: k for k, v in move_name_to_id.items()}

    train_step = create_internal_reasoning_sft_step_fn(
        net=net,
        opt=opt,
        env=env,
        state_to_str=state_to_str,
        tokenizer=tokenizer,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        move_name_to_id=move_name_to_id,
        valid_hard_token_ids=valid_hard_token_ids,
        soft_block_size=soft_block_size,
        soft_bptt_window=soft_bptt_window,
        max_cycles=max_cycles,
        batch_size=batch_size,
        accumulation_steps=accumulation_steps,
        max_grad_norm=max_grad_norm,
        normalize_by_sequence_length=normalize_by_sequence_length,
        use_bf16=use_bf16,
        think_token_id=think_token_id,
    )

    val_fn = create_internal_reasoning_sft_val_fn(
        net=net,
        state_to_str=state_to_str,
        tokenizer=tokenizer,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        soft_block_size=soft_block_size,
        use_bf16=use_bf16,
        val_episodes=val_episodes,
        val_batch_size=val_batch_size,
        val_reward_fn=val_reward_fn,
        think_token_id=think_token_id,
        move_id_to_name=move_id_to_name,
        valid_hard_token_ids=valid_hard_token_ids,
        max_cycles=max_cycles,
    )

    train_loop(
        max_episodes=max_episodes,
        save_ckpt_freq=save_ckpt_freq,
        val_freq=val_freq,
        net=net,
        opt=opt,
        train_step=train_step,
        val_envs=val_envs,
        val_fn=val_fn,
    )


def compute_variable_length_internal_reasoning_log_probs(
    *,
    net: BaseTransformer,
    prompt_token_ids: Integer[torch.Tensor, "B L"],
    attention_mask: Bool[torch.Tensor, "B L"],
    hard_token_ids: Integer[torch.Tensor, "B G C T_max"],
    hard_token_lengths: Integer[torch.Tensor, "B G C"],
    n_cycles: Integer[torch.Tensor, "B G"],
    soft_block_size: int,
    valid_hard_token_ids: list[int] | None = None,
    use_bf16: bool = False,
    soft_bptt_window: int | None = None,
    cycle_callback: Callable[
        [Float[torch.Tensor, " BG"], int], Float[torch.Tensor, " BG"]
    ]
    | None = None,
    think_token_id: int | None = None,
) -> tuple[Float[torch.Tensor, "B G C"], Bool[torch.Tensor, "B G C"]]:
    """Compute log probs for variable-length internal reasoning trajectories.

    Each cycle produces a variable number of hard tokens. Log probs are summed
    within each cycle to produce per-cycle log probs. This is the general form
    that subsumes the fixed-length (T_max=1) case used by maze environments.

    Parameters
    ----------
    net
        The transformer model.
    prompt_token_ids
        Prompt token IDs [B, L].
    attention_mask
        Attention mask for prompt [B, L].
    hard_token_ids
        Known hard tokens per cycle [B, G, C, T_max], right-padded.
    hard_token_lengths
        Number of actual hard tokens per cycle [B, G, C].
    n_cycles
        Actual number of cycles per sample [B, G].
    soft_block_size
        Number of soft forward passes per cycle.
    valid_hard_token_ids
        Optional token IDs to restrict logits to. When set, logits for tokens
        not in this list are set to -inf before log_softmax.
    use_bf16
        Whether to use bf16 autocast.
    soft_bptt_window
        Number of soft tokens at the end of each cycle's soft block to
        backpropagate through. None means full BPTT.
    cycle_callback
        Optional callback invoked after each cycle's summed log_probs are
        computed. Receives (cycle_log_probs [BG], cycle_index) and returns
        log_probs to store.
    think_token_id
        If set, use this token's embedding for soft passes instead of hidden
        state passthrough.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        (log_probs [B, G, C], completion_mask [B, G, C])
        log_probs are summed across tokens within each cycle.
    """
    if soft_bptt_window is None:
        soft_bptt_window = soft_block_size
    n_no_grad_soft = soft_block_size - soft_bptt_window
    B, G, C, T_max = hard_token_ids.shape
    BG = B * G
    device = prompt_token_ids.device

    flat_prompt = prompt_token_ids.repeat_interleave(G, dim=0)  # [BG, L]
    flat_mask = attention_mask.repeat_interleave(G, dim=0)  # [BG, L]
    flat_hard = hard_token_ids.view(BG, C, T_max)  # [BG, C, T_max]
    flat_lengths = hard_token_lengths.view(BG, C)  # [BG, C]
    flat_n_cycles = n_cycles.view(BG)  # [BG]

    L_prompt = flat_prompt.shape[1]
    max_actual_cycles = int(flat_n_cycles.max().item())
    max_seq_len = L_prompt + max_actual_cycles * (soft_block_size + T_max)

    std_kv_caches = [
        KVCache(
            max_seq_len=max_seq_len,
            num_heads=net.attn_num_kv_heads,
            head_dim=net.attn_head_d,
            device=device,
        )
        for _ in range(len(net.layers))
    ]

    valid_logit_mask: torch.Tensor | None = None
    if valid_hard_token_ids is not None:
        valid_logit_mask = torch.full((net.vocab_size,), float("-inf"), device=device)
        valid_logit_mask[valid_hard_token_ids] = 0.0

    with torch.no_grad():
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
        ):
            h = net(
                flat_prompt,
                kv_caches=std_kv_caches,
                attention_mask=flat_mask,
                return_hidden_states=True,
            )
    h = h[:, -1:]  # [BG, 1, D]

    grad_kv_caches: list[GradSafeKVCache | KVCache] = [
        GradSafeKVCache.from_standard(c) for c in std_kv_caches
    ]

    attn_mask = flat_mask

    if think_token_id is not None:
        think_embed = net.embed_tokens(
            torch.full((BG, 1), think_token_id, dtype=torch.long, device=device)
        ).detach()

    ones = torch.ones(BG, 1, dtype=torch.bool, device=device)

    all_log_probs = []
    for cycle in range(max_actual_cycles):
        h = h.detach()

        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
        ):
            if n_no_grad_soft > 0:
                with torch.no_grad():
                    for _ in range(n_no_grad_soft):
                        attn_mask = torch.cat([attn_mask, ones], dim=1)
                        if think_token_id is not None:
                            h = net(
                                think_embed,
                                kv_caches=grad_kv_caches,
                                attention_mask=attn_mask,
                                return_hidden_states=True,
                            )
                        else:
                            h = net(
                                h,
                                kv_caches=grad_kv_caches,
                                attention_mask=attn_mask,
                                return_hidden_states=True,
                            )
                            h = net.apply_soft_projection(h)
                h = h.detach()

            for _ in range(soft_bptt_window):
                attn_mask = torch.cat([attn_mask, ones], dim=1)
                if think_token_id is not None:
                    h = net(
                        think_embed,
                        kv_caches=grad_kv_caches,
                        attention_mask=attn_mask,
                        return_hidden_states=True,
                    )
                else:
                    h = net(
                        h,
                        kv_caches=grad_kv_caches,
                        attention_mask=attn_mask,
                        return_hidden_states=True,
                    )
                    h = net.apply_soft_projection(h)

            max_t = int(flat_lengths[:, cycle].max().item())
            cycle_lp = torch.zeros(BG, device=device)

            for t in range(max_t):
                pos_valid = (t < flat_lengths[:, cycle]).unsqueeze(1)
                attn_mask = torch.cat([attn_mask, pos_valid], dim=1)
                h_out = net(
                    h,
                    kv_caches=grad_kv_caches,
                    attention_mask=attn_mask,
                    return_hidden_states=True,
                )
                logits = net.lm_head(h_out)  # [BG, 1, V]

                logits_squeezed = logits.squeeze(1).float()  # [BG, V]
                if valid_logit_mask is not None:
                    logits_squeezed = logits_squeezed + valid_logit_mask.unsqueeze(0)
                lp_all = torch.log_softmax(logits_squeezed, dim=-1)
                target_token = flat_hard[:, cycle, t]  # [BG]
                lp = lp_all[torch.arange(BG, device=device), target_token]  # [BG]

                token_mask = (t < flat_lengths[:, cycle]) & (cycle < flat_n_cycles)
                lp = torch.where(token_mask, lp, torch.zeros_like(lp))
                cycle_lp = cycle_lp + lp

                with torch.no_grad():
                    h = net.embed_tokens(target_token.unsqueeze(1))

        if cycle_callback is not None:
            cycle_lp = cycle_callback(cycle_lp, cycle)

        all_log_probs.append(cycle_lp)

        for gc in grad_kv_caches:
            assert isinstance(gc, GradSafeKVCache)
            gc.freeze()

        if think_token_id is None and cycle < max_actual_cycles - 1:
            last_valid_idx = (flat_lengths[:, cycle] - 1).clamp(min=0)
            last_valid_tok = (
                flat_hard[:, cycle].gather(1, last_valid_idx.unsqueeze(1)).squeeze(1)
            )
            with torch.no_grad():
                h = net.embed_tokens(last_valid_tok.unsqueeze(1))

    log_probs = torch.stack(all_log_probs, dim=1)  # [BG, max_actual_cycles]

    if max_actual_cycles < C:
        pad_lp = torch.zeros(BG, C - max_actual_cycles, device=device)
        log_probs = torch.cat([log_probs, pad_lp], dim=1)

    log_probs = log_probs.view(B, G, C)

    cycle_indices = (
        torch.arange(C, device=device).unsqueeze(0).unsqueeze(0).expand(B, G, C)
    )
    completion_mask = cycle_indices < n_cycles.unsqueeze(-1)

    return log_probs, completion_mask


def make_variable_length_sft_per_cycle_backward_callback(
    *,
    B: int,
    completion_mask: Bool[torch.Tensor, "B C"],
    hard_token_lengths: Integer[torch.Tensor, "B C"],
    n_cycles: Integer[torch.Tensor, " B"],
    normalize_by_sequence_length: bool,
    loss_scale: float,
) -> Callable[[Float[torch.Tensor, " B"], int], Float[torch.Tensor, " B"]]:
    """Build a cycle_callback for variable-length SFT that calls backward per cycle.

    When normalize_by_sequence_length is True, normalizes by total hard tokens
    (not cycle count).
    """
    if normalize_by_sequence_length:
        C = hard_token_lengths.shape[1]
        cycle_mask = torch.arange(C, device=completion_mask.device).unsqueeze(
            0
        ) < n_cycles.unsqueeze(-1)
        total_tokens = (
            (hard_token_lengths * cycle_mask).sum(dim=-1).clamp(min=1).float()
        )
    else:
        total_tokens = torch.ones(B, device=completion_mask.device)

    def callback(
        lp: Float[torch.Tensor, " B"], cycle_idx: int
    ) -> Float[torch.Tensor, " B"]:
        mask_c = completion_mask[:, cycle_idx]  # [B]
        cycle_loss = -(lp * mask_c / total_tokens).mean()
        (cycle_loss * loss_scale).backward()
        return lp.detach()

    return callback


def create_countdown_sft_step_fn(
    *,
    net: BaseTransformer,
    opt: torch.optim.Optimizer,
    env: Env[T, A],
    state_to_str: Callable[[T], str],
    tokenizer: Tokenizer,
    pad_token_id: int,
    eos_token_id: int,
    soft_block_size: int,
    soft_bptt_window: int | None,
    max_cycles: int,
    batch_size: int,
    accumulation_steps: int,
    max_grad_norm: float,
    normalize_by_sequence_length: bool,
    use_bf16: bool,
    think_token_id: int | None,
):
    device = next(net.parameters()).device

    def _step(step: int):
        nonlocal net
        opt.zero_grad()
        step_nll_sum = 0.0
        step_solution_lengths: list[int] = []

        for _ in range(accumulation_steps):
            env_responses = [env.reset() for _ in range(batch_size)]

            all_cycle_token_ids: list[list[list[int]]] = []
            for er in env_responses:
                solution = er.data.solution
                assert solution is not None, "CountdownEnv must provide solution"
                cycle_ids_list: list[list[int]] = []
                for s in solution:
                    text = s.format_step()
                    ids = tokenizer.encode(text, add_special_tokens=False).ids
                    cycle_ids_list.append(list(ids))
                equation = build_countdown_equation(
                    er.data.numbers, solution, er.data.target
                )
                equation_ids = tokenizer.encode(equation, add_special_tokens=False).ids
                equation_ids = list(equation_ids) + [eos_token_id]
                cycle_ids_list.append(equation_ids)

                total_tokens = sum(len(c) for c in cycle_ids_list)
                step_solution_lengths.append(total_tokens)

                if len(cycle_ids_list) > max_cycles:
                    cycle_ids_list = cycle_ids_list[:max_cycles]

                all_cycle_token_ids.append(cycle_ids_list)

            n_c = [len(c) for c in all_cycle_token_ids]
            max_c = max(n_c)
            max_t = max(len(tok) for sample in all_cycle_token_ids for tok in sample)

            hard_token_ids = torch.full(
                (batch_size, 1, max_c, max_t),
                pad_token_id,
                dtype=torch.long,
                device=device,
            )
            hard_token_lengths_tensor = torch.zeros(
                batch_size, 1, max_c, dtype=torch.long, device=device
            )
            n_cycles_tensor = torch.tensor(
                n_c, dtype=torch.long, device=device
            ).unsqueeze(1)

            for b, cycles in enumerate(all_cycle_token_ids):
                for c_idx, tok_ids in enumerate(cycles):
                    hard_token_ids[b, 0, c_idx, : len(tok_ids)] = torch.tensor(
                        tok_ids, device=device
                    )
                    hard_token_lengths_tensor[b, 0, c_idx] = len(tok_ids)

            prompt_token_ids, attn_mask = _encode_prompts(
                env_responses, state_to_str, tokenizer, pad_token_id, batch_size, device
            )

            cycle_indices = (
                torch.arange(max_c, device=device)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(batch_size, 1, max_c)
            )
            completion_mask = cycle_indices < n_cycles_tensor.unsqueeze(-1)

            callback = make_variable_length_sft_per_cycle_backward_callback(
                B=batch_size,
                completion_mask=completion_mask[:, 0],
                hard_token_lengths=hard_token_lengths_tensor[:, 0],
                n_cycles=n_cycles_tensor[:, 0],
                normalize_by_sequence_length=normalize_by_sequence_length,
                loss_scale=1.0 / accumulation_steps,
            )

            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
            ):
                log_probs, _ = compute_variable_length_internal_reasoning_log_probs(
                    net=net,
                    prompt_token_ids=prompt_token_ids,
                    attention_mask=attn_mask,
                    hard_token_ids=hard_token_ids,
                    hard_token_lengths=hard_token_lengths_tensor,
                    n_cycles=n_cycles_tensor,
                    soft_block_size=soft_block_size,
                    use_bf16=use_bf16,
                    soft_bptt_window=soft_bptt_window,
                    cycle_callback=callback,
                    think_token_id=think_token_id,
                )

            log_probs_flat = log_probs[:, 0]
            completion_mask_flat = completion_mask[:, 0]
            masked_lp = log_probs_flat.detach() * completion_mask_flat
            if normalize_by_sequence_length:
                total_toks = torch.zeros(batch_size, device=device)
                for b in range(batch_size):
                    nc = n_cycles_tensor[b, 0].item()
                    total_toks[b] = (
                        hard_token_lengths_tensor[b, 0, :nc].sum().clamp(min=1).float()
                    )
                per_seq_nll = -(masked_lp.sum(dim=-1) / total_toks)
            else:
                per_seq_nll = -masked_lp.sum(dim=-1)
            step_nll_sum += per_seq_nll.mean().item() / accumulation_steps

        grad_norm_val: float | None = None
        if max_grad_norm > 0:
            params = [p for group in opt.param_groups for p in group["params"]]
            grad_norm_val = torch.nn.utils.clip_grad_norm_(
                params, max_norm=max_grad_norm
            ).item()
        opt.step()

        metrics: dict[str, Any] = {
            "sft/nll_loss": step_nll_sum,
            "sft/mean_solution_length": (
                sum(step_solution_lengths) / len(step_solution_lengths)
            ),
        }
        if grad_norm_val is not None:
            metrics["sft/grad_norm"] = grad_norm_val
        if net.soft_projection_alpha is not None:
            metrics["sft/soft_projection_alpha"] = net.soft_projection_alpha.item()

        return StepFunctionReturn(
            n_episodes_processed=batch_size * accumulation_steps, metrics=metrics
        )

    return _step


def create_variable_length_internal_reasoning_sft_val_fn(
    *,
    net: BaseTransformer,
    state_to_str: Callable,
    tokenizer: Tokenizer,
    pad_token_id: int,
    eos_token_id: int,
    separator_token_id: int,
    soft_block_size: int,
    max_cycles: int,
    max_tokens_per_cycle: int,
    use_bf16: bool,
    val_episodes: int,
    val_batch_size: int,
    val_reward_fn: RewardFn,
    think_token_id: int | None,
    pass_at_k_samples: int = 0,
    pass_at_k_temperature: float = 0.7,
):
    def _val(env: Env):
        from dialectic.rl.evaluate import evaluate_variable_length_internal_reasoning

        return evaluate_variable_length_internal_reasoning(
            net=net,
            env=env,
            reward_fn=val_reward_fn,
            state_to_str=state_to_str,
            tokenizer=tokenizer,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            separator_token_id=separator_token_id,
            soft_block_size=soft_block_size,
            max_cycles=max_cycles,
            max_tokens_per_cycle=max_tokens_per_cycle,
            max_episodes=val_episodes,
            batch_size=val_batch_size,
            temperature=0.0,
            use_bf16=use_bf16,
            n_examples=val_episodes,
            think_token_id=think_token_id,
            pass_at_k_samples=pass_at_k_samples,
            pass_at_k_temperature=pass_at_k_temperature,
        )

    return _val


def train_variable_length_internal_reasoning_sft(
    *,
    net: BaseTransformer,
    opt: torch.optim.Optimizer,
    env: Env[T, A],
    state_to_str: Callable[[T], str],
    tokenizer: Tokenizer,
    pad_token_id: int,
    eos_token_id: int,
    separator_token_id: int,
    soft_block_size: int,
    soft_bptt_window: int | None = None,
    max_cycles: int,
    max_tokens_per_cycle: int = 20,
    max_episodes: int,
    batch_size: int,
    accumulation_steps: int,
    max_grad_norm: float,
    normalize_by_sequence_length: bool,
    use_bf16: bool,
    save_ckpt_freq: int = sys.maxsize,
    val_reward_fn: RewardFn,
    val_episodes: int,
    val_batch_size: int,
    val_freq: int = 0,
    val_envs: Sequence[Env],
    think_token_id: int | None,
    pass_at_k_samples: int = 0,
    pass_at_k_temperature: float = 0.7,
) -> None:
    if use_bf16:
        net = net.to(dtype=torch.bfloat16)

    train_step = create_countdown_sft_step_fn(
        net=net,
        opt=opt,
        env=env,
        state_to_str=state_to_str,
        tokenizer=tokenizer,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        soft_block_size=soft_block_size,
        soft_bptt_window=soft_bptt_window,
        max_cycles=max_cycles,
        batch_size=batch_size,
        accumulation_steps=accumulation_steps,
        max_grad_norm=max_grad_norm,
        normalize_by_sequence_length=normalize_by_sequence_length,
        use_bf16=use_bf16,
        think_token_id=think_token_id,
    )

    val_fn = create_variable_length_internal_reasoning_sft_val_fn(
        net=net,
        state_to_str=state_to_str,
        tokenizer=tokenizer,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        separator_token_id=separator_token_id,
        soft_block_size=soft_block_size,
        max_cycles=max_cycles,
        max_tokens_per_cycle=max_tokens_per_cycle,
        use_bf16=use_bf16,
        val_episodes=val_episodes,
        val_batch_size=val_batch_size,
        val_reward_fn=val_reward_fn,
        think_token_id=think_token_id,
        pass_at_k_samples=pass_at_k_samples,
        pass_at_k_temperature=pass_at_k_temperature,
    )

    train_loop(
        max_episodes=max_episodes,
        save_ckpt_freq=save_ckpt_freq,
        val_freq=val_freq,
        net=net,
        opt=opt,
        train_step=train_step,
        val_envs=val_envs,
        val_fn=val_fn,
    )


def create_sft_step_fn(
    *,
    net: BaseTransformer,
    opt: torch.optim.Optimizer,
    env: Env,
    state_to_str: Callable,
    tokenizer: Tokenizer,
    pad_token_id: int,
    eos_token_id: int,
    soft_block_size: int,
    soft_bptt_window: int | None,
    max_answer_tokens: int,
    batch_size: int,
    accumulation_steps: int,
    max_grad_norm: float,
    normalize_by_sequence_length: bool = True,
    use_bf16: bool = True,
):
    device = next(net.parameters()).device

    def _step(step: int):
        nonlocal net
        opt.zero_grad()
        step_nll_sum = 0.0
        step_answer_lengths: list[int] = []

        for _ in range(accumulation_steps):
            env_responses = [env.reset() for _ in range(batch_size)]

            prompt_token_ids, attention_mask = _encode_prompts(
                env_responses, state_to_str, tokenizer, pad_token_id, batch_size, device
            )

            answer_ids_list: list[list[int]] = []
            for er in env_responses:
                ans_enc = tokenizer.encode(er.data.answer, add_special_tokens=False)
                ans_ids = ans_enc.ids + [eos_token_id]
                if len(ans_ids) > max_answer_tokens:
                    ans_ids = ans_ids[:max_answer_tokens]
                answer_ids_list.append(ans_ids)
                step_answer_lengths.append(len(ans_ids))

            max_a = max(len(a) for a in answer_ids_list)
            answer_token_ids = torch.full(
                (batch_size, max_a), pad_token_id, dtype=torch.long, device=device
            )
            answer_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
            for b, aids in enumerate(answer_ids_list):
                answer_token_ids[b, : len(aids)] = torch.tensor(aids, device=device)
                answer_lengths[b] = len(aids)

            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
            ):
                log_probs, completion_mask = compute_soft_prefill_log_probs(
                    net=net,
                    prompt_token_ids=prompt_token_ids,
                    attention_mask=attention_mask,
                    answer_token_ids=answer_token_ids,
                    answer_lengths=answer_lengths,
                    soft_block_size=soft_block_size,
                    pad_token_id=pad_token_id,
                    use_bf16=use_bf16,
                    soft_bptt_window=soft_bptt_window,
                )

            masked_lp = log_probs * completion_mask
            if normalize_by_sequence_length:
                seq_lens = completion_mask.sum(dim=-1).clamp(min=1).float()
                per_seq_nll = -(masked_lp.sum(dim=-1) / seq_lens)
            else:
                per_seq_nll = -masked_lp.sum(dim=-1)

            loss = per_seq_nll.mean() / accumulation_steps
            loss.backward()
            step_nll_sum += per_seq_nll.mean().item() / accumulation_steps

        grad_norm_val: float | None = None
        if max_grad_norm > 0:
            params = [p for group in opt.param_groups for p in group["params"]]
            grad_norm_val = torch.nn.utils.clip_grad_norm_(
                params, max_norm=max_grad_norm
            ).item()
        opt.step()

        metrics: dict[str, Any] = {
            "sft/nll_loss": step_nll_sum,
            "sft/mean_answer_length": (
                sum(step_answer_lengths) / len(step_answer_lengths)
            ),
        }
        if grad_norm_val is not None:
            metrics["sft/grad_norm"] = grad_norm_val
        if net.soft_projection_alpha is not None:
            metrics["sft/soft_projection_alpha"] = net.soft_projection_alpha.item()

        return StepFunctionReturn(
            n_episodes_processed=batch_size * accumulation_steps, metrics=metrics
        )

    return _step


def create_sft_val_fn(
    *,
    net: BaseTransformer,
    state_to_str: Callable,
    tokenizer: Tokenizer,
    pad_token_id: int,
    eos_token_id: int,
    soft_block_size: int,
    max_answer_tokens: int,
    use_bf16: bool,
    val_episodes: int,
    val_batch_size: int,
):
    def _val(env: Env):
        return evaluate_soft_prefill(
            net=net,
            env=env,
            state_to_str=state_to_str,
            answer_extractor=lambda state: state.answer,
            tokenizer=tokenizer,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            soft_block_size=soft_block_size,
            max_new_tokens=max_answer_tokens,
            max_episodes=val_episodes,
            batch_size=val_batch_size,
            temperature=0,
            use_bf16=use_bf16,
            n_examples=val_episodes,
        )

    return _val


# single step
def train_internal_reasoning_single_step_sft(
    *,
    net: BaseTransformer,
    opt: torch.optim.Optimizer,
    env: Env,
    state_to_str: Callable,
    tokenizer: Tokenizer,
    pad_token_id: int,
    eos_token_id: int,
    soft_block_size: int,
    soft_bptt_window: int | None,
    max_answer_tokens: int,
    max_episodes: int,
    batch_size: int,
    accumulation_steps: int,
    max_grad_norm: float,
    normalize_by_sequence_length: bool,
    use_bf16: bool,
    save_ckpt_freq: int = sys.maxsize,
    val_envs: Sequence[Env],
    val_batch_size: int,
    val_episodes: int,
    val_freq: int = 0,
) -> None:
    if use_bf16:
        net = net.to(dtype=torch.bfloat16)
    train_step = create_sft_step_fn(
        net=net,
        opt=opt,
        env=env,
        state_to_str=state_to_str,
        tokenizer=tokenizer,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        soft_block_size=soft_block_size,
        soft_bptt_window=soft_bptt_window,
        max_answer_tokens=max_answer_tokens,
        batch_size=batch_size,
        accumulation_steps=accumulation_steps,
        max_grad_norm=max_grad_norm,
        normalize_by_sequence_length=normalize_by_sequence_length,
        use_bf16=use_bf16,
    )

    val_fn = create_sft_val_fn(
        net=net,
        state_to_str=state_to_str,
        tokenizer=tokenizer,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        soft_block_size=soft_block_size,
        max_answer_tokens=max_answer_tokens,
        use_bf16=use_bf16,
        val_episodes=val_episodes,
        val_batch_size=val_batch_size,
    )

    train_loop(
        max_episodes=max_episodes,
        save_ckpt_freq=save_ckpt_freq,
        val_freq=val_freq,
        net=net,
        opt=opt,
        train_step=train_step,
        val_envs=val_envs,
        val_fn=val_fn,
    )
