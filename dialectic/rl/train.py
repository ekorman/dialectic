import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable

import extty
import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Integer
from tokenizers import Tokenizer

from dialectic.llm.base import BaseTransformer
from dialectic.llm.generate import PreFill
from dialectic.rl.env import Env
from dialectic.rl.evaluate import evaluate
from dialectic.rl.reward import RewardFn
from dialectic.rl.rollout import generate_rollout_batch, generate_soft_rollout_batch
from dialectic.rl.types import A, E, RewardResult, T


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


@dataclass
class ValidationConfig:
    envs: list[Env]
    reward_fn: RewardFn
    state_to_str: Callable
    extractor: Callable[[str], Any]
    tokenizer: Tokenizer
    eos_token_id: int
    pad_token_id: int
    max_episodes: int
    batch_size: int
    max_tokens_generated: int
    use_bf16: bool


@torch.no_grad()
def run_validation(
    *,
    net: BaseTransformer,
    val_config: ValidationConfig,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    reward_means: list[float] = []

    was_training = net.training
    net.eval()

    for env in val_config.envs:
        env.reseed()
        label = str(env)
        result, examples = evaluate(
            net=net,
            env=env,
            reward_fn=val_config.reward_fn,
            state_to_str=val_config.state_to_str,
            tokenizer=val_config.tokenizer,
            eos_token_id=val_config.eos_token_id,
            pad_token_id=val_config.pad_token_id,
            extractor=val_config.extractor,
            max_tokens_generated=val_config.max_tokens_generated,
            max_episodes=val_config.max_episodes,
            batch_size=val_config.batch_size,
            group_size=1,
            temperature=0.0,
            use_bf16=val_config.use_bf16,
        )

        metrics[f"val/{label}/reward_mean"] = result.reward_mean
        metrics[f"val/{label}/reward_std"] = result.reward_std
        for comp_name, comp_val in result.component_means.items():
            metrics[f"val/{label}/reward/{comp_name}"] = comp_val
        if examples:
            metrics[f"val/{label}/example"] = extty.BatchExample(
                prompts=[e.prompt for e in examples],
                responses=[e.responses for e in examples],
                rewards=[e.rewards for e in examples],
            )
        reward_means.append(result.reward_mean)

    if reward_means:
        metrics["val/reward_mean"] = sum(reward_means) / len(reward_means)

    if was_training:
        net.train()

    return metrics


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
    update_ref_net_batch_cadence: int,
    batch_size: int,
    group_size: int,
    normalize_by_sequence_length: bool,
    accumulation_steps: int,
    max_grad_norm: float,
    use_bf16: bool,
    save_ckpt_freq: int,
    advantage_fn: Callable[[Float[torch.Tensor, "G B"]], Float[torch.Tensor, "G B"]],
    val_config: ValidationConfig | None = None,
    val_freq: int = 0,
) -> None:
    device = next(net.parameters()).device
    n_episodes = 0
    step = 0

    if mu > 1 and eps is None:
        raise RuntimeError(
            "Should not have `mu` > 1 when not doing PPO style training."
        )

    ref_net = None
    while n_episodes < max_episodes:
        if beta != 0 and (step % update_ref_net_batch_cadence == 0):
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

        t_opt = time.perf_counter() - t_opt_start

        step += 1
        n_episodes += batch_size * accumulation_steps

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

        if extty._active_run is not None:
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
                        "train/hard_lp_std": sum(
                            mb["hard_lp_std"] for mb in micro_batches
                        )
                        / len(micro_batches),
                        "train/soft_lp_mean": sum(
                            mb["soft_lp_mean"] for mb in micro_batches
                        )
                        / len(micro_batches),
                        "train/soft_lp_std": sum(
                            mb["soft_lp_std"] for mb in micro_batches
                        )
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
                **{
                    f"train/reward/{name}": mean
                    for name, mean in component_means.items()
                },
            }
            if total_kl_loss is not None:
                metrics["train/kl_loss"] = total_kl_loss / mu
            if grad_norm_val is not None:
                metrics["train/grad_norm"] = grad_norm_val
            if logprob_recompute_max_diffs:
                metrics["train/logprob_recompute_max_diff"] = max(
                    logprob_recompute_max_diffs
                )
            extty.log(metrics, step=step)

            if step % save_ckpt_freq == 0:
                extty.save_checkpoint(
                    step=step,
                    state_dict=net.state_dict(),
                    optimizer_state_dict=opt.state_dict(),
                )

        if val_config is not None and val_freq > 0 and step % val_freq == 0:
            val_metrics = run_validation(net=net, val_config=val_config)
            extty.log(val_metrics, step=step)

    if step % save_ckpt_freq != 0 and extty._active_run is not None:
        extty.save_checkpoint(
            step=step,
            state_dict=net.state_dict(),
            optimizer_state_dict=opt.state_dict(),
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
    update_ref_net_batch_cadence: int,
    batch_size: int,
    group_size: int,
    temperature: float,
    advantage_fn: Callable[[Float[torch.Tensor, "G B"]], Float[torch.Tensor, "G B"]],
    normalize_by_sequence_length: bool,
    accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    logprob_chunk_size: int = 64,
    use_bf16: bool = True,
    save_ckpt_freq: int = sys.maxsize,
    val_config: ValidationConfig | None = None,
    val_freq: int = 0,
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
        val_config=val_config,
        val_freq=val_freq,
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
            chunk_logits.reshape(BG_c * L_chunk, V),
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
    update_ref_net_batch_cadence: int,
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
    accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    logprob_chunk_size: int = 64,
    use_bf16: bool = True,
    save_ckpt_freq: int = sys.maxsize,
    val_config: ValidationConfig | None = None,
    val_freq: int = 0,
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
        val_config=val_config,
        val_freq=val_freq,
    )
