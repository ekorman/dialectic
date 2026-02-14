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
from dialectic.llm.generate import PreFill, generate_hard_tokens, generate_soft_tokens
from dialectic.rl.env import Env
from dialectic.rl.reward import RewardFn
from dialectic.rl.types import A, E, EnvResponse, RewardResult, T


def _embedding_rms_norm(net: BaseTransformer) -> float:
    W = net.embed_tokens.weight
    return W.pow(2).mean().sqrt().item()


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
    full_mask = _expand_attention_mask(attention_mask, batch_size, group_size, seq_len)

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

    # Normalize per-sequence to avoid length bias: longer sequences should not
    # contribute more to the loss just because they have more tokens.
    # Each sequence's contribution is its mean (over tokens), then we average
    # across all sequences.
    sequence_lengths = completion_mask.sum(dim=-1, keepdim=True)  # [B, G, 1]
    ppo_obj_per_seq = (ppo_obj * completion_mask).sum(
        dim=-1, keepdim=True
    ) / sequence_lengths
    kl_loss_per_seq = (kl_loss * completion_mask).sum(
        dim=-1, keepdim=True
    ) / sequence_lengths

    ppo_loss_scalar = -ppo_obj_per_seq.mean()
    kl_loss_scalar = kl_loss_per_seq.mean()

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

    # Hard-token entropy stats for comparison with soft runs.
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
    ):
        stacked = stack_and_pad(
            tensors=rollout.completion_token_ids, pad_token_id=pad_token_id
        )
        logits = compute_logits_of_group(
            net=net,
            input_ids=stacked,
            attention_mask=rollout.attention_mask,
        )
        l_prompt = rollout.attention_mask.shape[1]
        logits = logits[:, :, l_prompt - 1 : -1]
        logp = torch.log_softmax(logits, dim=-1)
        entropy = -(logp.exp() * logp).sum(-1)
        if torch.any(completion_mask):
            ent_vals = entropy[completion_mask]
            hard_ent_mean = ent_vals.mean().item()
            hard_ent_std = ent_vals.std().item()
        else:
            hard_ent_mean = float("nan")
            hard_ent_std = float("nan")

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
        "hard_ent_mean": hard_ent_mean,
        "hard_ent_std": hard_ent_std,
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
    eps: float,
    mu: int,
    max_episodes: int,
    update_ref_net_batch_cadence: int,
    batch_size: int,
    group_size: int,
    normalize_advantages: bool,
    accumulation_steps: int,
    max_grad_norm: float,
    use_bf16: bool,
    save_ckpt_freq: int,
) -> None:
    device = next(net.parameters()).device
    n_episodes = 0
    step = 0

    while n_episodes < max_episodes:
        if step % update_ref_net_batch_cadence == 0:
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
                    log_probs, _ = recompute_log_probs_fn(net, micro_batch)

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
            extty.log(
                {
                    "train/loss": total_loss / mu,
                    "train/ppo_loss": total_ppo_loss / mu,
                    "train/kl_loss": total_kl_loss / mu,
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
                            "train/hard_entropy_mean": sum(
                                mb["hard_ent_mean"] for mb in micro_batches
                            )
                            / len(micro_batches),
                            "train/hard_entropy_std": sum(
                                mb["hard_ent_std"] for mb in micro_batches
                            )
                            / len(micro_batches),
                            "train/soft_entropy_mean": sum(
                                mb["soft_ent_mean"] for mb in micro_batches
                            )
                            / len(micro_batches),
                            "train/soft_entropy_std": sum(
                                mb["soft_ent_std"] for mb in micro_batches
                            )
                            / len(micro_batches),
                            "train/entropy_mean": sum(
                                mb["entropy_mean"] for mb in micro_batches
                            )
                            / len(micro_batches),
                            "train/entropy_std": sum(
                                mb["entropy_std"] for mb in micro_batches
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
                    **(
                        {
                            "train/entropy_mean": sum(
                                mb["hard_ent_mean"] for mb in micro_batches
                            )
                            / len(micro_batches),
                            "train/entropy_std": sum(
                                mb["hard_ent_std"] for mb in micro_batches
                            )
                            / len(micro_batches),
                        }
                        if "hard_ent_mean" in micro_batches[0]
                        else {}
                    ),
                },
                step=step,
            )

            if step % save_ckpt_freq == 0:
                extty.save_checkpoint(
                    step=step,
                    state_dict=net.state_dict(),
                    optimizer_state_dict=opt.state_dict(),
                )

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
    if use_bf16:
        net = net.to(dtype=torch.bfloat16)

    def collect_fn(net: BaseTransformer, ref_net: BaseTransformer) -> dict:
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
        normalize_advantages=normalize_advantages,
        accumulation_steps=accumulation_steps,
        max_grad_norm=max_grad_norm,
        use_bf16=use_bf16,
        save_ckpt_freq=save_ckpt_freq,
    )


def stack_and_pad_soft(
    *,
    tokens: list[Float[torch.Tensor, "B L V"]],
    noise: list[Float[torch.Tensor, "B L D"]],
    hard_masks: list[Bool[torch.Tensor, "B L"]],
    pad_token_id: int,
) -> tuple[
    Float[torch.Tensor, "B G L V"],
    Float[torch.Tensor, "B G L D"],
    Bool[torch.Tensor, "B G L"],
    Bool[torch.Tensor, "B G L"],
]:
    max_len = max(t.shape[1] for t in tokens)
    batch_size = tokens[0].shape[0]
    group_size = len(tokens)
    V = tokens[0].shape[2]
    D = noise[0].shape[2]
    device = tokens[0].device

    pad_one_hot = torch.nn.functional.one_hot(
        torch.tensor(pad_token_id, device=device), V
    ).to(dtype=tokens[0].dtype)

    stacked_tokens = pad_one_hot.expand(batch_size, group_size, max_len, V).clone()
    stacked_noise = torch.zeros(
        batch_size, group_size, max_len, D, device=device, dtype=noise[0].dtype
    )
    # Padded positions default to hard=True; harmless since non_pad_mask excludes them from the loss.
    stacked_masks = torch.ones(
        batch_size, group_size, max_len, device=device, dtype=torch.bool
    )
    non_pad_mask = torch.zeros(
        batch_size, group_size, max_len, device=device, dtype=torch.bool
    )

    for g, (t, n, m) in enumerate(zip(tokens, noise, hard_masks)):
        L = t.shape[1]
        stacked_tokens[:, g, :L] = t
        stacked_noise[:, g, :L] = n
        stacked_masks[:, g, :L] = m
        non_pad_mask[:, g, :L] = True

    return stacked_tokens, stacked_noise, stacked_masks, non_pad_mask


def stack_and_pad_actions(
    *,
    actions: list[Float[torch.Tensor, "B L D"]],
    max_len: int,
) -> Float[torch.Tensor, "B G L D"]:
    batch_size = actions[0].shape[0]
    group_size = len(actions)
    D = actions[0].shape[2]
    device = actions[0].device
    dtype = actions[0].dtype

    stacked_actions = torch.zeros(
        batch_size, group_size, max_len, D, device=device, dtype=dtype
    )

    for g, a in enumerate(actions):
        L = a.shape[1]
        stacked_actions[:, g, :L] = a

    return stacked_actions


def compute_soft_log_probs(
    *,
    net: BaseTransformer,
    attention_mask: Bool[torch.Tensor, "B L_prompt"],
    completion_tokens: list[Float[torch.Tensor, "B L V"]],
    completion_noise: list[Float[torch.Tensor, "B L D"]],
    completion_action_embeddings: list[Float[torch.Tensor, "B L D"]] | None = None,
    hard_tokens_mask: list[Bool[torch.Tensor, "B L"]],
    noise_std: float,
    temperature: float,
    pad_token_id: int,
    chunk_size: int = 0,
    return_entropy: bool = False,
) -> tuple[Float[torch.Tensor, "B G L_c"], Bool[torch.Tensor, "B G L_c"]]:
    l_prompt = attention_mask.shape[1]
    stacked_tokens, stacked_noise, stacked_masks, non_pad_mask = stack_and_pad_soft(
        tokens=completion_tokens,
        noise=completion_noise,
        hard_masks=hard_tokens_mask,
        pad_token_id=pad_token_id,
    )
    stacked_actions = None
    if completion_action_embeddings is not None:
        stacked_actions = stack_and_pad_actions(
            actions=completion_action_embeddings, max_len=stacked_tokens.shape[2]
        )

    if chunk_size > 0:
        log_probs, entropy = _compute_soft_log_probs_chunked(
            net=net,
            stacked_tokens=stacked_tokens,
            stacked_noise=stacked_noise,
            stacked_masks=stacked_masks,
            stacked_actions=stacked_actions,
            attention_mask=attention_mask,
            l_prompt=l_prompt,
            noise_std=noise_std,
            temperature=temperature,
            chunk_size=chunk_size,
            return_entropy=return_entropy,
        )
    else:
        log_probs, entropy = _compute_soft_log_probs_full(
            net=net,
            stacked_tokens=stacked_tokens,
            stacked_noise=stacked_noise,
            stacked_masks=stacked_masks,
            stacked_actions=stacked_actions,
            attention_mask=attention_mask,
            l_prompt=l_prompt,
            noise_std=noise_std,
            temperature=temperature,
            return_entropy=return_entropy,
        )

    # Treat explicit pad tokens as masked-out (finished sequences emit pad tokens).
    shadow_ids = stacked_tokens.argmax(-1)
    pad_mask = shadow_ids != pad_token_id
    completion_mask = (non_pad_mask & pad_mask)[:, :, l_prompt:]
    if return_entropy:
        return log_probs, completion_mask, entropy
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


def _compute_soft_log_probs_full(
    *,
    net: BaseTransformer,
    stacked_tokens: Float[torch.Tensor, "B G L V"],
    stacked_noise: Float[torch.Tensor, "B G L D"],
    stacked_masks: Bool[torch.Tensor, "B G L"],
    stacked_actions: Float[torch.Tensor, "B G L D"] | None,
    attention_mask: Bool[torch.Tensor, "B L_prompt"],
    l_prompt: int,
    noise_std: float,
    temperature: float,
    return_entropy: bool = False,
) -> tuple[Float[torch.Tensor, "B G L_c"], Float[torch.Tensor, "B G L_c"] | None]:
    batch_size, group_size, seq_len, V = stacked_tokens.shape
    BG = batch_size * group_size

    flat_tokens = stacked_tokens.view(BG, seq_len, V)
    flat_noise = stacked_noise.view(BG, seq_len, -1)
    flat_actions = (
        stacked_actions.view(BG, seq_len, -1) if stacked_actions is not None else None
    )

    full_mask = _expand_attention_mask(attention_mask, batch_size, group_size, seq_len)

    logits = net(
        flat_tokens,
        return_all_logits=True,
        attention_mask=full_mask,
        soft_token_noise=flat_noise,
    )

    logits = logits[:, l_prompt - 1 : -1]
    comp_tokens = flat_tokens[:, l_prompt:]
    comp_noise = flat_noise[:, l_prompt:]
    comp_masks = stacked_masks.view(BG, seq_len)[:, l_prompt:]
    completion_len = seq_len - l_prompt
    entropy = None

    W = net.embed_tokens.weight

    shadow_ids = comp_tokens.argmax(-1)
    hard_lp = -torch.nn.functional.cross_entropy(
        logits.reshape(BG * completion_len, V),
        shadow_ids.reshape(BG * completion_len),
        reduction="none",
    ).reshape(BG, completion_len)

    # compute Gaussian log-probs in float32 to avoid bfloat16 overflow in the
    # backward pass: 1/σ² can be ~10k and amplifies gradients beyond bf16 range
    device_type = comp_tokens.device.type
    with torch.amp.autocast(device_type=device_type, enabled=False):
        if flat_actions is not None:
            e_action = flat_actions[:, l_prompt:].float()
        else:
            e_action = comp_tokens.float() @ W.float() + comp_noise.float()
        mu_new = torch.softmax(logits.float() / temperature, dim=-1) @ W.float()
        # Normalize by embedding dimension to keep scale comparable to hard log-probs.
        soft_lp = -0.5 * ((e_action - mu_new) ** 2).mean(-1) / (noise_std**2)
        if return_entropy:
            logp = torch.log_softmax(logits.float() / temperature, dim=-1)
            entropy = -(logp.exp() * logp).sum(-1)

    log_probs = torch.where(comp_masks, hard_lp, soft_lp)
    log_probs = log_probs.view(batch_size, group_size, completion_len)
    if return_entropy:
        entropy = entropy.view(batch_size, group_size, completion_len)
        return log_probs, entropy
    return log_probs, None


def _compute_soft_log_probs_chunked(
    *,
    net: BaseTransformer,
    stacked_tokens: Float[torch.Tensor, "B G L V"],
    stacked_noise: Float[torch.Tensor, "B G L D"],
    stacked_masks: Bool[torch.Tensor, "B G L"],
    stacked_actions: Float[torch.Tensor, "B G L D"] | None,
    attention_mask: Bool[torch.Tensor, "B L_prompt"],
    l_prompt: int,
    noise_std: float,
    temperature: float,
    chunk_size: int,
    return_entropy: bool = False,
) -> tuple[Float[torch.Tensor, "B G L_c"], Float[torch.Tensor, "B G L_c"] | None]:
    batch_size, group_size, seq_len, V = stacked_tokens.shape
    BG = batch_size * group_size

    flat_tokens = stacked_tokens.view(BG, seq_len, V)
    flat_noise = stacked_noise.view(BG, seq_len, -1)
    flat_actions = (
        stacked_actions.view(BG, seq_len, -1) if stacked_actions is not None else None
    )

    full_mask = _expand_attention_mask(attention_mask, batch_size, group_size, seq_len)

    hidden_states = net(
        flat_tokens,
        attention_mask=full_mask,
        return_hidden_states=True,
        soft_token_noise=flat_noise,
    )

    completion_len = seq_len - l_prompt
    hidden_for_completion = hidden_states[:, l_prompt - 1 : -1]
    comp_tokens = flat_tokens[:, l_prompt:]
    comp_noise = flat_noise[:, l_prompt:]
    comp_masks = stacked_masks.view(BG, seq_len)[:, l_prompt:]
    W = net.embed_tokens.weight

    log_probs_list = []
    entropy_list = [] if return_entropy else None
    for chunk_start in range(0, completion_len, chunk_size):
        chunk_end = min(chunk_start + chunk_size, completion_len)

        chunk_hidden = hidden_for_completion[:, chunk_start:chunk_end]
        chunk_logits = net.lm_head(chunk_hidden)
        chunk_comp_tokens = comp_tokens[:, chunk_start:chunk_end]
        chunk_comp_noise = comp_noise[:, chunk_start:chunk_end]
        chunk_masks = comp_masks[:, chunk_start:chunk_end]
        chunk_actions = (
            flat_actions[:, l_prompt + chunk_start : l_prompt + chunk_end]
            if flat_actions is not None
            else None
        )

        BG_c, L_chunk, _ = chunk_logits.shape

        chunk_shadow_ids = chunk_comp_tokens.argmax(-1)
        hard_lp = -torch.nn.functional.cross_entropy(
            chunk_logits.reshape(BG_c * L_chunk, V),
            chunk_shadow_ids.reshape(BG_c * L_chunk),
            reduction="none",
        ).reshape(BG_c, L_chunk)

        device_type = chunk_comp_tokens.device.type
        with torch.amp.autocast(device_type=device_type, enabled=False):
            if chunk_actions is not None:
                e_action = chunk_actions.float()
            else:
                e_action = (
                    chunk_comp_tokens.float() @ W.float() + chunk_comp_noise.float()
                )
            mu_new = (
                torch.softmax(chunk_logits.float() / temperature, dim=-1) @ W.float()
            )
            # Normalize by embedding dimension to keep scale comparable to hard log-probs.
            soft_lp = -0.5 * ((e_action - mu_new) ** 2).mean(-1) / (noise_std**2)
            if return_entropy:
                logp = torch.log_softmax(chunk_logits.float() / temperature, dim=-1)
                entropy = -(logp.exp() * logp).sum(-1)

        chunk_log_probs = torch.where(chunk_masks, hard_lp, soft_lp)
        log_probs_list.append(chunk_log_probs)
        if return_entropy:
            entropy_list.append(entropy)
        del chunk_logits

    log_probs_flat = torch.cat(log_probs_list, dim=1).view(
        batch_size, group_size, completion_len
    )
    if return_entropy:
        entropy_flat = torch.cat(entropy_list, dim=1).view(
            batch_size, group_size, completion_len
        )
        return log_probs_flat, entropy_flat
    return log_probs_flat, None


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
    switch_to_hard_tokens_condition: torch.Tensor | None = None,
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
        min_soft_steps=min_soft_steps,
        prefill=prefill,
        use_bf16=use_bf16,
    )

    device = next(net.parameters()).device
    t_logprobs_start = time.perf_counter()
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
    ):
        ref_log_probs, completion_mask = compute_soft_log_probs(
            net=ref_net,
            attention_mask=rollout.attention_mask,
            completion_tokens=rollout.completion_tokens,
            completion_noise=rollout.completion_noise,
            completion_action_embeddings=rollout.completion_action_embeddings,
            hard_tokens_mask=rollout.hard_tokens_mask,
            noise_std=rollout.noise_std,
            temperature=rollout.temperature,
            pad_token_id=pad_token_id,
            chunk_size=logprob_chunk_size,
        )
        old_log_probs, _, entropy = compute_soft_log_probs(
            net=net,
            attention_mask=rollout.attention_mask,
            completion_tokens=rollout.completion_tokens,
            completion_noise=rollout.completion_noise,
            completion_action_embeddings=rollout.completion_action_embeddings,
            hard_tokens_mask=rollout.hard_tokens_mask,
            noise_std=rollout.noise_std,
            temperature=rollout.temperature,
            pad_token_id=pad_token_id,
            chunk_size=logprob_chunk_size,
            return_entropy=True,
        )
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

        shadow_ids = rollout.completion_tokens[g].argmax(-1)
        comp_pad_ok = shadow_ids[:, l_prompt:] != pad_token_id
        pad_len = min(comp_pad_ok.shape[1], Lc)
        pad_ok_mask[:, g, :pad_len] = comp_pad_ok[:, :pad_len]

    if __debug__:
        if torch.any(completion_mask & ~pad_ok_mask):
            raise RuntimeError("completion_mask includes padded tokens in soft rollout")

    hard_completion_ratio = (
        (hard_completion_mask & completion_mask).sum().float()
        / completion_mask.sum().float().clamp_min(1.0)
    ).item()

    def _masked_stats(values: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
        if not torch.any(mask):
            return float("nan"), float("nan")
        vals = values[mask]
        return vals.mean().item(), vals.std().item()

    hard_mask = hard_completion_mask & completion_mask
    soft_mask = (~hard_completion_mask) & completion_mask
    hard_lp_mean, hard_lp_std = _masked_stats(old_log_probs, hard_mask)
    soft_lp_mean, soft_lp_std = _masked_stats(old_log_probs, soft_mask)
    hard_ent_mean, hard_ent_std = _masked_stats(entropy, hard_mask)
    soft_ent_mean, soft_ent_std = _masked_stats(entropy, soft_mask)
    all_ent_mean, all_ent_std = _masked_stats(entropy, completion_mask)

    return {
        "prompts": rollout.prompts,
        "env_responses": rollout.env_responses,
        "attention_mask": rollout.attention_mask,
        "completion_tokens": rollout.completion_tokens,
        "completion_action_embeddings": rollout.completion_action_embeddings,
        "completion_noise": rollout.completion_noise,
        "hard_tokens_mask": rollout.hard_tokens_mask,
        "noise_std": rollout.noise_std,
        "temperature": rollout.temperature,
        "output_strs": rollout.output_strs,
        "reward_results": rollout.reward_results,
        "rewards": rollout.rewards,
        "ref_log_probs": ref_log_probs,
        "old_log_probs": old_log_probs,
        "completion_mask": completion_mask,
        "hard_completion_ratio": hard_completion_ratio,
        "hard_lp_mean": hard_lp_mean,
        "hard_lp_std": hard_lp_std,
        "soft_lp_mean": soft_lp_mean,
        "soft_lp_std": soft_lp_std,
        "hard_ent_mean": hard_ent_mean,
        "hard_ent_std": hard_ent_std,
        "soft_ent_mean": soft_ent_mean,
        "soft_ent_std": soft_ent_std,
        "entropy_mean": all_ent_mean,
        "entropy_std": all_ent_std,
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
    eps: float,
    mu: int,
    max_tokens_generated: int,
    max_episodes: int,
    update_ref_net_batch_cadence: int,
    batch_size: int,
    group_size: int,
    temperature: float,
    noise_std: float,
    switch_to_hard_tokens_condition: torch.Tensor | None = None,
    min_soft_steps: int = 0,
    prefill: PreFill | None = None,
    normalize_advantages: bool = True,
    accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    logprob_chunk_size: int = 64,
    use_bf16: bool = True,
    save_ckpt_freq: int = sys.maxsize,
) -> None:
    if use_bf16:
        net = net.to(dtype=torch.bfloat16)

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
            noise_std=noise_std,
            switch_to_hard_tokens_condition=switch_to_hard_tokens_condition,
            min_soft_steps=min_soft_steps,
            prefill=prefill,
            logprob_chunk_size=logprob_chunk_size,
            use_bf16=use_bf16,
        )

    def recompute_fn(
        net: BaseTransformer, mb: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return compute_soft_log_probs(
            net=net,
            attention_mask=mb["attention_mask"],
            completion_tokens=mb["completion_tokens"],
            completion_noise=mb["completion_noise"],
            completion_action_embeddings=mb["completion_action_embeddings"],
            hard_tokens_mask=mb["hard_tokens_mask"],
            noise_std=mb["noise_std"],
            temperature=mb["temperature"],
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
        normalize_advantages=normalize_advantages,
        accumulation_steps=accumulation_steps,
        max_grad_norm=max_grad_norm,
        use_bf16=use_bf16,
        save_ckpt_freq=save_ckpt_freq,
    )
