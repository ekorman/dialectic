import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable

import extty
import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Integer
from tokenizers import Tokenizer

from dialectic.llm.qwen import Qwen, generate_from_tokens
from dialectic.rl.env import Env
from dialectic.rl.reward import RewardFn
from dialectic.rl.types import A, E, EnvResponse, T


@dataclass
class GRPOMetrics:
    losses: list[float] = field(default_factory=list)
    mean_rewards: list[float] = field(default_factory=list)


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
    net: nn.Module,
    attention_mask: Integer[torch.Tensor, "B L_prompt"],
    completion_token_ids: list[Integer[torch.Tensor, "B L_completion"]],
    pad_token_id: int,
) -> tuple[Float[torch.Tensor, "B G L_new"], Bool[torch.Tensor, "B G L_new"]]:
    l_prompt = attention_mask.shape[1]
    stacked = stack_and_pad(tensors=completion_token_ids, pad_token_id=pad_token_id)
    logits: Float[torch.Tensor, "B G L_completion VC"] = compute_logits_of_group(
        net=net,
        input_ids=stacked,
        attention_mask=attention_mask,
    )

    # get only the new part
    logits = logits[:, :, l_prompt - 1 : -1]
    log_probs = logits.log_softmax(-1)
    only_completion = stacked[:, :, l_prompt:]
    log_probs = log_probs.take_along_dim(only_completion.unsqueeze(-1), -1).squeeze(-1)

    completion_mask = only_completion != pad_token_id

    return log_probs, completion_mask


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
) -> float:
    batch_size, g = log_probs.shape[:2]
    if old_log_probs is None:
        old_log_probs = log_probs.detach()

    advs: Float[torch.Tensor, "B G"] = compute_advantages(
        rewards, normalize=normalize_advantages
    ).T.view(batch_size, g, 1)
    ratio = (log_probs - old_log_probs).exp()
    unclipped = ratio * advs
    clipped = torch.clip(ratio, 1 - eps, 1 + eps) * advs

    ppo_loss = torch.min(unclipped, clipped)

    kl = torch.exp(ref_log_probs - log_probs) - (ref_log_probs - log_probs) - 1

    per_token_loss = -ppo_loss + beta * kl
    loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
    opt.zero_grad()
    loss.backward()
    opt.step()

    return loss.item()


def train_grpo(
    *,
    net: Qwen,
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
    mu: int,  # number of optimizations per batch
    max_tokens_generated: int,
    max_episodes: int,
    update_ref_net_batch_cadence: int,
    batch_size: int,
    group_size: int,
    temperature: float,
    normalize_advantages: bool = True,
) -> None:
    n_episodes = 0
    n_batches = 0

    while n_episodes < max_episodes:
        if n_batches % update_ref_net_batch_cadence == 0:
            ref_net = deepcopy(net)

        env_responses = get_batch(env, batch_size)
        prompts = [state_to_str(resp.data) for resp in env_responses]
        device = next(net.parameters()).device

        tokenizer.enable_padding(direction="left")
        tokens = tokenizer.encode_batch(prompts)
        # shape [B, L]
        # Model expects True = masked/padding, tokenizer gives 1 = attend, so invert
        attention_mask = (
            torch.tensor([t.attention_mask for t in tokens], device=device) == 0
        )

        token_ids = torch.tensor([t.ids for t in tokens], device=device)

        # generate `group_size` many completions for each batch
        # list of length `group_size`
        net.eval()
        completion_token_ids: list[Integer[torch.Tensor, "B L"]] = [
            generate_from_tokens(
                net=net,
                token_ids=token_ids,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                sampling_strategy="sample",
                temperature=temperature,
                attention_mask=attention_mask,
                use_kv_cache=True,
                max_tokens_generated=max_tokens_generated,
            )
            for _ in range(group_size)
        ]
        net.train()

        prompt_len = token_ids.shape[1]

        # outer list has length G, inner last has length B
        output_strs: list[list[str]] = [
            tokenizer.decode_batch(c[:, prompt_len:].tolist())
            for c in completion_token_ids
        ]
        rewards: Float[torch.Tensor, "G B"] = torch.tensor(
            [
                [
                    reward_fn(
                        env_response=env_response,
                        raw_model_output=s,
                        extracted_model_output=extractor(s),
                    )
                    for s, env_response in zip(group_batch, env_responses)
                ]
                for group_batch in output_strs
            ],
            device=device,
        )

        with torch.inference_mode():
            ref_log_probs, completion_mask = compute_log_probs(
                net=ref_net,
                attention_mask=attention_mask,
                completion_token_ids=completion_token_ids,
                pad_token_id=pad_token_id,
            )
        completion_mask = completion_mask.clone()

        old_log_probs = None
        total_loss = 0
        for _ in range(mu):
            log_probs, _ = compute_log_probs(
                net=net,
                attention_mask=attention_mask,
                completion_token_ids=completion_token_ids,
                pad_token_id=pad_token_id,
            )

            total_loss += grpo_step(
                opt=opt,
                log_probs=log_probs,
                old_log_probs=old_log_probs,
                ref_log_probs=ref_log_probs,
                completion_mask=completion_mask,
                beta=beta,
                eps=eps,
                rewards=rewards,
                normalize_advantages=normalize_advantages,
            )

            old_log_probs = log_probs.detach()

        n_batches += 1
        n_episodes += len(prompts)

        print("finished batch")

        completion_len_mean = sum(
            [len(s) for batch_output_strs in output_strs for s in batch_output_strs]
        ) / (len(output_strs) * len(output_strs[0]))

        # need to flip B/G for responses
        examples = extty.BatchExample(
            prompts=prompts,
            responses=[
                [output_strs[j][i] for j in range(len(output_strs))]
                for i in range(len(output_strs[0]))
            ],
        )

        extty.log(
            {
                "train/loss": total_loss / mu,
                "train/reward_mean": rewards.mean().item(),
                "train/reward_std": rewards.std().item(),
                "train/completion_len_mean": completion_len_mean,
                "train/example": examples,
            },
            step=n_episodes,
        )
        print("done with episode")
