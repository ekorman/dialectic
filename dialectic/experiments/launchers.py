import sys
from dataclasses import dataclass
from functools import partial
from typing import Callable, Literal

import extty
import torch

from dialectic.experiments.prompts import PromptCollection
from dialectic.experiments.reward_fns import get_coutdown_reward_fn
from dialectic.llm.base import BaseTransformer
from dialectic.llm.registry import MODEL_REGISTRY
from dialectic.llm.templates import Message
from dialectic.rl.env import Countdown, CountdownEnv, Env, MathEnv, MathState, MazeState
from dialectic.rl.extractors import extract_from_answer_tags
from dialectic.rl.math import MathDatasetConfig
from dialectic.rl.reward import RewardFn
from dialectic.rl.train import (
    ValidationConfig,
    grpo_advantage,
    rloo_advantage,
    train_grpo,
)


@dataclass
class TrainParams:
    model_name: str
    batch_size: int
    lr: float
    accumulation_steps: int
    max_grad_norm: float
    max_episodes: int
    max_tokens_generated: int
    val_freq: int
    val_episodes: int
    val_batch_size: int
    compile_model: bool
    seed: int
    use_bf_16: bool
    temperature: float
    logprob_chunk_size: int
    save_ckpt_freq: int = sys.maxsize


@dataclass
class GRPOParams:
    group_size: int
    advantage_fn_type: Literal["grpo", "rloo"]
    normalize_advantages: bool
    normalize_by_sequence_length: bool
    mu: int
    update_ref_net_batch_cadence: int
    eps: float
    beta: float


@dataclass
class SoftRecurrentParams:
    soft_block_size: int
    soft_bptt_window: int
    soft_projection: bool
    soft_projection_alpha_init: float
    soft_projection_rank: int
    max_cycles: int


@dataclass
class STHTParams:
    noise_std: float


@dataclass
class RewardParams:
    answer_tags_weight: float
    think_tags_weight: float


@dataclass
class CountdownParams:
    n_larges: int | list[int]
    n_total: int | list[int]
    n_ops: int | list


def load_model_and_opt(
    *, train_params: TrainParams, **kwargs
) -> tuple[BaseTransformer, torch.optim.Optimizer]:
    model_info = MODEL_REGISTRY[train_params.model_name]
    net = model_info.load_net(**kwargs)
    if train_params.compile_model:
        net.compile()
    opt = torch.optim.AdamW(net.parameters(), lr=train_params.lr)
    return net, opt


def get_state_to_str(
    *,
    format_messages: Callable[[list[Message], bool], str],
    system_prompt: str | None = None,
    assistant_prefill: str | None = None,
):
    def _state_to_str(data: Countdown | MazeState | MathState) -> str:
        msgs = []
        if system_prompt:
            msgs.append(Message(role="system", content=system_prompt))
        msgs.append(Message(role="user", content=data.prompt))
        ret = format_messages(msgs, True)
        if assistant_prefill:
            ret += assistant_prefill
        return ret

    return _state_to_str


def _train_grpo(
    *,
    train_params: TrainParams,
    grpo_params: GRPOParams,
    env: Env,
    prompt_collection: PromptCollection,
    reward_fn: RewardFn,
    extractor: Callable[[str], str | None],
    val_envs: list[Env],
):
    if grpo_params.advantage_fn_type == "grpo":
        advantage_fn = partial(
            grpo_advantage, normalize=grpo_params.normalize_advantages
        )
    elif grpo_params.advantage_fn_type == "rloo":
        advantage_fn = rloo_advantage
    else:
        raise ValueError(
            f"Got unknown advantage function type {grpo_params.advantage_fn_type}"
        )

    model_info = MODEL_REGISTRY[train_params.model_name]
    net, opt = load_model_and_opt(train_params=train_params)
    format_messages = model_info.format_messages

    state_to_str = get_state_to_str(
        format_messages=format_messages,
        system_prompt=prompt_collection.system_prompt,
        assistant_prefill=prompt_collection.assistant_prefill,
    )
    tokenizer = model_info.load_tokenizer()

    val_config = ValidationConfig(
        envs=val_envs,
        reward_fn=reward_fn,
        state_to_str=state_to_str,
        extractor=extractor,
        tokenizer=tokenizer,
        eos_token_id=model_info.eos_token_id,
        pad_token_id=model_info.pad_token_id,
        max_episodes=train_params.max_episodes,
        batch_size=train_params.batch_size,
        max_tokens_generated=train_params.max_tokens_generated,
        use_bf16=train_params.use_bf_16,
        internal_reasoning=False,
        soft_prefill=False,
        answer_extractor=None,
    )

    train_grpo(
        net=net,
        opt=opt,
        env=env,
        reward_fn=reward_fn,
        state_to_str=state_to_str,
        tokenizer=tokenizer,
        eos_token_id=model_info.eos_token_id,
        pad_token_id=model_info.pad_token_id,
        extractor=extractor,
        beta=grpo_params.beta,
        eps=grpo_params.eps,
        mu=grpo_params.mu,
        max_tokens_generated=train_params.max_tokens_generated,
        max_episodes=train_params.max_episodes,
        update_ref_net_batch_cadence=grpo_params.update_ref_net_batch_cadence,
        batch_size=train_params.batch_size,
        group_size=grpo_params.group_size,
        temperature=train_params.temperature,
        advantage_fn=advantage_fn,
        normalize_by_sequence_length=grpo_params.normalize_by_sequence_length,
        accumulation_steps=train_params.accumulation_steps,
        max_grad_norm=train_params.max_grad_norm,
        logprob_chunk_size=train_params.logprob_chunk_size,
        use_bf16=train_params.use_bf_16,
        save_ckpt_freq=train_params.save_ckpt_freq,
        val_config=val_config,
        val_freq=train_params.val_freq,
    )


@extty.experiment(project="grpo-countdown")
def train_grpo_countdown(
    *,
    train_params: TrainParams,
    grpo_params: GRPOParams,
    reward_params: RewardParams,
    prompt_collection: PromptCollection,
    countdown_params: CountdownParams,
):
    n_ops = countdown_params.n_ops
    n_larges = countdown_params.n_larges
    n_total = countdown_params.n_total

    env = CountdownEnv(
        seed=train_params.seed,
        n_larges=n_larges,
        n_total=n_total,
        n_ops=n_ops,
        prompt_template=prompt_collection.env_prompt,
    )
    reward_fn = get_coutdown_reward_fn(
        answer_tags_weight=reward_params.answer_tags_weight,
        think_tags_weight=reward_params.think_tags_weight,
    )
    extractor = extract_from_answer_tags

    n_ops_list = [n_ops] if isinstance(n_ops, int) else n_ops
    n_total_list = [n_total] if isinstance(n_total, int) else n_total
    n_larges_list = [n_larges] if isinstance(n_larges, int) else n_larges
    val_envs = [
        CountdownEnv(
            seed=2026 + i,
            n_larges=n_larges_list[i],
            n_total=n_total_list[i],
            n_ops=n_ops_list[i],
            prompt_template=prompt_collection.env_prompt,
        )
        for i in range(len(n_ops_list))
    ]

    return _train_grpo(
        train_params=train_params,
        grpo_params=grpo_params,
        env=env,
        prompt_collection=prompt_collection,
        reward_fn=reward_fn,
        extractor=extractor,
        val_envs=val_envs,
    )


def _train_sft():
    val_config = ValidationConfig(
        envs=val_envs,
        reward_fn=reward_fn,
        state_to_str=state_to_str,
        extractor=lambda x: x,
        tokenizer=tokenizer,
        eos_token_id=model_info.eos_token_id,
        pad_token_id=model_info.pad_token_id,
        max_episodes=val_episodes,
        batch_size=val_batch_size,
        max_tokens_generated=max_tokens,
        use_bf16=use_bf16,
        soft_prefill=True,
        answer_extractor=lambda state: state.answer,
        max_new_tokens=max_answer_tokens,
        soft_block_size=soft_block_size,
    )


@extty.experiment(project="sft-math")
def train_sft_math(
    *,
    train_params: TrainParams,
    math_config: MathDatasetConfig,
    prompt_collection: PromptCollection,
):
    env = MathEnv(config=math_config)
    val_envs = [MathEnv(config=math_config, seed=2026)]
