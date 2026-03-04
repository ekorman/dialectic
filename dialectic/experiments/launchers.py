import sys
from dataclasses import dataclass
from functools import partial
from typing import Callable, Literal

import extty
import torch

from dialectic.experiments.prompts import (
    MAZE_INTERNAL_REASONING_PROMPT,
    PromptCollection,
)
from dialectic.experiments.reward_fns import get_coutdown_reward_fn, get_maze_reward_fn
from dialectic.llm.base import BaseTransformer
from dialectic.llm.registry import MODEL_REGISTRY
from dialectic.llm.templates import Message
from dialectic.rl.env import (
    Countdown,
    CountdownEnv,
    Env,
    MathEnv,
    MathState,
    MazeEnv,
    MazeState,
)
from dialectic.rl.extractors import extract_from_answer_tags
from dialectic.rl.math import MathDatasetConfig
from dialectic.rl.maze import MazeConfig
from dialectic.rl.reward import RewardFn
from dialectic.rl.train import (
    grpo_advantage,
    rloo_advantage,
    train_grpo,
    train_internal_reasoning_sft,
    train_internal_reasoning_single_step_sft,
    train_soft_grpo,
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
    val_batch_size: int
    val_episodes: int
    val_envs: int
    val_freq: int
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
class SoftGRPOParams:
    noise_std: float
    normalize_soft_pdf_by_dim: bool


@dataclass
class SoftParams:
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
class MazeRewardParams:
    validity_weight: float
    distance_weight: float


@dataclass
class CountdownParams:
    n_larges: int | list[int]
    n_total: int | list[int]
    n_ops: int | list


@dataclass
class SingleStepSFTParams:
    max_answer_tokens: int
    normalize_by_sequence_length: bool


@dataclass
class MultiStepSFTParams:
    move_name_to_id: dict[str, int]
    valid_hard_token_ids: list[int]
    normalize_by_sequence_length: bool
    think_token_id: int | None


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
        val_batch_size=train_params.val_batch_size,
        val_episodes=train_params.val_episodes,
        val_freq=train_params.val_freq,
        val_envs=val_envs,
    )


def _get_countdown_env_reward_fn_extractor_val_envs(
    train_params: TrainParams,
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
    return env, reward_fn, extractor, val_envs


@extty.experiment(project="grpo-countdown")
def train_grpo_countdown(
    *,
    train_params: TrainParams,
    grpo_params: GRPOParams,
    reward_params: RewardParams,
    prompt_collection: PromptCollection,
    countdown_params: CountdownParams,
):
    env, reward_fn, extractor, val_envs = (
        _get_countdown_env_reward_fn_extractor_val_envs(
            train_params=train_params,
            reward_params=reward_params,
            prompt_collection=prompt_collection,
            countdown_params=countdown_params,
        )
    )

    return _train_grpo(
        train_params=train_params,
        grpo_params=grpo_params,
        env=env,
        prompt_collection=prompt_collection,
        reward_fn=reward_fn,
        extractor=extractor,
        val_envs=val_envs,
    )


def _train_internal_reasoning_single_step_sft(
    train_params: TrainParams,
    soft_params: SoftParams,
    sft_params: SingleStepSFTParams,
    env: Env,
    prompt_collection: PromptCollection,
    val_envs: list[Env],
):
    model_info = MODEL_REGISTRY[train_params.model_name]
    net, opt = load_model_and_opt(train_params=train_params)
    tokenizer = model_info.load_tokenizer()

    state_to_str = get_state_to_str(
        format_messages=model_info.format_messages,
        system_prompt=prompt_collection.system_prompt,
        assistant_prefill=prompt_collection.assistant_prefill,
    )

    train_internal_reasoning_single_step_sft(
        net=net,
        opt=opt,
        env=env,
        state_to_str=state_to_str,
        tokenizer=tokenizer,
        pad_token_id=model_info.pad_token_id,
        eos_token_id=model_info.eos_token_id,
        soft_block_size=soft_params.soft_block_size,
        soft_bptt_window=soft_params.soft_bptt_window,
        max_answer_tokens=sft_params.max_answer_tokens,
        max_episodes=train_params.max_episodes,
        batch_size=train_params.batch_size,
        accumulation_steps=train_params.accumulation_steps,
        max_grad_norm=train_params.max_grad_norm,
        normalize_by_sequence_length=sft_params.normalize_by_sequence_length,
        use_bf16=train_params.use_bf_16,
        save_ckpt_freq=train_params.save_ckpt_freq,
        val_envs=val_envs,
        val_batch_size=train_params.val_batch_size,
        val_episodes=train_params.val_episodes,
        val_freq=train_params.val_freq,
    )


@extty.experiment(project="sft-math")
def train_sft_math(
    *,
    train_params: TrainParams,
    soft_params: SoftParams,
    sft_params: SingleStepSFTParams,
    math_config: MathDatasetConfig,
    prompt_collection: PromptCollection,
):
    env = MathEnv(config=math_config)
    val_envs = [MathEnv(config=math_config, seed=2026)]
    return _train_internal_reasoning_single_step_sft(
        train_params=train_params,
        soft_params=soft_params,
        sft_params=sft_params,
        env=env,
        prompt_collection=prompt_collection,
        val_envs=val_envs,
    )


def _train_soft_grpo(
    *,
    train_params: TrainParams,
    grpo_params: GRPOParams,
    soft_grpo_params: SoftGRPOParams,
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

    train_soft_grpo(
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
        val_batch_size=train_params.val_batch_size,
        val_episodes=train_params.val_episodes,
        val_freq=train_params.val_freq,
        val_envs=val_envs,
        noise_std=soft_grpo_params.noise_std,
        normalize_soft_pdf_by_dim=soft_grpo_params.normalize_soft_pdf_by_dim,
    )


@extty.experiment(project="soft-grpo-countdown")
def train_soft_grpo_countdown(
    *,
    train_params: TrainParams,
    grpo_params: GRPOParams,
    soft_grpo_params: SoftGRPOParams,
    reward_params: RewardParams,
    prompt_collection: PromptCollection,
    countdown_params: CountdownParams,
):
    env, reward_fn, extractor, val_envs = (
        _get_countdown_env_reward_fn_extractor_val_envs(
            train_params=train_params,
            reward_params=reward_params,
            prompt_collection=prompt_collection,
            countdown_params=countdown_params,
        )
    )
    return _train_soft_grpo(
        train_params=train_params,
        grpo_params=grpo_params,
        soft_grpo_params=soft_grpo_params,
        env=env,
        prompt_collection=prompt_collection,
        reward_fn=reward_fn,
        extractor=extractor,
        val_envs=val_envs,
    )


def _train_internal_reasoning_sft(
    train_params: TrainParams,
    env: Env,
    soft_params: SoftParams,
    multistep_sft_params: MultiStepSFTParams,
    prompt_collection: PromptCollection,
    val_reward_fn: RewardFn,
    val_envs: list[Env],
):
    model_info = MODEL_REGISTRY[train_params.model_name]
    net, opt = load_model_and_opt(train_params=train_params)
    tokenizer = model_info.load_tokenizer()

    state_to_str = get_state_to_str(
        format_messages=model_info.format_messages,
        system_prompt=prompt_collection.system_prompt,
        assistant_prefill=prompt_collection.assistant_prefill,
    )

    return train_internal_reasoning_sft(
        net=net,
        opt=opt,
        env=env,
        state_to_str=state_to_str,
        tokenizer=tokenizer,
        pad_token_id=model_info.pad_token_id,
        eos_token_id=model_info.eos_token_id,
        move_name_to_id=multistep_sft_params.move_name_to_id,
        valid_hard_token_ids=multistep_sft_params.valid_hard_token_ids,
        soft_block_size=soft_params.soft_block_size,
        soft_bptt_window=soft_params.soft_bptt_window,
        max_cycles=soft_params.max_cycles,
        max_episodes=train_params.max_episodes,
        batch_size=train_params.batch_size,
        accumulation_steps=train_params.accumulation_steps,
        max_grad_norm=train_params.max_grad_norm,
        normalize_by_sequence_length=multistep_sft_params.normalize_by_sequence_length,
        use_bf16=train_params.use_bf_16,
        save_ckpt_freq=train_params.save_ckpt_freq,
        val_reward_fn=val_reward_fn,
        val_episodes=train_params.val_episodes,
        val_batch_size=train_params.val_batch_size,
        val_freq=train_params.val_freq,
        val_envs=val_envs,
        think_token_id=multistep_sft_params.think_token_id,
    )


# multistep
@extty.experiment(project="sft-maze")
def train_sft_maze(
    train_params: TrainParams,
    multistep_sft_params: MultiStepSFTParams,
    soft_params: SoftParams,
    reward_params: RewardParams,
    maze_reward_params: MazeRewardParams,
    maze_config: MazeConfig,  # TODO: move this with the other params
):
    prompt_collection = MAZE_INTERNAL_REASONING_PROMPT
    env = MazeEnv(
        config=maze_config,
        prompt_template=prompt_collection.env_prompt,
        seed=train_params.seed,
    )
    reward_fn = get_maze_reward_fn(
        answer_tags_weight=reward_params.answer_tags_weight,
        validity_weight=maze_reward_params.validity_weight,
        distance_weight=maze_reward_params.distance_weight,
        think_tags_weight=reward_params.think_tags_weight,
    )

    val_envs = [
        MazeEnv(
            config=maze_config,
            prompt_template=MAZE_INTERNAL_REASONING_PROMPT.env_prompt,
            seed=2026,
        )
    ]

    return _train_internal_reasoning_sft(
        train_params=train_params,
        env=env,
        soft_params=soft_params,
        multistep_sft_params=multistep_sft_params,
        prompt_collection=prompt_collection,
        val_reward_fn=reward_fn,
        val_envs=val_envs,
    )


# TODO: maze GRPO
