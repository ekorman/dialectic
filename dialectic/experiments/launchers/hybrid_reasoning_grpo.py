from functools import partial

import extty
import torch
from tokenizers import Tokenizer

from dialectic.experiments.arg_parser import Experiment, run_experiments_parser
from dialectic.experiments.envs import (
    get_maze_env_reward_fn_extractor_val_envs,
    get_state_to_str,
)
from dialectic.experiments.launchers import extract_separator_token_id
from dialectic.experiments.models import load_model_and_opt
from dialectic.experiments.params import (
    CountdownParams,
    GRPOParams,
    HybridReasoningParams,
    MazeRewardParams,
    RewardParams,
    TrainParams,
)
from dialectic.experiments.prompts import (
    COUNTDOWN_INTERNAL_REASONING_PROMPT,
    MAZE_INTERNAL_REASONING_PROMPT,
    PromptCollection,
)
from dialectic.llm.registry import MODEL_REGISTRY, ModelInfo
from dialectic.rl.env import CountdownEnv, Env
from dialectic.rl.maze import MazeConfig
from dialectic.rl.reward import RewardFn, countdown_hybrid_correct, weighted_reward
from dialectic.rl.train import (
    grpo_advantage,
    rloo_advantage,
    train_internal_reasoning_grpo,
    train_variable_length_internal_reasoning_grpo,
)


def _get_valid_hard_token_ids_and_move_id_to_name(
    model_info: ModelInfo, tokenizer: Tokenizer
):
    valid_hard_token_ids: list[int] = []
    move_id_to_name: dict[int, str] = {}

    direction_words = ["up", "down", "left", "right"]
    for word in direction_words:
        ids = tokenizer.encode(word, add_special_tokens=False).ids
        assert len(ids) == 1, f"'{word}' tokenizes to {len(ids)} tokens, expected 1"
        tid = ids[0]
        valid_hard_token_ids.append(tid)
        move_id_to_name[tid] = word
    valid_hard_token_ids.append(model_info.eos_token_id)

    return valid_hard_token_ids, move_id_to_name


def _train_hybrid_reasoning_grpo(
    *,
    train_params: TrainParams,
    grpo_params: GRPOParams,
    env: Env,
    prompt_collection: PromptCollection,
    hybrid_reasoning_params: HybridReasoningParams,
    reward_fn: RewardFn,
    val_envs: list[Env],
):
    torch.manual_seed(train_params.seed)

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
    net, opt = load_model_and_opt(
        train_params=train_params,
        soft_projection=hybrid_reasoning_params.soft_projection,
        soft_projection_alpha_init=hybrid_reasoning_params.soft_projection_alpha_init,
    )
    format_messages = model_info.format_messages

    state_to_str = get_state_to_str(
        format_messages=format_messages,
        system_prompt=prompt_collection.system_prompt,
        assistant_prefill=prompt_collection.assistant_prefill,
    )
    tokenizer = model_info.load_tokenizer()

    valid_hard_token_ids, move_id_to_name = (
        _get_valid_hard_token_ids_and_move_id_to_name(
            model_info=model_info, tokenizer=tokenizer
        )
    )

    train_internal_reasoning_grpo(
        net=net,
        opt=opt,
        env=env,
        reward_fn=reward_fn,
        state_to_str=state_to_str,
        tokenizer=tokenizer,
        eos_token_id=model_info.eos_token_id,
        pad_token_id=model_info.pad_token_id,
        extractor=lambda moves: moves if moves else None,
        beta=grpo_params.beta,
        eps=grpo_params.eps,
        mu=grpo_params.mu,
        move_id_to_name=move_id_to_name,
        valid_hard_token_ids=valid_hard_token_ids,
        max_episodes=train_params.max_episodes,
        update_ref_net_batch_cadence=grpo_params.update_ref_net_batch_cadence,
        batch_size=train_params.batch_size,
        group_size=grpo_params.group_size,
        temperature=train_params.temperature,
        soft_block_size=hybrid_reasoning_params.soft_block_size,
        soft_bptt_window=hybrid_reasoning_params.soft_bptt_window,
        max_cycles=hybrid_reasoning_params.max_cycles,
        think_token_id=hybrid_reasoning_params.think_token_id,
        advantage_fn=advantage_fn,
        normalize_by_sequence_length=grpo_params.normalize_by_sequence_length,
        accumulation_steps=train_params.accumulation_steps,
        max_grad_norm=train_params.max_grad_norm,
        use_bf16=train_params.use_bf16,
        save_ckpt_freq=train_params.save_ckpt_freq,
        val_batch_size=train_params.val_batch_size,
        val_episodes=train_params.val_episodes,
        val_freq=train_params.val_freq,
        val_envs=val_envs,
        warmup_steps=train_params.warmup_steps,
    )


@extty.experiment(project="hybrid-reasoning-grpo-maze")
def train_hybrid_reasoning_grpo_maze(
    train_params: TrainParams,
    grpo_params: GRPOParams,
    reward_params: RewardParams,
    hybrid_reasoning_params: HybridReasoningParams,
    maze_reward_params: MazeRewardParams,
    maze_config: MazeConfig,
):
    prompt_collection = MAZE_INTERNAL_REASONING_PROMPT

    env, reward_fn, _, val_envs = get_maze_env_reward_fn_extractor_val_envs(
        train_params=train_params,
        reward_params=reward_params,
        maze_reward_params=maze_reward_params,
        maze_config=maze_config,
        prompt_collection=prompt_collection,
    )

    return _train_hybrid_reasoning_grpo(
        train_params=train_params,
        grpo_params=grpo_params,
        env=env,
        prompt_collection=prompt_collection,
        reward_fn=reward_fn,
        val_envs=val_envs,
        hybrid_reasoning_params=hybrid_reasoning_params,
    )


@extty.experiment(project="hybrid-reasoning-grpo-countdown")
def train_hybrid_reasoning_grpo_countdown(
    train_params: TrainParams,
    grpo_params: GRPOParams,
    hybrid_reasoning_params: HybridReasoningParams,
    countdown_params: CountdownParams,
):
    torch.manual_seed(train_params.seed)

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

    n_ops = countdown_params.n_ops
    n_larges = countdown_params.n_larges
    n_total = countdown_params.n_total

    n_ops_list = [n_ops] if isinstance(n_ops, int) else n_ops
    n_total_list = [n_total] if isinstance(n_total, int) else n_total
    n_larges_list = [n_larges] if isinstance(n_larges, int) else n_larges
    lengths = {len(n_ops_list), len(n_total_list), len(n_larges_list)}
    if len(lengths) != 1:
        raise ValueError(
            f"n_ops, n_total, and n_larges must have the same length when lists, "
            f"got {len(n_ops_list)}, {len(n_total_list)}, {len(n_larges_list)}"
        )

    model_info = MODEL_REGISTRY[train_params.model_name]
    net, opt = load_model_and_opt(
        train_params=train_params,
        soft_projection=hybrid_reasoning_params.soft_projection,
        soft_projection_alpha_init=hybrid_reasoning_params.soft_projection_alpha_init,
        soft_projection_rank=hybrid_reasoning_params.soft_projection_rank,
    )
    tokenizer = model_info.load_tokenizer()

    separator_token_id = extract_separator_token_id(tokenizer)

    prompt_collection = COUNTDOWN_INTERNAL_REASONING_PROMPT

    state_to_str = get_state_to_str(
        format_messages=model_info.format_messages,
        system_prompt=prompt_collection.system_prompt,
        assistant_prefill=prompt_collection.assistant_prefill,
    )

    env = CountdownEnv(
        seed=train_params.seed,
        n_larges=n_larges,
        n_total=n_total,
        n_ops=n_ops,
        prompt_template=prompt_collection.env_prompt,
    )

    reward_fn = weighted_reward([("correct", 1.0, countdown_hybrid_correct)])

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

    train_variable_length_internal_reasoning_grpo(
        net=net,
        opt=opt,
        env=env,
        reward_fn=reward_fn,
        state_to_str=state_to_str,
        tokenizer=tokenizer,
        eos_token_id=model_info.eos_token_id,
        pad_token_id=model_info.pad_token_id,
        separator_token_id=separator_token_id,
        beta=grpo_params.beta,
        eps=grpo_params.eps,
        mu=grpo_params.mu,
        max_episodes=train_params.max_episodes,
        update_ref_net_batch_cadence=grpo_params.update_ref_net_batch_cadence,
        batch_size=train_params.batch_size,
        group_size=grpo_params.group_size,
        temperature=train_params.temperature,
        soft_block_size=hybrid_reasoning_params.soft_block_size,
        soft_bptt_window=hybrid_reasoning_params.soft_bptt_window,
        max_cycles=hybrid_reasoning_params.max_cycles,
        max_tokens_per_cycle=hybrid_reasoning_params.max_tokens_per_cycle,
        think_token_id=hybrid_reasoning_params.think_token_id,
        advantage_fn=advantage_fn,
        normalize_by_sequence_length=grpo_params.normalize_by_sequence_length,
        accumulation_steps=train_params.accumulation_steps,
        max_grad_norm=train_params.max_grad_norm,
        use_bf16=train_params.use_bf16,
        save_ckpt_freq=train_params.save_ckpt_freq,
        val_batch_size=train_params.val_batch_size,
        val_episodes=train_params.val_episodes,
        val_freq=train_params.val_freq,
        val_envs=val_envs,
        pass_at_k_samples=hybrid_reasoning_params.pass_at_k_samples,
        pass_at_k_temperature=hybrid_reasoning_params.pass_at_k_temperature,
        warmup_steps=train_params.warmup_steps,
    )


if __name__ == "__main__":
    run_experiments_parser(
        [
            Experiment(
                env_name="maze",
                fn=train_hybrid_reasoning_grpo_maze,
                include_prompt_collection_id=False,
            ),
            Experiment(
                env_name="countdown",
                fn=train_hybrid_reasoning_grpo_countdown,
                include_prompt_collection_id=False,
            ),
        ]
    )
