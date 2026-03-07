import argparse
from functools import partial
from typing import Callable

import extty

from dialectic.experiments.arg_parser import (
    create_subparser,
    load_dc_from_arg_parser_args,
)
from dialectic.experiments.envs import (
    get_countdown_env_reward_fn_extractor_val_envs,
    get_maze_env_reward_fn_extractor_val_envs,
    get_state_to_str,
)
from dialectic.experiments.models import load_model_and_opt
from dialectic.experiments.params import (
    CountdownParams,
    GRPOParams,
    MazeRewardParams,
    RewardParams,
    TrainParams,
)
from dialectic.experiments.prompts import (
    MAZE_INTERNAL_REASONING_PROMPT,
    PROMPT_COLLECTIONS,
    PromptCollection,
)
from dialectic.llm.registry import MODEL_REGISTRY
from dialectic.rl.env import Env
from dialectic.rl.maze import MazeConfig
from dialectic.rl.reward import RewardFn
from dialectic.rl.train import grpo_advantage, rloo_advantage, train_grpo


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
        use_bf16=train_params.use_bf16,
        save_ckpt_freq=train_params.save_ckpt_freq,
        val_batch_size=train_params.val_batch_size,
        val_episodes=train_params.val_episodes,
        val_freq=train_params.val_freq,
        val_envs=val_envs,
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
    env, reward_fn, extractor, val_envs = (
        get_countdown_env_reward_fn_extractor_val_envs(
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


@extty.experiment(project="grpo-maze")
def train_grpo_maze(
    train_params: TrainParams,
    grpo_params: GRPOParams,
    reward_params: RewardParams,
    maze_reward_params: MazeRewardParams,
    maze_config: MazeConfig,
):
    env, reward_fn, extractor, val_envs = get_maze_env_reward_fn_extractor_val_envs(
        train_params=train_params,
        reward_params=reward_params,
        maze_reward_params=maze_reward_params,
        maze_config=maze_config,
        prompt_collection=MAZE_INTERNAL_REASONING_PROMPT,
    )

    return _train_grpo(
        train_params=train_params,
        grpo_params=grpo_params,
        env=env,
        prompt_collection=MAZE_INTERNAL_REASONING_PROMPT,
        reward_fn=reward_fn,
        extractor=extractor,
        val_envs=val_envs,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="env")

    # used for all environments
    shared_dcs = [TrainParams, GRPOParams, RewardParams]

    maze_parser = create_subparser(
        "maze", subparsers, shared_dcs + [MazeRewardParams, MazeConfig]
    )
    countdown_parser = create_subparser(
        "countdown", subparsers, shared_dcs + [CountdownParams]
    )

    args = parser.parse_args()

    train_params = load_dc_from_arg_parser_args(TrainParams, args)
    grpo_params = load_dc_from_arg_parser_args(GRPOParams, args)
    reward_params = load_dc_from_arg_parser_args(RewardParams, args)

    if args.env == "maze":
        maze_reward_params = load_dc_from_arg_parser_args(MazeRewardParams, args)
        maze_config = load_dc_from_arg_parser_args(MazeConfig, args)

        train_grpo_maze(
            train_params=train_params,
            grpo_params=grpo_params,
            reward_params=reward_params,
            maze_reward_params=maze_reward_params,
            maze_config=maze_config,
        )
    elif args.env == "countdown":
        prompt_collection = PROMPT_COLLECTIONS["countdown"][args.prompt_collection_id]

        countdown_params = load_dc_from_arg_parser_args(CountdownParams, args)

        train_grpo_countdown(
            train_params=train_params,
            grpo_params=grpo_params,
            reward_params=reward_params,
            countdown_params=countdown_params,
            prompt_collection=prompt_collection,
        )
    else:
        raise ValueError(f"Unexpected environment {args.env}")

# TODO: check against run
# grpo-countdown/2026-02-17_16-50-13_182b
# ../ex/tui/target/debug/extty run -- uv run train_scripts/grpo_countdown.py --seed 20 --prompt-collections-id 2 --no-qwen-thinking --max-episodes 10000 --max-tokens 300 --n-total 4,3 --n-larges 1,1 --n-ops 3,2 --temperature 1.0 --eps 0.1 --model qwen3-0.6b --think-tags-weight 0.0
