from functools import partial
from typing import Callable

import extty
import torch

from dialectic.experiments.arg_parser import Experiment, run_experiments_parser
from dialectic.experiments.envs import (
    get_maze_env_reward_fn_extractor_val_envs,
    get_state_to_str,
    load_countdown_dataset_artifacts,
)
from dialectic.experiments.models import load_model_and_opt
from dialectic.experiments.params import (
    GRPOParams,
    MazeRewardParams,
    RewardParams,
    TrainParams,
)
from dialectic.experiments.prompts import PromptCollection
from dialectic.experiments.reward_fns import get_countdown_reward_fn
from dialectic.llm.registry import MODEL_REGISTRY
from dialectic.rl.dataset_env import DatasetEnv
from dialectic.rl.env import Env
from dialectic.rl.extractors import extract_from_answer_tags
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
        warmup_steps=train_params.warmup_steps,
    )


@extty.experiment(project="grpo-countdown")
def train_grpo_countdown(
    *,
    train_params: TrainParams,
    grpo_params: GRPOParams,
    reward_params: RewardParams,
    prompt_collection: PromptCollection,
    dataset_artifacts: list[str],
):
    all_problems = load_countdown_dataset_artifacts(
        dataset_artifacts, prompt_template=prompt_collection.env_prompt
    )
    train_problems = [
        resp for resp, extra in all_problems if extra.get("split") == "train"
    ]
    val_problems = [resp for resp, extra in all_problems if extra.get("split") == "val"]

    if not train_problems:
        raise ValueError("No training problems found (split='train')")
    print(f"Train: {len(train_problems)}, Val: {len(val_problems)}")

    env: Env = DatasetEnv(
        train_problems, seed=train_params.seed, label="countdown_train"
    )
    val_envs: list[Env] = (
        [DatasetEnv(val_problems, seed=2026, label="countdown_val")]
        if val_problems
        else []
    )

    reward_fn = get_countdown_reward_fn(
        answer_tags_weight=reward_params.answer_tags_weight,
        think_tags_weight=reward_params.think_tags_weight,
    )

    return _train_grpo(
        train_params=train_params,
        grpo_params=grpo_params,
        env=env,
        prompt_collection=prompt_collection,
        reward_fn=reward_fn,
        extractor=extract_from_answer_tags,
        val_envs=val_envs,
    )


@extty.experiment(project="grpo-maze")
def train_grpo_maze(
    train_params: TrainParams,
    grpo_params: GRPOParams,
    reward_params: RewardParams,
    maze_reward_params: MazeRewardParams,
    maze_config: MazeConfig,
    prompt_collection: PromptCollection,
):
    env, reward_fn, extractor, val_envs = get_maze_env_reward_fn_extractor_val_envs(
        train_params=train_params,
        reward_params=reward_params,
        maze_reward_params=maze_reward_params,
        maze_config=maze_config,
        prompt_collection=prompt_collection,
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


if __name__ == "__main__":
    run_experiments_parser(
        [
            Experiment(
                env_name="maze",
                fn=train_grpo_maze,
                include_prompt_collection_id=True,
            ),
            Experiment(
                env_name="countdown",
                fn=train_grpo_countdown,
                include_prompt_collection_id=True,
                include_dataset_glob=True,
            ),
        ]
    )
