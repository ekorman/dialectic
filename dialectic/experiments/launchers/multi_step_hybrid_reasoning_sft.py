import extty
import torch
from tokenizers import Tokenizer

from dialectic.experiments.arg_parser import Experiment, run_experiments_parser
from dialectic.experiments.envs import get_state_to_str
from dialectic.experiments.models import load_model_and_opt
from dialectic.experiments.params import (
    HybridReasoningParams,
    MazeRewardParams,
    MultiStepSFTParams,
    RewardParams,
    TrainParams,
)
from dialectic.experiments.prompts import (
    MAZE_INTERNAL_REASONING_PROMPT,
    PromptCollection,
)
from dialectic.experiments.reward_fns import get_maze_reward_fn
from dialectic.llm.registry import MODEL_REGISTRY, ModelInfo
from dialectic.rl.env import Env, MazeEnv
from dialectic.rl.extractors import extract_maze_moves
from dialectic.rl.maze import MazeConfig
from dialectic.rl.reward import RewardFn
from dialectic.rl.train import train_internal_reasoning_sft


def _get_valid_hard_token_ids_and_move_name_to_id(
    model_info: ModelInfo, tokenizer: Tokenizer
):
    valid_hard_token_ids: list[int] = []
    move_name_to_id: dict[str, int] = {}

    direction_words = ["up", "down", "left", "right"]
    for word in direction_words:
        ids = tokenizer.encode(word, add_special_tokens=False).ids
        assert len(ids) == 1, f"'{word}' tokenizes to {len(ids)} tokens, expected 1"
        tid = ids[0]
        valid_hard_token_ids.append(tid)
        move_name_to_id[word] = tid
    valid_hard_token_ids.append(model_info.eos_token_id)

    return valid_hard_token_ids, move_name_to_id


def _get_maze_env_reward_fn_extractor_val_envs(
    train_params: TrainParams,
    reward_params: RewardParams,
    maze_reward_params: MazeRewardParams,
    maze_config: MazeConfig,
    prompt_collection: PromptCollection,
):
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

    return env, reward_fn, extract_maze_moves, val_envs


def _train_internal_reasoning_multi_step_sft(
    train_params: TrainParams,
    env: Env,
    multistep_sft_params: MultiStepSFTParams,
    hr_params: HybridReasoningParams,
    prompt_collection: PromptCollection,
    val_reward_fn: RewardFn,
    val_envs: list[Env],
):
    model_info = MODEL_REGISTRY[train_params.model_name]
    net, opt = load_model_and_opt(
        train_params=train_params,
        soft_projection=hr_params.soft_projection,
        soft_projection_alpha_init=hr_params.soft_projection_alpha_init,
        soft_projection_rank=hr_params.soft_projection_rank,
    )
    tokenizer = model_info.load_tokenizer()

    valid_hard_token_ids, move_name_to_id = (
        _get_valid_hard_token_ids_and_move_name_to_id(
            model_info=model_info, tokenizer=tokenizer
        )
    )

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
        move_name_to_id=move_name_to_id,
        valid_hard_token_ids=valid_hard_token_ids,
        soft_block_size=hr_params.soft_block_size,
        soft_bptt_window=hr_params.soft_bptt_window,
        max_cycles=hr_params.max_cycles,
        max_episodes=train_params.max_episodes,
        batch_size=train_params.batch_size,
        accumulation_steps=train_params.accumulation_steps,
        max_grad_norm=train_params.max_grad_norm,
        normalize_by_sequence_length=multistep_sft_params.normalize_by_sequence_length,
        use_bf16=train_params.use_bf16,
        save_ckpt_freq=train_params.save_ckpt_freq,
        val_reward_fn=val_reward_fn,
        val_episodes=train_params.val_episodes,
        val_batch_size=train_params.val_batch_size,
        val_freq=train_params.val_freq,
        val_envs=val_envs,
        think_token_id=multistep_sft_params.think_token_id,
    )


@extty.experiment(project="sft-hybrid-reasoning-maze")
def train_hybrid_reasoning_sft_maze(
    train_params: TrainParams,
    multistep_sft_params: MultiStepSFTParams,
    hr_params: HybridReasoningParams,
    reward_params: RewardParams,
    maze_reward_params: MazeRewardParams,
    maze_config: MazeConfig,  # TODO: move this with the other params
):
    torch.manual_seed(train_params.seed)

    prompt_collection = MAZE_INTERNAL_REASONING_PROMPT

    env, reward_fn, _, val_envs = _get_maze_env_reward_fn_extractor_val_envs(
        train_params=train_params,
        reward_params=reward_params,
        maze_reward_params=maze_reward_params,
        maze_config=maze_config,
        prompt_collection=prompt_collection,
    )

    return _train_internal_reasoning_multi_step_sft(
        train_params=train_params,
        env=env,
        hr_params=hr_params,
        multistep_sft_params=multistep_sft_params,
        prompt_collection=prompt_collection,
        val_reward_fn=reward_fn,
        val_envs=val_envs,
    )


if __name__ == "__main__":
    run_experiments_parser(
        [
            Experiment(
                env_name="maze",
                fn=train_hybrid_reasoning_sft_maze,
                include_prompt_collection_id=False,
            ),
        ]
    )
