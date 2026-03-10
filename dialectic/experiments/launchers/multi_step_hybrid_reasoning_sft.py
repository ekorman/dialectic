import extty
import torch
from tokenizers import Tokenizer

from dialectic.experiments.arg_parser import Experiment, run_experiments_parser
from dialectic.experiments.envs import (
    get_maze_env_reward_fn_extractor_val_envs,
    get_state_to_str,
)
from dialectic.experiments.models import load_model_and_opt
from dialectic.experiments.params import (
    CountdownParams,
    HybridReasoningParams,
    MazeRewardParams,
    MultiStepSFTParams,
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
    train_internal_reasoning_sft,
    train_variable_length_internal_reasoning_sft,
)


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
        think_token_id=hr_params.think_token_id,
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

    env, reward_fn, _, val_envs = get_maze_env_reward_fn_extractor_val_envs(
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


SEPARATOR_STR = "|"


@extty.experiment(project="sft-hybrid-reasoning-countdown")
def train_hybrid_reasoning_sft_countdown(
    train_params: TrainParams,
    multistep_sft_params: MultiStepSFTParams,
    hr_params: HybridReasoningParams,
    countdown_params: CountdownParams,
):
    torch.manual_seed(train_params.seed)

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

    prompt_collection = COUNTDOWN_INTERNAL_REASONING_PROMPT

    model_info = MODEL_REGISTRY[train_params.model_name]
    net, opt = load_model_and_opt(
        train_params=train_params,
        soft_projection=hr_params.soft_projection,
        soft_projection_alpha_init=hr_params.soft_projection_alpha_init,
        soft_projection_rank=hr_params.soft_projection_rank,
    )
    tokenizer = model_info.load_tokenizer()

    separator_ids = tokenizer.encode(SEPARATOR_STR, add_special_tokens=False).ids
    assert len(separator_ids) == 1, (
        f"'|' tokenizes to {len(separator_ids)} tokens, expected 1"
    )
    separator_token_id = separator_ids[0]

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

    return train_variable_length_internal_reasoning_sft(
        net=net,
        opt=opt,
        env=env,
        state_to_str=state_to_str,
        tokenizer=tokenizer,
        pad_token_id=model_info.pad_token_id,
        eos_token_id=model_info.eos_token_id,
        separator_token_id=separator_token_id,
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
        val_reward_fn=reward_fn,
        val_episodes=train_params.val_episodes,
        val_batch_size=train_params.val_batch_size,
        val_freq=train_params.val_freq,
        val_envs=val_envs,
        think_token_id=hr_params.think_token_id,
    )


if __name__ == "__main__":
    run_experiments_parser(
        [
            Experiment(
                env_name="maze",
                fn=train_hybrid_reasoning_sft_maze,
                include_prompt_collection_id=False,
            ),
            Experiment(
                env_name="countdown",
                fn=train_hybrid_reasoning_sft_countdown,
                include_prompt_collection_id=False,
            ),
        ]
    )
