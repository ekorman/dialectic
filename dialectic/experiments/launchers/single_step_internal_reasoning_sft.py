from typing import Sequence

import extty

from dialectic.experiments.arg_parser import Experiment, run_experiments_parser
from dialectic.experiments.envs import get_state_to_str
from dialectic.experiments.models import load_model_and_opt
from dialectic.experiments.params import (
    InternalReasoningParams,
    MathEnvParams,
    SingleStepSFTParams,
    TrainParams,
)
from dialectic.experiments.prompts import PromptCollection
from dialectic.llm.registry import MODEL_REGISTRY
from dialectic.rl.env import Env, MathEnv
from dialectic.rl.train import train_internal_reasoning_single_step_sft


def _train_internal_reasoning_single_step_sft(
    train_params: TrainParams,
    sft_params: SingleStepSFTParams,
    ir_params: InternalReasoningParams,
    env: Env,
    prompt_collection: PromptCollection,
    val_envs: Sequence[Env],
):
    model_info = MODEL_REGISTRY[train_params.model_name]
    net, opt = load_model_and_opt(
        train_params=train_params,
        soft_projection=ir_params.soft_projection,
        soft_projection_alpha_init=ir_params.soft_projection_alpha_init,
        soft_projection_rank=ir_params.soft_projection_rank,
    )
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
        soft_block_size=ir_params.soft_block_size,
        soft_bptt_window=ir_params.soft_bptt_window,
        max_answer_tokens=sft_params.max_answer_tokens,
        max_episodes=train_params.max_episodes,
        batch_size=train_params.batch_size,
        accumulation_steps=train_params.accumulation_steps,
        max_grad_norm=train_params.max_grad_norm,
        normalize_by_sequence_length=sft_params.normalize_by_sequence_length,
        use_bf16=train_params.use_bf16,
        save_ckpt_freq=train_params.save_ckpt_freq,
        val_envs=val_envs,
        val_batch_size=train_params.val_batch_size,
        val_episodes=train_params.val_episodes,
        val_freq=train_params.val_freq,
    )


@extty.experiment(project="sft-internal-reasoning-math")
def train_internal_reasoning_sft_math(
    *,
    train_params: TrainParams,
    sft_params: SingleStepSFTParams,
    ir_params: InternalReasoningParams,
    math_env_params: MathEnvParams,
    prompt_collection: PromptCollection,
):
    env = MathEnv(
        direct_arithmetic_prob=math_env_params.direct_arithmetic_prob,
        twostep_arithmetic_prob=math_env_params.twostep_arithmetic_prob,
        word_problem_prob=math_env_params.word_problem_prob,
        number_properties_prob=math_env_params.number_properties_prob,
        difficulty=math_env_params.difficulty,
        seed=train_params.seed,
    )
    val_envs = [
        MathEnv(
            direct_arithmetic_prob=math_env_params.direct_arithmetic_prob,
            twostep_arithmetic_prob=math_env_params.twostep_arithmetic_prob,
            word_problem_prob=math_env_params.word_problem_prob,
            number_properties_prob=math_env_params.number_properties_prob,
            difficulty=math_env_params.difficulty,
            seed=2026,
        )
    ]

    return _train_internal_reasoning_single_step_sft(
        train_params=train_params,
        sft_params=sft_params,
        ir_params=ir_params,
        env=env,
        prompt_collection=prompt_collection,
        val_envs=val_envs,
    )


if __name__ == "__main__":
    run_experiments_parser(
        [
            Experiment(
                env_name="math",
                fn=train_internal_reasoning_sft_math,
                include_prompt_collection_id=True,
            ),
        ]
    )
