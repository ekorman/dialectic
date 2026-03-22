from dialectic.experiments.arg_parser import Experiment, run_experiments_parser
from dialectic.experiments.envs import (
    get_countdown_env_reward_fn_extractor_val_envs,
    get_state_to_str,
)
from dialectic.experiments.models import load_model
from dialectic.experiments.params import CountdownParams, RewardParams, TrainParams
from dialectic.experiments.prompts import PromptCollection
from dialectic.llm.registry import MODEL_REGISTRY
from dialectic.rl.evaluate import evaluate


def evaluate_grpo_countdown(
    *,
    train_params: TrainParams,
    reward_params: RewardParams,
    prompt_collection: PromptCollection,
    countdown_params: CountdownParams,
):
    _, reward_fn, extractor, val_envs = get_countdown_env_reward_fn_extractor_val_envs(
        train_params=train_params,
        reward_params=reward_params,
        prompt_collection=prompt_collection,
        countdown_params=countdown_params,
    )

    model_info = MODEL_REGISTRY[train_params.model_name]
    net = load_model(train_params=train_params)
    format_messages = model_info.format_messages
    state_to_str = get_state_to_str(
        format_messages=format_messages,
        system_prompt=prompt_collection.system_prompt,
        assistant_prefill=prompt_collection.assistant_prefill,
    )
    tokenizer = model_info.load_tokenizer()

    for env in val_envs:
        eval_result = evaluate(
            net=net,
            env=env,
            reward_fn=reward_fn,
            state_to_str=state_to_str,
            tokenizer=tokenizer,
            eos_token_id=model_info.eos_token_id,
            pad_token_id=model_info.pad_token_id,
            extractor=extractor,
            max_tokens_generated=train_params.max_tokens_generated,
            max_episodes=train_params.val_episodes,
            temperature=train_params.temperature,
        )
        print(eval_result[0])


if __name__ == "__main__":
    run_experiments_parser(
        [
            Experiment(
                env_name="countdown",
                fn=evaluate_grpo_countdown,
                include_prompt_collection_id=True,
            )
        ]
    )
