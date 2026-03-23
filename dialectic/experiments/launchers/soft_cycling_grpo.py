from functools import partial

import extty
import torch

from dialectic.experiments.arg_parser import Experiment, run_experiments_parser
from dialectic.experiments.envs import get_state_to_str
from dialectic.experiments.models import load_model_and_opt
from dialectic.experiments.params import (
    CountdownParams,
    GRPOParams,
    RewardParams,
    SoftCyclingParams,
    TrainParams,
)
from dialectic.experiments.prompts import ENV_PROMPT_WITH_SCRATCH_TAGS, PromptCollection
from dialectic.llm.registry import MODEL_REGISTRY
from dialectic.rl.env import CountdownEnv
from dialectic.rl.extractors import extract_from_answer_tags
from dialectic.rl.grammar import build_countdown_grammar_specs
from dialectic.rl.reward import (
    answer_tags,
    countdown_correct,
    scratch_tags,
    weighted_reward,
)
from dialectic.rl.train import grpo_advantage, rloo_advantage, train_soft_cycling_grpo

COUNTDOWN_SOFT_CYCLING_PROMPT = PromptCollection(
    system_prompt=None,
    env_prompt=ENV_PROMPT_WITH_SCRATCH_TAGS,
    assistant_prefill=None,
)


@extty.experiment(project="soft-cycling-grpo-countdown")
def train_soft_cycling_grpo_countdown(
    train_params: TrainParams,
    grpo_params: GRPOParams,
    reward_params: RewardParams,
    soft_cycling_params: SoftCyclingParams,
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

    model_info = MODEL_REGISTRY[train_params.model_name]
    net, opt = load_model_and_opt(train_params=train_params)
    tokenizer = model_info.load_tokenizer()

    prompt_collection = COUNTDOWN_SOFT_CYCLING_PROMPT
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

    reward_components = [("correct", 1.0, countdown_correct)]
    if reward_params.answer_tags_weight > 0:
        reward_components.append(
            ("answer_tags", reward_params.answer_tags_weight, answer_tags)
        )
    if reward_params.scratch_tags_weight > 0:
        reward_components.append(
            ("scratch_tags", reward_params.scratch_tags_weight, scratch_tags)
        )
    reward_fn = weighted_reward(reward_components)

    grammar_specs = build_countdown_grammar_specs(tokenizer)

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

    train_soft_cycling_grpo(
        net=net,
        opt=opt,
        env=env,
        reward_fn=reward_fn,
        state_to_str=state_to_str,
        tokenizer=tokenizer,
        extractor=extract_from_answer_tags,
        eos_token_id=model_info.eos_token_id,
        pad_token_id=model_info.pad_token_id,
        grammar_specs=grammar_specs,
        temperature=train_params.temperature,
        max_tokens_generated=train_params.max_tokens_generated,
        max_cycles=soft_cycling_params.max_cycles,
        max_tokens_per_cycle=soft_cycling_params.max_tokens_per_cycle,
        beta=grpo_params.beta,
        eps=grpo_params.eps,
        mu=grpo_params.mu,
        max_episodes=train_params.max_episodes,
        update_ref_net_batch_cadence=grpo_params.update_ref_net_batch_cadence,
        batch_size=train_params.batch_size,
        group_size=grpo_params.group_size,
        use_gumbel=soft_cycling_params.use_gumbel,
        soft_token_noise_std=None,
        min_soft_steps=soft_cycling_params.min_soft_steps,
        max_soft_steps_per_cycle=soft_cycling_params.max_soft_steps_per_cycle,
        soft_bptt_window=soft_cycling_params.soft_bptt_window,
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


if __name__ == "__main__":
    run_experiments_parser(
        [
            Experiment(
                env_name="countdown",
                fn=train_soft_cycling_grpo_countdown,
                include_prompt_collection_id=False,
            ),
        ]
    )
