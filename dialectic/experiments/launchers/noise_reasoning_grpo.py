from functools import partial

import extty
import torch

from dialectic.experiments.arg_parser import Experiment, run_experiments_parser
from dialectic.experiments.envs import get_state_to_str
from dialectic.experiments.models import load_model_and_opt
from dialectic.experiments.params import (
    CountdownParams,
    GRPOParams,
    NoiseReasoningParams,
    RewardParams,
    TrainParams,
)
from dialectic.experiments.prompts import (
    ENV_PROMPT_COUNTDOWN_MINIMAL,
    ENV_PROMPT_WITH_SCRATCH_TAGS,
    PromptCollection,
)
from dialectic.llm.registry import MODEL_REGISTRY
from dialectic.rl.env import CountdownEnv
from dialectic.rl.extractors import extract_countdown_answer
from dialectic.rl.grammar import (
    build_countdown_cycle_grammar_factory,
    build_countdown_simple_cycle_grammar_factory,
)
from dialectic.rl.reward import (
    answer_tags,
    countdown_correct,
    scratch_tags,
    weighted_reward,
)
from dialectic.rl.train import (
    grpo_advantage,
    rloo_advantage,
    train_noise_reasoning_grpo,
)

PROMPT_STYLES = {
    "scratch_tags": PromptCollection(
        system_prompt=None,
        env_prompt=ENV_PROMPT_WITH_SCRATCH_TAGS,
        assistant_prefill=None,
    ),
    "minimal": PromptCollection(
        system_prompt=None,
        env_prompt=ENV_PROMPT_COUNTDOWN_MINIMAL,
        assistant_prefill=None,
    ),
}


@extty.experiment(project="noise-reasoning-grpo-countdown")
def train_noise_reasoning_grpo_countdown(
    train_params: TrainParams,
    grpo_params: GRPOParams,
    reward_params: RewardParams,
    noise_reasoning_params: NoiseReasoningParams,
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

    adapter = None
    registers = None
    extra_params: list[torch.nn.Parameter] = []

    if noise_reasoning_params.use_noise_adapter:
        from dialectic.llm.noise_adapter import NoiseAdapter

        adapter = NoiseAdapter(
            d_model=net.d,
            n_heads=noise_reasoning_params.adapter_n_heads,
            d_ff=noise_reasoning_params.adapter_d_ff,
        ).to(next(net.parameters()).device)
        extra_params.extend(adapter.parameters())

    if (
        noise_reasoning_params.use_registers
        and noise_reasoning_params.n_noise_per_cycle > 0
    ):
        registers = torch.nn.Parameter(
            torch.randn(
                noise_reasoning_params.n_noise_per_cycle,
                net.d,
                device=next(net.parameters()).device,
            )
            * 0.02
        )
        extra_params.append(registers)

    if extra_params or noise_reasoning_params.freeze_base_model:
        if noise_reasoning_params.freeze_base_model:
            net.requires_grad_(False)
            opt = torch.optim.AdamW(extra_params, lr=train_params.lr)
        else:
            opt = torch.optim.AdamW(
                list(net.parameters()) + extra_params,
                lr=train_params.lr,
            )

    prompt_style = noise_reasoning_params.prompt_style
    if prompt_style == "minimal":
        grammar_factory = build_countdown_simple_cycle_grammar_factory(tokenizer)
    else:
        grammar_factory = build_countdown_cycle_grammar_factory(tokenizer)

    prompt_collection = PROMPT_STYLES[prompt_style]
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

    train_noise_reasoning_grpo(
        net=net,
        opt=opt,
        env=env,
        reward_fn=reward_fn,
        state_to_str=state_to_str,
        tokenizer=tokenizer,
        extractor=extract_countdown_answer,
        eos_token_id=model_info.eos_token_id,
        pad_token_id=model_info.pad_token_id,
        grammar_factory=grammar_factory,
        n_noise_per_cycle=noise_reasoning_params.n_noise_per_cycle,
        noise_std=noise_reasoning_params.noise_std,
        temperature=train_params.temperature,
        max_cycles=noise_reasoning_params.max_cycles,
        min_cycles=noise_reasoning_params.min_cycles,
        max_tokens_per_cycle=noise_reasoning_params.max_tokens_per_cycle,
        evict_noise_kv=noise_reasoning_params.evict_noise_kv,
        noise_adapter=adapter,
        registers=registers,
        beta=grpo_params.beta,
        eps=grpo_params.eps,
        mu=grpo_params.mu,
        max_episodes=train_params.max_episodes,
        update_ref_net_batch_cadence=grpo_params.update_ref_net_batch_cadence,
        batch_size=train_params.batch_size,
        group_size=grpo_params.group_size,
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
                fn=train_noise_reasoning_grpo_countdown,
                include_prompt_collection_id=False,
            ),
        ]
    )
