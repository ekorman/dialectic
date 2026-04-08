"""Evaluate a grpo-countdown checkpoint using the same eval as training."""

import argparse

import extty
import torch

from dialectic.experiments.envs import (
    get_countdown_env_reward_fn_extractor_val_envs,
    get_state_to_str,
)
from dialectic.experiments.params import CountdownParams, RewardParams, TrainParams
from dialectic.experiments.prompts import PromptCollection
from dialectic.llm.registry import MODEL_REGISTRY
from dialectic.llm.utils import get_default_device
from dialectic.log import log
from dialectic.rl.evaluate import EvaluationResult, evaluate


def main():
    parser = argparse.ArgumentParser(description="Evaluate a grpo-countdown checkpoint")
    parser.add_argument("run_name", help="Name of the grpo-countdown run")
    parser.add_argument("step", type=int, help="Checkpoint step number")
    parser.add_argument(
        "--val-episodes",
        type=int,
        default=None,
        help="Override number of val episodes (default: use run config)",
    )
    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=None,
        help="Override val batch size (default: use run config)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0 for greedy)",
    )
    parser.add_argument(
        "--no-examples",
        action="store_true",
        help="Suppress printing individual examples",
    )
    args = parser.parse_args()

    project = "grpo-countdown"
    run_data = extty.get_run(project, args.run_name)
    config = run_data.config

    train_cfg = config["train_params"]
    reward_cfg = config["reward_params"]
    countdown_cfg = config["countdown_params"]
    prompt_cfg = config["prompt_collection"]

    train_params = TrainParams(
        model_name=train_cfg["model_name"],
        batch_size=train_cfg["batch_size"],
        lr=train_cfg["lr"],
        accumulation_steps=train_cfg["accumulation_steps"],
        max_grad_norm=train_cfg["max_grad_norm"],
        max_episodes=train_cfg["max_episodes"],
        max_tokens_generated=train_cfg["max_tokens_generated"],
        compile_model=train_cfg["compile_model"],
        use_bf16=train_cfg["use_bf16"],
        seed=train_cfg["seed"],
        temperature=train_cfg["temperature"],
        val_batch_size=args.val_batch_size or train_cfg["val_batch_size"],
        val_episodes=args.val_episodes or train_cfg["val_episodes"],
        val_freq=train_cfg["val_freq"],
    )

    reward_params = RewardParams(
        answer_tags_weight=reward_cfg["answer_tags_weight"],
        think_tags_weight=reward_cfg["think_tags_weight"],
    )

    countdown_params = CountdownParams(
        n_larges=countdown_cfg["n_larges"],
        n_total=countdown_cfg["n_total"],
        n_ops=countdown_cfg["n_ops"],
    )

    prompt_collection = PromptCollection(
        system_prompt=prompt_cfg["system_prompt"],
        env_prompt=prompt_cfg["env_prompt"],
        assistant_prefill=prompt_cfg["assistant_prefill"],
    )

    _, reward_fn, extractor, val_envs = get_countdown_env_reward_fn_extractor_val_envs(
        train_params=train_params,
        reward_params=reward_params,
        prompt_collection=prompt_collection,
        countdown_params=countdown_params,
    )

    model_info = MODEL_REGISTRY[train_params.model_name]
    net = model_info.load_net(pretrained_weights=False)

    log.info(f"Loading checkpoint from {project}/{args.run_name}, step {args.step}")
    ckpt = extty.load_checkpoint_from(
        project=project,
        run_name=args.run_name,
        step=args.step,
        load_optimizer=False,
    )
    net.load_state_dict(ckpt["model_state_dict"])

    device = get_default_device()
    net = net.to(device)
    net.eval()
    log.info(f"Model loaded on {device}")

    tokenizer = model_info.load_tokenizer()
    state_to_str = get_state_to_str(
        format_messages=model_info.format_messages,
        system_prompt=prompt_collection.system_prompt,
        assistant_prefill=prompt_collection.assistant_prefill,
    )

    summary: list[tuple[str, EvaluationResult]] = []

    for val_env in val_envs:
        val_env.reseed()
        label = str(val_env)
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {label}")
        print(f"{'=' * 60}")

        with torch.no_grad():
            result, examples = evaluate(
                net=net,
                env=val_env,
                reward_fn=reward_fn,
                state_to_str=state_to_str,
                tokenizer=tokenizer,
                eos_token_id=model_info.eos_token_id,
                pad_token_id=model_info.pad_token_id,
                extractor=extractor,
                max_tokens_generated=train_params.max_tokens_generated,
                max_episodes=train_params.val_episodes,
                batch_size=train_params.val_batch_size,
                group_size=1,
                temperature=args.temperature,
                use_bf16=train_params.use_bf16,
                n_examples=train_params.val_episodes,
                show_progress=True,
            )

        summary.append((label, result))

        print(f"\nReward mean: {result.reward_mean:.4f}")
        print(f"Reward std:  {result.reward_std:.4f}")
        print(f"Episodes:    {result.n_episodes}")
        for comp_name, comp_val in result.component_means.items():
            print(f"  {comp_name}: {comp_val:.4f}")

        if not args.no_examples:
            print("\n--- Examples ---")
            for i, ex in enumerate(examples):
                print(f"\n[Example {i + 1}]")
                print(f"Prompt: {ex.prompt[:200]}...")
                print(f"Response: {ex.responses[0]}")
                print(f"Rewards: {ex.rewards[0]}")

    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    print(f"{'Env':<40} {'Mean':>10} {'Std':>10}")
    print("-" * 60)
    for label, result in summary:
        print(f"{label:<40} {result.reward_mean:>10.4f} {result.reward_std:>10.4f}")
    overall_mean = sum(r.reward_mean for _, r in summary) / len(summary)
    print("-" * 60)
    print(f"{'Overall':<40} {overall_mean:>10.4f}")


if __name__ == "__main__":
    main()
