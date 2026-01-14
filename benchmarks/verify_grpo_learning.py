"""Verify GRPO can learn on the Countdown task with Qwen-0.6B.

Based on TinyZero: https://github.com/Jiayi-Pan/TinyZero

Usage:
    uv run python benchmarks/verify_grpo_learning.py
    uv run python benchmarks/verify_grpo_learning.py --device mps
    uv run python benchmarks/verify_grpo_learning.py --max-episodes 100
"""

import argparse
from pathlib import Path

import extty
import torch
from tokenizers import Tokenizer

from dialectic.llm.qwen import load_qwen_06b
from dialectic.rl.env import CountdownEnv
from dialectic.rl.extractors import extract_from_answer_tags
from dialectic.rl.reward import CountdownRewardFn, CountdownWithFormatRewardFn
from dialectic.rl.train import train_grpo
from dialectic.rl.types import Countdown


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def countdown_state_to_str(data: Countdown) -> str:
    return data.prompt


def main():
    parser = argparse.ArgumentParser(description="Verify GRPO learning on Countdown")
    parser.add_argument("--device", default=None, help="Device (default: auto-detect)")
    parser.add_argument(
        "--max-episodes", type=int, default=50, help="Max training episodes"
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--group-size", type=int, default=2, help="Group size for GRPO")
    parser.add_argument(
        "--max-tokens", type=int, default=100, help="Max tokens to generate"
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument(
        "--beta", type=float, default=0.04, help="KL penalty coefficient"
    )
    parser.add_argument(
        "--weights", default="weights/qwen3-0.6b.pth", help="Path to model weights"
    )
    parser.add_argument(
        "--binary-reward",
        action="store_true",
        help="Use binary reward (1.0 for correct, 0.0 otherwise). Default uses format shaping.",
    )
    args = parser.parse_args()

    device = args.device or get_device()
    print(f"Device: {device}")

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"Error: Weights not found at {weights_path}")
        print("Download or generate weights first.")
        return

    print("Loading Qwen-0.6B...")
    net = load_qwen_06b()
    net.load_state_dict(
        torch.load(weights_path, map_location=device, weights_only=True)
    )
    net = net.to(device)
    print(
        f"Model loaded: {sum(p.numel() for p in net.parameters()) / 1e6:.1f}M parameters"
    )

    tokenizer = Tokenizer.from_file("qwen-tokenizer/tokenizer.json")

    env = CountdownEnv(
        num_operands=3,
        min_number=1,
        max_number=10,
        max_target=20,
    )

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    if args.binary_reward:
        reward_fn = CountdownRewardFn()
        reward_type = "binary (1.0 correct, 0.0 otherwise)"
    else:
        reward_fn = CountdownWithFormatRewardFn()
        reward_type = "format-shaped (partial credit for structure)"

    print()
    print("=" * 60)
    print("GRPO Training on Countdown Task")
    print("=" * 60)
    print(f"Batch size: {args.batch_size}")
    print(f"Group size: {args.group_size}")
    print(f"Max episodes: {args.max_episodes}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Learning rate: {args.lr}")
    print(f"KL beta: {args.beta}")
    print(f"Reward: {reward_type}")
    print("Task: Easy countdown (3 numbers, 1-10, target 1-20)")
    print("=" * 60)
    print()

    extty.init("grpo-learning", config=vars(args))

    metrics = train_grpo(
        net=net,
        opt=opt,
        env=env,
        reward_fn=reward_fn,
        state_to_str=countdown_state_to_str,
        tokenizer=tokenizer,
        eos_token_id=151645,  # <|im_end|>
        pad_token_id=151643,
        extractor=extract_from_answer_tags,
        beta=args.beta,
        eps=0.2,
        mu=1,
        max_tokens_generated=args.max_tokens,
        max_episodes=args.max_episodes,
        update_ref_net_batch_cadence=10,
        batch_size=args.batch_size,
        group_size=args.group_size,
        temperature=0.7,
        verbose=True,
    )

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    n_batches = len(metrics.mean_rewards)
    print(f"Completed {n_batches} batches")

    if n_batches >= 10:
        early_rewards = metrics.mean_rewards[:5]
        late_rewards = metrics.mean_rewards[-5:]
        early_mean = sum(early_rewards) / len(early_rewards)
        late_mean = sum(late_rewards) / len(late_rewards)

        print(f"Early mean reward (first 5 batches): {early_mean:.4f}")
        print(f"Late mean reward (last 5 batches): {late_mean:.4f}")
        print(f"Improvement: {late_mean - early_mean:+.4f}")

        if late_mean > early_mean:
            print("\nLearning detected: rewards improved over training")
        else:
            print("\nNo clear learning signal detected")
    elif n_batches > 0:
        print(
            f"Mean reward: {sum(metrics.mean_rewards) / len(metrics.mean_rewards):.4f}"
        )
        print("(Run with more episodes for learning comparison)")
    else:
        print("No batches completed")


if __name__ == "__main__":
    main()
