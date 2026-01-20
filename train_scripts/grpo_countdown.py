"""Verify GRPO can learn on the Countdown task with Qwen-0.6B.

Based on TinyZero: https://github.com/Jiayi-Pan/TinyZero

Usage:
    uv run python benchmarks/verify_grpo_learning.py
    uv run python benchmarks/verify_grpo_learning.py --device mps
    uv run python benchmarks/verify_grpo_learning.py --max-episodes 100
"""

import argparse
import importlib.util

import extty
import torch
from tokenizers import Tokenizer

from dialectic.llm.qwen import load_qwen_06b
from dialectic.llm.utils import get_default_device
from dialectic.rl.env import CountdownEnv
from dialectic.rl.extractors import extract_from_answer_tags
from dialectic.rl.reward import CountdownRewardFn, CountdownWithFormatRewardFn
from dialectic.rl.train import train_grpo
from dialectic.rl.types import Countdown


def _is_modal_installed():
    return importlib.util.find_spec("modal") is not None


def _check_inside_modal_fn():
    if _is_modal_installed():
        import modal

        return modal.current_function_call_id()
    return False


bucket_name = "model-weights"
r2_account_id = "a64c6da180648dd944675d311c296763"


def countdown_state_to_str(data: Countdown) -> str:
    return data.prompt


TIMEOUT_HOURS = 1


def train(
    *,
    device: str | None = None,
    max_episodes: int = 1000,
    batch_size: int = 8,
    group_size: int = 8,
    max_tokens: int = 1024,
    lr: float = 1e-5,
    beta: float = 0.04,
    weights_path: str = "/weights/qwen3-0.6b.pth",
    tokenizer_path: str = "/weights/tokenizer.json",
    binary_reward: bool = False,
    num_operands: int = 2,
    mu: int = 1,
    accumulation_steps: int = 4,
    max_grad_norm: float = 1.0,
):
    extty.init(
        "grpo-learning",
        config={
            "max_episodes": max_episodes,
            "batch_size": batch_size,
            "group_size": group_size,
            "max_tokens": max_tokens,
            "lr": lr,
            "beta": beta,
            "binary_reward": binary_reward,
            "mu": mu,
            "num_operands": num_operands,
            "accumulation_steps": accumulation_steps,
            "max_grad_norm": max_grad_norm,
        },
        server=_check_inside_modal_fn(),
    )

    net = load_qwen_06b()
    net.load_state_dict(
        torch.load(weights_path, map_location=device, weights_only=True)
    )

    print(
        f"Model loaded: {sum(p.numel() for p in net.parameters()) / 1e6:.1f}M parameters"
    )

    opt = torch.optim.Adam(net.parameters(), lr=lr)

    if binary_reward:
        reward_fn = CountdownRewardFn()
    else:
        reward_fn = CountdownWithFormatRewardFn()

    tokenizer = Tokenizer.from_file(tokenizer_path)

    env = CountdownEnv(
        num_operands=num_operands,
        min_number=1,
        max_number=10,
    )
    device = device or get_default_device()
    print(f"device: {device}")
    net = net.to(device)

    try:
        train_grpo(
            net=net,
            opt=opt,
            env=env,
            reward_fn=reward_fn,
            state_to_str=countdown_state_to_str,
            tokenizer=tokenizer,
            eos_token_id=151645,  # <|im_end|>
            pad_token_id=151643,
            extractor=extract_from_answer_tags,
            beta=beta,
            eps=0.2,
            mu=mu,
            max_tokens_generated=max_tokens,
            max_episodes=max_episodes,
            update_ref_net_batch_cadence=10,
            batch_size=batch_size,
            group_size=group_size,
            temperature=0.7,
            accumulation_steps=accumulation_steps,
            max_grad_norm=max_grad_norm,
        )
    finally:
        extty.finish()


if _is_modal_installed():
    import modal

    app = modal.App()
    secret = modal.Secret.from_name(
        "r2-secret", required_keys=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
    )

    image = modal.Image.debian_slim().uv_sync().add_local_python_source("dialectic")

    train_modal = app.function(
        image=image,
        gpu="A100-80GB",
        volumes={
            "/weights": modal.CloudBucketMount(
                bucket_name=bucket_name,
                bucket_endpoint_url=f"https://{r2_account_id}.r2.cloudflarestorage.com",
                secret=secret,
                read_only=True,
            )
        },
        timeout=60 * 60 * TIMEOUT_HOURS,
    )(train)


def main():
    parser = argparse.ArgumentParser(description="Verify GRPO learning on Countdown")
    parser.add_argument("--device", default=None, help="Device (default: auto-detect)")
    parser.add_argument(
        "--max-episodes", type=int, default=1000, help="Max training episodes"
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--group-size", type=int, default=8, help="Group size for GRPO")
    parser.add_argument(
        "--max-tokens", type=int, default=1024, help="Max tokens to generate"
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument(
        "--beta", type=float, default=0.04, help="KL penalty coefficient"
    )
    parser.add_argument(
        "--weights-path", default="weights/qwen3-0.6b.pth", help="Path to model weights"
    )
    parser.add_argument(
        "--tokenizer-path",
        default="qwen-tokenizer/tokenizer.json",
        help="Path to tokenizer",
    )
    parser.add_argument(
        "--binary-reward",
        action="store_true",
        help="Use binary reward (1.0 for correct, 0.0 otherwise). Default uses format shaping.",
    )
    parser.add_argument(
        "--num-operands", type=int, default=2, help="Number of operands"
    )
    parser.add_argument(
        "--mu", type=int, default=1, help="Optimization passes per batch"
    )
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping",
    )
    args = parser.parse_args()

    train(
        device=args.device,
        max_episodes=args.max_episodes,
        batch_size=args.batch_size,
        group_size=args.group_size,
        max_tokens=args.max_tokens,
        lr=args.lr,
        beta=args.beta,
        weights_path=args.weights_path,
        tokenizer_path=args.tokenizer_path,
        binary_reward=args.binary_reward,
        num_operands=args.num_operands,
        mu=args.mu,
        accumulation_steps=args.accumulation_steps,
        max_grad_norm=args.max_grad_norm,
    )


if __name__ == "__main__":
    main()
