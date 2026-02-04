"""Verify GRPO can learn on the Countdown task with Qwen-0.6B.

Based on TinyZero: https://github.com/Jiayi-Pan/TinyZero

Usage:
    uv run python benchmarks/verify_grpo_learning.py
    uv run python benchmarks/verify_grpo_learning.py --device mps
    uv run python benchmarks/verify_grpo_learning.py --max-episodes 100
"""

import argparse
import importlib.util
import os
import random

import extty
import torch
from dotenv import load_dotenv
from tokenizers import Tokenizer

from dialectic.llm.qwen import load_qwen_06b
from dialectic.llm.templates import Message, get_qwen_input_text_from_messages
from dialectic.llm.utils import get_default_device
from dialectic.rl.env import Countdown, CountdownEnv
from dialectic.rl.extractors import extract_from_answer_tags
from dialectic.rl.reward import CountdownRewardFn, CountdownWithFormatRewardFn
from dialectic.rl.train import train_grpo

load_dotenv()


def _is_modal_installed():
    return importlib.util.find_spec("modal") is not None


def _check_inside_modal_fn():
    if _is_modal_installed():
        import modal

        return modal.current_function_call_id()
    return False


def get_state_to_str(enable_thinking: bool):
    def _state_to_str(data: Countdown) -> str:
        reasoning_tag = "think" if enable_thinking else "reasoning"
        ret = get_qwen_input_text_from_messages(
            [Message(role="user", content=data.prompt)],
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

        ret += f"Let me solve this step by step\n<{reasoning_tag}>"

        return ret

    return _state_to_str


def get_prompt_template(enable_thinking: bool):
    reasoning_tag = "think" if enable_thinking else "reasoning"
    return (
        "Using the numbers {numbers}, create an equation that equals {target}. "
        "You can use basic arithmetic operations (+, -, *, /) and each number at most once. "
        f"Show your reasoning in <{reasoning_tag}></{reasoning_tag}> tags."
        "Put your final equation in <answer></answer> tags, for example <answer> (1 + 2) / 3 </answer>."
    )


MODAL_TIMEOUT_HOURS = int(os.getenv("MODAL_TIMEOUT_HOURS", 1))


@extty.experiment(project="grpo-countdown", server=_check_inside_modal_fn)
def train(
    *,
    device: str | None = None,
    max_episodes: int = 1000,
    batch_size: int = 2,
    group_size: int = 8,
    max_tokens: int = 1024,
    lr: float = 1e-5,
    beta: float = 0.04,
    weights_path: str = "/weights/qwen3/qwen3-0.6b.pth",
    tokenizer_path: str = "/weights/qwen3/tokenizer.json",
    binary_reward: bool = False,
    n_larges: int = 2,
    n_total: int = 6,
    n_ops: int = 5,
    seed: int,
    compile_model: bool,
    mu: int = 1,
    accumulation_steps: int = 16,
    update_ref_net_batch_cadence: int = 100,
    max_grad_norm: float = 1.0,
    logprob_chunk_size: int = 64,
    use_bf16: bool = True,
    use_qwen_thinking: bool = False,
):
    torch.manual_seed(seed)
    net = load_qwen_06b()
    net.load_state_dict(
        torch.load(weights_path, map_location=device, weights_only=True)
    )

    if compile_model:
        net.compile()

    print(
        f"Model loaded: {sum(p.numel() for p in net.parameters()) / 1e6:.1f}M parameters"
    )

    opt = torch.optim.Adam(net.parameters(), lr=lr)

    if binary_reward:
        reward_fn = CountdownRewardFn()
    else:
        reward_fn = CountdownWithFormatRewardFn(
            "think" if use_qwen_thinking else "reasoning"
        )

    tokenizer = Tokenizer.from_file(tokenizer_path)

    env = CountdownEnv(
        seed=seed,
        n_larges=n_larges,
        n_total=n_total,
        n_ops=n_ops,
        prompt_template=get_prompt_template(enable_thinking=use_qwen_thinking),
    )
    device = device or get_default_device()
    print(f"device: {device}")
    net = net.to(device)

    state_to_str = get_state_to_str(enable_thinking=use_qwen_thinking)

    try:
        train_grpo(
            net=net,
            opt=opt,
            env=env,
            reward_fn=reward_fn,
            state_to_str=state_to_str,
            tokenizer=tokenizer,
            eos_token_id=151645,  # <|im_end|>
            pad_token_id=151643,
            extractor=extract_from_answer_tags,
            beta=beta,
            eps=0.2,
            mu=mu,
            max_tokens_generated=max_tokens,
            max_episodes=max_episodes,
            update_ref_net_batch_cadence=update_ref_net_batch_cadence,
            batch_size=batch_size,
            group_size=group_size,
            temperature=0.7,
            accumulation_steps=accumulation_steps,
            max_grad_norm=max_grad_norm,
            logprob_chunk_size=logprob_chunk_size,
            use_bf16=use_bf16,
        )
    finally:
        extty.finish()


if _is_modal_installed():
    import modal

    app = modal.App()
    secret = modal.Secret.from_name(
        "r2-secret", required_keys=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
    )

    # get `extty` variables
    s3_conf = extty.S3Config.load()
    if s3_conf is not None:
        extty_env_dict = {
            "EXTTY_S3_BUCKET": s3_conf.bucket,
            "EXTTY_S3_PREFIX": s3_conf.prefix,
            "EXTTY_S3_REGION": s3_conf.region,
            "EXTTY_S3_ACCESS_KEY_ID": s3_conf.access_key_id,
            "EXTTY_S3_SECRET_ACCESS_KEY": s3_conf.secret_access_key,
            "EXTTY_S3_ENDPOINT_URL": s3_conf.endpoint_url,
        }
    else:
        extty_env_dict = {}

    image = (
        modal.Image.debian_slim()
        .uv_sync()
        .add_local_python_source("dialectic", "extty")
    )

    bucket_name = os.environ.get("WEIGHTS_BUCKET_NAME")
    bucket_endpoint_url = os.environ.get("WEIGHTS_BUCKET_ENDPOINT_URL")

    train_modal = app.function(
        image=image,
        gpu="A100-80GB",
        volumes={
            "/weights": modal.CloudBucketMount(
                bucket_name=bucket_name,
                bucket_endpoint_url=bucket_endpoint_url,
                secret=secret,
                read_only=True,
            )
        },
        timeout=60 * 60 * MODAL_TIMEOUT_HOURS,
        secrets=[modal.Secret.from_dict(extty_env_dict)],
    )(train)


def main():
    parser = argparse.ArgumentParser(description="Verify GRPO learning on Countdown")
    parser.add_argument("--device", default=None, help="Device (default: auto-detect)")
    parser.add_argument(
        "--max-episodes", type=int, default=1000, help="Max training episodes"
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--group-size", type=int, default=8, help="Group size for GRPO")
    parser.add_argument(
        "--max-tokens", type=int, default=1024, help="Max tokens to generate"
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument(
        "--beta", type=float, default=0.04, help="KL penalty coefficient"
    )
    parser.add_argument(
        "--weights-path", default="qwen3/qwen3-0.6b.pth", help="Path to model weights"
    )
    parser.add_argument(
        "--tokenizer-path",
        default="qwen3/tokenizer.json",
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
        default=16,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping",
    )
    parser.add_argument(
        "--logprob-chunk-size",
        type=int,
        default=64,
        help="Chunk size for log prob computation (0 to disable chunking)",
    )
    parser.add_argument(
        "--update-ref-net-batch-cadence",
        type=int,
        default=100,
        help="How often to update the reference net",
    )

    parser.add_argument(
        "--use-bf16",
        action="store_true",
        default=True,
        help="Use bf16 mixed precision training (default: True)",
    )
    parser.add_argument(
        "--no-bf16",
        dest="use_bf16",
        action="store_false",
        help="Disable bf16 mixed precision training",
    )

    parser.add_argument(
        "--use-qwen-thinking",
        action="store_true",
        default=False,
        help="Use Qwen's out-of-the-box thinking mode",
    )
    parser.add_argument(
        "--no-qwen-thinking",
        dest="use_qwen_thinking",
        action="store_false",
        help="Do not use Qwen's out-of-the-box thinking mode",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=random.choice(range(1000)),
        help="Chunk size for log prob computation (0 to disable chunking)",
    )

    parser.add_argument("--compile-model", action="store_true", help="compile model")
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
        n_ops=args.num_operands,
        mu=args.mu,
        accumulation_steps=args.accumulation_steps,
        update_ref_net_batch_cadence=args.update_ref_net_batch_cadence,
        max_grad_norm=args.max_grad_norm,
        logprob_chunk_size=args.logprob_chunk_size,
        use_bf16=args.use_bf16,
        use_qwen_thinking=args.use_qwen_thinking,
        compile_model=args.compile_model,
    )


if __name__ == "__main__":
    main()
