"""Verify GRPO can learn on the Countdown task with Qwen-0.6B.

Based on TinyZero: https://github.com/Jiayi-Pan/TinyZero

Usage:
    uv run python benchmarks/verify_grpo_learning.py
    uv run python benchmarks/verify_grpo_learning.py --device mps
    uv run python benchmarks/verify_grpo_learning.py --max-episodes 100
"""

import argparse
from dataclasses import asdict

import extty
import torch
from tokenizers import Tokenizer

from dialectic.llm.qwen import load_qwen_06b
from dialectic.llm.templates import Message, get_qwen_input_text_from_messages
from dialectic.llm.utils import get_default_device
from dialectic.rl.env import Countdown, CountdownEnv
from dialectic.rl.evaluate import evaluate
from dialectic.rl.extractors import extract_from_answer_tags
from dialectic.rl.reward import CountdownRewardFn, CountdownWithFormatRewardFn


def get_state_to_str(enable_thinking: bool):
    def _state_to_str(data: Countdown) -> str:
        return get_qwen_input_text_from_messages(
            [Message(role="user", content=data.prompt)],
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

    return _state_to_str


def get_prompt_template(enable_thinking: bool):
    if enable_thinking:
        return (
            "Using the numbers {numbers}, create an equation that equals {target}. "
            "You can use +, -, *, / and each number at most once. "
            "Show your reasoning in <think></think> tags. Please be concise and give just one solution."
            "Put your final equation in <answer></answer> tags. "
            "For example, if the equation is 3+5*2, respond with <answer>3+5*2</answer>."
        )
    return (
        "Using the numbers {numbers}, create an equation that equals {target}. "
        "You can use +, -, *, / and each number at most once. "
        "Show your reasoning in <reasoning></reasoning> tags. Please be concise and give just one solution."
        "Put your final equation in <answer></answer> tags. "
        "For example, if the equation is 3+5*2, respond with <reasoning>[detailed reasoning explanations]</reasoning><answer>3+5*2</answer>."
    )


@extty.evaluation(
    "grpo",
    model="qwen",
    model_config_kwargs=["weights_path"],
    eval_config_kwargs=["batch_size"],
)
def eval(
    *,
    device: str | None = None,
    max_tokens: int = 1024,
    batch_size: int = 2,
    group_size: int = 8,
    weights_path: str = "/weights/qwen3-0.6b.pth",
    tokenizer_path: str = "/weights/tokenizer.json",
    binary_reward: bool = False,
    num_operands: int = 2,
    use_bf16: bool = True,
    use_qwen_thinking: bool = False,
    max_episodes: int,
    n_examples: int,
):
    net = load_qwen_06b()
    net.load_state_dict(
        torch.load(weights_path, map_location=device, weights_only=True)
    )

    print(
        f"Model loaded: {sum(p.numel() for p in net.parameters()) / 1e6:.1f}M parameters"
    )

    if binary_reward:
        reward_fn = CountdownRewardFn()
    else:
        reward_fn = CountdownWithFormatRewardFn(
            "think" if use_qwen_thinking else "reasoning"
        )

    tokenizer = Tokenizer.from_file(tokenizer_path)

    env = CountdownEnv(
        num_operands=num_operands,
        min_number=1,
        max_number=10,
        prompt_template=get_prompt_template(enable_thinking=use_qwen_thinking),
    )
    device = device or get_default_device()
    print(f"device: {device}")
    net = net.to(device)

    state_to_str = get_state_to_str(enable_thinking=use_qwen_thinking)

    eval_result, examples = evaluate(
        net=net,
        tokenizer=tokenizer,
        env=env,
        reward_fn=reward_fn,
        state_to_str=state_to_str,
        eos_token_id=151645,
        pad_token_id=151643,
        extractor=extract_from_answer_tags,
        max_tokens_generated=max_tokens,
        max_episodes=max_episodes,
        batch_size=batch_size,
        group_size=group_size,
        temperature=0.7,
        use_bf16=use_bf16,
        n_examples=n_examples,
    )

    return asdict(eval_result), examples


def main():
    parser = argparse.ArgumentParser(description="Verify GRPO learning on Countdown")
    parser.add_argument("--device", default=None, help="Device (default: auto-detect)")
    parser.add_argument(
        "--max-episodes", type=int, default=1000, help="Max training episodes"
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--group-size", type=int, default=4, help="Group size for GRPO")
    parser.add_argument(
        "--max-tokens", type=int, default=1024, help="Max tokens to generate"
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
        "--n-examples", type=int, default=10, help="Number of examples to return"
    )
    args = parser.parse_args()

    eval(
        device=args.device,
        max_episodes=args.max_episodes,
        batch_size=args.batch_size,
        group_size=args.group_size,
        max_tokens=args.max_tokens,
        weights_path=args.weights_path,
        tokenizer_path=args.tokenizer_path,
        binary_reward=args.binary_reward,
        num_operands=args.num_operands,
        use_bf16=args.use_bf16,
        use_qwen_thinking=args.use_qwen_thinking,
        n_examples=args.n_examples,
    )


if __name__ == "__main__":
    main()
