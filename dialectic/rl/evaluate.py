"""
Evaluate a Qwen model on ArithmeticEnv or GSM8kEnv.

Usage:
    # Evaluate on arithmetic (default, easy settings)
    python -m dialectic.rl.evaluate --env arithmetic

    # Evaluate on arithmetic with custom settings
    python -m dialectic.rl.evaluate --env arithmetic --min-value 0 --max-value 50 --num-episodes 100

    # Evaluate on GSM8k
    python -m dialectic.rl.evaluate --env gsm8k --data-path path/to/gsm8k.jsonl
"""

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
from tokenizers import Tokenizer
from tqdm import tqdm

from dialectic.llm.qwen import Qwen, generate_from_chat, load_qwen_06b
from dialectic.llm.tokenizer import Message
from dialectic.rl.env import ArithmeticEnv, Env, GSM8kEnv


@dataclass
class EpisodeResult:
    """Result from a single evaluation episode."""

    question: str
    correct_answer: str
    model_response: str
    extracted_answer: str | None
    correct: bool


@dataclass
class EvalResult:
    """Structured output from model evaluation."""

    accuracy: float
    correct: int
    total: int
    episodes: list[EpisodeResult] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"EvalResult(accuracy={self.accuracy:.2%}, correct={self.correct}/{self.total})"


def load_model(weights_path: str, device: str) -> tuple[Qwen, Tokenizer]:
    """Load Qwen model and tokenizer."""
    print(f"Loading model from {weights_path}...")
    model = load_qwen_06b()
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer_path = (
        Path(__file__).parent.parent.parent / "qwen-tokenizer" / "tokenizer.json"
    )
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    print(f"Model loaded on {device}")
    return model, tokenizer


def get_model_response(
    model: Qwen,
    tokenizer: Tokenizer,
    question: str,
    device: str,
    max_tokens: int = 256,
    system_prompt: str | None = None,
    disable_thinking: bool = False,
) -> str:
    """Get model's response to a question."""
    messages = []
    if system_prompt:
        messages.append(Message(role="system", content=system_prompt))

    if disable_thinking:
        question += " /no_think"

    messages.append(Message(role="user", content=question))

    response = generate_from_chat(
        net=model,
        tokenizer=tokenizer,
        messages=messages,
        max_tokens_generated=max_tokens,
        device=device,
    )

    # Response is a list, get the first one and extract assistant's answer
    full_response = response[0]

    print("full_response: \n", full_response)

    # Try multiple patterns to extract assistant response
    # The tokenizer may decode special tokens differently
    for marker in ["<|im_start|>assistant\n", "<|im_start|>assistant", "assistant\n"]:
        if marker in full_response:
            answer = full_response.split(marker)[-1]
            break
    else:
        answer = full_response

    # Remove end token if present
    for end_marker in ["<|im_end|>", "<|endoftext|>"]:
        if end_marker in answer:
            answer = answer.split(end_marker)[0]

    # Remove thinking block if present - we want just the final answer
    if "</think>" in answer:
        answer = answer.split("</think>")[-1]

    return answer.strip()


ARITHMETIC_SYSTEM_PROMPT = """\
You are a calculator. Solve the arithmetic problem and give the final answer.

Format your response exactly like this example:
User: What is 2 + 3?
Assistant: 2 + 3 = 5

#### 5

Always end with "#### " followed by just the number."""


GSM8K_SYSTEM_PROMPT = """\
You are a helpful math tutor. Solve the word problem step by step, then give the final numerical answer.

Format your response exactly like this example:
User: John has 5 apples. He buys 3 more. How many apples does he have?
Assistant: John starts with 5 apples.
He buys 3 more apples.
Total apples = 5 + 3 = 8

#### 8

Always end with "#### " followed by just the final number."""


def extract_final_answer(text: str) -> str | None:
    """
    Extract the final answer from model response.

    Tries multiple patterns in order of preference.

    Parameters
    ----------
    text : str
        Model response text.

    Returns
    -------
    str or None
        Extracted number or None if pattern not found.
    """
    patterns = [
        r"####\s*(-?\d+(?:\.\d+)?)",  # GSM8k style: #### 42
        r"=\s*(-?\d+(?:\.\d+)?)\s*$",  # Trailing equals: = 42
        r"(?:answer|result)(?:\s+is)?[:\s]+(-?\d+(?:\.\d+)?)",  # "answer is 42"
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1)

    return None


def evaluate(
    env: Env[str, str],
    generate: Callable[[str], str],
    *,
    num_episodes: int,
    seed: int = 42,
    extract_answer: Callable[[str], str | None] = extract_final_answer,
    get_correct_answer: Callable[[dict[str, Any], dict[str, Any]], str] | None = None,
    verbose: bool = True,
) -> EvalResult:
    """
    Evaluate a model on an RL environment.

    Parameters
    ----------
    env : Env[str, str]
        Environment implementing reset() and step().
    generate : Callable[[str], str]
        Function that takes a question and returns the model's response.
    num_episodes : int
        Number of episodes to evaluate.
    seed : int
        Random seed for reproducibility.
    extract_answer : Callable[[str], str | None]
        Function to extract answer from model response.
    get_correct_answer : Callable[[dict, dict], str] | None
        Function to get correct answer from (reset_info, step_info).
        If None, tries reset_info["answer"] then step_info["correct_answer"].
    verbose : bool
        Whether to print progress and results.

    Returns
    -------
    EvalResult
        Structured evaluation results.
    """
    correct = 0
    total = 0
    episodes: list[EpisodeResult] = []

    iterator = range(num_episodes)
    if verbose:
        iterator = tqdm(iterator, desc="Evaluating")

    for i in iterator:
        episode_seed = seed + i if i == 0 else None
        question, reset_info = env.reset(seed=episode_seed)

        if verbose:
            print("question:\n", question)

        response = generate(question)

        extracted = extract_answer(response)
        action = extracted if extracted else response
        _, reward, _, _, step_info = env.step(action)

        if get_correct_answer:
            correct_answer = get_correct_answer(reset_info, step_info)
        else:
            correct_answer = reset_info.get("answer") or step_info.get(
                "correct_answer", ""
            )

        is_correct = reward == 1.0
        correct += int(is_correct)
        total += 1

        episodes.append(
            EpisodeResult(
                question=question,
                correct_answer=str(correct_answer),
                model_response=response,
                extracted_answer=extracted or step_info.get("proposed_answer"),
                correct=is_correct,
            )
        )

    accuracy = correct / total if total > 0 else 0.0

    if verbose:
        print(f"\n{'=' * 50}")
        print("Evaluation Results")
        print(f"{'=' * 50}")
        print(f"Episodes: {total}")
        print(f"Correct: {correct}")
        print(f"Accuracy: {accuracy:.2%}")

    return EvalResult(
        accuracy=accuracy,
        correct=correct,
        total=total,
        episodes=episodes,
    )


def evaluate_arithmetic(
    *,
    model: Qwen,
    tokenizer: Tokenizer,
    device: str,
    num_episodes: int = 100,
    min_value: int = 0,
    max_value: int = 20,
    operations: tuple[str, ...] = ("+",),
    num_operands: int = 2,
    max_tokens: int = 64,
    seed: int = 42,
    disable_thinking: bool = False,
) -> EvalResult:
    """Evaluate model on arithmetic environment."""
    env = ArithmeticEnv(
        min_value=min_value,
        max_value=max_value,
        operations=operations,
        num_operands=num_operands,
    )

    def generate(question: str) -> str:
        return get_model_response(
            model=model,
            tokenizer=tokenizer,
            question=question,
            device=device,
            max_tokens=max_tokens,
            system_prompt=ARITHMETIC_SYSTEM_PROMPT,
            disable_thinking=disable_thinking,
        )

    return evaluate(env, generate, num_episodes=num_episodes, seed=seed)


def evaluate_gsm8k(
    *,
    model: Qwen,
    tokenizer: Tokenizer,
    device: str,
    data_path: str,
    num_episodes: int | None = None,
    max_tokens: int = 512,
    seed: int = 42,
    disable_thinking: bool = False,
) -> EvalResult:
    """Evaluate model on GSM8k environment."""
    env = GSM8kEnv(data_path=data_path, shuffle=True)

    if num_episodes is None:
        num_episodes = len(env.problems)

    def generate(question: str) -> str:
        return get_model_response(
            model=model,
            tokenizer=tokenizer,
            question=question,
            device=device,
            max_tokens=max_tokens,
            system_prompt=GSM8K_SYSTEM_PROMPT,
            disable_thinking=disable_thinking,
        )

    return evaluate(
        env,
        generate,
        num_episodes=num_episodes,
        seed=seed,
        get_correct_answer=lambda _, step_info: step_info["correct_answer"],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen model on RL environments"
    )
    parser.add_argument(
        "--env",
        type=str,
        choices=["arithmetic", "gsm8k"],
        default="arithmetic",
        help="Environment to evaluate on",
    )
    parser.add_argument(
        "--weights-path",
        type=str,
        default="weights/qwen3-0.6b.pth",
        help="Path to model weights",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run on",
    )
    parser.add_argument(
        "--num-episodes", type=int, default=50, help="Number of episodes"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--max-tokens", type=int, default=128, help="Max tokens to generate"
    )

    # Arithmetic-specific args
    parser.add_argument("--min-value", type=int, default=0, help="Min operand value")
    parser.add_argument("--max-value", type=int, default=20, help="Max operand value")
    parser.add_argument(
        "--operations",
        type=str,
        nargs="+",
        default=["+"],
        help="Operations to use (e.g., + - * /)",
    )
    parser.add_argument(
        "--num-operands", type=int, default=2, help="Number of operands"
    )
    parser.add_argument(
        "--disable-thinking", action="store_true", help="Do not use thinking mode"
    )

    # GSM8k-specific args
    parser.add_argument("--data-path", type=str, help="Path to GSM8k JSONL file")

    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model(args.weights_path, args.device)

    if args.env == "arithmetic":
        evaluate_arithmetic(
            model=model,
            tokenizer=tokenizer,
            device=args.device,
            num_episodes=args.num_episodes,
            min_value=args.min_value,
            max_value=args.max_value,
            operations=tuple(args.operations),
            num_operands=args.num_operands,
            max_tokens=args.max_tokens,
            seed=args.seed,
            disable_thinking=args.disable_thinking,
        )
    elif args.env == "gsm8k":
        if not args.data_path:
            parser.error("--data-path is required for gsm8k environment")
        evaluate_gsm8k(
            model=model,
            tokenizer=tokenizer,
            device=args.device,
            data_path=args.data_path,
            num_episodes=args.num_episodes,
            max_tokens=args.max_tokens,
            seed=args.seed,
            disable_thinking=args.disable_thinking,
        )


if __name__ == "__main__":
    main()
