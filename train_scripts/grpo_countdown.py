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
import sys
from dataclasses import dataclass
from functools import partial
from typing import Callable, Literal

import extty
import torch
from dotenv import load_dotenv
from tokenizers import Tokenizer

from dialectic.artifacts import Artifact, get_artifact
from dialectic.llm.llama import (
    LLAMA_32_TOKENIZER,
    load_llama_32_1b_instruct,
    load_llama_32_3b_instruct,
)
from dialectic.llm.qwen import load_qwen3_06b
from dialectic.llm.templates import (
    Message,
    get_llama_input_text_from_messages,
    get_qwen_input_text_from_messages,
)
from dialectic.llm.utils import get_default_device
from dialectic.rl.env import Countdown, CountdownEnv
from dialectic.rl.extractors import extract_from_answer_tags
from dialectic.rl.reward import (
    answer_tags,
    countdown_correct,
    think_tags,
    weighted_reward,
)
from dialectic.rl.train import (
    ValidationConfig,
    grpo_advantage,
    rloo_advantage,
    train_grpo,
    train_soft_grpo,
)

load_dotenv()

REASONING_TAG = "reasoning"


@dataclass
class ModelInfo:
    net_factory: Callable
    tokenizer: str | Artifact
    eos_token_id: int
    pad_token_id: int
    format_messages: Callable[[list[Message], bool], str]

    def load_net(self):
        return self.net_factory()

    def load_tokenizer(self) -> Tokenizer:
        if isinstance(self.tokenizer, str):
            return Tokenizer.from_pretrained(self.tokenizer)
        return Tokenizer.from_file(str(get_artifact(self.tokenizer)[0]))


MODEL_REGISTRY: dict[str, ModelInfo] = {
    "qwen3-0.6b": ModelInfo(
        net_factory=lambda: load_qwen3_06b(True),
        tokenizer="Qwen/Qwen3-0.6B",
        eos_token_id=151645,
        pad_token_id=151643,
        format_messages=lambda msgs, gen: get_qwen_input_text_from_messages(
            msgs, gen, enable_thinking=False
        ),
    ),
    "llama-3.2-1b-instruct": ModelInfo(
        net_factory=lambda: load_llama_32_1b_instruct(True),
        tokenizer=LLAMA_32_TOKENIZER,
        eos_token_id=128009,
        pad_token_id=128009,
        format_messages=lambda msgs, gen: get_llama_input_text_from_messages(msgs, gen),
    ),
    "llama-3.2-3b-instruct": ModelInfo(
        net_factory=lambda: load_llama_32_3b_instruct(True),
        tokenizer=LLAMA_32_TOKENIZER,
        eos_token_id=128009,
        pad_token_id=128009,
        format_messages=lambda msgs, gen: get_llama_input_text_from_messages(msgs, gen),
    ),
}

# system prompt from Soft Tokens Hard Truths paper (but using answer tags instead of boxed)
STHT_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves"
    " it. The assistant first shows the complete reasoning process step by step, then provides the final"
    " answer in <answer></answer> tags. The assistant must always follow the format: 'User: [question] Assistant:"
    " [detailed reasoning] The final answer is: <answer>[answer]</answer>.'"
)
SIMPLE_SYSTEM_PROMPT = (
    "You are a helpful assistant. When asked a question to solve you first show your complete "
    "reasoning process step by step and then provide the user with the answer in the specified format."
)

ENV_PROMPT_WITH_REASONING_TAGS = (
    "Using the numbers {numbers}, create an equation that equals {target}. "
    "You can use basic arithmetic operations (+, -, *, /) and each number exactly once. "
    f"Show your reasoning in <{REASONING_TAG}></{REASONING_TAG}> tags."
    " Put your final equation in <answer></answer> tags, for example <answer> (1 + 2) / 3 </answer>."
)
ENV_PROMPT_WITHOUT_REASONING_TAGS = (
    "Using the numbers {numbers}, create an equation that equals {target}. "
    "You can use basic arithmetic operations (+, -, *, /) and each number exactly once. Put your final"
    " equation in <answer></answer> tags, for example <answer> (1 + 2) / 3 </answer>. "
)


@dataclass
class PromptCollection:
    system_prompt: str | None
    env_prompt: str
    assistant_prefill: str | None


PROMPT_COLLECTIONS: list[PromptCollection] = [
    PromptCollection(
        system_prompt=None,
        env_prompt=ENV_PROMPT_WITH_REASONING_TAGS,
        assistant_prefill=f"Let me solve this step by step\n<{REASONING_TAG}>",
    ),
    PromptCollection(
        system_prompt=STHT_SYSTEM_PROMPT,
        env_prompt=ENV_PROMPT_WITHOUT_REASONING_TAGS,
        assistant_prefill=None,
    ),
    PromptCollection(
        system_prompt=SIMPLE_SYSTEM_PROMPT,
        env_prompt=ENV_PROMPT_WITHOUT_REASONING_TAGS,
        assistant_prefill="Let me solve this step by step.",
    ),
]


def _is_modal_installed():
    return importlib.util.find_spec("modal") is not None


def get_state_to_str(
    *,
    format_messages: Callable[[list[Message], bool], str],
    system_prompt: str | None = None,
    assistant_prefill: str | None = None,
):
    def _state_to_str(data: Countdown) -> str:
        # TODO: add system prompt here
        msgs = []
        if system_prompt:
            msgs.append(Message(role="system", content=system_prompt))
        msgs.append(Message(role="user", content=data.prompt))
        ret = format_messages(msgs, True)
        if assistant_prefill:
            ret += assistant_prefill
        return ret

    return _state_to_str


def get_reward_fn(answer_tags_weight: float, think_tags_weight: float):
    components = [
        ("correct", 1.0, countdown_correct),
        ("answer_tags", answer_tags_weight, answer_tags),
    ]
    if think_tags_weight > 0:
        components.append(
            (
                "think_tags",
                think_tags_weight,
                think_tags(REASONING_TAG, prefilled_open=True),
            )
        )
    reward_fn = weighted_reward(components)
    return reward_fn


# update prompt? especially for soft tokens using <reasoning> tags don't make sense


MODAL_TIMEOUT_HOURS = int(os.getenv("MODAL_TIMEOUT_HOURS", 1))


@extty.experiment(project="grpo-countdown")
def train(
    *,
    model_name: str = "qwen3-0.6b",
    device: str | None = None,
    max_episodes: int = 1000,
    eps: float | None,
    batch_size: int = 2,
    group_size: int = 8,
    max_tokens: int = 700,
    lr: float = 1e-5,
    beta: float = 0.04,
    n_larges: int | list[int] = 2,
    n_total: int | list[int] = 6,
    n_ops: int | list[int] = 5,
    seed: int,
    advantage_fn_type: Literal["grpo", "rloo"],
    compile_model: bool = False,
    mu: int = 1,
    accumulation_steps: int = 16,
    update_ref_net_batch_cadence: int = 100,
    max_grad_norm: float = 1.0,
    logprob_chunk_size: int = 64,
    use_bf16: bool = True,
    use_qwen_thinking: bool = False,
    save_ckpt_freq: int = sys.maxsize,
    soft_tokens: bool = False,
    noise_std: float = 0.33,
    temperature: float = 0.7,
    min_soft_steps: int = 0,
    answer_tags_weight: float = 0.1,
    think_tags_weight: float = 0.05,
    env_prompt_template: str,
    system_prompt: str | None,
    assistant_prefill: str | None,
    normalize_advantages: bool = True,
    normalize_by_sequence_length: bool,
    val_freq: int = 0,
    val_episodes: int = 50,
    val_batch_size: int = 4,
):
    assert model_name in MODEL_REGISTRY
    torch.manual_seed(seed)

    model_info = MODEL_REGISTRY[model_name]
    net = model_info.load_net()
    tokenizer = model_info.load_tokenizer()

    if compile_model:
        net.compile()

    print(
        f"Model loaded: {sum(p.numel() for p in net.parameters()) / 1e6:.1f}M parameters"
    )

    opt = torch.optim.AdamW(net.parameters(), lr=lr)

    if use_qwen_thinking:

        def format_messages(msgs: list[Message], gen: bool) -> str:
            return get_qwen_input_text_from_messages(msgs, gen, enable_thinking=True)
    else:
        format_messages = model_info.format_messages

    reward_fn = get_reward_fn(
        answer_tags_weight=answer_tags_weight, think_tags_weight=think_tags_weight
    )

    env = CountdownEnv(
        seed=seed,
        n_larges=n_larges,
        n_total=n_total,
        n_ops=n_ops,
        prompt_template=env_prompt_template,
    )
    device = device or get_default_device()
    print(f"device: {device}")
    net = net.to(device)

    state_to_str = get_state_to_str(
        format_messages=format_messages,
        system_prompt=system_prompt,
        assistant_prefill=assistant_prefill,
    )

    answer_tag_ids = tokenizer.encode("<answer>", add_special_tokens=False).ids
    switch_condition = torch.tensor(answer_tag_ids)
    max_tokens_prefill = torch.tensor(answer_tag_ids)

    if advantage_fn_type == "grpo":
        advantage_fn = partial(grpo_advantage, normalize=normalize_advantages)
    elif advantage_fn_type == "rloo":
        advantage_fn = rloo_advantage
    else:
        raise ValueError(f"Got unknown advantage function type {advantage_fn_type}")

    val_config: ValidationConfig | None = None
    if val_freq > 0:
        n_ops_list = [n_ops] if isinstance(n_ops, int) else n_ops
        n_total_list = [n_total] if isinstance(n_total, int) else n_total
        n_larges_list = [n_larges] if isinstance(n_larges, int) else n_larges
        val_envs = []
        for i in range(len(n_ops_list)):
            val_envs.append(
                CountdownEnv(
                    seed=2026 + i,
                    n_larges=n_larges_list[i],
                    n_total=n_total_list[i],
                    n_ops=n_ops_list[i],
                    prompt_template=env_prompt_template,
                )
            )
        val_config = ValidationConfig(
            envs=val_envs,
            reward_fn=reward_fn,
            state_to_str=state_to_str,
            extractor=extract_from_answer_tags,
            tokenizer=tokenizer,
            eos_token_id=model_info.eos_token_id,
            pad_token_id=model_info.pad_token_id,
            max_episodes=val_episodes,
            batch_size=val_batch_size,
            max_tokens_generated=max_tokens,
            use_bf16=use_bf16,
        )

    try:
        if soft_tokens:
            train_soft_grpo(
                net=net,
                opt=opt,
                env=env,
                reward_fn=reward_fn,
                state_to_str=state_to_str,
                tokenizer=tokenizer,
                eos_token_id=model_info.eos_token_id,
                pad_token_id=model_info.pad_token_id,
                extractor=extract_from_answer_tags,
                beta=beta,
                eps=eps,
                mu=mu,
                max_tokens_generated=max_tokens,
                max_episodes=max_episodes,
                update_ref_net_batch_cadence=update_ref_net_batch_cadence,
                batch_size=batch_size,
                group_size=group_size,
                temperature=temperature,
                noise_std=noise_std,
                switch_to_hard_tokens_condition=switch_condition,
                max_tokens_prefill=max_tokens_prefill,
                max_tokens_prefill_steps_before_end=20,
                min_soft_steps=min_soft_steps,
                accumulation_steps=accumulation_steps,
                max_grad_norm=max_grad_norm,
                logprob_chunk_size=logprob_chunk_size,
                use_bf16=use_bf16,
                save_ckpt_freq=save_ckpt_freq,
                advantage_fn=advantage_fn,
                normalize_by_sequence_length=normalize_by_sequence_length,
                val_config=val_config,
                val_freq=val_freq,
            )
        else:
            train_grpo(
                net=net,
                opt=opt,
                env=env,
                reward_fn=reward_fn,
                state_to_str=state_to_str,
                tokenizer=tokenizer,
                eos_token_id=model_info.eos_token_id,
                pad_token_id=model_info.pad_token_id,
                extractor=extract_from_answer_tags,
                beta=beta,
                eps=eps,
                mu=mu,
                max_tokens_generated=max_tokens,
                max_episodes=max_episodes,
                update_ref_net_batch_cadence=update_ref_net_batch_cadence,
                batch_size=batch_size,
                group_size=group_size,
                temperature=temperature,
                accumulation_steps=accumulation_steps,
                max_grad_norm=max_grad_norm,
                logprob_chunk_size=logprob_chunk_size,
                use_bf16=use_bf16,
                save_ckpt_freq=save_ckpt_freq,
                advantage_fn=advantage_fn,
                normalize_by_sequence_length=normalize_by_sequence_length,
                val_config=val_config,
                val_freq=val_freq,
            )
    finally:
        extty.finish()


if _is_modal_installed():
    import modal

    app = modal.App()

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

    train_modal = app.function(
        image=image,
        gpu="A100-80GB",
        timeout=60 * 60 * MODAL_TIMEOUT_HOURS,
        secrets=[modal.Secret.from_dict(extty_env_dict)],
    )(train)


def main():
    parser = argparse.ArgumentParser(description="Verify GRPO learning on Countdown")

    parser.add_argument("--model", type=str, default="qwen3-0.6b")
    parser.add_argument("--device", default=None, help="Device (default: auto-detect)")
    parser.add_argument(
        "--max-episodes", type=int, default=1000, help="Max training episodes"
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--group-size", type=int, default=8, help="Group size for GRPO")
    parser.add_argument("--advantage-fn-type", type=str, default="grpo")
    parser.add_argument(
        "--max-tokens", type=int, default=1024, help="Max tokens to generate"
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument(
        "--beta", type=float, default=0.04, help="KL penalty coefficient"
    )
    parser.add_argument("--eps", type=float, help="clip coefficient")
    parser.add_argument(
        "--n-ops",
        type=lambda s: [int(x) for x in s.split(",")] if "," in s else int(s),
        default=5,
        help="Number of ops for Countdown (comma-separated for multiple configs)",
    )
    parser.add_argument(
        "--n-total",
        type=lambda s: [int(x) for x in s.split(",")] if "," in s else int(s),
        default=6,
        help="Total numbers for Countdown (comma-separated for multiple configs)",
    )
    parser.add_argument(
        "--n-larges",
        type=lambda s: [int(x) for x in s.split(",")] if "," in s else int(s),
        default=2,
        help="Number of large numbers for Countdown (comma-separated for multiple configs)",
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
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--save-ckpt-freq",
        type=int,
        default=sys.maxsize,
        help="How often to checkpoint",
    )

    parser.add_argument("--compile-model", action="store_true", help="compile model")
    parser.add_argument(
        "--soft-tokens",
        action="store_true",
        default=False,
        help="Use soft token generation (train_soft_grpo)",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.33,
        help="Noise scale as a multiplier of the embedding RMS norm (only used with --soft-tokens)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for generation",
    )
    parser.add_argument(
        "--min-soft-steps",
        type=int,
        default=0,
        help="Minimum number of soft-token steps before switching to hard tokens",
    )
    parser.add_argument(
        "--answer-tags-weight",
        type=float,
        default=0.1,
        help="Reward weight for answer tag formatting",
    )
    parser.add_argument(
        "--think-tags-weight",
        type=float,
        default=0.05,
        help="Reward weight for thinking tags formatting",
    )

    parser.add_argument(
        "--normalize-advantages",
        action="store_true",
        default=True,
        help="Normalize advantages by standard deviation",
    )
    parser.add_argument(
        "--no-normalize-advantages",
        dest="normalize_advantages",
        action="store_false",
        help="Do not normalize advantages by standard deviation",
    )
    parser.add_argument("--prompt-collections-id", type=int, required=True)
    parser.add_argument(
        "--normalize-by-sequence-length",
        action="store_true",
        default=True,
        help="Normalize per-token loss by sequence length",
    )
    parser.add_argument(
        "--no-normalize-by-sequence-length",
        dest="normalize_by_sequence_length",
        action="store_false",
        help="Do not normalize per-token loss by sequence length",
    )

    parser.add_argument(
        "--val-freq",
        type=int,
        default=50,
        help="Validate every N steps (0 = disabled)",
    )
    parser.add_argument(
        "--val-episodes",
        type=int,
        default=100,
        help="Episodes per validation env",
    )
    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=4,
        help="Batch size for validation",
    )

    args = parser.parse_args()

    prompt_collection: PromptCollection = PROMPT_COLLECTIONS[args.prompt_collections_id]

    train(
        model_name=args.model,
        device=args.device,
        max_episodes=args.max_episodes,
        batch_size=args.batch_size,
        group_size=args.group_size,
        max_tokens=args.max_tokens,
        lr=args.lr,
        beta=args.beta,
        n_larges=args.n_larges,
        n_ops=args.n_ops,
        n_total=args.n_total,
        mu=args.mu,
        accumulation_steps=args.accumulation_steps,
        update_ref_net_batch_cadence=args.update_ref_net_batch_cadence,
        max_grad_norm=args.max_grad_norm,
        logprob_chunk_size=args.logprob_chunk_size,
        use_bf16=args.use_bf16,
        use_qwen_thinking=args.use_qwen_thinking,
        compile_model=args.compile_model,
        seed=args.seed,
        advantage_fn_type=args.advantage_fn_type,
        soft_tokens=args.soft_tokens,
        noise_std=args.noise_std,
        temperature=args.temperature,
        min_soft_steps=args.min_soft_steps,
        answer_tags_weight=args.answer_tags_weight,
        eps=args.eps,
        normalize_advantages=args.normalize_advantages,
        normalize_by_sequence_length=args.normalize_by_sequence_length,
        system_prompt=prompt_collection.system_prompt,
        env_prompt_template=prompt_collection.env_prompt,
        assistant_prefill=prompt_collection.assistant_prefill,
        think_tags_weight=args.think_tags_weight,
        val_freq=args.val_freq,
        val_episodes=args.val_episodes,
        val_batch_size=args.val_batch_size,
    )


if __name__ == "__main__":
    main()
