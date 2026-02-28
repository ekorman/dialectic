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
from dialectic.llm.qwen import load_qwen3_06b, load_qwen3_17b
from dialectic.llm.templates import (
    Message,
    get_llama_input_text_from_messages,
    get_qwen_input_text_from_messages,
)
from dialectic.llm.utils import get_default_device
from dialectic.rl.env import Countdown, CountdownEnv, MazeEnv, MazeState
from dialectic.rl.extractors import extract_from_answer_tags, extract_maze_moves
from dialectic.rl.maze import MazeConfig
from dialectic.rl.reward import (
    answer_tags,
    countdown_correct,
    maze_correct,
    maze_distance,
    maze_validity,
    think_tags,
    weighted_reward,
)
from dialectic.rl.train import (
    ValidationConfig,
    grpo_advantage,
    rloo_advantage,
    train_grpo,
    train_internal_reasoning_grpo,
    train_internal_reasoning_sft,
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

    def load_net(self, **kwargs):
        return self.net_factory(**kwargs)

    def load_tokenizer(self) -> Tokenizer:
        if isinstance(self.tokenizer, str):
            return Tokenizer.from_pretrained(self.tokenizer)
        return Tokenizer.from_file(str(get_artifact(self.tokenizer)[0]))


MODEL_REGISTRY: dict[str, ModelInfo] = {
    "qwen3-0.6b": ModelInfo(
        net_factory=lambda **kw: load_qwen3_06b(True, **kw),
        tokenizer="Qwen/Qwen3-0.6B",
        eos_token_id=151645,
        pad_token_id=151643,
        format_messages=lambda msgs, gen: get_qwen_input_text_from_messages(
            msgs, gen, enable_thinking=False
        ),
    ),
    "qwen3-1.7b": ModelInfo(
        net_factory=lambda **kw: load_qwen3_17b(True, **kw),
        tokenizer="Qwen/Qwen3-1.7B",
        eos_token_id=151645,
        pad_token_id=151643,
        format_messages=lambda msgs, gen: get_qwen_input_text_from_messages(
            msgs, gen, enable_thinking=False
        ),
    ),
    "llama-3.2-1b-instruct": ModelInfo(
        net_factory=lambda **kw: load_llama_32_1b_instruct(True, **kw),
        tokenizer=LLAMA_32_TOKENIZER,
        eos_token_id=128009,
        pad_token_id=128009,
        format_messages=lambda msgs, gen: get_llama_input_text_from_messages(msgs, gen),
    ),
    "llama-3.2-3b-instruct": ModelInfo(
        net_factory=lambda **kw: load_llama_32_3b_instruct(True, **kw),
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


MAZE_INTERNAL_REASONING_PROMPT = PromptCollection(
    system_prompt=None,
    env_prompt=(
        "Navigate the maze from Start to Goal. "
        "Each line shows a cell and the directions you can move from it.\n\n"
        "{maze}"
    ),
    assistant_prefill=None,
)

PROMPT_COLLECTIONS: dict[str, list[PromptCollection]] = {
    "countdown": [
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
    ],
    "maze": [
        PromptCollection(
            system_prompt=None,
            env_prompt=(
                "Navigate the maze from Start to Goal. "
                "Each line shows a cell and the directions you can move from it.\n\n"
                "{maze}\n\n"
                f"Show your reasoning in <{REASONING_TAG}></{REASONING_TAG}> tags. "
                "Put your moves in <answer></answer> tags as a comma-separated list, "
                "for example <answer>right, down, right, down</answer>."
            ),
            assistant_prefill=f"Let me solve this step by step\n<{REASONING_TAG}>",
        ),
        PromptCollection(
            system_prompt=STHT_SYSTEM_PROMPT,
            env_prompt=(
                "Navigate the maze from Start to Goal. "
                "Each line shows a cell and the directions you can move from it.\n\n"
                "{maze}\n\n"
                "Put your moves in <answer></answer> tags as a comma-separated list, "
                "for example <answer>right, down, right, down</answer>."
            ),
            assistant_prefill=None,
        ),
        PromptCollection(
            system_prompt=SIMPLE_SYSTEM_PROMPT,
            env_prompt=(
                "Navigate the maze from Start to Goal. "
                "Each line shows a cell and the directions you can move from it.\n\n"
                "{maze}\n\n"
                "Put your moves in <answer></answer> tags as a comma-separated list, "
                "for example <answer>right, down, right, down</answer>."
            ),
            assistant_prefill="Let me solve this step by step.",
        ),
    ],
}


def _is_modal_installed():
    return importlib.util.find_spec("modal") is not None


def get_state_to_str(
    *,
    format_messages: Callable[[list[Message], bool], str],
    system_prompt: str | None = None,
    assistant_prefill: str | None = None,
):
    def _state_to_str(data: Countdown | MazeState) -> str:
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


def get_maze_reward_fn(
    answer_tags_weight: float,
    validity_weight: float,
    distance_weight: float,
    think_tags_weight: float,
):
    components = [
        ("correct", 1.0, maze_correct),
        ("distance", distance_weight, maze_distance),
        ("validity", validity_weight, maze_validity),
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
    return weighted_reward(components)


# update prompt? especially for soft tokens using <reasoning> tags don't make sense


MODAL_TIMEOUT_HOURS = int(os.getenv("MODAL_TIMEOUT_HOURS", 1))


@extty.experiment(project="hybrid-reasoning")
def train(
    *,
    env_type: Literal["countdown", "maze"] = "countdown",
    model_name: str = "qwen3-0.6b",
    device: str | None = None,
    max_episodes: int = 1000,
    eps: float | None,
    batch_size: int = 2,
    group_size: int = 8,
    max_tokens: int = 700,
    lr: float = 1e-5,
    beta: float = 0.04,
    # countdown params
    n_larges: int | list[int] = 2,
    n_total: int | list[int] = 6,
    n_ops: int | list[int] = 5,
    # maze params
    maze_height: int = 5,
    maze_width: int = 5,
    maze_openness: float = 0.0,
    maze_min_solution_length: int | None = None,
    maze_max_solution_length: int | None = None,
    maze_start_pos: str = "top_left",
    maze_goal_pos: str = "bottom_right",
    maze_validity_weight: float = 0.0,
    maze_distance_weight: float = 0.5,
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
    normalize_soft_pdf_by_dim: bool,
    val_freq: int = 0,
    val_episodes: int = 50,
    val_batch_size: int = 4,
    internal_reasoning: bool = False,
    soft_block_size: int = 4,
    soft_bptt_window: int | None = None,
    max_cycles: int = 30,
    soft_projection: bool = False,
    soft_projection_alpha_init: float = 1e-3,
    soft_projection_rank: int | None = None,
    sft: bool = False,
    think_token: str | None = None,
):
    assert model_name in MODEL_REGISTRY
    torch.manual_seed(seed)

    model_info = MODEL_REGISTRY[model_name]
    net = model_info.load_net(
        soft_projection=soft_projection,
        soft_projection_alpha_init=soft_projection_alpha_init,
        soft_projection_rank=soft_projection_rank,
    )
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

    if env_type == "countdown":
        env = CountdownEnv(
            seed=seed,
            n_larges=n_larges,
            n_total=n_total,
            n_ops=n_ops,
            prompt_template=env_prompt_template,
        )
        reward_fn = get_reward_fn(
            answer_tags_weight=answer_tags_weight, think_tags_weight=think_tags_weight
        )
        extractor = extract_from_answer_tags
    elif env_type == "maze":
        maze_config = MazeConfig(
            height=maze_height,
            width=maze_width,
            openness=maze_openness,
            min_solution_length=maze_min_solution_length,
            max_solution_length=maze_max_solution_length,
            start_pos=maze_start_pos,
            goal_pos=maze_goal_pos,
        )
        maze_prompt = (
            MAZE_INTERNAL_REASONING_PROMPT.env_prompt
            if internal_reasoning
            else env_prompt_template
        )
        env = MazeEnv(config=maze_config, prompt_template=maze_prompt, seed=seed)
        reward_fn = get_maze_reward_fn(
            answer_tags_weight=answer_tags_weight,
            validity_weight=maze_validity_weight,
            distance_weight=maze_distance_weight,
            think_tags_weight=think_tags_weight,
        )
        extractor = extract_maze_moves
    else:
        raise ValueError(f"Unknown env_type: {env_type}")

    device = device or get_default_device()
    print(f"device: {device}")
    net = net.to(device)

    if internal_reasoning:
        if env_type != "maze":
            raise ValueError("--internal-reasoning only supported with --env maze")
        state_to_str = get_state_to_str(
            format_messages=format_messages,
            system_prompt=MAZE_INTERNAL_REASONING_PROMPT.system_prompt,
            assistant_prefill=MAZE_INTERNAL_REASONING_PROMPT.assistant_prefill,
        )
    else:
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

    valid_hard_token_ids: list[int] = []
    move_id_to_name: dict[int, str] = {}
    ir_reward_fn = reward_fn
    ir_extractor = extractor
    if internal_reasoning:
        direction_words = ["up", "down", "left", "right"]
        for word in direction_words:
            ids = tokenizer.encode(word, add_special_tokens=False).ids
            assert len(ids) == 1, f"'{word}' tokenizes to {len(ids)} tokens, expected 1"
            tid = ids[0]
            valid_hard_token_ids.append(tid)
            move_id_to_name[tid] = word
        valid_hard_token_ids.append(model_info.eos_token_id)

        move_name_to_id: dict[str, int] = {v: k for k, v in move_id_to_name.items()}

        think_token_id: int | None = None
        if think_token:
            ids = tokenizer.encode(think_token, add_special_tokens=False).ids
            assert len(ids) == 1, (
                f"'{think_token}' must be a single token, got {len(ids)}"
            )
            think_token_id = ids[0]

        ir_reward_fn = get_maze_reward_fn(
            answer_tags_weight=0.0,
            validity_weight=maze_validity_weight,
            distance_weight=maze_distance_weight,
            think_tags_weight=0.0,
        )
        ir_extractor = lambda moves: moves if moves else None  # noqa: E731

    val_config: ValidationConfig | None = None
    if val_freq > 0:
        if env_type == "countdown":
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
        else:
            val_maze_prompt = (
                MAZE_INTERNAL_REASONING_PROMPT.env_prompt
                if internal_reasoning
                else env_prompt_template
            )
            val_envs = [
                MazeEnv(config=maze_config, prompt_template=val_maze_prompt, seed=2026)
            ]
        if internal_reasoning:
            val_config = ValidationConfig(
                envs=val_envs,
                reward_fn=ir_reward_fn,
                state_to_str=state_to_str,
                extractor=ir_extractor,
                tokenizer=tokenizer,
                eos_token_id=model_info.eos_token_id,
                pad_token_id=model_info.pad_token_id,
                max_episodes=val_episodes,
                batch_size=val_batch_size,
                max_tokens_generated=max_tokens,
                use_bf16=use_bf16,
                internal_reasoning=True,
                valid_hard_token_ids=valid_hard_token_ids,
                move_id_to_name=move_id_to_name,
                soft_block_size=soft_block_size,
                max_cycles=max_cycles,
                think_token_id=think_token_id,
            )
        else:
            val_config = ValidationConfig(
                envs=val_envs,
                reward_fn=reward_fn,
                state_to_str=state_to_str,
                extractor=extractor,
                tokenizer=tokenizer,
                eos_token_id=model_info.eos_token_id,
                pad_token_id=model_info.pad_token_id,
                max_episodes=val_episodes,
                batch_size=val_batch_size,
                max_tokens_generated=max_tokens,
                use_bf16=use_bf16,
            )

    try:
        if sft:
            assert internal_reasoning and env_type == "maze", (
                "--sft requires --internal-reasoning and --env maze"
            )
            train_internal_reasoning_sft(
                net=net,
                opt=opt,
                env=env,
                state_to_str=state_to_str,
                tokenizer=tokenizer,
                pad_token_id=model_info.pad_token_id,
                eos_token_id=model_info.eos_token_id,
                move_name_to_id=move_name_to_id,
                valid_hard_token_ids=valid_hard_token_ids,
                move_id_to_name=move_id_to_name,
                soft_block_size=soft_block_size,
                soft_bptt_window=soft_bptt_window,
                max_cycles=max_cycles,
                max_episodes=max_episodes,
                batch_size=batch_size,
                accumulation_steps=accumulation_steps,
                max_grad_norm=max_grad_norm,
                normalize_by_sequence_length=normalize_by_sequence_length,
                use_bf16=use_bf16,
                save_ckpt_freq=save_ckpt_freq,
                val_config=val_config,
                val_freq=val_freq,
                think_token_id=think_token_id,
            )
        elif internal_reasoning:
            train_internal_reasoning_grpo(
                net=net,
                opt=opt,
                env=env,
                reward_fn=ir_reward_fn,
                state_to_str=state_to_str,
                tokenizer=tokenizer,
                eos_token_id=model_info.eos_token_id,
                pad_token_id=model_info.pad_token_id,
                extractor=ir_extractor,
                move_id_to_name=move_id_to_name,
                valid_hard_token_ids=valid_hard_token_ids,
                soft_block_size=soft_block_size,
                soft_bptt_window=soft_bptt_window,
                max_cycles=max_cycles,
                beta=beta,
                eps=eps,
                mu=mu,
                max_episodes=max_episodes,
                update_ref_net_batch_cadence=update_ref_net_batch_cadence,
                batch_size=batch_size,
                group_size=group_size,
                temperature=temperature,
                accumulation_steps=accumulation_steps,
                max_grad_norm=max_grad_norm,
                use_bf16=use_bf16,
                save_ckpt_freq=save_ckpt_freq,
                advantage_fn=advantage_fn,
                normalize_by_sequence_length=normalize_by_sequence_length,
                val_config=val_config,
                val_freq=val_freq,
                think_token_id=think_token_id,
            )
        elif soft_tokens:
            train_soft_grpo(
                net=net,
                opt=opt,
                env=env,
                reward_fn=reward_fn,
                state_to_str=state_to_str,
                tokenizer=tokenizer,
                eos_token_id=model_info.eos_token_id,
                pad_token_id=model_info.pad_token_id,
                extractor=extractor,
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
                normalize_soft_pdf_by_dim=normalize_soft_pdf_by_dim,
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
                extractor=extractor,
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
    parser = argparse.ArgumentParser(description="GRPO training on Countdown or Maze")

    parser.add_argument(
        "--env",
        type=str,
        default="countdown",
        choices=["countdown", "maze"],
        help="Environment to train on",
    )
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
    # maze params
    parser.add_argument("--maze-height", type=int, default=5, help="Maze grid height")
    parser.add_argument("--maze-width", type=int, default=5, help="Maze grid width")
    parser.add_argument(
        "--maze-openness",
        type=float,
        default=0.0,
        help="Fraction of extra walls to remove (0.0=perfect maze)",
    )
    parser.add_argument("--maze-min-solution-length", type=int, default=None)
    parser.add_argument("--maze-max-solution-length", type=int, default=None)
    parser.add_argument("--maze-start-pos", type=str, default="top_left")
    parser.add_argument("--maze-goal-pos", type=str, default="bottom_right")
    parser.add_argument(
        "--maze-validity-weight",
        type=float,
        default=0.0,
        help="Reward weight for move validity",
    )
    parser.add_argument(
        "--maze-distance-weight",
        type=float,
        default=0.5,
        help="Reward weight for proximity to goal",
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
        "--internal-reasoning",
        action="store_true",
        default=False,
        help="Use internal reasoning with fixed soft/hard token interleaving (maze only)",
    )
    parser.add_argument(
        "--soft-block-size",
        type=int,
        default=4,
        help="Soft tokens per cycle for internal reasoning",
    )
    parser.add_argument(
        "--soft-bptt-window",
        type=int,
        default=None,
        help="Number of soft tokens to backprop through per cycle (default: all)",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=30,
        help="Maximum reasoning cycles for internal reasoning",
    )
    parser.add_argument(
        "--soft-projection",
        action="store_true",
        default=False,
        help="Add a learnable projection for soft token hidden states",
    )
    parser.add_argument(
        "--soft-projection-alpha-init",
        type=float,
        default=1e-3,
        help="Initial value for soft projection alpha parameter (default: 1e-3)",
    )
    parser.add_argument(
        "--soft-projection-rank",
        type=int,
        default=None,
        help="Low-rank factorization for soft projection (D->r->D). Requires --soft-projection.",
    )
    parser.add_argument(
        "--sft",
        action="store_true",
        default=False,
        help="Supervised fine-tuning with BFS ground-truth (requires --internal-reasoning --env maze)",
    )
    parser.add_argument(
        "--think-token",
        type=str,
        default=None,
        help="Use discrete think tokens instead of soft tokens (e.g., 'wait'). "
        "Requires --internal-reasoning.",
    )
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
    parser.add_argument("--prompt-collections-id", type=int, default=0)
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
        "--normalize-soft-pdf-by-dim",
        action="store_true",
        default=False,
        help="Normalize soft token probs by embedding dimension",
    )
    parser.add_argument(
        "--no-normalize-soft-pdf-by-dim",
        dest="normalize_soft_pdf_by_dim",
        action="store_false",
        help="Do not normalize soft token probs by embedding dimension",
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

    prompt_collection = PROMPT_COLLECTIONS[args.env][args.prompt_collections_id]

    train(
        env_type=args.env,
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
        maze_height=args.maze_height,
        maze_width=args.maze_width,
        maze_openness=args.maze_openness,
        maze_min_solution_length=args.maze_min_solution_length,
        maze_max_solution_length=args.maze_max_solution_length,
        maze_start_pos=args.maze_start_pos,
        maze_goal_pos=args.maze_goal_pos,
        maze_validity_weight=args.maze_validity_weight,
        maze_distance_weight=args.maze_distance_weight,
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
        normalize_soft_pdf_by_dim=args.normalize_soft_pdf_by_dim,
        internal_reasoning=args.internal_reasoning,
        soft_block_size=args.soft_block_size,
        soft_bptt_window=args.soft_bptt_window,
        max_cycles=args.max_cycles,
        soft_projection=args.soft_projection,
        soft_projection_alpha_init=args.soft_projection_alpha_init,
        soft_projection_rank=args.soft_projection_rank,
        sft=args.sft,
        think_token=args.think_token,
    )


if __name__ == "__main__":
    main()
