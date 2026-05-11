import sys
from functools import partial
from typing import Callable

import extty
import torch

from dialectic.distributed import (
    barrier,
    cleanup,
    get_device,
    get_rank,
    get_world_size,
    init_distributed,
    is_distributed,
    is_main_process,
    wrap_ddp,
)
from dialectic.experiments.arg_parser import Experiment, run_experiments_parser
from dialectic.experiments.envs import (
    get_state_to_str,
    load_countdown_dataset_artifacts,
)
from dialectic.experiments.models import load_model_and_opt
from dialectic.experiments.params import GRPOParams, RewardParams, TrainParams
from dialectic.experiments.prompts import PromptCollection
from dialectic.experiments.reward_fns import get_countdown_reward_fn
from dialectic.llm.registry import MODEL_REGISTRY
from dialectic.llm.vllm_weight_sync import build_vllm_for_training
from dialectic.log import log
from dialectic.rl.dataset_env import DatasetEnv
from dialectic.rl.env import Env
from dialectic.rl.extractors import extract_from_answer_tags
from dialectic.rl.reward import RewardFn
from dialectic.rl.train import grpo_advantage, rloo_advantage, train_grpo


def _resolve_val_episodes(val_episodes: int | None, val_envs: list[Env]) -> int:
    if val_episodes is not None:
        return val_episodes
    if val_envs and hasattr(val_envs[0], "problems"):
        return len(val_envs[0].problems)
    return sys.maxsize


def _train_grpo(
    *,
    train_params: TrainParams,
    grpo_params: GRPOParams,
    env: Env,
    prompt_collection: PromptCollection,
    reward_fn: RewardFn,
    extractor: Callable[[str], str | None],
    val_envs: list[Env],
):
    init_distributed()
    rank = get_rank()
    world_size = get_world_size()
    torch.manual_seed(train_params.seed + rank)
    device = get_device()

    batch_size = train_params.batch_size
    if batch_size % world_size != 0:
        raise ValueError(
            f"batch_size ({batch_size}) must be divisible by world_size ({world_size})"
        )
    local_batch_size = batch_size // world_size

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

    model_info = MODEL_REGISTRY[train_params.model_name]
    if is_distributed() and not is_main_process():
        barrier()
    net, opt = load_model_and_opt(
        model_name=train_params.model_name,
        device=device,
        use_bf16=train_params.use_bf16,
        compile_model=train_params.compile_model,
        load_opt=True,
        lr=train_params.lr,
    )
    if is_distributed():
        if is_main_process():
            barrier()
        from dialectic.distributed import get_local_rank

        net = wrap_ddp(net, get_local_rank())
    format_messages = model_info.format_messages

    state_to_str = get_state_to_str(
        format_messages=format_messages,
        system_prompt=prompt_collection.system_prompt,
        assistant_prefill=prompt_collection.assistant_prefill,
    )
    tokenizer = model_info.load_tokenizer()

    # Build the vLLM sampler engine once. Each rank gets its own engine on
    # its own LOCAL_RANK GPU; weights get pushed in-place after every
    # optimizer step via `sync_weights_to_vllm`, so we only pay engine init
    # cost once per run.
    vllm_max_model_len = train_params.max_tokens_generated + 1024
    llm = build_vllm_for_training(
        net,
        tokenizer=tokenizer,
        eos_token_id=model_info.eos_token_id,
        pad_token_id=model_info.pad_token_id,
        max_model_len=vllm_max_model_len,
        gpu_memory_utilization=0.3,
        dtype="bfloat16" if train_params.use_bf16 else "float16",
        seed=train_params.seed + rank,
    )

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
        beta=grpo_params.beta,
        eps=grpo_params.eps,
        mu=grpo_params.mu,
        max_tokens_generated=train_params.max_tokens_generated,
        max_episodes=train_params.max_episodes,
        update_ref_net_batch_cadence=grpo_params.update_ref_net_batch_cadence,
        batch_size=local_batch_size,
        group_size=grpo_params.group_size,
        temperature=train_params.temperature,
        advantage_fn=advantage_fn,
        normalize_by_sequence_length=grpo_params.normalize_by_sequence_length,
        accumulation_steps=train_params.accumulation_steps,
        max_grad_norm=train_params.max_grad_norm,
        logprob_chunk_size=train_params.logprob_chunk_size,
        use_bf16=train_params.use_bf16,
        save_ckpt_freq=train_params.save_ckpt_freq,
        val_batch_size=train_params.val_batch_size,
        val_episodes=_resolve_val_episodes(train_params.val_episodes, val_envs),
        val_freq=train_params.val_freq,
        val_envs=val_envs,
        warmup_steps=train_params.warmup_steps,
        llm=llm,
    )
    cleanup()


@extty.experiment(project="grpo-countdown")
def train_grpo_countdown(
    *,
    train_params: TrainParams,
    grpo_params: GRPOParams,
    reward_params: RewardParams,
    prompt_collection: PromptCollection,
    dataset_artifacts: list[str],
):
    init_distributed()
    rank = get_rank()

    all_problems = load_countdown_dataset_artifacts(
        dataset_artifacts, prompt_template=prompt_collection.env_prompt
    )
    train_problems = [
        resp for resp, extra in all_problems if extra.get("split") == "train"
    ]
    val_problems = [resp for resp, extra in all_problems if extra.get("split") == "val"]

    if not train_problems:
        raise ValueError("No training problems found (split='train')")
    log.info(f"Train: {len(train_problems)}, Val: {len(val_problems)}")

    env: Env = DatasetEnv(
        train_problems,
        seed=train_params.seed + rank * 10_000,
        label="countdown_train",
    )
    val_envs: list[Env] = (
        [DatasetEnv(val_problems, seed=2026, label="countdown_val")]
        if val_problems
        else []
    )

    reward_fn = get_countdown_reward_fn(
        answer_tags_weight=reward_params.answer_tags_weight,
        think_tags_weight=reward_params.think_tags_weight,
    )

    return _train_grpo(
        train_params=train_params,
        grpo_params=grpo_params,
        env=env,
        prompt_collection=prompt_collection,
        reward_fn=reward_fn,
        extractor=extract_from_answer_tags,
        val_envs=val_envs,
    )


if __name__ == "__main__":
    run_experiments_parser(
        [
            Experiment(
                env_name="countdown",
                fn=train_grpo_countdown,
                include_prompt_collection_id=True,
                include_dataset_glob=True,
            ),
        ]
    )
