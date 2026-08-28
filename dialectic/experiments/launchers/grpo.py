import random
import sys
from functools import partial
from typing import Callable

import extty
import torch

from dialectic.distributed import (
    barrier,
    cleanup,
    get_device,
    get_local_rank,
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
from dialectic.experiments.params import (
    GRPOParams,
    GSM8kParams,
    RewardParams,
    TrainParams,
)
from dialectic.experiments.prompts import PromptCollection
from dialectic.experiments.reward_fns import (
    get_countdown_reward_fn,
    get_gsm8k_reward_fn,
)
from dialectic.llm.lora import apply_lora, freeze_base_params, resolve_lora_targets
from dialectic.llm.registry import MODEL_REGISTRY
from dialectic.llm.vllm_weight_sync import build_vllm_for_training, sync_weights_to_vllm
from dialectic.log import log
from dialectic.rl.dataset_env import DatasetEnv
from dialectic.rl.env import Env, GSM8kEnv
from dialectic.rl.extractors import extract_from_answer_tags
from dialectic.rl.reward import RewardFn
from dialectic.rl.train import grpo_advantage, train_grpo


def _resolve_val_episodes(val_episodes: int | None, val_envs: list[Env]) -> int:
    if val_episodes is not None:
        return val_episodes
    if val_envs:
        env0 = val_envs[0]
        if hasattr(env0, "problems"):
            return len(env0.problems)
        if hasattr(env0, "data"):
            return len(env0.data)
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

    advantage_fn = partial(grpo_advantage, normalize=grpo_params.normalize_advantages)

    model_info = MODEL_REGISTRY[train_params.model_name]
    use_lora = train_params.lora_rank is not None

    if is_distributed() and not is_main_process():
        barrier()

    # With LoRA enabled, the start checkpoint decides the load path: a LoRA
    # checkpoint (wrapped `.base.` / `lora_` keys) only lines up AFTER
    # `apply_lora`, while a plain checkpoint warm-starts the base model
    # through `load_model_and_opt` as usual. Fetch it up front (inside the
    # rank-0 download barrier) to inspect its keys.
    start_ckpt: dict | None = None
    resume_lora_ckpt = False
    if use_lora and train_params.start_ckpt_run is not None:
        if train_params.start_ckpt_step is None:
            raise ValueError("--start-ckpt-step required with --start-ckpt-run")
        project, run_name = train_params.start_ckpt_run.rsplit("/", 1)
        ckpt: dict = extty.load_checkpoint_from(
            project=project,
            run_name=run_name,
            step=train_params.start_ckpt_step,
        )
        start_ckpt = ckpt
        resume_lora_ckpt = any("lora_" in k for k in ckpt["model_state_dict"])

    net, opt = load_model_and_opt(
        model_name=train_params.model_name,
        start_ckpt_run=None if resume_lora_ckpt else train_params.start_ckpt_run,
        start_ckpt_step=None if resume_lora_ckpt else train_params.start_ckpt_step,
        device=device,
        use_bf16=train_params.use_bf16,
        compile_model=train_params.compile_model,
        load_opt=not use_lora,
        lr=train_params.lr,
        weight_decay=train_params.weight_decay,
    )
    if is_distributed() and is_main_process():
        barrier()
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
    # cost once per run. Built BEFORE any LoRA wrapping: the HF export path
    # reads plain module attributes (e.g. `gate_proj.out_features`) and
    # plain state-dict keys. `lora_B` is zero-initialized, so the base
    # weights loaded here equal the merged weights at step 0.
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

    if use_lora:
        # Applied after `load_model_and_opt` has cast/moved the net so the
        # adapter layers inherit the right dtype/device from their base, and
        # before the optimizer is built so it only collects adapter params.
        assert train_params.lora_rank is not None
        apply_lora(
            net,
            rank=train_params.lora_rank,
            alpha=train_params.lora_alpha,
            dropout=train_params.lora_dropout,
            target_modules=resolve_lora_targets(train_params.lora_target_modules),
        )
        freeze_base_params(net)
        trainable_params = [param for param in net.parameters() if param.requires_grad]
        opt = torch.optim.AdamW(
            trainable_params,
            lr=train_params.lr,
            weight_decay=train_params.weight_decay,
        )
        if start_ckpt is not None and resume_lora_ckpt:
            state_dict = start_ckpt["model_state_dict"]
            rng_torch = state_dict.pop("_rng_torch", None)
            rng_python = state_dict.pop("_rng_python", None)
            rng_cuda = state_dict.pop("_rng_cuda", None)
            net.load_state_dict(state_dict)
            if "optimizer_state_dict" in start_ckpt:
                opt.load_state_dict(start_ckpt["optimizer_state_dict"])
            if rng_torch is not None:
                torch.random.set_rng_state(rng_torch)
            if rng_python is not None:
                random.setstate(rng_python)
            if rng_cuda is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state(rng_cuda)
            log.info(
                f"Resumed LoRA checkpoint from {train_params.start_ckpt_run} "
                f"step {train_params.start_ckpt_step}"
            )
            # The engine was built from base-only weights; push the resumed
            # adapters (merged) so step-1 rollouts run the resumed policy.
            sync_weights_to_vllm(llm, net)
        total_params = sum(param.numel() for param in net.parameters())
        trainable_count = sum(param.numel() for param in trainable_params)
        log.info(
            f"[rank {rank}] applied LoRA "
            f"(rank={train_params.lora_rank}, "
            f"alpha={train_params.lora_alpha}, "
            f"dropout={train_params.lora_dropout}, "
            f"target={train_params.lora_target_modules}); "
            f"total params: {total_params:,}, trainable: {trainable_count:,}"
        )

    if is_distributed():
        net = wrap_ddp(net, get_local_rank())

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


@extty.experiment(project="grpo-gsm8k")
def train_grpo_gsm8k(
    *,
    train_params: TrainParams,
    grpo_params: GRPOParams,
    reward_params: RewardParams,
    prompt_collection: PromptCollection,
    gsm8k_params: GSM8kParams,
):
    init_distributed()
    rank = get_rank()

    env = GSM8kEnv(
        path=gsm8k_params.train_path,
        prompt_template=prompt_collection.env_prompt,
        eval_mode=False,
        seed=train_params.seed + rank * 10_000,
    )
    val_gsm8k_envs = (
        [
            GSM8kEnv(
                path=gsm8k_params.val_path,
                prompt_template=prompt_collection.env_prompt,
                eval_mode=True,
            )
        ]
        if gsm8k_params.val_path
        else []
    )
    val_envs: list[Env] = list(val_gsm8k_envs)
    log.info(
        f"Train: {len(env.data)} problems"
        + (f", Val: {len(val_gsm8k_envs[0].data)}" if val_gsm8k_envs else "")
    )

    reward_fn = get_gsm8k_reward_fn(
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
            Experiment(
                env_name="gsm8k",
                fn=train_grpo_gsm8k,
                include_prompt_collection_id=True,
            ),
        ]
    )
