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
from dialectic.experiments.params import (
    BackwardParams,
    GRPOParams,
    RewardParams,
    TrainParams,
)
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


def _build_backward_loss_fn(
    *,
    net,
    llm,
    tokenizer,
    model_info,
    backward_params: BackwardParams,
    prompt_collection: PromptCollection,
    group_size: int,
    device,
):
    from vllm import SamplingParams, TokensPrompt

    from dialectic.experiments.envs import Message
    from dialectic.rl.inverse_cot_loss import compute_contrastive_loss
    from dialectic.rl.reward import _evaluate_and_verify_countdown

    backward_system_prompt = backward_params.backward_system_prompt

    def _make_backward_prompt(data) -> str:
        msgs = []
        msgs.append(Message(role="system", content=backward_system_prompt))
        msgs.append(Message(role="user", content=data.prompt))
        return model_info.format_messages(msgs, True)

    def _backward_loss_fn(micro_batch: dict) -> tuple[torch.Tensor, dict]:
        env_responses = micro_batch["env_responses"]
        output_strs = micro_batch["output_strs"]
        batch_size = len(env_responses)

        # Extract correct/incorrect answers from forward rollouts
        backward_prompts_list: list[str] = []
        backward_is_correct: list[bool] = []
        backward_group_boundaries: list[int] = []

        for b in range(batch_size):
            numbers = env_responses[b].data.numbers
            target = env_responses[b].data.target
            correct_answers: set[str] = set()
            incorrect_answers: set[str] = set()

            for g in range(group_size):
                comp_str = output_strs[g][b]
                extracted = extract_from_answer_tags(comp_str)
                if extracted is None:
                    continue
                is_correct = _evaluate_and_verify_countdown(extracted, numbers, target)
                if is_correct:
                    correct_answers.add(extracted)
                else:
                    incorrect_answers.add(extracted)

            if not correct_answers and not incorrect_answers:
                continue

            base_prompt = _make_backward_prompt(env_responses[b].data)
            group_count = 0
            for ans in correct_answers:
                backward_prompts_list.append(base_prompt + f" <answer> {ans} </answer>")
                backward_is_correct.append(True)
                group_count += 1
            for ans in incorrect_answers:
                backward_prompts_list.append(base_prompt + f" <answer> {ans} </answer>")
                backward_is_correct.append(False)
                group_count += 1

            prev = backward_group_boundaries[-1] if backward_group_boundaries else 0
            backward_group_boundaries.append(prev + group_count)

        zero = torch.tensor(0.0, device=device, requires_grad=True)
        if not backward_prompts_list:
            return zero, {"train/backward_loss": 0.0, "train/backward_n_prompts": 0}

        # Generate backward CoTs via vLLM
        tokenizer.no_padding()
        tokenizer.no_truncation()
        encs = tokenizer.encode_batch(backward_prompts_list)
        prompt_token_ids = [list(e.ids) for e in encs]

        sampling_params = SamplingParams(
            n=1,
            temperature=backward_params.backward_temperature,
            max_tokens=500,
            stop_token_ids=[model_info.eos_token_id],
        )
        vllm_outputs = llm.generate(
            [TokensPrompt(prompt_token_ids=ids) for ids in prompt_token_ids],
            sampling_params,
            use_tqdm=False,
        )
        cot_strs = [out.outputs[0].text for out in vllm_outputs]

        # Build backward training sequences
        all_ids: list[list[int]] = []
        all_prefix_lens: list[int] = []
        for i, cot_str in enumerate(cot_strs):
            p_ids = prompt_token_ids[i]
            c_ids = list(tokenizer.encode(cot_str).ids)
            all_ids.append(p_ids + c_ids + [model_info.eos_token_id])
            all_prefix_lens.append(len(p_ids))

        # Pad and build tensors
        max_len = max(len(s) for s in all_ids)
        padded = [s + [model_info.pad_token_id] * (max_len - len(s)) for s in all_ids]
        input_ids = torch.tensor(padded, dtype=torch.long, device=device)
        prefix_lengths = torch.tensor(all_prefix_lens, dtype=torch.long, device=device)
        actual_lengths = torch.tensor(
            [len(s) for s in all_ids], dtype=torch.long, device=device
        )
        positions = torch.arange(max_len, device=device).unsqueeze(0)
        loss_mask = (positions >= prefix_lengths.unsqueeze(1)) & (
            positions < actual_lengths.unsqueeze(1)
        )
        is_correct_t = torch.tensor(
            backward_is_correct, dtype=torch.bool, device=device
        )

        group_sizes = []
        prev = 0
        for boundary in backward_group_boundaries:
            group_sizes.append(boundary - prev)
            prev = boundary

        loss, metrics = compute_contrastive_loss(
            net,
            input_ids,
            prefix_lengths,
            loss_mask,
            is_correct_t,
            group_sizes,
            contrastive_weight=backward_params.contrastive_weight,
            contrastive_margin=backward_params.contrastive_margin,
        )

        scaled_loss = backward_params.backward_weight * loss
        bwd_metrics = {
            "train/backward_loss": scaled_loss.item(),
            "train/backward_nll": metrics["train/nll"],
            "train/backward_contrastive": metrics["train/contrastive_loss"],
            "train/backward_gap": metrics.get("train/contrastive_gap", 0.0),
            "train/backward_n_prompts": len(backward_prompts_list),
        }
        return scaled_loss, bwd_metrics

    return _backward_loss_fn


def _train_grpo(
    *,
    train_params: TrainParams,
    grpo_params: GRPOParams,
    env: Env,
    prompt_collection: PromptCollection,
    reward_fn: RewardFn,
    extractor: Callable[[str], str | None],
    val_envs: list[Env],
    backward_params: BackwardParams | None = None,
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

    bwd_loss_fn = None
    if backward_params is not None and backward_params.backward_weight > 0:
        from dialectic.distributed import unwrap_model

        bwd_loss_fn = _build_backward_loss_fn(
            net=unwrap_model(net),
            llm=llm,
            tokenizer=tokenizer,
            model_info=model_info,
            backward_params=backward_params,
            prompt_collection=prompt_collection,
            group_size=grpo_params.group_size,
            device=device,
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
        backward_loss_fn=bwd_loss_fn,
        llm=llm,
    )
    cleanup()


@extty.experiment(project="grpo-countdown")
def train_grpo_countdown(
    *,
    train_params: TrainParams,
    grpo_params: GRPOParams,
    reward_params: RewardParams,
    backward_params: BackwardParams,
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
        backward_params=backward_params,
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
