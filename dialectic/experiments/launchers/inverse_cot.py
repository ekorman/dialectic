import copy
import random

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
    wrap_ddp,
)
from dialectic.experiments.arg_parser import Experiment, run_experiments_parser
from dialectic.experiments.models import load_model_and_opt
from dialectic.experiments.params import InverseCotParams, TrainParams
from dialectic.experiments.prompts import PromptCollection
from dialectic.llm.generate import generate_hard_tokens
from dialectic.llm.lora import DEFAULT_TARGET_MODULES, apply_lora, freeze_base_params
from dialectic.llm.registry import MODEL_REGISTRY
from dialectic.log import log
from dialectic.rl.evaluate import EvaluationResult
from dialectic.rl.inverse_cot_data import (
    build_contrastive_batch,
    build_shuffled_batch,
    load_rollout_artifact,
    subsample_completions,
)
from dialectic.rl.inverse_cot_eval import compute_fcr
from dialectic.rl.inverse_cot_loss import compute_contrastive_loss, compute_nll_loss
from dialectic.training import StepFunctionReturn, train_loop


def _train_inverse_cot(
    *,
    train_params: TrainParams,
    inverse_cot_params: InverseCotParams,
    prompt_collection: PromptCollection,
):
    """Env-agnostic inverse-CoT training body.

    Consumes the pre-tokenized rollout artifact named by
    ``inverse_cot_params.p_rollout_artifact``; no env-specific code paths in
    the training loop or ``_val_fn``. Rollout artifacts are complete (every
    prompt, regardless of correctness mix); the loader enforces
    ``has_correct AND has_incorrect`` per train-split prompt (see
    ``dialectic/rl/inverse_cot_data.py``) — the contrastive loss needs both
    classes per group — while val passes through unfiltered, keeping the
    all-correct / all-incorrect buckets needed by FCR.
    """
    if inverse_cot_params.p_rollout_artifact is None:
        raise ValueError(
            "`p_rollout_artifact` is required: pass "
            "--inverse_cot_params.p-rollout-artifact"
        )
    init_distributed()
    rank = get_rank()
    world_size = get_world_size()
    # Per-rank seed offset de-correlates `torch.randint` draws so each rank
    # samples a different training sub-batch. Without this, DDP all-reduces
    # identical gradients across ranks and gains nothing from multi-GPU.
    torch.manual_seed(train_params.seed + rank)
    random.seed(train_params.seed + rank)
    device = get_device()
    log.info(f"[rank {rank}/{world_size}] distributed init complete, device={device}")

    batch_size = train_params.batch_size
    if batch_size % world_size != 0:
        raise ValueError(
            f"batch_size ({batch_size}) must be divisible by world_size ({world_size})"
        )
    local_batch_size = batch_size // world_size

    model_info = MODEL_REGISTRY[train_params.model_name]
    tokenizer = model_info.load_tokenizer()

    forward_run = inverse_cot_params.forward_ckpt_run
    forward_step = inverse_cot_params.forward_ckpt_step
    if forward_run is None or forward_step is None:
        raise ValueError(
            "Must specify forward model checkpoint via "
            "--forward-ckpt-run and --forward-ckpt-step"
        )

    # Barrier dance: non-main ranks wait at the first barrier so only rank 0
    # downloads / extracts the checkpoint. Once rank 0 has it on local disk
    # (via the extty cache), all ranks can load it independently without
    # racing on the download.
    if is_distributed() and not rank == 0:
        log.info(f"[rank {rank}] waiting at pre-checkpoint barrier for rank 0 download")
        barrier()
        log.info(f"[rank {rank}] released, loading forward model p from cache")
    else:
        log.info(f"[rank {rank}] loading forward model p (will populate cache)")

    p, _ = load_model_and_opt(
        model_name=train_params.model_name,
        start_ckpt_run=forward_run,
        start_ckpt_step=forward_step,
        device=device,
        use_bf16=train_params.use_bf16,
        compile_model=train_params.compile_model,
        load_opt=False,
    )
    if is_distributed() and rank == 0:
        log.info(f"[rank {rank}] download done, releasing post-checkpoint barrier")
        barrier()
    p.requires_grad_(False)
    p.eval()
    log.info(f"[rank {rank}] forward model p loaded and frozen")

    q = copy.deepcopy(p)
    q.requires_grad_(True)
    q.embed_tokens.requires_grad_(False)

    for layer in q.layers:
        layer.self_attn.causal = False

    q.use_gradient_checkpointing = inverse_cot_params.gradient_checkpointing
    if inverse_cot_params.freeze_lm_head:
        q.lm_head.requires_grad_(False)
    dtype = next(p.parameters()).dtype
    q = q.to(device=device, dtype=dtype)

    # Apply LoRA AFTER `q.to(device, dtype)` so adapter layers inherit the
    # right dtype/device from the base, and BEFORE the optimizer is built
    # so `trainable_params` only collects adapter params. `freeze_base_params`
    # overrides the earlier `requires_grad_` calls — when LoRA is on, only
    # the LoRA A/B matrices train, regardless of `freeze_lm_head` (lm_head
    # isn't a LoRA target so it gets frozen automatically).
    if inverse_cot_params.lora_rank is not None:
        if inverse_cot_params.lora_target_modules == "all":
            lora_targets = DEFAULT_TARGET_MODULES
        elif inverse_cot_params.lora_target_modules == "attn":
            lora_targets = ("q_proj", "k_proj", "v_proj", "o_proj")
        elif inverse_cot_params.lora_target_modules == "mlp":
            lora_targets = ("gate_proj", "up_proj", "down_proj")
        else:
            raise ValueError(
                f"Unknown lora_target_modules: {inverse_cot_params.lora_target_modules!r} "
                "(expected 'all', 'attn', or 'mlp')"
            )
        apply_lora(
            q,
            rank=inverse_cot_params.lora_rank,
            alpha=inverse_cot_params.lora_alpha,
            dropout=inverse_cot_params.lora_dropout,
            target_modules=lora_targets,
        )
        freeze_base_params(q)
        log.info(
            f"[rank {rank}] applied LoRA "
            f"(rank={inverse_cot_params.lora_rank}, "
            f"alpha={inverse_cot_params.lora_alpha}, "
            f"dropout={inverse_cot_params.lora_dropout}, "
            f"target={inverse_cot_params.lora_target_modules})"
        )

    trainable_params = [param for param in q.parameters() if param.requires_grad]
    opt = torch.optim.AdamW(
        trainable_params, lr=train_params.lr, weight_decay=train_params.weight_decay
    )

    start_step = 0
    if train_params.start_ckpt_run is not None:
        if train_params.start_ckpt_step is None:
            raise ValueError("--start-ckpt-step required with --start-ckpt-run")
        project, run_name = train_params.start_ckpt_run.rsplit("/", 1)
        ckpt = extty.load_checkpoint_from(
            project=project, run_name=run_name, step=train_params.start_ckpt_step
        )
        state_dict = ckpt["model_state_dict"]
        rng_torch = state_dict.pop("_rng_torch", None)
        rng_python = state_dict.pop("_rng_python", None)
        rng_cuda = state_dict.pop("_rng_cuda", None)
        q.load_state_dict(state_dict)
        if "optimizer_state_dict" in ckpt:
            log.info(
                f"Loading optimizer from from {train_params.start_ckpt_run} step {start_step}"
            )
            opt.load_state_dict(ckpt["optimizer_state_dict"])
        if rng_torch is not None:
            torch.random.set_rng_state(rng_torch)
        if rng_python is not None:
            random.setstate(rng_python)
        if rng_cuda is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(rng_cuda)
        start_step = train_params.start_ckpt_step
        log.info(f"Resumed q from {train_params.start_ckpt_run} step {start_step}")

    # Capture the bare InverseCotModel BEFORE wrapping in DDP. `q_raw` is
    # used for every forward path inside `_val_fn` (and `compute_fcr`)
    # because val runs only on rank 0 via `train_loop`'s `is_main_process`
    # gate — any forward through the DDP wrapper would trigger a
    # buffer-broadcast collective that non-main ranks (sitting at the
    # post-step `barrier()`) never enter, causing an NCCL hang. Training
    # forwards still go through `q` (the wrapped version) so gradient
    # all-reduce fires correctly on `.backward()`.
    q_raw = q
    if is_distributed():
        log.info(f"[rank {rank}] entering DDP wrap (broadcasts initial q params)")
        q = wrap_ddp(q, get_local_rank())
        log.info(f"[rank {rank}] DDP wrap complete")

    total_params = sum(param.numel() for param in q_raw.parameters())
    trainable_count = sum(param.numel() for param in trainable_params)
    log.info(
        f"q total params: {total_params:,}, trainable: {trainable_count:,} "
        f"(rank {rank}/{world_size}, local_batch_size={local_batch_size})"
    )

    log.info(
        f"[rank {rank}] loading rollout artifact "
        f"{inverse_cot_params.p_rollout_artifact}"
    )
    by_split = load_rollout_artifact(
        inverse_cot_params.p_rollout_artifact,
        tokenizer,
        max_cot_tokens=inverse_cot_params.max_cot_tokens,
        # Ensure post-filter prompts retain enough rollouts for
        # `subsample_completions`'s uniform-K assumption. Default to
        # `train_group_size` if set, otherwise 2 (the minimum for the
        # mixed-correctness constraint).
        min_completions_per_prompt=2,
    )
    rollout_data = by_split.get("train", [])
    val_rollout_data = by_split.get("val", [])
    log.info(
        f"[rank {rank}] rollout data ready: "
        f"train={len(rollout_data)} prompts, val={len(val_rollout_data)} prompts"
    )
    if not rollout_data:
        raise ValueError("No training data found (split='train')")

    val_dataset_label = (
        f"inverse_cot_val_n{len(val_rollout_data[: train_params.val_episodes or len(val_rollout_data)])}"
        if val_rollout_data
        else "inverse_cot_val"
    )

    def _train_step(_step_idx: int) -> StepFunctionReturn:
        q.train()
        opt.zero_grad()

        total_metrics: dict[str, float] = {}
        local_episodes = 0
        max_seq_len = 0

        # Sample all prompts for this step up front and subsample their
        # completions once, so we can length-sort across the full set
        # before splitting into accumulation-step micro-batches. The
        # prefix-LM attention is O(L²) in the batch's max sequence length,
        # so grouping length-homogeneous prompts into the same micro-batch
        # turns a single long-tail outlier from a per-step penalty into a
        # per-chunk penalty — other chunks get tighter padding.
        total_local_prompts = local_batch_size * train_params.accumulation_steps
        indices = torch.randint(len(rollout_data), (total_local_prompts,)).tolist()
        step_prompts = [rollout_data[i] for i in indices]
        if inverse_cot_params.train_group_size is not None:
            step_prompts = [
                subsample_completions(pr, inverse_cot_params.train_group_size)
                for pr in step_prompts
            ]

        def _max_seq_len(pr) -> int:
            # Upper bound on the longest sequence this prompt can emit in
            # `build_infonce_batch`: prompt + (longest answer) + (longest
            # cot) + eos. Conservative when the longest answer and longest
            # cot come from different completions, but correct as an
            # ordering key.
            longest_ans = max(len(c.answer_ids) for c in pr.completions)
            longest_cot = max(len(c.cot_ids) for c in pr.completions)
            return len(pr.prompt_ids) + longest_ans + longest_cot + 1

        step_prompts.sort(key=_max_seq_len)

        for chunk_idx in range(train_params.accumulation_steps):
            batch_prompts = step_prompts[
                chunk_idx * local_batch_size : (chunk_idx + 1) * local_batch_size
            ]

            input_ids, prefix_lengths, loss_mask, is_correct, group_sizes = (
                build_contrastive_batch(
                    batch_prompts,
                    model_info.eos_token_id,
                    model_info.pad_token_id,
                    device,
                )
            )

            loss, step_metrics = compute_contrastive_loss(
                q,
                input_ids,
                prefix_lengths,
                loss_mask,
                is_correct,
                group_sizes,
                contrastive_weight=inverse_cot_params.contrastive_weight,
                contrastive_margin=inverse_cot_params.contrastive_margin,
                logprob_chunk_size=train_params.logprob_chunk_size,
            )
            (loss / train_params.accumulation_steps).backward()

            n, l = input_ids.shape
            max_seq_len = max(max_seq_len, l)
            for k, v in step_metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + v
            local_episodes += local_batch_size

        # `clip_grad_norm_` returns the total grad norm *before* clipping,
        # which is the interesting quantity — watching whether it spikes
        # tells you if the optimizer is hitting instability or the loss
        # landscape is changing character. Matches what GRPO logs.
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable_params, train_params.max_grad_norm
        ).item()
        opt.step()

        metrics = {
            k: v / train_params.accumulation_steps for k, v in total_metrics.items()
        }
        # Report the global episode count (across all ranks) so the
        # `max_episodes` budget in `train_loop` converges at the same
        # wall-clock rate regardless of world_size.
        global_episodes = local_episodes * world_size
        metrics["train/episodes"] = global_episodes
        metrics["train/grad_norm"] = grad_norm
        metrics["train/max_seq_len"] = max_seq_len
        return StepFunctionReturn(n_episodes_processed=global_episodes, metrics=metrics)

    max_val = (
        train_params.val_episodes
        if train_params.val_episodes is not None
        else len(val_rollout_data)
    )

    def _val_fn() -> tuple[EvaluationResult, list[extty.Example]]:
        # Val runs only on rank 0 via `train_loop`'s `is_main_process` gate
        # (the whole val block is inside `if is_main_process() and
        # extty.has_active_run():` in `training.py`). Every forward pass
        # below therefore goes through `q_raw` — the bare InverseCotModel
        # captured BEFORE the DDP wrap — so no DDP collective fires on a
        # rank the other ranks never enter. Non-main ranks sit at the
        # post-step `barrier()` in `train_loop` while rank 0 runs val.
        q_raw.eval()

        # Deterministic val dataset across calls:
        # 1. Prompt selection is sequential over `val_rollout_data[:max_val]`
        #    rather than `torch.randint`-sampled — same prompts in the same
        #    order every val call.
        # 2. `subsample_completions` receives a dedicated `random.Random`
        #    seeded from the training seed, so completion subsampling within
        #    each prompt is also reproducible across val calls.
        # Generation paths (`generate_hard_tokens` for qualitative examples,
        # `compute_fcr`) still use the global torch RNG, so their sampled
        # outputs DO vary across val calls — that's intentional, it shows
        # how q_phi's output distribution evolves during training.
        val_rng = random.Random(train_params.seed + 1_000_000)

        all_nll: list[float] = []
        all_nll_correct: list[float] = []
        all_nll_incorrect: list[float] = []
        all_nll_shuffled: list[float] = []
        examples: list[extty.Example] = []

        n_episodes = 0
        while n_episodes < max_val:
            current_batch = min(train_params.val_batch_size, max_val - n_episodes)
            batch_prompts = val_rollout_data[n_episodes : n_episodes + current_batch]
            if not batch_prompts:
                break
            current_batch = len(batch_prompts)

            if inverse_cot_params.train_group_size is not None:
                batch_prompts = [
                    subsample_completions(
                        pr, inverse_cot_params.train_group_size, rng=val_rng
                    )
                    for pr in batch_prompts
                ]

            input_ids, prefix_lengths, loss_mask, is_correct, group_sizes = (
                build_contrastive_batch(
                    batch_prompts,
                    model_info.eos_token_id,
                    model_info.pad_token_id,
                    device,
                )
            )

            _, _, per_sample_nlls = compute_nll_loss(
                q_raw,
                input_ids,
                prefix_lengths,
                loss_mask,
                inverse_cot_params.normalize_by_sequence_length,
            )
            for sample_nll, correct in zip(per_sample_nlls, is_correct.tolist()):
                if correct:
                    all_nll_correct.append(sample_nll)
                else:
                    all_nll_incorrect.append(sample_nll)

            _, step_metrics = compute_contrastive_loss(
                q_raw,
                input_ids,
                prefix_lengths,
                loss_mask,
                is_correct,
                group_sizes,
                contrastive_weight=inverse_cot_params.contrastive_weight,
                contrastive_margin=inverse_cot_params.contrastive_margin,
                logprob_chunk_size=train_params.logprob_chunk_size,
            )
            all_nll.append(step_metrics["train/nll"])

            # shuffled answer diagnostic
            N = input_ids.shape[0]
            if N > 1:
                s_input_ids, s_prefix_lengths, s_loss_mask = build_shuffled_batch(
                    batch_prompts,
                    model_info.eos_token_id,
                    model_info.pad_token_id,
                    device,
                )
                _, nll_shuffled, _ = compute_nll_loss(
                    q_raw,
                    s_input_ids,
                    s_prefix_lengths,
                    s_loss_mask,
                    inverse_cot_params.normalize_by_sequence_length,
                )
                all_nll_shuffled.append(nll_shuffled)

            # qualitative examples (cap at 20). `build_contrastive_batch`
            # emits exactly `len(prompt.completions)` rows per prompt, so we
            # walk the batch using its returned per-prompt `group_sizes`.
            # The uniform-stride shortcut here used to assume every prompt
            # had `train_group_size` completions, but val splits are
            # unfiltered (artifacts are complete), so all-correct /
            # all-incorrect prompts pass through `subsample_completions`
            # unchanged and break that assumption — hence the per-prompt
            # stride below.
            if len(examples) < 20:
                ex_offset = 0
                for prompt_data, prompt_group_size in zip(batch_prompts, group_sizes):
                    if len(examples) >= 20:
                        break
                    prefix_len = prefix_lengths[ex_offset].item()
                    prefix_ids = input_ids[ex_offset, :prefix_len].unsqueeze(0)

                    q_completion = generate_hard_tokens(
                        net=q_raw,
                        token_ids=prefix_ids,
                        sampling_strategy="sample",
                        temperature=train_params.temperature,
                        eos_token_id=model_info.eos_token_id,
                        pad_token_id=model_info.pad_token_id,
                        max_tokens_generated=train_params.max_tokens_generated,
                        use_kv_cache=True,
                        use_bf16=train_params.use_bf16,
                    ).tokens

                    q_cot = tokenizer.decode(q_completion[0, prefix_len:].tolist())
                    target_cot = tokenizer.decode(prompt_data.completions[0].cot_ids)
                    prompt_str = tokenizer.decode(
                        input_ids[ex_offset, :prefix_len].tolist()
                    )
                    examples.append(
                        extty.Example(
                            prompt=prompt_str,
                            responses=[
                                f"[p's CoT] {target_cot}",
                                f"[q's CoT] {q_cot}",
                            ],
                            rewards=[
                                {"nll": step_metrics["train/nll"]},
                                {"nll": step_metrics["train/nll"]},
                            ],
                        )
                    )
                    ex_offset += prompt_group_size

            n_episodes += current_batch

        nll_correct_mean = (
            sum(all_nll_correct) / len(all_nll_correct) if all_nll_correct else 0.0
        )
        nll_incorrect_mean = (
            sum(all_nll_incorrect) / len(all_nll_incorrect)
            if all_nll_incorrect
            else 0.0
        )
        nll_shuffled_mean = (
            sum(all_nll_shuffled) / len(all_nll_shuffled) if all_nll_shuffled else 0.0
        )

        # Forward Consistency Rate (FCR) — uses `q_raw` for the same reason
        # the other val forwards do: val runs only on rank 0 and must not
        # touch the DDP wrapper.
        fcr_result = compute_fcr(
            p=p,
            q=q_raw,
            prompts=val_rollout_data[:max_val],
            tokenizer=tokenizer,
            eos_token_id=model_info.eos_token_id,
            pad_token_id=model_info.pad_token_id,
            max_tokens_generated=train_params.max_tokens_generated,
            batch_size=train_params.val_batch_size,
            use_bf16=train_params.use_bf16,
            q_temperature=train_params.temperature,
        )

        return EvaluationResult(
            n_episodes=n_episodes,
            reward_mean=fcr_result.fcr,
            reward_std=0.0,
            component_means={
                "nll_correct": nll_correct_mean,
                "nll_incorrect": nll_incorrect_mean,
                "nll_shuffled": nll_shuffled_mean,
                "fcr": fcr_result.fcr,
                "fcr_on_all_incorrect": fcr_result.fcr_on_all_incorrect,
                "n_on_all_incorrect": fcr_result.n_on_all_incorrect,
                "p_baseline": fcr_result.p_baseline,
                "fcr_on_hard": fcr_result.fcr_on_hard,
                "n_on_hard": fcr_result.n_on_hard,
            },
        ), examples

    log.info(
        f"[rank {rank}] entering train_loop "
        f"(start_step={start_step}, max_episodes={train_params.max_episodes}, "
        f"val_freq={train_params.val_freq})"
    )
    train_loop(
        max_episodes=train_params.max_episodes,
        save_ckpt_freq=train_params.save_ckpt_freq,
        val_freq=train_params.val_freq,
        net=q,
        opt=opt,
        train_step=_train_step,
        val_dataset_fn=_val_fn if val_rollout_data else None,
        val_dataset_label=val_dataset_label,
        start_step=start_step,
    )
    cleanup()


@extty.experiment(project="inverse-cot-countdown")
def train_inverse_cot_countdown(
    *,
    train_params: TrainParams,
    inverse_cot_params: InverseCotParams,
    prompt_collection: PromptCollection,
):
    return _train_inverse_cot(
        train_params=train_params,
        inverse_cot_params=inverse_cot_params,
        prompt_collection=prompt_collection,
    )


@extty.experiment(project="inverse-cot-gsm8k")
def train_inverse_cot_gsm8k(
    *,
    train_params: TrainParams,
    inverse_cot_params: InverseCotParams,
    prompt_collection: PromptCollection,
):
    return _train_inverse_cot(
        train_params=train_params,
        inverse_cot_params=inverse_cot_params,
        prompt_collection=prompt_collection,
    )


if __name__ == "__main__":
    run_experiments_parser(
        [
            Experiment(
                env_name="countdown",
                fn=train_inverse_cot_countdown,
                include_prompt_collection_id=True,
            ),
            Experiment(
                env_name="gsm8k",
                fn=train_inverse_cot_gsm8k,
                include_prompt_collection_id=True,
            ),
        ]
    )
