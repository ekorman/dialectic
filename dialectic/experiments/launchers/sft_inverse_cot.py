import json
import random
import time

import extty
import torch
import torch.nn.functional as F

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
from dialectic.experiments.launchers._eval_helpers import (
    load_run_config,
    rollout_artifact_from_q_config,
)
from dialectic.experiments.models import load_model_and_opt
from dialectic.experiments.params import SftParams
from dialectic.llm.registry import MODEL_REGISTRY
from dialectic.log import log
from dialectic.rl.evaluate import EvaluationResult
from dialectic.rl.inverse_cot_data import load_rollout_artifact
from dialectic.rl.inverse_cot_eval import (
    GradeFn,
    compute_baseline,
    expressions_match,
    gsm8k_match,
)
from dialectic.training import StepFunctionReturn, train_loop


def _read_artifact_jsonl(name: str) -> list[dict]:
    t0 = time.perf_counter()
    data = extty.load_artifact(name, cache=True)
    if not isinstance(data, bytes):
        raise ValueError(f"Expected bytes from artifact {name}, got {type(data)}")
    log.info(f"Downloaded artifact '{name}' in {time.perf_counter() - t0:.1f}s")

    t1 = time.perf_counter()
    entries = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
    log.info(f"Parsed {len(entries)} entries in {time.perf_counter() - t1:.1f}s")
    return entries


def _log_split_summary(
    by_split: dict[str, list[dict]],
    tokenizer,
    *,
    label: str,
) -> None:
    for split, examples in sorted(by_split.items()):
        log.info(f"  [{label}] {split}: {len(examples)} examples")
        if examples:
            ex = examples[0]
            decoded = tokenizer.decode(ex["full_ids"])
            log.info(f"  [{label}] Sample ({split}): {decoded[:500]!r}")
            log.info(
                f"  [{label}] prompt_ids len={len(ex['prompt_ids'])} response_ids len={len(ex['response_ids'])} total={len(ex['full_ids'])}"
            )


def _load_q_cot_artifact(
    name: str,
    tokenizer,
    eos_token_id: int,
) -> dict[str, list[dict]]:
    """Load a q-CoT artifact emitted by ``generate_q_cot.py``.

    Each entry has the schema ``{"prompt_ids", "cot", "answer", "split",
    "fcr_correct"}``. Only entries with ``fcr_correct=True`` are kept —
    we don't want to teach p to imitate CoTs that didn't actually prime
    it to the right answer.
    """

    tokenizer.no_padding()
    tokenizer.no_truncation()
    by_split: dict[str, list[dict]] = {}

    entries = _read_artifact_jsonl(name)
    t0 = time.perf_counter()
    n_dropped = 0
    for entry in entries:
        if not entry.get("fcr_correct", False):
            n_dropped += 1
            continue
        prompt_ids = entry["prompt_ids"]
        cot = entry["cot"]
        answer = entry["answer"]
        split = entry.get("split", "train")

        cot_str = f"<think>\n{cot.strip()}\n</think>\n\n"
        response_str = cot_str + f"<answer> {answer} </answer>"
        response_ids = tokenizer.encode(response_str).ids
        cot_len = len(tokenizer.encode(cot_str).ids)

        full_ids = prompt_ids + response_ids + [eos_token_id]

        by_split.setdefault(split, []).append(
            {
                "prompt_ids": prompt_ids,
                "response_ids": response_ids,
                "full_ids": full_ids,
                "prefix_len": len(prompt_ids),
                "cot_len": cot_len,
            }
        )

    log.info(
        f"q-CoT artifact '{name}': kept {sum(len(v) for v in by_split.values())} entries "
        f"(dropped {n_dropped} with fcr_correct=False) in "
        f"{time.perf_counter() - t0:.1f}s"
    )
    _log_split_summary(by_split, tokenizer, label="q_cot")
    return by_split


def _load_p_rollout_artifact(
    name: str,
    tokenizer,
    eos_token_id: int,
) -> dict[str, list[dict]]:
    """Load a p-rollout artifact emitted by ``generate_inverse_cot_rollouts.py``.

    Each entry has the schema ``{"prompt_str", "completions": [{"cot",
    "answer", "is_correct"}, ...], "split"}``. Only completions with
    ``is_correct=True`` are kept; each correct completion becomes its own
    training example.
    """

    tokenizer.no_padding()
    tokenizer.no_truncation()
    by_split: dict[str, list[dict]] = {}

    entries = _read_artifact_jsonl(name)
    t0 = time.perf_counter()

    prompt_strs: list[str] = []
    response_strs: list[str] = []
    cot_only_strs: list[str] = []
    splits: list[str] = []

    for entry in entries:
        split = entry.get("split", "train")
        for comp in entry["completions"]:
            if not comp["is_correct"]:
                continue
            cot_str = f"<think>\n{comp['cot'].strip()}\n</think>\n\n"
            prompt_strs.append(entry["prompt_str"])
            response_strs.append(cot_str + comp["answer"])
            cot_only_strs.append(cot_str)
            splits.append(split)

    log.info(
        f"Collected {len(prompt_strs)} correct completions in "
        f"{time.perf_counter() - t0:.1f}s"
    )
    t1 = time.perf_counter()
    prompt_encs = tokenizer.encode_batch(prompt_strs)
    response_encs = tokenizer.encode_batch(response_strs)
    cot_encs = tokenizer.encode_batch(cot_only_strs)
    log.info(
        f"Batch tokenized {len(prompt_strs)} pairs in {time.perf_counter() - t1:.1f}s"
    )

    for idx in range(len(prompt_strs)):
        prompt_ids = list(prompt_encs[idx].ids)
        response_ids = list(response_encs[idx].ids)
        cot_len = len(cot_encs[idx].ids)
        full_ids = prompt_ids + response_ids + [eos_token_id]
        by_split.setdefault(splits[idx], []).append(
            {
                "prompt_ids": prompt_ids,
                "response_ids": response_ids,
                "full_ids": full_ids,
                "prefix_len": len(prompt_ids),
                "cot_len": cot_len,
            }
        )

    _log_split_summary(by_split, tokenizer, label="p_rollout")
    return by_split


def _build_sft_batch(
    examples: list[dict], pad_token_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a padded batch from SFT examples.

    Returns (input_ids, loss_mask)
    """
    max_len = max(len(ex["full_ids"]) for ex in examples)

    padded = [
        ex["full_ids"] + [pad_token_id] * (max_len - len(ex["full_ids"]))
        for ex in examples
    ]
    input_ids = torch.tensor(padded, dtype=torch.long, device=device)

    prefix_lens = torch.tensor(
        [ex["prefix_len"] for ex in examples], dtype=torch.long, device=device
    )

    end_lens = torch.tensor(
        [len(ex["full_ids"]) for ex in examples], dtype=torch.long, device=device
    )
    positions = torch.arange(max_len, device=device).unsqueeze(0)
    loss_mask = (positions >= prefix_lens.unsqueeze(1)) & (
        positions < end_lens.unsqueeze(1)
    )

    return input_ids, loss_mask


def _maybe_trace_q_ckpt_from_q_cot_artifact(sft_params: SftParams) -> None:
    """If ``q_cot_artifact`` is set but ``q_ckpt_run`` isn't, look up the
    artifact's producing run (every extty artifact carries the
    ``run_project``/``run_name`` of the run that created it) and pull
    ``q_ckpt_run``/``q_ckpt_step`` from that run's ``gen_params``.

    This collapses the chain ``q_cot_artifact → generate-q-cot run →
    q_ckpt_run → inverse-cot run → forward_ckpt + p_rollout_artifact``
    so the user only needs to point at the q-cot artifact for everything
    downstream to fall into place.
    """
    if sft_params.q_cot_artifact is None or sft_params.q_ckpt_run is not None:
        return
    try:
        meta = extty.get_artifact(sft_params.q_cot_artifact)
    except (KeyError, FileNotFoundError):
        log.warning(
            f"q-cot artifact {sft_params.q_cot_artifact!r} not found; "
            "cannot trace q_ckpt_run from it"
        )
        return
    if meta.run_project is None or meta.run_name is None:
        log.warning(
            f"q-cot artifact {sft_params.q_cot_artifact!r} has no producing-run "
            "linkage; cannot trace q_ckpt_run from it"
        )
        return
    try:
        gen_run = extty.get_run(meta.run_project, meta.run_name)
    except FileNotFoundError:
        log.warning(
            f"q-cot's producing run {meta.run_project}/{meta.run_name} not reachable; "
            "cannot trace q_ckpt_run"
        )
        return
    gen_params = (
        gen_run.config.get("gen_params", {}) if isinstance(gen_run.config, dict) else {}
    )
    sft_params.q_ckpt_run = gen_params.get("q_ckpt_run")
    sft_params.q_ckpt_step = gen_params.get("q_ckpt_step")
    log.info(
        f"Traced q_ckpt from q_cot_artifact: "
        f"{sft_params.q_ckpt_run} step {sft_params.q_ckpt_step} "
        f"(via run {meta.run_project}/{meta.run_name})"
    )


def _maybe_auto_derive_from_q_ckpt(sft_params: SftParams) -> None:
    """When ``q_ckpt_run`` is set, fill in ``start_ckpt_run/step`` and
    ``p_rollout_artifact`` from q's training config — the natural SFT root
    is "the p that q was trained against, trained further on the same
    rollouts q saw."

    Explicit CLI values always win; this only fills holes. If
    ``q_ckpt_run`` is ``None`` or the run isn't reachable, this is a no-op
    and the user is expected to have set those fields themselves (validation
    will catch the omission).
    """
    if sft_params.q_ckpt_run is None:
        return
    q_config = load_run_config(sft_params.q_ckpt_run)
    if q_config is None:
        log.warning(
            f"q run {sft_params.q_ckpt_run!r} not reachable; skipping auto-derivation"
        )
        return

    q_icp = q_config.get("inverse_cot_params", {}) if isinstance(q_config, dict) else {}
    q_train = q_config.get("train_params", {}) if isinstance(q_config, dict) else {}

    if sft_params.model_name is None:
        sft_params.model_name = q_train.get("model_name")
    if sft_params.use_bf16 is None:
        if "use_bf16" in q_train:
            sft_params.use_bf16 = bool(q_train["use_bf16"])

    if sft_params.start_ckpt_run is None:
        sft_params.start_ckpt_run = q_icp.get("forward_ckpt_run")
    if sft_params.start_ckpt_step is None:
        sft_params.start_ckpt_step = q_icp.get("forward_ckpt_step")

    if sft_params.p_rollout_artifact is None:
        sft_params.p_rollout_artifact = rollout_artifact_from_q_config(q_config)

    log.info("Auto-derived from q ckpt:")
    log.info(f"  model_name:         {sft_params.model_name}")
    log.info(f"  use_bf16:           {sft_params.use_bf16}")
    log.info(
        f"  start_ckpt:         {sft_params.start_ckpt_run} step {sft_params.start_ckpt_step}"
    )
    log.info(f"  p_rollout_artifact: {sft_params.p_rollout_artifact}")


def _validate_sft_params(sft_params: SftParams) -> None:
    """Enforce launcher-level invariants before any GPU work happens.

    SFT contract:
    - ``start_ckpt_run`` / ``start_ckpt_step`` must be set. The dataclass
      itself defaults them to ``None`` (inherited from ``_StartCkptMixin``)
      so the same param shape works for launchers that don't need them,
      but SFT always needs a starting checkpoint — training from raw
      pretrained weights isn't the intended use.

    Data slots:
    - At least one of ``q_cot_artifact`` / ``p_rollout_artifact`` must be set.
    - If only ``q_cot_artifact`` is set, ``mix_ratio`` must be 1.0 (no rollout
      data to draw from).
    - If only ``p_rollout_artifact`` is set, ``mix_ratio`` must be 0.0.
    - ``mix_ratio`` must lie in ``[0, 1]``.
    """
    if sft_params.model_name is None:
        raise ValueError(
            "`model_name` is required (set --sft_params.model-name or pass a "
            "reachable --sft_params.q-ckpt-run for auto-derivation)"
        )
    if sft_params.use_bf16 is None:
        raise ValueError(
            "`use_bf16` is required (set --sft_params.use-bf16 / --no-sft_params.use-bf16 "
            "or pass a reachable --sft_params.q-ckpt-run for auto-derivation)"
        )
    if sft_params.start_ckpt_run is None or sft_params.start_ckpt_step is None:
        raise ValueError("`start_ckpt_run` and `start_ckpt_step` are required for SFT")

    has_q = sft_params.q_cot_artifact is not None
    has_p = sft_params.p_rollout_artifact is not None
    if not has_q and not has_p:
        raise ValueError(
            "At least one of `q_cot_artifact` or `p_rollout_artifact` must be set"
        )
    if not (0.0 <= sft_params.mix_ratio <= 1.0):
        raise ValueError(f"mix_ratio must be in [0, 1], got {sft_params.mix_ratio}")
    if not has_q and sft_params.mix_ratio != 0.0:
        raise ValueError(
            "q_cot_artifact is None; mix_ratio must be 0.0 for pure-rollout training"
        )
    if not has_p and sft_params.mix_ratio != 1.0:
        raise ValueError(
            "p_rollout_artifact is None; mix_ratio must be 1.0 for pure-q-CoT training"
        )


def _train_sft_inverse_cot(
    *,
    sft_params: SftParams,
    grade_fn: GradeFn,
    val_from_train_holdout: bool,
):
    """Env-agnostic SFT body. Env-specific behavior enters through two knobs:

    - ``grade_fn``: how the val-accuracy pass grades p's generations.
    - ``val_from_train_holdout``: where training-time validation data comes
      from. Countdown datasets carry a genuine train/val/test split, so the
      artifact's val split is a legitimate validation set (``False``). GSM8K
      only has official train/test — the pipeline loads the test set into
      the "val" slot, so using it for training-time validation would
      contaminate the final holdout; instead a random ``n_val_holdout``
      prompts are split off the train data (``True``).
    """
    init_distributed()
    rank = get_rank()
    world_size = get_world_size()
    torch.manual_seed(sft_params.seed + rank)
    random.seed(sft_params.seed + rank)
    device = get_device()

    batch_size = sft_params.batch_size
    if batch_size % world_size != 0:
        raise ValueError(
            f"batch_size ({batch_size}) must be divisible by world_size ({world_size})"
        )
    local_batch_size = batch_size // world_size

    model_info = MODEL_REGISTRY[sft_params.model_name]
    tokenizer = model_info.load_tokenizer()

    # Load model
    if is_distributed() and not rank == 0:
        barrier()
    p, _ = load_model_and_opt(
        model_name=sft_params.model_name,
        start_ckpt_run=sft_params.start_ckpt_run,
        start_ckpt_step=sft_params.start_ckpt_step,
        device=device,
        use_bf16=sft_params.use_bf16,
        compile_model=sft_params.compile_model,
        load_opt=False,
    )
    if is_distributed() and rank == 0:
        barrier()

    if sft_params.lora_rank is not None:
        from dialectic.llm.lora import (
            DEFAULT_TARGET_MODULES,
            apply_lora,
            freeze_base_params,
        )

        targets = DEFAULT_TARGET_MODULES
        if sft_params.lora_target_modules == "attn":
            targets = ("q_proj", "k_proj", "v_proj", "o_proj")
        elif sft_params.lora_target_modules == "mlp":
            targets = ("gate_proj", "up_proj", "down_proj")

        apply_lora(
            p,
            rank=sft_params.lora_rank,
            alpha=sft_params.lora_alpha,
            dropout=sft_params.lora_dropout,
            target_modules=targets,
        )
        freeze_base_params(p)

    trainable_params = [pa for pa in p.parameters() if pa.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=sft_params.lr)

    scheduler = None
    if sft_params.warmup_steps > 0:
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        warmup = LinearLR(
            opt, start_factor=1e-8, end_factor=1.0, total_iters=sft_params.warmup_steps
        )
        total_steps = sft_params.max_episodes // (
            sft_params.batch_size * sft_params.accumulation_steps
        )
        cosine = CosineAnnealingLR(
            opt, T_max=max(total_steps - sft_params.warmup_steps, 1)
        )
        scheduler = SequentialLR(
            opt, schedulers=[warmup, cosine], milestones=[sft_params.warmup_steps]
        )
        log.info(
            f"LR schedule: warmup {sft_params.warmup_steps} steps, cosine decay over {total_steps} total steps"
        )

    p_raw = p
    if is_distributed():
        p = wrap_ddp(p, get_local_rank())

    total_params = sum(pa.numel() for pa in p_raw.parameters())
    trainable_count = sum(pa.numel() for pa in trainable_params)
    log.info(
        f"p total params: {total_params:,}, trainable: {trainable_count:,} "
        f"(rank {rank}/{world_size}, local_batch_size={local_batch_size})"
    )

    # Load SFT data (rank 0 downloads first, others wait for cache).
    # q_cot_artifact and p_rollout_artifact are independent slots — fill
    # whichever ones are set; combine at batch-construction time via mix_ratio.
    if is_distributed() and rank != 0:
        barrier()

    q_cot_by_split: dict[str, list[dict]] = {}
    if sft_params.q_cot_artifact is not None:
        q_cot_by_split = _load_q_cot_artifact(
            sft_params.q_cot_artifact,
            tokenizer,
            eos_token_id=model_info.eos_token_id,
        )
    p_rollout_by_split: dict[str, list[dict]] = {}
    if sft_params.p_rollout_artifact is not None:
        p_rollout_by_split = _load_p_rollout_artifact(
            sft_params.p_rollout_artifact,
            tokenizer,
            eos_token_id=model_info.eos_token_id,
        )

    train_q_cot = q_cot_by_split.get("train", [])
    train_p_rollout = p_rollout_by_split.get("train", [])

    val_prompts = []
    if val_from_train_holdout:
        # Training-time validation comes from a random holdout of TRAIN
        # prompts — the artifact's "val" slot (GSM8K's official test set) is
        # never consumed during training so it stays clean for final test
        # metrics. The holdout is sampled at the prompt level (not the
        # example level) so a held-out prompt's other completions can't leak
        # into training, and with a rank-independent RNG (no ``+ rank``
        # offset) so every DDP rank excludes the same prompts.
        holdout_rng = random.Random(sft_params.seed)
        if sft_params.p_rollout_artifact is not None:
            # PreTokenizedPrompt objects retain the gold answer and the
            # rollout completions, so the held-out prompts can drive
            # fresh-generation val accuracy (compute_baseline) including
            # the hard-prompt bucket.
            rollout_splits = load_rollout_artifact(
                sft_params.p_rollout_artifact, tokenizer, filter_train_split=False
            )
            rollout_train_prompts = rollout_splits.get("train", [])
            n_holdout = min(sft_params.n_val_holdout, len(rollout_train_prompts))
            val_prompts = (
                holdout_rng.sample(rollout_train_prompts, n_holdout)
                if n_holdout > 0
                else []
            )
            holdout_keys = {tuple(pr.prompt_ids) for pr in val_prompts}
        else:
            # Pure q-CoT mode: no rollout artifact to source gold answers
            # from, so hold out at the q-cot prompt level. Val accuracy is
            # skipped (no equations available); val loss still runs.
            unique_keys = sorted({tuple(ex["prompt_ids"]) for ex in train_q_cot})
            n_holdout = min(sft_params.n_val_holdout, len(unique_keys))
            holdout_keys = (
                set(holdout_rng.sample(unique_keys, n_holdout))
                if n_holdout > 0
                else set()
            )

        def _split_holdout(pool: list[dict]) -> tuple[list[dict], list[dict]]:
            kept: list[dict] = []
            held: list[dict] = []
            for ex in pool:
                (held if tuple(ex["prompt_ids"]) in holdout_keys else kept).append(ex)
            return kept, held

        train_q_cot, val_q_cot = _split_holdout(train_q_cot)
        train_p_rollout, val_p_rollout = _split_holdout(train_p_rollout)
        val_data = val_q_cot + val_p_rollout

        log.info(
            f"Val holdout: {len(holdout_keys)} train prompts held out — "
            f"{len(val_q_cot)} q-cot + {len(val_p_rollout)} rollout examples for "
            f"val loss, {len(val_prompts)} prompts for val accuracy. "
            "Artifact val/test splits untouched (reserved for final test metrics)."
        )
    else:
        # The artifact carries a genuine train/val/test split (countdown):
        # use its val split for training-time validation; test remains the
        # final holdout. Val loss always comes from the rollout val split —
        # q-cot artifacts are train-only by construction (generate_q_cot
        # emits one artifact per split, and synthesis targets train), so
        # they never carry val data.
        val_data = p_rollout_by_split.get("val", [])
        if sft_params.p_rollout_artifact is not None:
            rollout_splits = load_rollout_artifact(
                sft_params.p_rollout_artifact, tokenizer
            )
            val_prompts = rollout_splits.get("val", [])
            log.info(f"Loaded {len(val_prompts)} val prompts for accuracy eval")

    if sft_params.mix_ratio > 0 and not train_q_cot:
        raise ValueError(
            "mix_ratio > 0 but q_cot artifact yielded zero training examples"
        )
    if sft_params.mix_ratio < 1 and not train_p_rollout:
        raise ValueError(
            "mix_ratio < 1 but p_rollout artifact yielded zero training examples"
        )

    log.info(
        f"Training mix: mix_ratio={sft_params.mix_ratio} "
        f"({sft_params.mix_ratio:.0%} q-cot from {len(train_q_cot)} examples, "
        f"{1 - sft_params.mix_ratio:.0%} rollout from {len(train_p_rollout)} examples)"
    )

    if is_distributed() and rank == 0:
        barrier()

    def _train_step(_step_idx: int) -> StepFunctionReturn:
        p.train()
        opt.zero_grad()

        total_loss_val = 0.0
        local_episodes = 0

        for _ in range(sft_params.accumulation_steps):
            # Split the local batch into q-CoT and rollout halves per mix_ratio.
            # Validation guarantees: if mix_ratio == 0 then train_q_cot is empty,
            # if mix_ratio == 1 then train_p_rollout is empty; the active slot
            # always has data.
            n_q = round(local_batch_size * sft_params.mix_ratio)
            n_p = local_batch_size - n_q
            batch_examples: list[dict] = []
            if n_q > 0:
                q_indices = torch.randint(len(train_q_cot), (n_q,)).tolist()
                batch_examples += [train_q_cot[i] for i in q_indices]
            if n_p > 0:
                p_indices = torch.randint(len(train_p_rollout), (n_p,)).tolist()
                batch_examples += [train_p_rollout[i] for i in p_indices]

            input_ids, loss_mask = _build_sft_batch(
                batch_examples, model_info.pad_token_id, device
            )

            logits = p(input_ids, return_all_logits=True)
            shift_logits = logits[:, :-1]
            shift_targets = input_ids[:, 1:]
            shift_mask = loss_mask[:, 1:]

            B, L, V = shift_logits.shape
            per_token_loss = F.cross_entropy(
                shift_logits.reshape(B * L, V),
                shift_targets.reshape(B * L),
                reduction="none",
            ).reshape(B, L)

            masked_loss = per_token_loss * shift_mask
            seq_lengths = shift_mask.sum(dim=1).clamp(min=1)
            nll_loss = (masked_loss.sum(dim=1) / seq_lengths).mean()

            loss = nll_loss

            (loss / sft_params.accumulation_steps).backward()

            total_loss_val += nll_loss.item()
            local_episodes += local_batch_size

        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable_params, sft_params.max_grad_norm
        ).item()
        opt.step()
        if scheduler is not None:
            scheduler.step()

        global_episodes = local_episodes * world_size
        metrics = {
            "train/nll_loss": total_loss_val / sft_params.accumulation_steps,
            "train/grad_norm": grad_norm,
            "train/episodes": global_episodes,
            "train/lr": opt.param_groups[0]["lr"],
        }

        metrics["train/loss"] = total_loss_val / sft_params.accumulation_steps

        return StepFunctionReturn(
            n_episodes_processed=global_episodes,
            metrics=metrics,
        )

    def _val_fn() -> tuple[EvaluationResult, list[extty.Example]]:
        p_raw.eval()

        # Merge LoRA weights for faster generation, restore after
        if sft_params.lora_rank is not None:
            from dialectic.llm.lora import LoRALinear

            lora_state: list[tuple[torch.nn.Module, str, LoRALinear]] = []
            for _, module in p_raw.named_modules():
                for attr_name, child in list(module.named_children()):
                    if isinstance(child, LoRALinear):
                        lora_state.append((module, attr_name, child))
                        merged = child.base
                        merged.weight.data += (
                            child.scaling * child.lora_B.weight @ child.lora_A.weight
                        )
                        setattr(module, attr_name, merged)

        max_val = sft_params.val_episodes or max(len(val_data), len(val_prompts))
        component_means: dict[str, float] = {}

        # Val loss on SFT data (if available)
        if val_data:
            val_examples = val_data[:max_val]
            total_val_loss = 0.0
            n_batches = 0
            for batch_start in range(0, len(val_examples), sft_params.val_batch_size):
                batch = val_examples[
                    batch_start : batch_start + sft_params.val_batch_size
                ]
                input_ids, loss_mask = _build_sft_batch(
                    batch, model_info.pad_token_id, device
                )
                with torch.no_grad():
                    logits = p_raw(input_ids, return_all_logits=True)
                shift_logits = logits[:, :-1]
                shift_targets = input_ids[:, 1:]
                shift_mask = loss_mask[:, 1:]
                n, l, v = shift_logits.shape
                per_token_loss = F.cross_entropy(
                    shift_logits.reshape(n * l, v),
                    shift_targets.reshape(n * l),
                    reduction="none",
                ).reshape(n, l)
                masked = per_token_loss * shift_mask
                seq_lengths = shift_mask.sum(dim=1).clamp(min=1)
                batch_loss = (masked.sum(dim=1) / seq_lengths).mean().item()
                total_val_loss += batch_loss
                n_batches += 1
            component_means["val_loss"] = total_val_loss / max(n_batches, 1)

        # Accuracy: have p generate from scratch on val prompts via vLLM
        examples: list[extty.Example] = []
        if val_prompts:
            eval_prompts = val_prompts[:max_val]
            val_rng_state = torch.random.get_rng_state()
            torch.manual_seed(42)
            baseline_result = compute_baseline(
                p=p_raw,
                prompts=eval_prompts,
                tokenizer=tokenizer,
                eos_token_id=model_info.eos_token_id,
                pad_token_id=model_info.pad_token_id,
                max_tokens_generated=sft_params.max_tokens_generated,
                batch_size=sft_params.val_batch_size,
                grade_fn=grade_fn,
                use_bf16=sft_params.use_bf16,
                temperature=sft_params.temperature,
                n_samples=sft_params.val_pass_at_n,
            )
            n_label = (
                f"pass@{sft_params.val_pass_at_n}"
                if sft_params.val_pass_at_n > 1
                else "accuracy"
            )
            component_means[n_label] = baseline_result.accuracy
            component_means[f"hard_{n_label}"] = baseline_result.hard_accuracy
            torch.random.set_rng_state(val_rng_state)

        n_label = (
            f"pass@{sft_params.val_pass_at_n}"
            if sft_params.val_pass_at_n > 1
            else "accuracy"
        )
        # Restore LoRA layers after val
        if sft_params.lora_rank is not None:
            for module, attr_name, lora_child in lora_state:
                # Undo the merge
                lora_child.base.weight.data -= (
                    lora_child.scaling
                    * lora_child.lora_B.weight
                    @ lora_child.lora_A.weight
                )
                setattr(module, attr_name, lora_child)

        return EvaluationResult(
            n_episodes=max_val,
            reward_mean=component_means.get(
                n_label,
                component_means.get("accuracy", component_means.get("val_loss", 0.0)),
            ),
            reward_std=0.0,
            component_means=component_means,
        ), examples

    has_val = bool(val_data) or bool(val_prompts)
    train_loop(
        max_episodes=sft_params.max_episodes,
        save_ckpt_freq=sft_params.save_ckpt_freq,
        val_freq=sft_params.val_freq,
        net=p,
        opt=opt,
        train_step=_train_step,
        val_dataset_fn=_val_fn if has_val else None,
        val_dataset_label="sft_val",
    )
    cleanup()


def _countdown_grade(extracted: str | None, gold: str) -> bool:
    return extracted is not None and expressions_match(extracted, gold)


@extty.experiment(project="sft-inverse-cot-countdown")
def train_sft_inverse_cot_countdown(
    *,
    sft_params: SftParams,
):
    # Countdown datasets carry a genuine train/val/test split — validate
    # against the artifact's val split; test remains the final holdout.
    _train_sft_inverse_cot(
        sft_params=sft_params,
        grade_fn=_countdown_grade,
        val_from_train_holdout=False,
    )


@extty.experiment(project="sft-inverse-cot-gsm8k")
def train_sft_inverse_cot_gsm8k(
    *,
    sft_params: SftParams,
):
    # GSM8K only has official train/test; the pipeline loads test into the
    # "val" slot, so training-time validation must come from a train holdout
    # to keep the test set clean for final metrics.
    _train_sft_inverse_cot(
        sft_params=sft_params,
        grade_fn=gsm8k_match,
        val_from_train_holdout=True,
    )


def _resolve_sft_kwargs(kwargs: dict) -> None:
    """Pre-launch resolver hook. Runs in ``run_experiments_parser`` before
    the experiment function is invoked, so the resolved values land in
    extty's config snapshot rather than appearing as ``None`` placeholders.
    """
    sft_params: SftParams = kwargs["sft_params"]
    _maybe_trace_q_ckpt_from_q_cot_artifact(sft_params)
    _maybe_auto_derive_from_q_ckpt(sft_params)
    _validate_sft_params(sft_params)


if __name__ == "__main__":
    run_experiments_parser(
        [
            Experiment(
                env_name="countdown",
                fn=train_sft_inverse_cot_countdown,
                include_prompt_collection_id=False,
                resolve_kwargs=_resolve_sft_kwargs,
            ),
            Experiment(
                env_name="gsm8k",
                fn=train_sft_inverse_cot_gsm8k,
                include_prompt_collection_id=False,
                resolve_kwargs=_resolve_sft_kwargs,
            ),
        ]
    )
