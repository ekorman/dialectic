from typing import Callable

import extty
import torch

from dialectic.experiments.arg_parser import Experiment, run_experiments_parser
from dialectic.experiments.launchers._eval_helpers import (
    load_run_config,
    pass_at_k,
    pass_at_ks,
)
from dialectic.experiments.params import GrpoEvalParams
from dialectic.llm.lora import DEFAULT_TARGET_MODULES, apply_lora, merge_lora
from dialectic.llm.registry import MODEL_REGISTRY
from dialectic.llm.vllm_weight_sync import build_vllm_for_training, sync_weights_to_vllm
from dialectic.log import log
from dialectic.rl.extractors import extract_from_answer_tags
from dialectic.rl.inverse_cot_data import load_rollout_artifact
from dialectic.rl.inverse_cot_eval import (
    HARD_PROMPT_THRESHOLD,
    expressions_match,
    gsm8k_match,
)

try:
    from vllm import SamplingParams, TokensPrompt
except ModuleNotFoundError:
    log.error(
        "vllm is required to run `eval_grpo.py`. please make sure dialectic is installed with the vllm extra."
    )


GradeFn = Callable[[str | None, str], bool]


def _bucket_metrics(bucket: list[tuple[int, int]], ks: list[int]) -> dict:
    """``pass_rate`` (per-sample mean) and unbiased ``pass@k`` for a prompt subset.

    ``bucket`` is a list of ``(n_drawn, n_correct)`` per prompt.
    """
    total = len(bucket)
    n_gen = sum(n for n, _ in bucket)
    return {
        "total": total,
        "pass_rate_at_n": sum(c for _, c in bucket) / max(n_gen, 1),
        "pass_at_k": {
            k: sum(pass_at_k(n, c, k) for n, c in bucket) / max(total, 1) for k in ks
        },
    }


def _available_ckpt_steps(ckpt_run: str) -> list[int]:
    """Saved checkpoint steps for ``ckpt_run`` (``project/run_name``), sorted."""
    project, run_name = ckpt_run.split("/")
    run = extty.get_run(project, run_name)
    return sorted(c.step for c in run.checkpoints)


def _resolve_ckpt_steps(spec: str | None, available: list[int]) -> list[int]:
    """Expand a ckpt-step spec into the concrete steps to evaluate.

    Specs (see ``GrpoEvalParams.ckpt_step``): ``None``/``"all"`` → every
    saved checkpoint; ``"a..b..s"`` → ``range(a, b+1, s)``; ``"a..b"`` → every
    saved checkpoint in ``[a, b]``; ``"x,y,z"`` → that list; ``"63000"`` →
    single step. Every spec is intersected with ``available`` (warning on
    requested-but-missing steps); an empty result raises.
    """
    available_set = set(available)
    spec = spec.strip() if spec is not None else None

    if spec is None or spec == "all":
        requested = list(available)
    elif ".." in spec:
        parts = spec.split("..")
        if len(parts) == 3:
            a, b, s = (int(p) for p in parts)
            requested = list(range(a, b + 1, s))
        elif len(parts) == 2:
            a, b = (int(p) for p in parts)
            requested = [x for x in available if a <= x <= b]
        else:
            raise ValueError(
                f"bad ckpt-step range {spec!r} (expected 'a..b' or 'a..b..step')"
            )
    elif "," in spec:
        requested = [int(p) for p in spec.split(",") if p.strip()]
    else:
        requested = [int(spec)]

    steps = [x for x in requested if x in available_set]
    missing = [x for x in requested if x not in available_set]
    if missing:
        log.warning(
            f"{len(missing)} requested step(s) have no saved checkpoint and are "
            f"skipped: {missing[:10]}{'...' if len(missing) > 10 else ''}"
        )
    if not steps:
        raise ValueError(
            f"ckpt-step spec {spec!r} resolved to no available checkpoints "
            f"(available: {available[:10]}{'...' if len(available) > 10 else ''})"
        )
    return steps


def _resolve_lora_targets(target_modules: str) -> tuple[str, ...]:
    if target_modules == "all":
        return DEFAULT_TARGET_MODULES
    if target_modules == "attn":
        return ("q_proj", "k_proj", "v_proj", "o_proj")
    if target_modules == "mlp":
        return ("gate_proj", "up_proj", "down_proj")
    raise ValueError(
        f"Unknown lora_target_modules: {target_modules!r} (expected 'all', 'attn', or 'mlp')"
    )


def _maybe_trace_ckpt_from_rollout_artifact(
    eval_params: GrpoEvalParams,
) -> None:
    """If ``ckpt_run`` is unset but ``p_rollout_artifact`` is given,
    trace ``p_rollout_artifact → producing run → rollout_gen_params.start_ckpt_*``
    to recover the GRPO checkpoint that produced the rollouts.
    """
    if eval_params.ckpt_run is not None:
        return
    if eval_params.p_rollout_artifact is None:
        return
    name = eval_params.p_rollout_artifact
    try:
        meta = extty.get_artifact(name)
    except (KeyError, FileNotFoundError):
        log.warning(f"Rollout artifact {name!r} not found; cannot trace ckpt from it")
        return
    if meta.run_project is None or meta.run_name is None:
        log.warning(
            f"Rollout artifact {name!r} has no producing-run linkage; cannot trace ckpt"
        )
        return
    try:
        rollout_run = extty.get_run(meta.run_project, meta.run_name)
    except FileNotFoundError:
        log.warning(
            f"Rollout's producing run {meta.run_project}/{meta.run_name} not reachable; "
            "cannot trace ckpt"
        )
        return
    rgp = (
        rollout_run.config.get("rollout_gen_params", {})
        if isinstance(rollout_run.config, dict)
        else {}
    )
    eval_params.ckpt_run = rgp.get("start_ckpt_run")
    traced_step = rgp.get("start_ckpt_step")
    # ckpt_step is a spec string; the rollout trace yields a single base step.
    eval_params.ckpt_step = str(traced_step) if traced_step is not None else None
    log.info(
        f"Traced ckpt from p_rollout_artifact: "
        f"{eval_params.ckpt_run} step {eval_params.ckpt_step} "
        f"(via run {meta.run_project}/{meta.run_name})"
    )


def _maybe_auto_derive_from_ckpt(eval_params: GrpoEvalParams) -> None:
    """Pull ``model_name`` / ``use_bf16`` (and p-LoRA, if the ckpt was an
    SFT/inverse-CoT run with LoRA) from the checkpoint's training config.
    No-op if ``ckpt_run`` is ``None`` or unreachable.
    """
    if eval_params.ckpt_run is None:
        return
    fwd_config = load_run_config(eval_params.ckpt_run)
    if fwd_config is None:
        return
    # model_name / use_bf16 live in train_params for GRPO checkpoints but in
    # sft_params for SFT checkpoints (which have no train_params). Check both.
    sections = [
        fwd_config[s]
        for s in ("train_params", "sft_params")
        if isinstance(fwd_config, dict) and isinstance(fwd_config.get(s), dict)
    ]

    def _first(key):
        for sec in sections:
            if sec.get(key) is not None:
                return sec[key]
        return None

    if eval_params.model_name is None:
        eval_params.model_name = _first("model_name")
    if eval_params.use_bf16 is None:
        bf16 = _first("use_bf16")
        if bf16 is not None:
            eval_params.use_bf16 = bool(bf16)

    # p-LoRA: if the ckpt's training run was SFT-with-LoRA or inverse-CoT
    # (neither applies for plain GRPO), pull the LoRA spec. Only fire when
    # the user hasn't set lora_rank explicitly.
    if eval_params.lora_rank is None and isinstance(fwd_config, dict):
        for section_name in ("inverse_cot_params", "sft_params", "train_params"):
            section = fwd_config.get(section_name)
            if isinstance(section, dict) and section.get("lora_rank") is not None:
                eval_params.lora_rank = section.get("lora_rank")
                if eval_params.lora_alpha == 16.0:  # the _LoraMixin default
                    eval_params.lora_alpha = float(section.get("lora_alpha", 16.0))
                if eval_params.lora_target_modules == "all":
                    eval_params.lora_target_modules = section.get(
                        "lora_target_modules", "all"
                    )
                break

    log.info("Auto-derived from ckpt:")
    log.info(f"  model_name:  {eval_params.model_name}")
    log.info(f"  use_bf16:    {eval_params.use_bf16}")
    log.info(
        f"  p LoRA:      rank={eval_params.lora_rank} "
        f"alpha={eval_params.lora_alpha} target={eval_params.lora_target_modules}"
    )


def _validate_grpo_eval_params(eval_params: GrpoEvalParams) -> None:
    if eval_params.p_rollout_artifact is None:
        raise ValueError(
            "`p_rollout_artifact` is required: pass --eval_params.p-rollout-artifact "
            "(the rollout artifact to eval against — also gives us the ckpt "
            "via producing-run metadata)"
        )
    if eval_params.ckpt_run is None:
        raise ValueError(
            "`ckpt_run` could not be resolved. Either pass --eval_params.ckpt-run "
            "or use a p_rollout_artifact whose producing run has "
            "`rollout_gen_params.start_ckpt_run` set."
        )
    # ckpt_step may be None here: it means "evaluate all saved checkpoints of
    # ckpt_run". The concrete step list is resolved in _eval_grpo.
    if eval_params.model_name is None:
        raise ValueError(
            "`model_name` is required (set --eval_params.model-name or pass a "
            "--eval_params.ckpt-run whose train_params carries model_name)"
        )
    if eval_params.use_bf16 is None:
        raise ValueError(
            "`use_bf16` is required (set --eval_params.use-bf16 or pass a "
            "--eval_params.ckpt-run whose train_params carries use_bf16)"
        )


def _eval_grpo(
    *,
    eval_params: GrpoEvalParams,
    grade_fn: GradeFn,
) -> None:
    # Validation/resolution ran in the launcher's resolve_kwargs hook before
    # extty snapshotted the config — at this point the non-step Optional fields
    # are concrete. ``ckpt_step`` may still be None (= "all saved checkpoints").
    # The asserts narrow the types for the type-checker.
    assert eval_params.p_rollout_artifact is not None
    assert eval_params.model_name is not None
    assert eval_params.ckpt_run is not None
    assert eval_params.use_bf16 is not None

    torch.manual_seed(eval_params.seed)
    model_info = MODEL_REGISTRY[eval_params.model_name]
    tokenizer = model_info.load_tokenizer()
    p_project, p_run_name = eval_params.ckpt_run.split("/")

    steps = _resolve_ckpt_steps(
        eval_params.ckpt_step, _available_ckpt_steps(eval_params.ckpt_run)
    )
    log.info(
        f"Evaluating {len(steps)} checkpoint(s) of {eval_params.ckpt_run}: {steps}"
    )

    # ---------------- data (loaded once, shared across checkpoints) -----------
    by_split = load_rollout_artifact(
        eval_params.p_rollout_artifact, tokenizer, filter_train_split=False
    )
    prompts = by_split.get(eval_params.split, [])
    prompts = [pr for pr in prompts if pr.equation is not None]
    if not prompts:
        raise ValueError(
            f"No prompts found for split={eval_params.split} with non-null `equation`"
        )
    if eval_params.max_prompts is not None:
        prompts = prompts[: eval_params.max_prompts]
    log.info(
        f"Evaluating on {len(prompts)} prompts "
        f"(split={eval_params.split}, n_samples={eval_params.n_samples}, "
        f"temperature={eval_params.temperature})"
    )

    sampling_params = SamplingParams(
        n=eval_params.n_samples,
        temperature=eval_params.temperature if eval_params.temperature > 0 else 0.0,
        max_tokens=eval_params.max_tokens_generated,
        stop_token_ids=[model_info.eos_token_id],
        seed=eval_params.seed,
    )
    ks = pass_at_ks(eval_params.n_samples)
    vllm_prompts = [TokensPrompt(prompt_token_ids=pr.prompt_ids) for pr in prompts]

    def _load_ckpt_model(step: int):
        """Load one checkpoint as a plain (LoRA-merged) model on CPU.

        Kept on CPU so it doesn't compete with the persistent vLLM engine for
        VRAM — its weights are pushed into the engine via ``sync_weights_to_vllm``.
        """
        net = model_info.load_net(pretrained_weights=False)
        ckpt = extty.load_checkpoint_from(
            project=p_project, run_name=p_run_name, step=step, load_optimizer=False
        )
        state = ckpt["model_state_dict"]
        for key in ("_rng_torch", "_rng_python", "_rng_cuda"):
            state.pop(key, None)
        if eval_params.lora_rank is not None:
            targets = _resolve_lora_targets(eval_params.lora_target_modules)
            apply_lora(
                net,
                rank=eval_params.lora_rank,
                alpha=eval_params.lora_alpha,
                target_modules=targets,
            )
            net.load_state_dict(state)
            merge_lora(net)
        else:
            net.load_state_dict(state)
        net.requires_grad_(False)
        net.eval()
        return net

    # ---------------- sweep: build engine once, hot-swap weights -------------
    llm = None
    for step in steps:
        log.info(f"=== checkpoint step {step} ===")
        model = _load_ckpt_model(step)
        if llm is None:
            # Build the engine from the first checkpoint; subsequent checkpoints
            # reuse it via sync_weights_to_vllm (enforce_eager makes that safe).
            llm = build_vllm_for_training(
                model,
                tokenizer=tokenizer,
                eos_token_id=model_info.eos_token_id,
                pad_token_id=model_info.pad_token_id,
                max_model_len=eval_params.max_tokens_generated + 1024,
                gpu_memory_utilization=eval_params.gpu_memory_utilization,
                dtype="bfloat16" if eval_params.use_bf16 else "float16",
                seed=eval_params.seed,
            )
        sync_weights_to_vllm(llm, model)
        del model

        vllm_outputs = llm.generate(vllm_prompts, sampling_params, use_tqdm=True)

        # Per prompt: (n drawn, c correct, is_hard, is_all_incorrect). Buckets:
        #   hard          — artifact_rate <= threshold (base solves rarely)
        #   all_incorrect — base NEVER solved it in its rollouts (rate == 0).
        # all_incorrect is the cleanest rescue test: rejection sampling (mix=0)
        # has zero training data for these prompts by construction, so any gain
        # there is pure q. pass@k for every k is estimated from the full n
        # samples via `pass_at_k` — no subsetting.
        counts: list[tuple[int, int, bool, bool]] = []
        for pr, out in zip(prompts, vllm_outputs):
            assert pr.equation is not None
            per_sample_correct = [
                grade_fn(extract_from_answer_tags(sample.text), pr.equation)
                for sample in out.outputs
            ]
            n_correct_artifact = sum(c.is_correct for c in pr.completions)
            artifact_rate = (
                n_correct_artifact / len(pr.completions) if pr.completions else 0.0
            )
            counts.append(
                (
                    len(per_sample_correct),
                    sum(per_sample_correct),
                    artifact_rate <= HARD_PROMPT_THRESHOLD,
                    n_correct_artifact == 0,
                )
            )

        all_b = _bucket_metrics([(n, c) for n, c, _, _ in counts], ks)
        hard_b = _bucket_metrics([(n, c) for n, c, h, _ in counts if h], ks)
        ai_b = _bucket_metrics([(n, c) for n, c, _, ai in counts if ai], ks)

        log.info(
            f"  step {step}: {all_b['total']} prompts, "
            f"{hard_b['total']} hard, {ai_b['total']} all-incorrect"
        )
        for label, b in [("all", all_b), ("hard", hard_b), ("all_incorrect", ai_b)]:
            log.info(
                f"    [{label}] pass_rate_at_n={b['pass_rate_at_n']:.4f}  "
                + "  ".join(f"pass@{k}={b['pass_at_k'][k]:.4f}" for k in ks)
            )

        if extty.has_active_run():
            # step = checkpoint step → extty's native step-axis is the learning
            # curve across checkpoints (no select-then-report).
            metrics: dict = {
                "n_samples": eval_params.n_samples,
                # all prompts (legacy key names preserved)
                "n_prompts": all_b["total"],
                "pass_rate_at_n": all_b["pass_rate_at_n"],
                **{f"pass_at_{k}": all_b["pass_at_k"][k] for k in ks},
                # hard bucket
                "hard_total": hard_b["total"],
                "hard_pass_rate_at_n": hard_b["pass_rate_at_n"],
                **{f"hard_pass_at_{k}": hard_b["pass_at_k"][k] for k in ks},
            }
            # all-incorrect bucket (only when present, to avoid a spurious
            # flat-zero curve on splits with none).
            if ai_b["total"] > 0:
                metrics["all_incorrect_total"] = ai_b["total"]
                metrics["all_incorrect_pass_rate_at_n"] = ai_b["pass_rate_at_n"]
                metrics.update(
                    {f"all_incorrect_pass_at_{k}": ai_b["pass_at_k"][k] for k in ks}
                )
            extty.log(metrics, step=step)

        # Free the ~1.2 GB local checkpoint cache (S3 copy untouched) so a long
        # sweep doesn't fill the disk. Best-effort: never abort the sweep on it.
        try:
            extty.delete_local_checkpoint(p_project, p_run_name, step)
        except Exception as exc:  # noqa: BLE001
            log.warning(f"could not delete local checkpoint {step}: {exc}")


def _countdown_grade(extracted: str | None, gold: str) -> bool:
    return extracted is not None and expressions_match(extracted, gold)


@extty.experiment(project="eval-grpo-countdown")
def eval_grpo_countdown(
    *,
    eval_params: GrpoEvalParams,
) -> None:
    _eval_grpo(eval_params=eval_params, grade_fn=_countdown_grade)


@extty.experiment(project="eval-grpo-gsm8k")
def eval_grpo_gsm8k(
    *,
    eval_params: GrpoEvalParams,
) -> None:
    _eval_grpo(eval_params=eval_params, grade_fn=gsm8k_match)


def _resolve_grpo_eval_kwargs(kwargs: dict) -> None:
    """Pre-launch resolver hook. Trace ckpt from the rollout artifact's
    producing run, then derive model_name/use_bf16/p-LoRA from the ckpt's
    training config. Mutates ``kwargs["eval_params"]`` in place so extty's
    config snapshot reflects the resolved values.
    """
    eval_params: GrpoEvalParams = kwargs["eval_params"]
    _maybe_trace_ckpt_from_rollout_artifact(eval_params)
    _maybe_auto_derive_from_ckpt(eval_params)
    _validate_grpo_eval_params(eval_params)


if __name__ == "__main__":
    run_experiments_parser(
        [
            Experiment(
                env_name="countdown",
                fn=eval_grpo_countdown,
                include_prompt_collection_id=False,
                resolve_kwargs=_resolve_grpo_eval_kwargs,
            ),
            Experiment(
                env_name="gsm8k",
                fn=eval_grpo_gsm8k,
                include_prompt_collection_id=False,
                resolve_kwargs=_resolve_grpo_eval_kwargs,
            ),
        ]
    )
