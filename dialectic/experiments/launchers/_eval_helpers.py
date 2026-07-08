"""Shared eval-launcher helpers: param resolution + LoRA loading.

Both ``eval_inverse_cot.py`` and ``eval_q_verifier.py`` need to:
1. Auto-derive most ``InverseCotEvalParams`` fields from the q run's extty
   config (so the CLI minimal-invocation is just ``--q-ckpt-run`` + step).
2. Optionally apply + merge LoRA adapters when loading checkpoints so the
   resulting model is a plain transformer (no LoRA wrapper layers).

Both ops are pure and have no env-specific logic, so they live here.
"""

from dataclasses import dataclass
from typing import Any

import extty

from dialectic.experiments.params import EvalCommonParams, InverseCotEvalParams
from dialectic.llm.lora import DEFAULT_TARGET_MODULES, apply_lora, merge_lora
from dialectic.log import log


@dataclass
class ResolvedEvalCommonParams:
    """Post-resolution view of ``EvalCommonParams`` (every field concrete)."""

    q_ckpt_run: str
    q_ckpt_step: int
    model_name: str
    forward_ckpt_run: str
    forward_ckpt_step: int
    use_bf16: bool
    lora_rank: int | None
    lora_alpha: float
    lora_target_modules: str
    q_lora_rank: int | None
    q_lora_alpha: float
    q_lora_target_modules: str
    p_rollout_artifact: str
    seed: int
    split: str
    max_prompts: int | None
    dump_scores: bool


@dataclass
class ResolvedInverseCotEvalParams(ResolvedEvalCommonParams):
    """Resolved view of ``InverseCotEvalParams`` (adds FCR-specific knobs)."""

    batch_size: int
    max_tokens_generated: int
    temperature: float
    n_samples: int
    gpu_memory_utilization: float


def load_run_config(ckpt_run: str) -> dict[str, Any] | None:
    """Fetch a run's extty config; return ``None`` if the run isn't local."""
    try:
        project, name = ckpt_run.split("/", 1)
    except ValueError:
        log.warning(
            f"Cannot parse run ref {ckpt_run!r} as 'project/name'; "
            "skipping auto-derivation"
        )
        return None
    try:
        run_data = extty.get_run(project, name)
    except FileNotFoundError:
        log.warning(
            f"Run {ckpt_run!r} not found locally; cannot auto-derive its config"
        )
        return None
    return run_data.config


def _pick_lora_section(config: dict[str, Any]) -> dict[str, Any]:
    """Find the param section in a run config that carries lora settings.

    GRPO's ``TrainParams`` has no LoRA fields, but SFT's ``SftParams`` and
    inverse-cot's ``InverseCotParams`` do. Scan candidate sections in order
    and return the first one with a ``lora_rank`` key; otherwise an empty
    dict (no LoRA was used).
    """
    for section_name in ("inverse_cot_params", "sft_params", "train_params"):
        section = config.get(section_name)
        if isinstance(section, dict) and "lora_rank" in section:
            return section
    return {}


def rollout_artifact_from_q_config(q_config: dict[str, Any]) -> str | None:
    """Recover the p-rollout artifact a q run was trained on.

    New q runs record it as ``inverse_cot_params.p_rollout_artifact``
    (single str); older runs recorded a top-level ``dataset_artifacts``
    list (via the since-removed ``--dataset-glob`` plumbing). Handle both
    so lineage tracing keeps working for existing checkpoints.
    """
    icp = q_config.get("inverse_cot_params")
    if isinstance(icp, dict) and icp.get("p_rollout_artifact"):
        return icp["p_rollout_artifact"]
    legacy = q_config.get("dataset_artifacts")
    if legacy:
        if len(legacy) > 1:
            log.warning(
                f"q run was trained on {len(legacy)} artifacts; using the "
                f"first ({legacy[0]!r}). Pass the artifact explicitly to "
                "override."
            )
        return legacy[0]
    return None


def resolve_eval_common_params(
    eval_params: EvalCommonParams,
) -> ResolvedEvalCommonParams:
    """Fill in every ``None`` field on the shared common-eval surface.

    Strategy:
    1. Load q's config from ``q_ckpt_run``. From it, take ``model_name``,
       ``use_bf16``, the q-LoRA settings (``inverse_cot_params.lora_*``),
       the forward-checkpoint reference, and the p-rollout artifact (the
       same rollout artifact q was trained on).
    2. Load the forward run's config (the GRPO/SFT run that produced p).
       Take p's LoRA settings from whichever param section has ``lora_rank``.
    3. Any explicit value the user set on the CLI overrides the derived one.
    """
    q_config = load_run_config(eval_params.q_ckpt_run)
    if q_config is None:
        raise ValueError(
            f"Cannot auto-derive eval params: q run {eval_params.q_ckpt_run!r} "
            "is not available locally. Either pull the run or specify all "
            "required fields explicitly on the CLI."
        )

    q_train = q_config.get("train_params", {}) if isinstance(q_config, dict) else {}
    q_icp = q_config.get("inverse_cot_params", {}) if isinstance(q_config, dict) else {}

    model_name = eval_params.model_name or q_train.get("model_name")
    if model_name is None:
        raise ValueError(
            "`model_name` is not set and was not found in the q run's "
            "train_params.model_name"
        )

    forward_ckpt_run = eval_params.forward_ckpt_run or q_icp.get("forward_ckpt_run")
    forward_ckpt_step = (
        eval_params.forward_ckpt_step
        if eval_params.forward_ckpt_step is not None
        else q_icp.get("forward_ckpt_step")
    )
    if forward_ckpt_run is None or forward_ckpt_step is None:
        raise ValueError(
            "`forward_ckpt_run`/`forward_ckpt_step` not set and not found in "
            "the q run's inverse_cot_params"
        )

    use_bf16 = (
        eval_params.use_bf16
        if eval_params.use_bf16 is not None
        else bool(q_train.get("use_bf16", True))
    )

    q_lora_rank = (
        eval_params.q_lora_rank
        if eval_params.q_lora_rank is not None
        else q_icp.get("lora_rank")
    )
    q_lora_alpha = (
        eval_params.q_lora_alpha
        if eval_params.q_lora_alpha is not None
        else float(q_icp.get("lora_alpha", 16.0))
    )
    q_lora_target_modules = eval_params.q_lora_target_modules or q_icp.get(
        "lora_target_modules", "all"
    )

    # p-LoRA: resolve from the forward run's config (recursive lookup). If
    # the forward run isn't reachable, fall back to "no LoRA" — this is the
    # right default for GRPO-trained p.
    p_lora_rank = eval_params.lora_rank
    p_lora_alpha = eval_params.lora_alpha
    p_lora_target_modules = eval_params.lora_target_modules
    if p_lora_rank is None and p_lora_alpha is None and p_lora_target_modules is None:
        fwd_config = load_run_config(forward_ckpt_run)
        if fwd_config is not None:
            fwd_lora = _pick_lora_section(fwd_config)
            p_lora_rank = fwd_lora.get("lora_rank")
            p_lora_alpha = float(fwd_lora.get("lora_alpha", 16.0))
            p_lora_target_modules = fwd_lora.get("lora_target_modules", "all")
    p_lora_alpha = p_lora_alpha if p_lora_alpha is not None else 16.0
    p_lora_target_modules = p_lora_target_modules or "all"

    p_rollout_artifact = eval_params.p_rollout_artifact
    if p_rollout_artifact is None and isinstance(q_config, dict):
        p_rollout_artifact = rollout_artifact_from_q_config(q_config)
    if p_rollout_artifact is None:
        raise ValueError(
            "`p_rollout_artifact` is not set and could not be derived from "
            "the q run's config."
        )

    resolved = ResolvedEvalCommonParams(
        q_ckpt_run=eval_params.q_ckpt_run,
        q_ckpt_step=eval_params.q_ckpt_step,
        model_name=model_name,
        forward_ckpt_run=forward_ckpt_run,
        forward_ckpt_step=forward_ckpt_step,
        use_bf16=use_bf16,
        lora_rank=p_lora_rank,
        lora_alpha=p_lora_alpha,
        lora_target_modules=p_lora_target_modules,
        q_lora_rank=q_lora_rank,
        q_lora_alpha=q_lora_alpha,
        q_lora_target_modules=q_lora_target_modules,
        p_rollout_artifact=p_rollout_artifact,
        seed=eval_params.seed,
        split=eval_params.split,
        max_prompts=eval_params.max_prompts,
        dump_scores=eval_params.dump_scores,
    )
    _log_resolved(resolved)
    return resolved


def resolve_inverse_cot_eval_params(
    eval_params: InverseCotEvalParams,
) -> ResolvedInverseCotEvalParams:
    """Resolve the FCR-specific param surface: common fields + sampling knobs.

    The common surface is resolved via :func:`resolve_eval_common_params`;
    FCR-specific knobs (``temperature``, ``n_samples``, vLLM settings) are
    pure CLI inputs — no derivation, so they're copied through verbatim.
    """
    common = resolve_eval_common_params(eval_params)
    return ResolvedInverseCotEvalParams(
        **common.__dict__,
        batch_size=eval_params.batch_size,
        max_tokens_generated=eval_params.max_tokens_generated,
        temperature=eval_params.temperature,
        n_samples=eval_params.n_samples,
        gpu_memory_utilization=eval_params.gpu_memory_utilization,
    )


def _log_resolved(resolved: ResolvedEvalCommonParams) -> None:
    log.info("Resolved eval params:")
    log.info(f"  model_name:           {resolved.model_name}")
    log.info(
        f"  forward_ckpt:         {resolved.forward_ckpt_run} step {resolved.forward_ckpt_step}"
    )
    log.info(
        f"  q_ckpt:               {resolved.q_ckpt_run} step {resolved.q_ckpt_step}"
    )
    log.info(f"  use_bf16:             {resolved.use_bf16}")
    log.info(
        f"  p LoRA:               rank={resolved.lora_rank} alpha={resolved.lora_alpha} target={resolved.lora_target_modules}"
    )
    log.info(
        f"  q LoRA:               rank={resolved.q_lora_rank} alpha={resolved.q_lora_alpha} target={resolved.q_lora_target_modules}"
    )
    log.info(f"  p_rollout_artifact:   {resolved.p_rollout_artifact}")


def resolve_lora_targets(target_modules: str) -> tuple[str, ...]:
    if target_modules == "all":
        return DEFAULT_TARGET_MODULES
    if target_modules == "attn":
        return ("q_proj", "k_proj", "v_proj", "o_proj")
    if target_modules == "mlp":
        return ("gate_proj", "up_proj", "down_proj")
    raise ValueError(
        f"Unknown lora_target_modules: {target_modules!r} (expected 'all', 'attn', or 'mlp')"
    )


def strip_rng_state(state: dict) -> None:
    for key in ("_rng_torch", "_rng_python", "_rng_cuda"):
        state.pop(key, None)


def load_with_optional_lora(
    net,
    state_dict: dict,
    *,
    lora_rank: int | None,
    lora_alpha: float,
    lora_target_modules: str,
) -> None:
    """Load ``state_dict`` into ``net`` with an optional LoRA merge step.

    LoRA-trained checkpoints carry adapter weights that only line up if the
    adapter layers are added to ``net`` *before* the state dict is loaded.
    After load the adapters get merged back into the base weights so vLLM
    (and inference paths in general) see a standard transformer.
    """
    if lora_rank is None:
        net.load_state_dict(state_dict)
        return
    targets = resolve_lora_targets(lora_target_modules)
    apply_lora(net, rank=lora_rank, alpha=lora_alpha, target_modules=targets)
    net.load_state_dict(state_dict)
    merge_lora(net)
    log.info(f"Loaded and merged LoRA weights (rank={lora_rank})")
