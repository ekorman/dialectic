import copy
import gc
from dataclasses import dataclass
from typing import Any, Callable

import extty
import torch

from dialectic.experiments.arg_parser import Experiment, run_experiments_parser
from dialectic.experiments.params import InverseCotEvalParams
from dialectic.llm.lora import DEFAULT_TARGET_MODULES, apply_lora, merge_lora
from dialectic.llm.registry import MODEL_REGISTRY
from dialectic.llm.vllm_loader import load_dialectic_qwen_as_vllm
from dialectic.log import log
from dialectic.rl.extractors import extract_from_answer_tags
from dialectic.rl.inverse_cot_data import PreTokenizedPrompt, load_rollout_artifacts
from dialectic.rl.inverse_cot_eval import (
    HARD_PROMPT_THRESHOLD,
    EvalFcrResult,
    expressions_match,
    gsm8k_match,
)


@dataclass
class _ResolvedEvalParams:
    """Holds the post-resolution view of ``InverseCotEvalParams``: every
    field is concrete (no ``None`` for derived values)."""

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
    dataset_artifacts: list[str]
    seed: int
    split: str
    batch_size: int
    max_tokens_generated: int
    temperature: float
    n_samples: int
    max_prompts: int | None
    gpu_memory_utilization: float


def _load_run_config(ckpt_run: str) -> dict[str, Any] | None:
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


def _resolve_eval_params(
    eval_params: InverseCotEvalParams,
    dataset_artifacts: list[str] | None,
) -> _ResolvedEvalParams:
    """Fill in every ``None`` field from the q run's extty config.

    Strategy:
    1. Load q's config from ``q_ckpt_run``. From it, take ``model_name``,
       ``use_bf16``, the q-LoRA settings (``inverse_cot_params.lora_*``),
       the forward-checkpoint reference, and the ``dataset_artifacts`` list
       (the same rollout artifact q was trained on).
    2. Load the forward run's config (the GRPO/SFT run that produced p).
       Take p's LoRA settings from whichever param section has ``lora_rank``.
    3. Any explicit value the user set on the CLI overrides the derived one.
    """
    q_config = _load_run_config(eval_params.q_ckpt_run)
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
        fwd_config = _load_run_config(forward_ckpt_run)
        if fwd_config is not None:
            fwd_lora = _pick_lora_section(fwd_config)
            p_lora_rank = fwd_lora.get("lora_rank")
            p_lora_alpha = float(fwd_lora.get("lora_alpha", 16.0))
            p_lora_target_modules = fwd_lora.get("lora_target_modules", "all")
    p_lora_alpha = p_lora_alpha if p_lora_alpha is not None else 16.0
    p_lora_target_modules = p_lora_target_modules or "all"

    if dataset_artifacts is None:
        cfg_artifacts = (
            q_config.get("dataset_artifacts") if isinstance(q_config, dict) else None
        )
        if not cfg_artifacts:
            raise ValueError(
                "No `dataset_artifacts` set via --dataset-glob and the q run's "
                "config has no `dataset_artifacts` field to fall back to."
            )
        dataset_artifacts = list(cfg_artifacts)

    resolved = _ResolvedEvalParams(
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
        dataset_artifacts=dataset_artifacts,
        seed=eval_params.seed,
        split=eval_params.split,
        batch_size=eval_params.batch_size,
        max_tokens_generated=eval_params.max_tokens_generated,
        temperature=eval_params.temperature,
        n_samples=eval_params.n_samples,
        max_prompts=eval_params.max_prompts,
        gpu_memory_utilization=eval_params.gpu_memory_utilization,
    )
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
    log.info(f"  dataset_artifacts:    {resolved.dataset_artifacts}")
    return resolved


try:
    from vllm import SamplingParams, TokensPrompt
except ModuleNotFoundError:
    log.error(
        "vllm is required to run `eval_inverse_cot.py`. please make sure dialectic is installed with the vllm extra."
    )


GradeFn = Callable[[str | None, str], bool]


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


def _strip_rng_state(state: dict) -> None:
    for key in ("_rng_torch", "_rng_python", "_rng_cuda"):
        state.pop(key, None)


def _load_with_optional_lora(
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
    sees a standard transformer.
    """
    if lora_rank is None:
        net.load_state_dict(state_dict)
        return
    targets = _resolve_lora_targets(lora_target_modules)
    apply_lora(net, rank=lora_rank, alpha=lora_alpha, target_modules=targets)
    net.load_state_dict(state_dict)
    merge_lora(net)
    log.info(f"Loaded and merged LoRA weights (rank={lora_rank})")


def _safe_div(num: float, den: float) -> float:
    return num / den if den > 0 else 0.0


def _aggregate_fcr(
    prompts: list[PreTokenizedPrompt],
    fcr_correct_by_prompt: list[list[bool]],
    baseline_correct_by_prompt: list[list[bool]],
) -> EvalFcrResult:
    """Reduce per-prompt × per-sample correctness grids to ``EvalFcrResult``.

    Buckets:
    - ``all_incorrect``: prompts where every pre-generated rollout completion
      in the artifact was incorrect (q's rescue surface).
    - ``hard``: prompts whose artifact p-rate ≤ ``HARD_PROMPT_THRESHOLD``.

    ``*_pass_rate_at_n`` are pooled means over N×P samples in the bucket;
    ``*_pass_at_n`` are means over prompts of ``any(...)``; ``*_at_1`` use
    only the first q-CoT per prompt for back-compat with training-time FCR.
    The same bucketed reduction is applied to ``baseline_correct_by_prompt``
    so freshly-sampled p can be compared head-to-head with q-guided p
    inside each bucket (matters when eval-time N > artifact group size).
    """
    P = len(prompts)
    N = max((len(row) for row in fcr_correct_by_prompt), default=1)

    p_artifact_rates = [
        sum(c.is_correct for c in pr.completions) / len(pr.completions)
        if pr.completions
        else 0.0
        for pr in prompts
    ]
    all_incorrect_mask = [
        bool(pr.completions) and all(not c.is_correct for c in pr.completions)
        for pr in prompts
    ]
    hard_mask = [r <= HARD_PROMPT_THRESHOLD for r in p_artifact_rates]

    def _bucket_metrics(
        grid: list[list[bool]], mask: list[bool] | None
    ) -> tuple[float, float, float, int]:
        idxs = [i for i in range(P) if mask is None or mask[i]]
        bucket_total = len(idxs)
        if bucket_total == 0:
            return 0.0, 0.0, 0.0, 0
        total_samples = sum(len(grid[i]) for i in idxs)
        total_correct = sum(sum(grid[i]) for i in idxs)
        any_correct = sum(1 for i in idxs if any(grid[i]))
        at_1 = sum(1 for i in idxs if grid[i] and grid[i][0])
        return (
            _safe_div(at_1, bucket_total),
            _safe_div(total_correct, total_samples),
            _safe_div(any_correct, bucket_total),
            bucket_total,
        )

    fcr_at_1, fcr_pr, fcr_any, _ = _bucket_metrics(fcr_correct_by_prompt, None)
    ai_at_1, ai_pr, ai_any, ai_total = _bucket_metrics(
        fcr_correct_by_prompt, all_incorrect_mask
    )
    hard_at_1, hard_pr, hard_any, hard_total = _bucket_metrics(
        fcr_correct_by_prompt, hard_mask
    )

    _, base_pr, base_any, _ = _bucket_metrics(baseline_correct_by_prompt, None)
    _, base_pr_ai, base_any_ai, _ = _bucket_metrics(
        baseline_correct_by_prompt, all_incorrect_mask
    )
    _, base_pr_hard, base_any_hard, _ = _bucket_metrics(
        baseline_correct_by_prompt, hard_mask
    )

    p_baseline_artifact = _safe_div(sum(p_artifact_rates), P)

    return EvalFcrResult(
        n_prompts=P,
        n_samples=N,
        fcr_at_1=fcr_at_1,
        fcr_pass_rate_at_n=fcr_pr,
        fcr_pass_at_n=fcr_any,
        fcr_all_incorrect_at_1=ai_at_1,
        fcr_all_incorrect_pass_rate_at_n=ai_pr,
        fcr_all_incorrect_pass_at_n=ai_any,
        fcr_all_incorrect_total=ai_total,
        fcr_hard_at_1=hard_at_1,
        fcr_hard_pass_rate_at_n=hard_pr,
        fcr_hard_pass_at_n=hard_any,
        fcr_hard_total=hard_total,
        p_baseline_artifact=p_baseline_artifact,
        p_pass_rate_at_n=base_pr,
        p_pass_at_n=base_any,
        p_pass_rate_at_n_on_all_incorrect=base_pr_ai,
        p_pass_at_n_on_all_incorrect=base_any_ai,
        p_pass_rate_at_n_on_hard=base_pr_hard,
        p_pass_at_n_on_hard=base_any_hard,
        fcr_pass_rate_lift=fcr_pr - base_pr,
        fcr_pass_at_n_lift=fcr_any - base_any,
        fcr_pass_rate_lift_on_all_incorrect=ai_pr - base_pr_ai,
        fcr_pass_at_n_lift_on_all_incorrect=ai_any - base_any_ai,
        fcr_pass_rate_lift_on_hard=hard_pr - base_pr_hard,
        fcr_pass_at_n_lift_on_hard=hard_any - base_any_hard,
    )


def _eval_inverse_cot(
    *,
    eval_params: InverseCotEvalParams,
    dataset_artifacts: list[str] | None,
    grade_fn: GradeFn,
) -> None:
    cfg = _resolve_eval_params(eval_params, dataset_artifacts)

    torch.manual_seed(cfg.seed)
    model_info = MODEL_REGISTRY[cfg.model_name]
    tokenizer = model_info.load_tokenizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if cfg.use_bf16 else torch.float32

    # ---------------- p ----------------
    log.info(f"Loading p from {cfg.forward_ckpt_run} step {cfg.forward_ckpt_step}")
    p = model_info.load_net(pretrained_weights=False)
    p_project, p_run_name = cfg.forward_ckpt_run.split("/")
    p_ckpt = extty.load_checkpoint_from(
        project=p_project,
        run_name=p_run_name,
        step=cfg.forward_ckpt_step,
        load_optimizer=False,
    )
    p_state = p_ckpt["model_state_dict"]
    _strip_rng_state(p_state)
    _load_with_optional_lora(
        p,
        p_state,
        lora_rank=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_target_modules=cfg.lora_target_modules,
    )
    p = p.to(device=device, dtype=dtype)
    p.requires_grad_(False)
    p.eval()

    # ---------------- q ----------------
    q = copy.deepcopy(p)
    for layer in q.layers:
        layer.self_attn.causal = False
    q = q.to(device=device, dtype=dtype)

    log.info(f"Loading q from {cfg.q_ckpt_run} step {cfg.q_ckpt_step}")
    q_project, q_run_name = cfg.q_ckpt_run.split("/")
    q_ckpt = extty.load_checkpoint_from(
        project=q_project,
        run_name=q_run_name,
        step=cfg.q_ckpt_step,
        load_optimizer=False,
    )
    q_state = q_ckpt["model_state_dict"]
    _strip_rng_state(q_state)
    _load_with_optional_lora(
        q,
        q_state,
        lora_rank=cfg.q_lora_rank,
        lora_alpha=cfg.q_lora_alpha,
        lora_target_modules=cfg.q_lora_target_modules,
    )
    q.requires_grad_(False)
    q.eval()

    # ---------------- data ----------------
    by_split = load_rollout_artifacts(
        cfg.dataset_artifacts, tokenizer, filter_train_split=False
    )
    prompts = by_split.get(cfg.split, [])
    prompts = [pr for pr in prompts if pr.equation is not None]
    if not prompts:
        raise ValueError(
            f"No prompts found for split={cfg.split} with non-null `equation`"
        )
    if cfg.max_prompts is not None:
        prompts = prompts[: cfg.max_prompts]
    log.info(
        f"Evaluating on {len(prompts)} prompts "
        f"(split={cfg.split}, n_samples={cfg.n_samples})"
    )

    # ---------------- q -> vLLM, sample N CoTs per prompt ----------------
    log.info(
        f"Exporting q to vLLM (gpu_memory_utilization={cfg.gpu_memory_utilization})"
    )
    q_vllm = load_dialectic_qwen_as_vllm(
        q,
        tokenizer=tokenizer,
        eos_token_id=model_info.eos_token_id,
        pad_token_id=model_info.pad_token_id,
        max_model_len=cfg.max_tokens_generated + 1024,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        dtype="bfloat16" if cfg.use_bf16 else "float16",
        seed=cfg.seed,
    )
    del q
    torch.cuda.empty_cache()

    q_prefix_ids = [
        pr.prompt_ids + tokenizer.encode(f" <answer> {pr.equation} </answer>").ids
        for pr in prompts
    ]
    q_sampling = SamplingParams(
        n=cfg.n_samples,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens_generated,
        stop_token_ids=[model_info.eos_token_id],
        seed=cfg.seed,
    )
    log.info(f"Generating {cfg.n_samples} q-CoTs per prompt")
    q_outputs = q_vllm.generate(
        [TokensPrompt(prompt_token_ids=ids) for ids in q_prefix_ids],
        q_sampling,
        use_tqdm=True,
    )
    q_cot_strs: list[list[str]] = [
        [sample.text for sample in out.outputs] for out in q_outputs
    ]

    del q_vllm
    gc.collect()
    torch.cuda.empty_cache()

    # ---------------- p -> vLLM, greedy FCR + fresh baseline ----------------
    log.info(
        f"Exporting p to vLLM (gpu_memory_utilization={cfg.gpu_memory_utilization})"
    )
    p_vllm = load_dialectic_qwen_as_vllm(
        p,
        tokenizer=tokenizer,
        eos_token_id=model_info.eos_token_id,
        pad_token_id=model_info.pad_token_id,
        max_model_len=cfg.max_tokens_generated + 1024,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        dtype="bfloat16" if cfg.use_bf16 else "float16",
        seed=cfg.seed,
    )
    del p
    torch.cuda.empty_cache()

    # FCR path: prime p with prompt + <think>q_cot</think> for every (prompt, sample)
    primed_strs: list[str] = []
    for pr, cots in zip(prompts, q_cot_strs):
        prompt_str = tokenizer.decode(pr.prompt_ids)
        for cot in cots:
            primed_strs.append(
                prompt_str + "<think>\n" + cot.strip() + "\n</think>\n\n"
            )

    p_greedy_sampling = SamplingParams(
        n=1,
        temperature=0.0,
        max_tokens=cfg.max_tokens_generated,
        stop_token_ids=[model_info.eos_token_id],
        seed=cfg.seed,
    )
    log.info(f"Greedy-decoding p on {len(primed_strs)} primed prompts (FCR path)")
    p_fcr_outputs = p_vllm.generate(primed_strs, p_greedy_sampling, use_tqdm=True)

    # Reshape flat output back to [P][N] grading grid
    fcr_correct_by_prompt: list[list[bool]] = []
    cursor = 0
    for pr, cots in zip(prompts, q_cot_strs):
        per_prompt: list[bool] = []
        for _ in cots:
            answer_text = p_fcr_outputs[cursor].outputs[0].text
            extracted = extract_from_answer_tags(answer_text)
            assert pr.equation is not None
            per_prompt.append(grade_fn(extracted, pr.equation))
            cursor += 1
        fcr_correct_by_prompt.append(per_prompt)

    # Fresh p baseline (no q): sample N answers per prompt at `temperature`
    log.info(
        f"Sampling fresh p baseline ({cfg.n_samples} per prompt at temperature={cfg.temperature})"
    )
    p_baseline_sampling = SamplingParams(
        n=cfg.n_samples,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens_generated,
        stop_token_ids=[model_info.eos_token_id],
        seed=cfg.seed + 1,
    )
    p_baseline_outputs = p_vllm.generate(
        [TokensPrompt(prompt_token_ids=pr.prompt_ids) for pr in prompts],
        p_baseline_sampling,
        use_tqdm=True,
    )

    baseline_correct_by_prompt: list[list[bool]] = []
    for pr, out in zip(prompts, p_baseline_outputs):
        per_prompt = []
        for sample in out.outputs:
            extracted = extract_from_answer_tags(sample.text)
            assert pr.equation is not None
            per_prompt.append(grade_fn(extracted, pr.equation))
        baseline_correct_by_prompt.append(per_prompt)

    del p_vllm
    gc.collect()
    torch.cuda.empty_cache()

    # ---------------- aggregate + log ----------------
    result = _aggregate_fcr(prompts, fcr_correct_by_prompt, baseline_correct_by_prompt)

    log.info(
        f"Results ({result.n_prompts} prompts, N={result.n_samples}, "
        f"temp={cfg.temperature}):"
    )
    log.info(f"  fcr_at_1:                            {result.fcr_at_1:.4f}")
    log.info(f"  fcr_pass_rate_at_n:                  {result.fcr_pass_rate_at_n:.4f}")
    log.info(f"  fcr_pass_at_n:                       {result.fcr_pass_at_n:.4f}")
    log.info(f"  p_baseline_artifact:                 {result.p_baseline_artifact:.4f}")
    log.info(f"  p_pass_rate_at_n (fresh):            {result.p_pass_rate_at_n:.4f}")
    log.info(f"  p_pass_at_n (fresh):                 {result.p_pass_at_n:.4f}")
    log.info(f"  fcr_pass_rate_lift:                  {result.fcr_pass_rate_lift:+.4f}")
    log.info(f"  fcr_pass_at_n_lift:                  {result.fcr_pass_at_n_lift:+.4f}")
    log.info(f"  all_incorrect bucket ({result.fcr_all_incorrect_total} prompts):")
    log.info(
        f"    fcr_pass_rate_at_n:              {result.fcr_all_incorrect_pass_rate_at_n:.4f}"
    )
    log.info(
        f"    fcr_pass_at_n:                   {result.fcr_all_incorrect_pass_at_n:.4f}"
    )
    log.info(
        f"    p_pass_rate_at_n:                {result.p_pass_rate_at_n_on_all_incorrect:.4f}"
    )
    log.info(
        f"    p_pass_at_n:                     {result.p_pass_at_n_on_all_incorrect:.4f}"
    )
    log.info(
        f"    fcr_pass_rate_lift:              {result.fcr_pass_rate_lift_on_all_incorrect:+.4f}"
    )
    log.info(
        f"    fcr_pass_at_n_lift:              {result.fcr_pass_at_n_lift_on_all_incorrect:+.4f}"
    )
    log.info(f"  hard bucket ({result.fcr_hard_total} prompts):")
    log.info(
        f"    fcr_pass_rate_at_n:              {result.fcr_hard_pass_rate_at_n:.4f}"
    )
    log.info(f"    fcr_pass_at_n:                   {result.fcr_hard_pass_at_n:.4f}")
    log.info(
        f"    p_pass_rate_at_n:                {result.p_pass_rate_at_n_on_hard:.4f}"
    )
    log.info(f"    p_pass_at_n:                     {result.p_pass_at_n_on_hard:.4f}")
    log.info(
        f"    fcr_pass_rate_lift:              {result.fcr_pass_rate_lift_on_hard:+.4f}"
    )
    log.info(
        f"    fcr_pass_at_n_lift:              {result.fcr_pass_at_n_lift_on_hard:+.4f}"
    )

    if extty.has_active_run():
        extty.log(
            {
                "n_prompts": result.n_prompts,
                "n_samples": result.n_samples,
                "fcr_at_1": result.fcr_at_1,
                "fcr_pass_rate_at_n": result.fcr_pass_rate_at_n,
                "fcr_pass_at_n": result.fcr_pass_at_n,
                "fcr_all_incorrect_at_1": result.fcr_all_incorrect_at_1,
                "fcr_all_incorrect_pass_rate_at_n": result.fcr_all_incorrect_pass_rate_at_n,
                "fcr_all_incorrect_pass_at_n": result.fcr_all_incorrect_pass_at_n,
                "fcr_all_incorrect_total": result.fcr_all_incorrect_total,
                "fcr_hard_at_1": result.fcr_hard_at_1,
                "fcr_hard_pass_rate_at_n": result.fcr_hard_pass_rate_at_n,
                "fcr_hard_pass_at_n": result.fcr_hard_pass_at_n,
                "fcr_hard_total": result.fcr_hard_total,
                "p_baseline_artifact": result.p_baseline_artifact,
                "p_pass_rate_at_n": result.p_pass_rate_at_n,
                "p_pass_at_n": result.p_pass_at_n,
                "p_pass_rate_at_n_on_all_incorrect": result.p_pass_rate_at_n_on_all_incorrect,
                "p_pass_at_n_on_all_incorrect": result.p_pass_at_n_on_all_incorrect,
                "p_pass_rate_at_n_on_hard": result.p_pass_rate_at_n_on_hard,
                "p_pass_at_n_on_hard": result.p_pass_at_n_on_hard,
                "fcr_pass_rate_lift": result.fcr_pass_rate_lift,
                "fcr_pass_at_n_lift": result.fcr_pass_at_n_lift,
                "fcr_pass_rate_lift_on_all_incorrect": result.fcr_pass_rate_lift_on_all_incorrect,
                "fcr_pass_at_n_lift_on_all_incorrect": result.fcr_pass_at_n_lift_on_all_incorrect,
                "fcr_pass_rate_lift_on_hard": result.fcr_pass_rate_lift_on_hard,
                "fcr_pass_at_n_lift_on_hard": result.fcr_pass_at_n_lift_on_hard,
            },
            step=0,
        )


def _countdown_grade(extracted: str | None, gold: str) -> bool:
    return extracted is not None and expressions_match(extracted, gold)


@extty.experiment(project="eval-inverse-cot-countdown")
def eval_inverse_cot_countdown(
    *,
    eval_params: InverseCotEvalParams,
    dataset_artifacts: list[str] | None = None,
) -> None:
    _eval_inverse_cot(
        eval_params=eval_params,
        dataset_artifacts=dataset_artifacts,
        grade_fn=_countdown_grade,
    )


@extty.experiment(project="eval-inverse-cot-gsm8k")
def eval_inverse_cot_gsm8k(
    *,
    eval_params: InverseCotEvalParams,
    dataset_artifacts: list[str] | None = None,
) -> None:
    _eval_inverse_cot(
        eval_params=eval_params,
        dataset_artifacts=dataset_artifacts,
        grade_fn=gsm8k_match,
    )


if __name__ == "__main__":
    run_experiments_parser(
        [
            Experiment(
                env_name="countdown",
                fn=eval_inverse_cot_countdown,
                include_dataset_glob=True,
                include_prompt_collection_id=False,
                dataset_glob_required=False,
            ),
            Experiment(
                env_name="gsm8k",
                fn=eval_inverse_cot_gsm8k,
                include_dataset_glob=True,
                include_prompt_collection_id=False,
                dataset_glob_required=False,
            ),
        ]
    )
