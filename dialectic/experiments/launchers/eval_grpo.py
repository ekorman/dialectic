import math
from typing import Callable

import extty
import torch

from dialectic.experiments.arg_parser import Experiment, run_experiments_parser
from dialectic.experiments.launchers._eval_helpers import (
    load_run_config,
)
from dialectic.experiments.params import GrpoEvalParams
from dialectic.llm.lora import DEFAULT_TARGET_MODULES, apply_lora, merge_lora
from dialectic.llm.registry import MODEL_REGISTRY
from dialectic.llm.vllm_loader import load_dialectic_qwen_as_vllm
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


def _pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al. 2021): ``1 - C(n-c, k) / C(n, k)``.

    Estimates the probability that at least one of k samples is correct,
    given c correct out of n drawn — using all n samples for every k, so
    it's both unbiased and lower-variance than grading any k-subset.
    At k == n it reduces exactly to ``any``-of-n.
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def _pass_at_ks(n_samples: int) -> list[int]:
    """Powers of two up to ``n_samples``, plus ``n_samples`` itself."""
    ks = []
    k = 1
    while k < n_samples:
        ks.append(k)
        k *= 2
    ks.append(n_samples)
    return ks


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
    eval_params.ckpt_step = rgp.get("start_ckpt_step")
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
    train = fwd_config.get("train_params", {}) if isinstance(fwd_config, dict) else {}
    if eval_params.model_name is None:
        eval_params.model_name = train.get("model_name")
    if eval_params.use_bf16 is None and "use_bf16" in train:
        eval_params.use_bf16 = bool(train["use_bf16"])

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
    if eval_params.ckpt_run is None or eval_params.ckpt_step is None:
        raise ValueError(
            "`ckpt_run`/`ckpt_step` could not be resolved. Either "
            "pass them explicitly or use a p_rollout_artifact whose producing run "
            "has `rollout_gen_params.start_ckpt_run` set."
        )
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
    # extty snapshotted the config — at this point all the Optional fields
    # are concrete. The asserts narrow the types for the type-checker.
    assert eval_params.p_rollout_artifact is not None
    assert eval_params.model_name is not None
    assert eval_params.ckpt_run is not None
    assert eval_params.ckpt_step is not None
    assert eval_params.use_bf16 is not None

    torch.manual_seed(eval_params.seed)
    model_info = MODEL_REGISTRY[eval_params.model_name]
    tokenizer = model_info.load_tokenizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if eval_params.use_bf16 else torch.float32

    log.info(f"Loading p from {eval_params.ckpt_run} step {eval_params.ckpt_step}")
    p = model_info.load_net(pretrained_weights=False)
    p_project, p_run_name = eval_params.ckpt_run.split("/")
    p_ckpt = extty.load_checkpoint_from(
        project=p_project,
        run_name=p_run_name,
        step=eval_params.ckpt_step,
        load_optimizer=False,
    )
    p_state = p_ckpt["model_state_dict"]
    for key in ("_rng_torch", "_rng_python", "_rng_cuda"):
        p_state.pop(key, None)

    if eval_params.lora_rank is not None:
        targets = _resolve_lora_targets(eval_params.lora_target_modules)
        apply_lora(
            p,
            rank=eval_params.lora_rank,
            alpha=eval_params.lora_alpha,
            target_modules=targets,
        )
        p.load_state_dict(p_state)
        merge_lora(p)
        log.info(f"Loaded and merged LoRA weights (rank={eval_params.lora_rank})")
    else:
        p.load_state_dict(p_state)

    p = p.to(device=device, dtype=dtype)
    p.requires_grad_(False)
    p.eval()

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

    llm = load_dialectic_qwen_as_vllm(
        p,
        tokenizer=tokenizer,
        eos_token_id=model_info.eos_token_id,
        pad_token_id=model_info.pad_token_id,
        max_model_len=eval_params.max_tokens_generated + 1024,
        gpu_memory_utilization=eval_params.gpu_memory_utilization,
        dtype="bfloat16" if eval_params.use_bf16 else "float16",
        seed=eval_params.seed,
    )
    del p
    torch.cuda.empty_cache()

    sampling_params = SamplingParams(
        n=eval_params.n_samples,
        temperature=eval_params.temperature if eval_params.temperature > 0 else 0.0,
        max_tokens=eval_params.max_tokens_generated,
        stop_token_ids=[model_info.eos_token_id],
        seed=eval_params.seed,
    )
    vllm_outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=pr.prompt_ids) for pr in prompts],
        sampling_params,
        use_tqdm=True,
    )

    # Per prompt: (n drawn, c correct, is_hard). pass@k for every k is then
    # estimated from the full n samples via `_pass_at_k` — no subsetting.
    counts: list[tuple[int, int, bool]] = []
    for pr, out in zip(prompts, vllm_outputs):
        assert pr.equation is not None
        per_sample_correct = [
            grade_fn(extract_from_answer_tags(sample.text), pr.equation)
            for sample in out.outputs
        ]
        artifact_rate = (
            sum(c.is_correct for c in pr.completions) / len(pr.completions)
            if pr.completions
            else 0.0
        )
        is_hard = artifact_rate <= HARD_PROMPT_THRESHOLD
        counts.append((len(per_sample_correct), sum(per_sample_correct), is_hard))

    total = len(counts)
    hard_counts = [(n, c) for n, c, is_hard in counts if is_hard]
    hard_total = len(hard_counts)

    pass_rate_at_n = sum(c for n, c, _ in counts) / max(sum(n for n, _, _ in counts), 1)
    hard_pass_rate_at_n = sum(c for _, c in hard_counts) / max(
        sum(n for n, _ in hard_counts), 1
    )

    ks = _pass_at_ks(eval_params.n_samples)
    pass_at_k = {
        k: sum(_pass_at_k(n, c, k) for n, c, _ in counts) / max(total, 1) for k in ks
    }
    hard_pass_at_k = {
        k: sum(_pass_at_k(n, c, k) for n, c in hard_counts) / max(hard_total, 1)
        for k in ks
    }

    log.info(
        f"Results ({total} prompts, {hard_total} hard, "
        f"n_samples={eval_params.n_samples}, temp={eval_params.temperature}):"
    )
    log.info(f"  pass_rate_at_n:      {pass_rate_at_n:.4f}")
    log.info(f"  hard_pass_rate_at_n: {hard_pass_rate_at_n:.4f}")
    for k in ks:
        log.info(
            f"  pass_at_{k}: {pass_at_k[k]:.4f}   hard_pass_at_{k}: {hard_pass_at_k[k]:.4f}"
        )

    if extty.has_active_run():
        extty.log(
            {
                "n_prompts": total,
                "n_samples": eval_params.n_samples,
                "pass_rate_at_n": pass_rate_at_n,
                "hard_pass_rate_at_n": hard_pass_rate_at_n,
                "hard_total": hard_total,
                **{f"pass_at_{k}": pass_at_k[k] for k in ks},
                **{f"hard_pass_at_{k}": hard_pass_at_k[k] for k in ks},
            },
            step=0,
        )


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
