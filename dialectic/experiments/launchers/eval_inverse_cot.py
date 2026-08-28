import copy
import gc
from typing import Callable

import extty
import torch
from tqdm import tqdm

from dialectic.experiments.arg_parser import Experiment, run_experiments_parser
from dialectic.experiments.launchers._eval_helpers import (
    load_with_optional_lora,
    pass_at_k,
    pass_at_ks,
    resolve_inverse_cot_eval_params,
    strip_rng_state,
)
from dialectic.experiments.params import InverseCotEvalParams
from dialectic.llm.generate import generate_hard_tokens
from dialectic.llm.registry import MODEL_REGISTRY
from dialectic.llm.vllm_loader import load_dialectic_qwen_as_vllm
from dialectic.log import log
from dialectic.rl.extractors import extract_from_answer_tags
from dialectic.rl.inverse_cot_data import PreTokenizedPrompt, load_rollout_artifact
from dialectic.rl.inverse_cot_eval import (
    HARD_PROMPT_THRESHOLD,
    EvalFcrResult,
    expressions_match,
    gsm8k_match,
)

try:
    from vllm import SamplingParams, TokensPrompt
except ModuleNotFoundError:
    log.error(
        "vllm is required to run `eval_inverse_cot.py`. please make sure dialectic is installed with the vllm extra."
    )


GradeFn = Callable[[str | None, str], bool]


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

    ks = pass_at_ks(N)

    def _bucket_metrics(
        grid: list[list[bool]], mask: list[bool] | None
    ) -> tuple[float, float, float, int, dict[int, float]]:
        idxs = [i for i in range(P) if mask is None or mask[i]]
        bucket_total = len(idxs)
        if bucket_total == 0:
            return 0.0, 0.0, 0.0, 0, {k: 0.0 for k in ks}
        total_samples = sum(len(grid[i]) for i in idxs)
        total_correct = sum(sum(grid[i]) for i in idxs)
        any_correct = sum(1 for i in idxs if any(grid[i]))
        at_1 = sum(1 for i in idxs if grid[i] and grid[i][0])
        pass_k = {
            k: sum(pass_at_k(len(grid[i]), sum(grid[i]), k) for i in idxs)
            / bucket_total
            for k in ks
        }
        return (
            _safe_div(at_1, bucket_total),
            _safe_div(total_correct, total_samples),
            _safe_div(any_correct, bucket_total),
            bucket_total,
            pass_k,
        )

    fcr_at_1, fcr_pr, fcr_any, _, fcr_pk = _bucket_metrics(fcr_correct_by_prompt, None)
    ai_at_1, ai_pr, ai_any, ai_total, ai_pk = _bucket_metrics(
        fcr_correct_by_prompt, all_incorrect_mask
    )
    hard_at_1, hard_pr, hard_any, hard_total, hard_pk = _bucket_metrics(
        fcr_correct_by_prompt, hard_mask
    )

    _, base_pr, base_any, _, base_pk = _bucket_metrics(baseline_correct_by_prompt, None)
    _, base_pr_ai, base_any_ai, _, base_pk_ai = _bucket_metrics(
        baseline_correct_by_prompt, all_incorrect_mask
    )
    _, base_pr_hard, base_any_hard, _, base_pk_hard = _bucket_metrics(
        baseline_correct_by_prompt, hard_mask
    )

    p_baseline_artifact = _safe_div(sum(p_artifact_rates), P)

    return EvalFcrResult(
        n_prompts=P,
        n_samples=N,
        fcr_at_1=fcr_at_1,
        fcr_pass_rate_at_n=fcr_pr,
        fcr_pass_at_n=fcr_any,
        fcr_at_1_on_all_incorrect=ai_at_1,
        fcr_pass_rate_at_n_on_all_incorrect=ai_pr,
        fcr_pass_at_n_on_all_incorrect=ai_any,
        n_on_all_incorrect=ai_total,
        fcr_at_1_on_hard=hard_at_1,
        fcr_pass_rate_at_n_on_hard=hard_pr,
        fcr_pass_at_n_on_hard=hard_any,
        n_on_hard=hard_total,
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
        fcr_pass_at_k=fcr_pk,
        fcr_pass_at_k_on_all_incorrect=ai_pk,
        fcr_pass_at_k_on_hard=hard_pk,
        p_pass_at_k=base_pk,
        p_pass_at_k_on_all_incorrect=base_pk_ai,
        p_pass_at_k_on_hard=base_pk_hard,
    )


def _eval_inverse_cot(
    *,
    eval_params: InverseCotEvalParams,
    grade_fn: GradeFn,
) -> None:
    cfg = resolve_inverse_cot_eval_params(eval_params)

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
    strip_rng_state(p_state)
    load_with_optional_lora(
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
    strip_rng_state(q_state)
    load_with_optional_lora(
        q,
        q_state,
        lora_rank=cfg.q_lora_rank,
        lora_alpha=cfg.q_lora_alpha,
        lora_target_modules=cfg.q_lora_target_modules,
    )
    q.requires_grad_(False)
    q.eval()

    # ---------------- data ----------------
    by_split = load_rollout_artifact(
        cfg.p_rollout_artifact, tokenizer, filter_train_split=False
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

    # ---------------- q via native PyTorch (prefix-LM attention) ----------------
    # vLLM serves a Qwen3ForCausalLM as causal-only, but q was trained with
    # prefix-LM (bidirectional over prompt + answer) — the ``causal=False`` flag
    # on q.layers is a runtime in-memory state, not part of the HF export. Using
    # vLLM for q therefore feeds it an attention pattern it wasn't trained on
    # and the resulting CoTs are garbage at early checkpoints (and silently
    # off-distribution at later ones). Match ``compute_fcr`` 's training-time
    # regime: run q natively with ``attention_mask=prefix_mask``, KV-cached.
    log.info(
        f"Sampling {cfg.n_samples} q-CoTs per prompt via native PyTorch (prefix-LM)"
    )

    prefix_ids_per_prompt: list[list[int]] = [
        pr.prompt_ids + tokenizer.encode(f" <answer> {pr.equation} </answer>").ids
        for pr in prompts
    ]

    # Flat (prompt_idx, sample_idx, prefix_ids) job list — N samples per prompt.
    jobs: list[tuple[int, int, list[int]]] = []
    for p_idx in range(len(prompts)):
        for s_idx in range(cfg.n_samples):
            jobs.append((p_idx, s_idx, prefix_ids_per_prompt[p_idx]))

    q_cot_strs: list[list[str]] = [["" for _ in range(cfg.n_samples)] for _ in prompts]

    n_batches = (len(jobs) + cfg.batch_size - 1) // cfg.batch_size
    for batch_start in tqdm(
        range(0, len(jobs), cfg.batch_size),
        total=n_batches,
        desc="q sampling (native)",
    ):
        batch = jobs[batch_start : batch_start + cfg.batch_size]
        B = len(batch)
        max_prefix_len = max(len(j[2]) for j in batch)

        q_prefix_ids_tensor = torch.full(
            (B, max_prefix_len),
            model_info.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        q_prefix_mask = torch.zeros(B, max_prefix_len, dtype=torch.bool, device=device)
        for i, (_, _, prefix_ids) in enumerate(batch):
            offset = max_prefix_len - len(prefix_ids)
            q_prefix_ids_tensor[i, offset:] = torch.tensor(prefix_ids, device=device)
            q_prefix_mask[i, offset:] = True

        with torch.no_grad():
            q_completions = generate_hard_tokens(
                net=q,
                token_ids=q_prefix_ids_tensor,
                sampling_strategy="sample",
                temperature=cfg.temperature,
                eos_token_id=model_info.eos_token_id,
                pad_token_id=model_info.pad_token_id,
                max_tokens_generated=cfg.max_tokens_generated,
                use_kv_cache=True,
                use_bf16=cfg.use_bf16,
                attention_mask=q_prefix_mask,
            ).tokens

        for i, (p_idx, s_idx, _) in enumerate(batch):
            cot_tokens = q_completions[i, max_prefix_len:].tolist()
            q_cot_strs[p_idx][s_idx] = tokenizer.decode(cot_tokens)

    del q
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
    for label, pk in (
        ("fcr", result.fcr_pass_at_k),
        ("fcr[hard]", result.fcr_pass_at_k_on_hard),
        ("fcr[all_incorrect]", result.fcr_pass_at_k_on_all_incorrect),
        ("p", result.p_pass_at_k),
        ("p[hard]", result.p_pass_at_k_on_hard),
        ("p[all_incorrect]", result.p_pass_at_k_on_all_incorrect),
    ):
        log.info(
            f"  {label + ':':22s}"
            + "  ".join(f"pass@{k}={v:.4f}" for k, v in sorted(pk.items()))
        )
    log.info(f"  p_baseline_artifact:                 {result.p_baseline_artifact:.4f}")
    log.info(f"  p_pass_rate_at_n (fresh):            {result.p_pass_rate_at_n:.4f}")
    log.info(f"  p_pass_at_n (fresh):                 {result.p_pass_at_n:.4f}")
    log.info(f"  fcr_pass_rate_lift:                  {result.fcr_pass_rate_lift:+.4f}")
    log.info(f"  fcr_pass_at_n_lift:                  {result.fcr_pass_at_n_lift:+.4f}")
    log.info(f"  all_incorrect bucket ({result.n_on_all_incorrect} prompts):")
    log.info(
        f"    fcr_pass_rate_at_n:              {result.fcr_pass_rate_at_n_on_all_incorrect:.4f}"
    )
    log.info(
        f"    fcr_pass_at_n:                   {result.fcr_pass_at_n_on_all_incorrect:.4f}"
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
    log.info(f"  hard bucket ({result.n_on_hard} prompts):")
    log.info(
        f"    fcr_pass_rate_at_n:              {result.fcr_pass_rate_at_n_on_hard:.4f}"
    )
    log.info(f"    fcr_pass_at_n:                   {result.fcr_pass_at_n_on_hard:.4f}")
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
                "fcr_at_1_on_all_incorrect": result.fcr_at_1_on_all_incorrect,
                "fcr_pass_rate_at_n_on_all_incorrect": result.fcr_pass_rate_at_n_on_all_incorrect,
                "fcr_pass_at_n_on_all_incorrect": result.fcr_pass_at_n_on_all_incorrect,
                "n_on_all_incorrect": result.n_on_all_incorrect,
                "fcr_at_1_on_hard": result.fcr_at_1_on_hard,
                "fcr_pass_rate_at_n_on_hard": result.fcr_pass_rate_at_n_on_hard,
                "fcr_pass_at_n_on_hard": result.fcr_pass_at_n_on_hard,
                "n_on_hard": result.n_on_hard,
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
                **{f"fcr_pass_at_{k}": v for k, v in result.fcr_pass_at_k.items()},
                **{
                    f"fcr_pass_at_{k}_on_hard": v
                    for k, v in result.fcr_pass_at_k_on_hard.items()
                },
                **{
                    f"fcr_pass_at_{k}_on_all_incorrect": v
                    for k, v in result.fcr_pass_at_k_on_all_incorrect.items()
                },
                **{f"p_pass_at_{k}": v for k, v in result.p_pass_at_k.items()},
                **{
                    f"p_pass_at_{k}_on_hard": v
                    for k, v in result.p_pass_at_k_on_hard.items()
                },
                **{
                    f"p_pass_at_{k}_on_all_incorrect": v
                    for k, v in result.p_pass_at_k_on_all_incorrect.items()
                },
            },
            step=0,
        )


def _countdown_grade(extracted: str | None, gold: str) -> bool:
    return extracted is not None and expressions_match(extracted, gold)


@extty.experiment(project="eval-inverse-cot-countdown")
def eval_inverse_cot_countdown(*, eval_params: InverseCotEvalParams) -> None:
    _eval_inverse_cot(eval_params=eval_params, grade_fn=_countdown_grade)


@extty.experiment(project="eval-inverse-cot-gsm8k")
def eval_inverse_cot_gsm8k(*, eval_params: InverseCotEvalParams) -> None:
    _eval_inverse_cot(eval_params=eval_params, grade_fn=gsm8k_match)


if __name__ == "__main__":
    run_experiments_parser(
        [
            Experiment(
                env_name="countdown",
                fn=eval_inverse_cot_countdown,
                include_prompt_collection_id=False,
            ),
            Experiment(
                env_name="gsm8k",
                fn=eval_inverse_cot_gsm8k,
                include_prompt_collection_id=False,
            ),
        ]
    )
