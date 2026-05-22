"""Evaluate q as a verifier: does q(CoT | answer, P) separate correct from incorrect rollouts?

For each prompt in the rollout data:
1. For each completion (CoT + answer from p):
   - Compute q(CoT | extracted_answer, P) — log prob of the CoT under q conditioned on p's own answer
   - Compute p(CoT, answer | P) — log prob under causal p as a baseline
2. Compare scores for correct vs incorrect completions
3. Report: ROC AUC, PR AUC, score gap, and accuracy of "pick highest q-score" as a
   selection strategy — for both q and p_self, with q-vs-p_self lifts.

The p_self baseline answers the meaningful question: did training q on
``p(CoT | answer)`` make it a better verifier than just asking p its own
confidence? If ``q_roc_auc <= p_self_roc_auc`` then q is just rediscovering p's
intrinsic uncertainty and isn't pulling its weight as a separate model.

Correctness is read from the rollout artifact's ``is_correct`` flag, so this
launcher is env-agnostic — same body for countdown and gsm8k.
"""

import copy
import random

import extty
import torch
import torch.nn.functional as F

from dialectic.experiments.arg_parser import Experiment, run_experiments_parser
from dialectic.experiments.launchers._eval_helpers import (
    load_with_optional_lora,
    resolve_eval_common_params,
    strip_rng_state,
)
from dialectic.experiments.params import EvalCommonParams
from dialectic.llm.inverse_cot import create_prefix_lm_mask
from dialectic.llm.registry import MODEL_REGISTRY
from dialectic.log import log
from dialectic.rl.extractors import extract_from_answer_tags
from dialectic.rl.inverse_cot_data import load_rollout_artifacts


def _import_sklearn():
    """Lazy-import sklearn so the verifier still loads (with a clear error
    at call time) when the ``eval`` dependency group isn't installed."""
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        return roc_auc_score, average_precision_score
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "scikit-learn is required for verifier AUC metrics. Install it via "
            "`uv sync --group eval` (or add the `eval` dependency group)."
        ) from exc


@torch.no_grad()
def _score_q_prefix_lm(
    q,
    full_ids: list[int],
    prefix_len: int,
    device: torch.device,
) -> float:
    """q-score: avg log p_q(cot + eos | prompt + answer) under prefix-LM attention."""
    input_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
    prefix_lengths = torch.tensor([prefix_len], dtype=torch.long, device=device)
    attention_mask = create_prefix_lm_mask(prefix_lengths, len(full_ids), device)
    logits = q(input_tensor, attention_mask=attention_mask, return_all_logits=True)
    shift_logits = logits[0, prefix_len - 1 : -1]
    shift_targets = input_tensor[0, prefix_len:]
    log_probs = -F.cross_entropy(shift_logits.float(), shift_targets, reduction="none")
    return log_probs.mean().item()


@torch.no_grad()
def _score_p_causal(
    p,
    full_ids: list[int],
    prefix_len: int,
    device: torch.device,
) -> float:
    """p-score: avg log p_p(cot + answer + eos | prompt) under default causal attention."""
    input_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
    logits = p(input_tensor, return_all_logits=True)
    shift_logits = logits[0, prefix_len - 1 : -1]
    shift_targets = input_tensor[0, prefix_len:]
    log_probs = -F.cross_entropy(shift_logits.float(), shift_targets, reduction="none")
    return log_probs.mean().item()


def _eval_q_verifier(
    *,
    eval_params: EvalCommonParams,
    dataset_artifacts: list[str] | None,
) -> None:
    roc_auc_score, average_precision_score = _import_sklearn()

    cfg = resolve_eval_common_params(eval_params, dataset_artifacts)

    torch.manual_seed(cfg.seed)
    model_info = MODEL_REGISTRY[cfg.model_name]
    tokenizer = model_info.load_tokenizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if cfg.use_bf16 else torch.float32

    # Load p (and keep it alive — needed for the p_self baseline).
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

    by_split = load_rollout_artifacts(
        cfg.dataset_artifacts, tokenizer, filter_train_split=False
    )
    prompts = by_split.get(cfg.split, [])
    if not prompts:
        raise ValueError(f"No data found for split={cfg.split}")
    if cfg.max_prompts is not None and len(prompts) > cfg.max_prompts:
        rng = random.Random(cfg.seed)
        prompts = rng.sample(prompts, cfg.max_prompts)
    log.info(f"Evaluating on {len(prompts)} prompts (split={cfg.split})")

    tokenizer.no_padding()
    tokenizer.no_truncation()

    # Per-completion arrays for AUC computation.
    all_q_scores: list[float] = []
    all_p_scores: list[float] = []
    all_is_correct: list[int] = []

    correct_q_scores: list[float] = []
    incorrect_q_scores: list[float] = []
    correct_p_scores: list[float] = []
    incorrect_p_scores: list[float] = []
    correct_cot_lens: list[int] = []
    incorrect_cot_lens: list[int] = []
    correct_has_answer_tag = 0
    incorrect_has_answer_tag = 0

    n_skipped_too_long = 0
    n_prompts_evaluated = 0
    q_best_of_n_correct = 0
    p_best_of_n_correct = 0
    best_of_n_total = 0
    random_correct = 0
    random_total = 0
    selection_rng = random.Random(cfg.seed + 1)
    examples: list[extty.Example] = []

    for pr_idx, pr in enumerate(prompts):
        if not pr.completions or pr.equation is None:
            continue

        prompt_str = tokenizer.decode(pr.prompt_ids)

        # Per-completion (q_score, p_score, is_correct, response_str) tuples
        scored: list[tuple[float, float, bool, str]] = []

        for comp in pr.completions:
            answer_str = tokenizer.decode(comp.answer_ids)
            cot_str = tokenizer.decode(comp.cot_ids)
            extracted_answer = extract_from_answer_tags(answer_str)
            has_answer_tag = extracted_answer is not None and len(extracted_answer) > 0

            cot_len = len(comp.cot_ids)

            # q sees prompt + answer as prefix; scores the cot
            q_prefix_ids = pr.prompt_ids + comp.answer_ids
            q_full_ids = q_prefix_ids + comp.cot_ids + [model_info.eos_token_id]
            q_prefix_len = len(q_prefix_ids)

            # p sees prompt only as prefix; scores cot + answer (matches p's
            # actual generation order: <think>cot</think> then <answer>...</answer>)
            p_prefix_ids = pr.prompt_ids
            p_full_ids = (
                p_prefix_ids
                + comp.cot_ids
                + comp.answer_ids
                + [model_info.eos_token_id]
            )
            p_prefix_len = len(p_prefix_ids)

            if len(q_full_ids) > 2048 or len(p_full_ids) > 2048:
                n_skipped_too_long += 1
                continue

            q_score = _score_q_prefix_lm(q, q_full_ids, q_prefix_len, device)
            p_score = _score_p_causal(p, p_full_ids, p_prefix_len, device)

            is_correct = comp.is_correct
            response_str = (
                f"[q={q_score:.3f}] [p={p_score:.3f}] [correct={is_correct}] "
                f"[answer={answer_str}]\n{cot_str}"
            )
            scored.append((q_score, p_score, is_correct, response_str))

            all_q_scores.append(q_score)
            all_p_scores.append(p_score)
            all_is_correct.append(int(is_correct))

            if is_correct:
                correct_q_scores.append(q_score)
                correct_p_scores.append(p_score)
                correct_cot_lens.append(cot_len)
                if has_answer_tag:
                    correct_has_answer_tag += 1
            else:
                incorrect_q_scores.append(q_score)
                incorrect_p_scores.append(p_score)
                incorrect_cot_lens.append(cot_len)
                if has_answer_tag:
                    incorrect_has_answer_tag += 1

        if not scored:
            continue

        n_prompts_evaluated += 1

        # Best-of-N selection for both rankers
        q_best = max(scored, key=lambda x: x[0])
        p_best = max(scored, key=lambda x: x[1])
        best_of_n_total += 1
        if q_best[2]:
            q_best_of_n_correct += 1
        if p_best[2]:
            p_best_of_n_correct += 1

        # Random baseline: pick a random completion (seeded for reproducibility)
        random_comp = selection_rng.choice(scored)
        random_total += 1
        if random_comp[2]:
            random_correct += 1

        if len(examples) < 20:
            sorted_comps = sorted(scored, key=lambda x: x[0], reverse=True)
            shown = sorted_comps[:2] + sorted_comps[-2:]
            examples.append(
                extty.Example(
                    prompt=prompt_str,
                    responses=[c[3] for c in shown],
                    rewards=[
                        {"q_score": c[0], "p_score": c[1], "correct": float(c[2])}
                        for c in shown
                    ],
                    groundtruth=pr.equation,
                )
            )

        if pr_idx % 50 == 0 and pr_idx > 0:
            log.info(
                f"  [{pr_idx}/{len(prompts)}] "
                f"q_correct_mean={sum(correct_q_scores) / max(len(correct_q_scores), 1):.3f} "
                f"q_incorrect_mean={sum(incorrect_q_scores) / max(len(incorrect_q_scores), 1):.3f} "
                f"q_best_of_n={q_best_of_n_correct}/{best_of_n_total} "
                f"p_best_of_n={p_best_of_n_correct}/{best_of_n_total}"
            )

    # ------------- Aggregate -------------
    n_completions = len(all_is_correct)
    n_correct = sum(all_is_correct)
    n_incorrect = n_completions - n_correct
    prevalence_correct = n_correct / max(n_completions, 1)
    prevalence_incorrect = n_incorrect / max(n_completions, 1)

    correct_q_mean = sum(correct_q_scores) / max(len(correct_q_scores), 1)
    incorrect_q_mean = sum(incorrect_q_scores) / max(len(incorrect_q_scores), 1)
    q_gap = correct_q_mean - incorrect_q_mean
    correct_p_mean = sum(correct_p_scores) / max(len(correct_p_scores), 1)
    incorrect_p_mean = sum(incorrect_p_scores) / max(len(incorrect_p_scores), 1)
    p_gap = correct_p_mean - incorrect_p_mean

    q_best_of_n_acc = q_best_of_n_correct / max(best_of_n_total, 1)
    p_best_of_n_acc = p_best_of_n_correct / max(best_of_n_total, 1)
    random_acc = random_correct / max(random_total, 1)

    # AUCs require both classes to be present. With "correct" as the positive
    # label, ROC AUC = P(q_score(correct) > q_score(incorrect)), PR AUC =
    # area under precision-recall when ranking by score. Flipping the score
    # gives the same ROC AUC but the "incorrect as positive" PR AUC, which
    # is the right framing for the filtering use case.
    can_auc = n_correct > 0 and n_incorrect > 0
    if can_auc:
        y_true = all_is_correct
        q_roc_auc = float(roc_auc_score(y_true, all_q_scores))
        p_roc_auc = float(roc_auc_score(y_true, all_p_scores))
        q_pr_auc_on_correct = float(average_precision_score(y_true, all_q_scores))
        p_pr_auc_on_correct = float(average_precision_score(y_true, all_p_scores))
        # Flip labels and scores to compute PR AUC with "incorrect" as positive
        y_true_incorrect = [1 - c for c in all_is_correct]
        q_pr_auc_on_incorrect = float(
            average_precision_score(y_true_incorrect, [-s for s in all_q_scores])
        )
        p_pr_auc_on_incorrect = float(
            average_precision_score(y_true_incorrect, [-s for s in all_p_scores])
        )
    else:
        q_roc_auc = p_roc_auc = 0.0
        q_pr_auc_on_correct = p_pr_auc_on_correct = 0.0
        q_pr_auc_on_incorrect = p_pr_auc_on_incorrect = 0.0
        log.warning(
            "Only one class present in the eval set; AUC metrics will be uninformative."
        )

    # ------------- Log -------------
    log.info(f"\nResults ({n_prompts_evaluated} prompts, {n_completions} completions):")
    log.info(
        f"  prevalence_correct = {prevalence_correct:.4f}, "
        f"prevalence_incorrect = {prevalence_incorrect:.4f}"
    )
    log.info("")
    log.info(
        "  Discrimination (higher is better; lift_vs_p_self > 0 = q beats p alone):"
    )
    log.info(
        f"    roc_auc:            q={q_roc_auc:.4f}  p_self={p_roc_auc:.4f}  "
        f"lift={q_roc_auc - p_roc_auc:+.4f}  (random floor = 0.5)"
    )
    log.info(
        f"    pr_auc on correct:  q={q_pr_auc_on_correct:.4f}  p_self={p_pr_auc_on_correct:.4f}  "
        f"lift={q_pr_auc_on_correct - p_pr_auc_on_correct:+.4f}  "
        f"(random floor = {prevalence_correct:.4f})"
    )
    log.info(
        f"    pr_auc on incorrect:q={q_pr_auc_on_incorrect:.4f}  p_self={p_pr_auc_on_incorrect:.4f}  "
        f"lift={q_pr_auc_on_incorrect - p_pr_auc_on_incorrect:+.4f}  "
        f"(random floor = {prevalence_incorrect:.4f})"
    )
    log.info("")
    log.info("  Selection (best-of-N):")
    log.info(
        f"    q   : {q_best_of_n_acc:.4f} ({q_best_of_n_correct}/{best_of_n_total})"
    )
    log.info(
        f"    p_self: {p_best_of_n_acc:.4f} ({p_best_of_n_correct}/{best_of_n_total})"
    )
    log.info(f"    random: {random_acc:.4f} ({random_correct}/{random_total})")
    log.info(
        f"    lift (q − p_self):  {q_best_of_n_acc - p_best_of_n_acc:+.4f}    "
        f"lift (q − random): {q_best_of_n_acc - random_acc:+.4f}"
    )
    log.info("")
    log.info("  Score gaps (correct − incorrect):")
    log.info(
        f"    q     : {q_gap:.4f} (correct={correct_q_mean:.4f} incorrect={incorrect_q_mean:.4f})"
    )
    log.info(
        f"    p_self: {p_gap:.4f} (correct={correct_p_mean:.4f} incorrect={incorrect_p_mean:.4f})"
    )

    log.info("\n  Diagnostics:")
    log.info(f"    Skipped (too long): {n_skipped_too_long}")
    log.info(
        f"    Correct: {correct_has_answer_tag}/{n_correct} have answer tags "
        f"({correct_has_answer_tag / max(n_correct, 1):.0%})"
    )
    log.info(
        f"    Incorrect: {incorrect_has_answer_tag}/{n_incorrect} have answer tags "
        f"({incorrect_has_answer_tag / max(n_incorrect, 1):.0%})"
    )
    avg_correct_cot = sum(correct_cot_lens) / max(len(correct_cot_lens), 1)
    avg_incorrect_cot = sum(incorrect_cot_lens) / max(len(incorrect_cot_lens), 1)
    log.info(
        f"    Avg CoT length — correct: {avg_correct_cot:.0f}, incorrect: {avg_incorrect_cot:.0f}"
    )

    if extty.has_active_run():
        metrics: dict = {
            "n_prompts": n_prompts_evaluated,
            "n_completions": n_completions,
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "prevalence_correct": prevalence_correct,
            "prevalence_incorrect": prevalence_incorrect,
            # q metrics
            "q_score_correct_mean": correct_q_mean,
            "q_score_incorrect_mean": incorrect_q_mean,
            "q_score_gap": q_gap,
            "q_roc_auc": q_roc_auc,
            "q_pr_auc_on_correct": q_pr_auc_on_correct,
            "q_pr_auc_on_incorrect": q_pr_auc_on_incorrect,
            "q_best_of_n_accuracy": q_best_of_n_acc,
            # p_self baseline metrics
            "p_self_score_correct_mean": correct_p_mean,
            "p_self_score_incorrect_mean": incorrect_p_mean,
            "p_self_score_gap": p_gap,
            "p_self_roc_auc": p_roc_auc,
            "p_self_pr_auc_on_correct": p_pr_auc_on_correct,
            "p_self_pr_auc_on_incorrect": p_pr_auc_on_incorrect,
            "p_self_best_of_n_accuracy": p_best_of_n_acc,
            # random selection baseline
            "random_best_of_n_accuracy": random_acc,
            # apples-to-apples lifts vs p_self
            "roc_auc_lift_vs_p_self": q_roc_auc - p_roc_auc,
            "pr_auc_on_correct_lift_vs_p_self": q_pr_auc_on_correct
            - p_pr_auc_on_correct,
            "pr_auc_on_incorrect_lift_vs_p_self": q_pr_auc_on_incorrect
            - p_pr_auc_on_incorrect,
            "best_of_n_lift_vs_p_self": q_best_of_n_acc - p_best_of_n_acc,
            # diagnostics
            "avg_correct_cot_len": avg_correct_cot,
            "avg_incorrect_cot_len": avg_incorrect_cot,
            "correct_answer_tag_pct": correct_has_answer_tag / max(n_correct, 1),
            "incorrect_answer_tag_pct": incorrect_has_answer_tag / max(n_incorrect, 1),
            "n_skipped_too_long": n_skipped_too_long,
        }
        if examples:
            metrics["examples"] = extty.BatchExample(
                prompts=[e.prompt for e in examples],
                responses=[e.responses for e in examples],
                rewards=[e.rewards for e in examples],
                groundtruth=[e.groundtruth for e in examples],
            )
        extty.log(metrics, step=0)


@extty.experiment(project="eval-q-verifier-countdown")
def eval_q_verifier_countdown(
    *,
    eval_params: EvalCommonParams,
    dataset_artifacts: list[str] | None = None,
) -> None:
    _eval_q_verifier(eval_params=eval_params, dataset_artifacts=dataset_artifacts)


@extty.experiment(project="eval-q-verifier-gsm8k")
def eval_q_verifier_gsm8k(
    *,
    eval_params: EvalCommonParams,
    dataset_artifacts: list[str] | None = None,
) -> None:
    _eval_q_verifier(eval_params=eval_params, dataset_artifacts=dataset_artifacts)


if __name__ == "__main__":
    run_experiments_parser(
        [
            Experiment(
                env_name="countdown",
                fn=eval_q_verifier_countdown,
                include_dataset_glob=True,
                include_prompt_collection_id=False,
                dataset_glob_required=False,
            ),
            Experiment(
                env_name="gsm8k",
                fn=eval_q_verifier_gsm8k,
                include_dataset_glob=True,
                include_prompt_collection_id=False,
                dataset_glob_required=False,
            ),
        ]
    )
