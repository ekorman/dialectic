"""Evaluate q as a verifier: does q(CoT | answer, P) separate correct from incorrect rollouts?

For each prompt in the rollout data:
1. For each completion (CoT + answer from p):
   - Compute q(CoT | extracted_answer, P) — log prob of the CoT under q conditioned on p's own answer
   - Compute p(CoT, answer | P) — log prob under causal p as a baseline
2. Compare scores for correct vs incorrect completions
3. Report: ROC AUC, PR AUC, score gap, and best-of-N selection accuracy — for
   q against two baselines: ``p_self`` and ``length`` (see below).

Two baselines frame every metric:

- ``p_self``: score each completion by p's own causal log prob. Answers "did
  training q on ``p(CoT | answer)`` beat just asking p its confidence?" If
  ``q_roc_auc <= p_self_roc_auc`` then q is just rediscovering p's intrinsic
  uncertainty. (In practice p ≈ non-discriminating: the completions are p's
  own samples, so its avg per-token log prob is ~flat across correctness.)
- ``length``: score by ``-cot_len`` (shorter CoT ⇒ predict correct). In
  countdown, wrong rollouts ramble, so length alone is a strong but
  content-blind selector — and pooled AUC is dominated by it. The number that
  actually measures q's reasoning signal is its lift over ``length``, not over
  ``p_self`` or random.
- ``self_consist``: model-free majority vote — score each completion by how
  many of the prompt's completions agree on its answer VALUE (env-specific:
  countdown evaluates the expression, gsm8k parses the number). This is the
  cheap RIVAL method, not a confound control: if q can't beat it on fixed-N
  selection, q's edge has to come from its per-item capabilities (early-stop,
  N=1, abstention) rather than raw selection accuracy.

Because pooled AUC over-credits the length confound, the HEADLINE selection
metric is best-of-N on MIXED prompts only (those with both a correct and an
incorrect completion) — all-correct/all-incorrect prompts are dropped since
every ranker scores identically there. A length-stratified split (does q still
pick right when the shortest CoT is wrong?) isolates q's non-length signal.

Correctness is read from the rollout artifact's ``is_correct`` flag, so this
launcher is env-agnostic — same body for countdown and gsm8k.
"""

import copy
import random
from collections import Counter
from typing import Callable, Hashable

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
from dialectic.rl.inverse_cot_data import load_rollout_artifact

AnswerValueFn = Callable[[str | None], Hashable | None]
"""Maps an extracted answer string to a hashable vote value for the
self-consistency baseline, or ``None`` when it can't be parsed (those
completions cast no vote). Env-specific: countdown evaluates the expression so
that all target-hitting answers collapse to one value; gsm8k parses the number."""


def _countdown_answer_value(ans: str | None) -> Hashable | None:
    """Evaluate a countdown answer expression to its numeric value so that
    distinct-but-equivalent correct expressions (e.g. ``(8+75)+(25*10)`` and
    ``((25*10)+75+8)``) vote together on the same target value."""
    if ans is None:
        return None
    try:
        return round(float(eval(ans, {"__builtins__": {}}, {})), 6)
    except Exception:
        return None


def _gsm8k_answer_value(ans: str | None) -> Hashable | None:
    """Parse a gsm8k answer to a numeric value (thousands-commas stripped), so
    ``"1,000"`` and ``"1000"`` vote together."""
    if ans is None:
        return None
    try:
        return round(float(ans.replace(",", "").strip()), 6)
    except Exception:
        return None


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
    *, eval_params: EvalCommonParams, answer_value_fn: AnswerValueFn
) -> None:
    roc_auc_score, average_precision_score = _import_sklearn()

    cfg = resolve_eval_common_params(eval_params)

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

    by_split = load_rollout_artifact(
        cfg.p_rollout_artifact, tokenizer, filter_train_split=False
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
    # CoT token length per completion — drives the length-only confound
    # baseline (shorter CoT ⇒ predict correct).
    all_cot_len: list[int] = []
    # Self-consistency score per completion — fraction of the prompt's
    # completions that agree on this completion's answer value (a model-free
    # majority-vote baseline). Prompt-normalized so it's comparable in the
    # pooled AUC.
    all_sc_score: list[float] = []

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
    length_best_of_n_correct = 0
    sc_best_of_n_correct = 0
    best_of_n_total = 0
    random_correct = 0
    random_total = 0
    # Mixed-only selection (prompts with BOTH a correct and an incorrect
    # completion) — the headline selection metric. All-correct/all-incorrect
    # prompts are dropped because every ranker scores identically there.
    mixed_total = 0
    q_mixed_correct = 0
    p_mixed_correct = 0
    length_mixed_correct = 0
    sc_mixed_correct = 0
    random_mixed_correct = 0
    # Length-stratified within mixed prompts: split by whether the length
    # ranker's own pick is correct. q's accuracy on the "adversarial" stratum
    # (shortest CoT is wrong) is q signal that length cannot explain.
    len_aligned_total = 0
    len_adv_total = 0
    q_len_aligned_correct = 0
    q_len_adv_correct = 0
    p_len_adv_correct = 0
    sc_len_adv_correct = 0
    random_len_adv_correct = 0
    selection_rng = random.Random(cfg.seed + 1)
    examples: list[extty.Example] = []
    # Per-completion score dump (only when cfg.dump_scores). One entry per
    # evaluated prompt, each a list of its completions' records — the prompt
    # grouping is what the early-stopping Pareto simulation needs.
    per_prompt_scores: list[list[dict]] = []

    for pr_idx, pr in enumerate(prompts):
        if not pr.completions or pr.equation is None:
            continue

        prompt_str = tokenizer.decode(pr.prompt_ids)

        # Per-completion (q_score, p_score, is_correct, response_str, cot_len,
        # answer_value) tuples. answer_value drives the self-consistency vote.
        scored: list[tuple[float, float, bool, str, int, Hashable | None]] = []

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
            answer_value = answer_value_fn(extracted_answer)
            response_str = (
                f"[q={q_score:.3f}] [p={p_score:.3f}] [correct={is_correct}] "
                f"[answer={answer_str}]\n{cot_str}"
            )
            scored.append(
                (q_score, p_score, is_correct, response_str, cot_len, answer_value)
            )

            all_q_scores.append(q_score)
            all_p_scores.append(p_score)
            all_is_correct.append(int(is_correct))
            all_cot_len.append(cot_len)

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

        if cfg.dump_scores:
            per_prompt_scores.append(
                [
                    {
                        "q": x[0],
                        "p": x[1],
                        "correct": bool(x[2]),
                        "val": x[5],
                        "cot_len": x[4],
                    }
                    for x in scored
                ]
            )

        # Best-of-N selection for each ranker. `length` is a confound
        # baseline — "pick the shortest CoT". In countdown, wrong rollouts
        # ramble, so length alone is a strong but content-blind selector; q
        # only earns its keep by beating it. `random` is seeded for
        # reproducibility.
        q_best = max(scored, key=lambda x: x[0])
        p_best = max(scored, key=lambda x: x[1])
        length_best = min(scored, key=lambda x: x[4])
        random_comp = selection_rng.choice(scored)

        # Self-consistency (majority vote): score each completion by how many
        # of the prompt's completions agree on its answer VALUE — a model-free
        # rival, not a confound control. Pick the modal value, ties broken
        # randomly. Completions with an unparseable answer (value None) cast no
        # vote. The per-completion normalized vote share feeds the pooled AUC.
        value_counts = Counter(x[5] for x in scored if x[5] is not None)
        n_comp = len(scored)
        sc_votes = [
            value_counts.get(x[5], 0) if x[5] is not None else 0 for x in scored
        ]
        for v in sc_votes:
            all_sc_score.append(v / n_comp)
        max_votes = max(sc_votes, default=0)
        sc_best = selection_rng.choice(
            [x for x, v in zip(scored, sc_votes) if v == max_votes]
        )

        best_of_n_total += 1
        random_total += 1
        if q_best[2]:
            q_best_of_n_correct += 1
        if p_best[2]:
            p_best_of_n_correct += 1
        if length_best[2]:
            length_best_of_n_correct += 1
        if sc_best[2]:
            sc_best_of_n_correct += 1
        if random_comp[2]:
            random_correct += 1

        # Mixed-only selection (headline metric): on all-correct or
        # all-incorrect prompts every ranker's pick lands the same, so they
        # only dilute the comparison. Restrict to prompts with both classes.
        has_correct = any(x[2] for x in scored)
        has_incorrect = any(not x[2] for x in scored)
        if has_correct and has_incorrect:
            mixed_total += 1
            if q_best[2]:
                q_mixed_correct += 1
            if p_best[2]:
                p_mixed_correct += 1
            if length_best[2]:
                length_mixed_correct += 1
            if sc_best[2]:
                sc_mixed_correct += 1
            if random_comp[2]:
                random_mixed_correct += 1

            # Length-stratified (residual-confound isolation): split mixed
            # prompts by whether the length ranker's own pick is correct. On
            # the adversarial stratum the shortest CoT is wrong, so any
            # accuracy q keeps there is content signal NOT explained by length
            # (length acc is 1.0 on aligned, 0.0 on adversarial by construction).
            if length_best[2]:
                len_aligned_total += 1
                if q_best[2]:
                    q_len_aligned_correct += 1
            else:
                len_adv_total += 1
                if q_best[2]:
                    q_len_adv_correct += 1
                if p_best[2]:
                    p_len_adv_correct += 1
                if sc_best[2]:
                    sc_len_adv_correct += 1
                if random_comp[2]:
                    random_len_adv_correct += 1

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
    length_best_of_n_acc = length_best_of_n_correct / max(best_of_n_total, 1)
    sc_best_of_n_acc = sc_best_of_n_correct / max(best_of_n_total, 1)
    random_acc = random_correct / max(random_total, 1)

    q_mixed_acc = q_mixed_correct / max(mixed_total, 1)
    p_mixed_acc = p_mixed_correct / max(mixed_total, 1)
    length_mixed_acc = length_mixed_correct / max(mixed_total, 1)
    sc_mixed_acc = sc_mixed_correct / max(mixed_total, 1)
    random_mixed_acc = random_mixed_correct / max(mixed_total, 1)

    q_len_aligned_acc = q_len_aligned_correct / max(len_aligned_total, 1)
    q_len_adv_acc = q_len_adv_correct / max(len_adv_total, 1)
    p_len_adv_acc = p_len_adv_correct / max(len_adv_total, 1)
    sc_len_adv_acc = sc_len_adv_correct / max(len_adv_total, 1)
    random_len_adv_acc = random_len_adv_correct / max(len_adv_total, 1)

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
        # Length-only confound baseline: shorter CoT ⇒ correct, so the
        # "correct-positive" score is -cot_len (and +cot_len for incorrect).
        length_roc_auc = float(roc_auc_score(y_true, [-l for l in all_cot_len]))
        length_pr_auc_on_correct = float(
            average_precision_score(y_true, [-l for l in all_cot_len])
        )
        length_pr_auc_on_incorrect = float(
            average_precision_score(y_true_incorrect, all_cot_len)
        )
        # Self-consistency: higher vote share ⇒ predict correct. (Only ROC-AUC
        # is kept — the PR-AUC split was pruned from the logged metrics.)
        sc_roc_auc = float(roc_auc_score(y_true, all_sc_score))
    else:
        q_roc_auc = p_roc_auc = 0.0
        q_pr_auc_on_correct = p_pr_auc_on_correct = 0.0
        q_pr_auc_on_incorrect = p_pr_auc_on_incorrect = 0.0
        length_roc_auc = 0.0
        length_pr_auc_on_correct = length_pr_auc_on_incorrect = 0.0
        sc_roc_auc = 0.0
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
        "  Discrimination (pooled over all completions — inflated by CoT-length "
        "confound; see length column):"
    )
    log.info(
        f"    roc_auc:            q={q_roc_auc:.4f}  p_self={p_roc_auc:.4f}  "
        f"length={length_roc_auc:.4f}  self_consist={sc_roc_auc:.4f}  "
        f"lift_vs_len={q_roc_auc - length_roc_auc:+.4f}  "
        f"lift_vs_sc={q_roc_auc - sc_roc_auc:+.4f}  (random floor = 0.5)"
    )
    log.info(
        f"    pr_auc on correct:  q={q_pr_auc_on_correct:.4f}  p_self={p_pr_auc_on_correct:.4f}  "
        f"length={length_pr_auc_on_correct:.4f}  "
        f"lift_vs_p={q_pr_auc_on_correct - p_pr_auc_on_correct:+.4f}  "
        f"lift_vs_len={q_pr_auc_on_correct - length_pr_auc_on_correct:+.4f}  "
        f"(random floor = {prevalence_correct:.4f})"
    )
    log.info(
        f"    pr_auc on incorrect:q={q_pr_auc_on_incorrect:.4f}  p_self={p_pr_auc_on_incorrect:.4f}  "
        f"length={length_pr_auc_on_incorrect:.4f}  "
        f"lift_vs_p={q_pr_auc_on_incorrect - p_pr_auc_on_incorrect:+.4f}  "
        f"lift_vs_len={q_pr_auc_on_incorrect - length_pr_auc_on_incorrect:+.4f}  "
        f"(random floor = {prevalence_incorrect:.4f})"
    )
    log.info("")
    log.info("  Selection — best-of-N, ALL prompts (incl. all-correct/all-incorrect):")
    log.info(
        f"    q     : {q_best_of_n_acc:.4f} ({q_best_of_n_correct}/{best_of_n_total})"
    )
    log.info(
        f"    p_self: {p_best_of_n_acc:.4f} ({p_best_of_n_correct}/{best_of_n_total})"
    )
    log.info(
        f"    length: {length_best_of_n_acc:.4f} ({length_best_of_n_correct}/{best_of_n_total})"
    )
    log.info(
        f"    self_consist: {sc_best_of_n_acc:.4f} ({sc_best_of_n_correct}/{best_of_n_total})"
    )
    log.info(f"    random: {random_acc:.4f} ({random_correct}/{random_total})")
    log.info("")
    log.info(
        f"  Selection — best-of-N, MIXED prompts only ({mixed_total}/{best_of_n_total}) "
        "[HEADLINE — all-correct/all-incorrect dropped; the pick can't matter there]:"
    )
    log.info(f"    q           : {q_mixed_acc:.4f} ({q_mixed_correct}/{mixed_total})")
    log.info(f"    p_self      : {p_mixed_acc:.4f} ({p_mixed_correct}/{mixed_total})")
    log.info(
        f"    length      : {length_mixed_acc:.4f} ({length_mixed_correct}/{mixed_total})"
    )
    log.info(f"    self_consist: {sc_mixed_acc:.4f} ({sc_mixed_correct}/{mixed_total})")
    log.info(
        f"    random      : {random_mixed_acc:.4f} ({random_mixed_correct}/{mixed_total})"
    )
    log.info(
        f"    lift (q − self_consist): {q_mixed_acc - sc_mixed_acc:+.4f}    "
        f"lift (q − length): {q_mixed_acc - length_mixed_acc:+.4f}    "
        f"lift (q − p_self): {q_mixed_acc - p_mixed_acc:+.4f}"
    )
    log.info("")
    log.info(
        "  Length-stratified (mixed prompts; isolates q signal beyond CoT length):"
    )
    log.info(
        f"    length-aligned     (shortest CoT correct, {len_aligned_total} prompts): "
        f"q={q_len_aligned_acc:.4f}  (length=1.0000 by construction)"
    )
    log.info(
        f"    length-adversarial (shortest CoT wrong,   {len_adv_total} prompts): "
        f"q={q_len_adv_acc:.4f}  self_consist={sc_len_adv_acc:.4f}  "
        f"p_self={p_len_adv_acc:.4f}  random={random_len_adv_acc:.4f}  "
        f"(length=0.0000 by construction)"
    )
    log.info(
        "    -> q accuracy on the adversarial stratum is q's content signal that "
        "CoT length cannot explain; self_consist is the model-free rival there."
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
        # Lean metric set — one series per claim we actually make. The console
        # log above stays verbose (means, gaps, PR-AUCs, per-baseline lifts);
        # this dict is what becomes plottable series, so it's pruned to the
        # confound story: pooled AUC is inflated -> q beats the length confound
        # -> q beats the self-consistency rival -> q's signal survives where
        # length is adversarial.
        metrics: dict = {
            # data + confound evidence
            "n_prompts": n_prompts_evaluated,
            "n_completions": n_completions,
            "n_correct": n_correct,
            "prevalence_correct": prevalence_correct,
            "q_score_gap": q_gap,
            "avg_correct_cot_len": avg_correct_cot,
            "avg_incorrect_cot_len": avg_incorrect_cot,
            "n_skipped_too_long": n_skipped_too_long,
            # pooled AUC — kept only to show the length inflation (not a
            # quality measure). p_self_roc_auc documents p's flat/<0.5 artifact.
            "q_roc_auc": q_roc_auc,
            "length_roc_auc": length_roc_auc,
            "self_consist_roc_auc": sc_roc_auc,
            "p_self_roc_auc": p_roc_auc,
            # full-set selection anchor
            "q_best_of_n_accuracy": q_best_of_n_acc,
            # HEADLINE — mixed-only best-of-N (all-correct/all-incorrect dropped)
            "n_mixed_prompts": mixed_total,
            "q_mixed_best_of_n_accuracy": q_mixed_acc,
            "self_consist_mixed_best_of_n_accuracy": sc_mixed_acc,
            "length_mixed_best_of_n_accuracy": length_mixed_acc,
            "p_self_mixed_best_of_n_accuracy": p_mixed_acc,
            "random_mixed_best_of_n_accuracy": random_mixed_acc,
            "mixed_best_of_n_lift_vs_length": q_mixed_acc - length_mixed_acc,
            "mixed_best_of_n_lift_vs_self_consist": q_mixed_acc - sc_mixed_acc,
            # residual-confound isolation — length-adversarial stratum
            "n_len_aligned_prompts": len_aligned_total,
            "n_len_adversarial_prompts": len_adv_total,
            "q_len_aligned_best_of_n_accuracy": q_len_aligned_acc,
            "q_len_adversarial_best_of_n_accuracy": q_len_adv_acc,
            "self_consist_len_adversarial_best_of_n_accuracy": sc_len_adv_acc,
            "random_len_adversarial_best_of_n_accuracy": random_len_adv_acc,
        }
        if examples:
            metrics["examples"] = extty.BatchExample(
                prompts=[e.prompt for e in examples],
                responses=[e.responses for e in examples],
                rewards=[e.rewards for e in examples],
                groundtruth=[e.groundtruth for e in examples],
            )
        extty.log(metrics, step=0)

    if cfg.dump_scores and extty.has_active_run():
        import json
        import os
        import tempfile

        # Artifact names are globally unique in extty, so bake the q checkpoint
        # AND rollout-artifact identity into the name — same q evaluated on two
        # rollout sets (e.g. GSM-Symbolic p1 vs p2, both split "train") must
        # not collide.
        q_run_slug = cfg.q_ckpt_run.split("/")[-1]
        rollout_slug = cfg.p_rollout_artifact.rsplit("-", 1)[-1]
        artifact_name = (
            f"q-verifier-scores-{cfg.split}-{rollout_slug}-{q_run_slug}"
            f"-s{cfg.q_ckpt_step}"
        )
        jsonl = "\n".join(json.dumps(rec) for rec in per_prompt_scores)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fh:
            fh.write(jsonl)
            tmp_path = fh.name
        try:
            meta = extty.save_artifact(
                artifact_name,
                tmp_path,
                description=(
                    f"Per-completion q/p scores on {cfg.split} split "
                    f"({len(per_prompt_scores)} prompts) for offline early-stopping "
                    "Pareto analysis."
                ),
                metadata={
                    "q_ckpt_run": cfg.q_ckpt_run,
                    "q_ckpt_step": cfg.q_ckpt_step,
                    "p_rollout_artifact": cfg.p_rollout_artifact,
                    "split": cfg.split,
                    "max_prompts": cfg.max_prompts,
                },
            )
        finally:
            os.unlink(tmp_path)
        log.info(f"Saved per-completion scores artifact: {meta.name}")


@extty.experiment(project="eval-q-verifier-countdown")
def eval_q_verifier_countdown(*, eval_params: EvalCommonParams) -> None:
    _eval_q_verifier(eval_params=eval_params, answer_value_fn=_countdown_answer_value)


@extty.experiment(project="eval-q-verifier-gsm8k")
def eval_q_verifier_gsm8k(*, eval_params: EvalCommonParams) -> None:
    _eval_q_verifier(eval_params=eval_params, answer_value_fn=_gsm8k_answer_value)


if __name__ == "__main__":
    run_experiments_parser(
        [
            Experiment(
                env_name="countdown",
                fn=eval_q_verifier_countdown,
                include_prompt_collection_id=False,
            ),
            Experiment(
                env_name="gsm8k",
                fn=eval_q_verifier_gsm8k,
                include_prompt_collection_id=False,
            ),
        ]
    )
