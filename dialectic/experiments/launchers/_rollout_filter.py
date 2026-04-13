"""Shared parsing / grading / group-filter logic for rollout launchers.

Both the pure-PyTorch and vLLM-backed variants of the inverse-CoT rollout
launcher call into this helper after their respective generation step. Keeping
it here means both launchers agree on what counts as "kept" and what the
per-batch log line looks like.
"""

from dataclasses import dataclass

from dialectic.log import log
from dialectic.rl.env import Countdown
from dialectic.rl.extractors import extract_from_answer_tags, parse_cot_and_answer
from dialectic.rl.reward import _evaluate_and_verify_countdown
from dialectic.rl.types import EnvResponse


@dataclass
class _BatchCounts:
    correct: int = 0
    incorrect: int = 0
    unparsed: int = 0


def filter_countdown_rollouts(
    *,
    prompts: list[str],
    env_responses: list[EnvResponse[Countdown]],
    extra_fields: list[dict],
    completions_by_problem: list[list[str]],
    n_pos_min: int,
    n_neg_min: int,
) -> list[dict]:
    """Parse countdown rollouts and keep prompts with enough pos/neg samples.

    Parameters
    ----------
    prompts
        Chat-templated prompt strings, one per problem.
    env_responses
        Per-problem ``Countdown`` env data (numbers + target).
    extra_fields
        Passthrough dicts (e.g. ``split``, ``equation``) merged into the
        returned entries.
    completions_by_problem
        Outer axis is problems (length == ``len(prompts)``), inner axis is
        the group of samples drawn for that problem.
    n_pos_min, n_neg_min
        Minimum counts of correct / incorrect samples required for a prompt
        to be emitted.

    Returns
    -------
    list[dict]
        JSONL-ready entries for the kept prompts.
    """
    counts = _BatchCounts()
    kept: list[dict] = []

    for b, comp_strs in enumerate(completions_by_problem):
        group_completions: list[dict] = []
        for comp_str in comp_strs:
            parsed = parse_cot_and_answer(comp_str)
            if parsed is None:
                counts.unparsed += 1
                continue
            cot, answer = parsed
            if not cot.strip():
                counts.unparsed += 1
                continue

            extracted = extract_from_answer_tags(comp_str)
            is_correct = extracted is not None and _evaluate_and_verify_countdown(
                extracted,
                env_responses[b].data.numbers,
                env_responses[b].data.target,
            )
            if is_correct:
                counts.correct += 1
            else:
                counts.incorrect += 1
            group_completions.append(
                {"cot": cot, "answer": answer, "is_correct": is_correct}
            )

        n_pos = sum(1 for c in group_completions if c["is_correct"])
        n_neg = sum(1 for c in group_completions if not c["is_correct"])
        if n_pos >= n_pos_min and n_neg >= n_neg_min:
            kept.append(
                {
                    "prompt_str": prompts[b],
                    "numbers": env_responses[b].data.numbers,
                    "target": env_responses[b].data.target,
                    "completions": group_completions,
                    **extra_fields[b],
                }
            )

    total = counts.correct + counts.incorrect + counts.unparsed
    log.info(
        f"Batch: {counts.correct}/{total} correct, "
        f"{counts.incorrect}/{total} incorrect, "
        f"{counts.unparsed}/{total} unparsed, "
        f"{len(kept)}/{len(prompts)} prompts kept"
    )
    return kept
