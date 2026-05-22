"""Shared parsing / grading / group-filter logic for rollout launchers.

The vLLM rollout launcher calls into this helper after its generation step.
Each env (countdown, GSM8K, …) plugs in two callables:

- ``grade_fn(env_response, extracted) -> bool`` — decides ``is_correct``
  given the gold problem data and the string extracted from
  ``<answer>…</answer>``.
- ``entry_extra_fn(env_response) -> dict`` — emits env-specific fields
  baked into the kept JSONL entry (e.g. countdown's ``numbers`` / ``target``).

Everything else — parsing the completion into ``(cot, answer)``, counting
per-batch stats, enforcing ``n_pos_min`` / ``n_neg_min``, building the
output entry — is shared in ``_filter_rollouts``.
"""

from dataclasses import dataclass
from typing import Callable

from dialectic.log import log
from dialectic.rl.env import Countdown, MathState
from dialectic.rl.extractors import extract_from_answer_tags, parse_cot_and_answer
from dialectic.rl.inverse_cot_eval import gsm8k_match
from dialectic.rl.reward import _evaluate_and_verify_countdown
from dialectic.rl.types import EnvResponse


@dataclass
class _BatchCounts:
    correct: int = 0
    incorrect: int = 0
    unparsed: int = 0


def _filter_rollouts(
    *,
    prompts: list[str],
    env_responses: list[EnvResponse],
    extra_fields: list[dict],
    completions_by_problem: list[list[str]],
    n_pos_min: int,
    n_neg_min: int,
    grade_fn: Callable[[EnvResponse, str | None], bool],
    entry_extra_fn: Callable[[EnvResponse], dict],
) -> list[dict]:
    """Parse a batch of rollouts and keep prompts with enough pos/neg samples.

    Parameters
    ----------
    prompts
        Chat-templated prompt strings, one per problem.
    env_responses
        Per-problem env data; passed through to ``grade_fn`` and
        ``entry_extra_fn`` (the only two env-aware callbacks).
    extra_fields
        Passthrough dicts (e.g. ``split``, ``equation``) merged into the
        returned entries.
    completions_by_problem
        Outer axis is problems (length == ``len(prompts)``), inner axis is
        the group of samples drawn for that problem.
    n_pos_min, n_neg_min
        Minimum counts of correct / incorrect samples required for a prompt
        to be emitted.
    grade_fn
        Returns whether the model's extracted ``<answer>…</answer>`` content
        is correct for the given problem.
    entry_extra_fn
        Returns env-specific fields to embed in the emitted JSONL entry.
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
            is_correct = grade_fn(env_responses[b], extracted)
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
                    "completions": group_completions,
                    **entry_extra_fn(env_responses[b]),
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


def filter_countdown_rollouts(
    *,
    prompts: list[str],
    env_responses: list[EnvResponse[Countdown]],
    extra_fields: list[dict],
    completions_by_problem: list[list[str]],
    n_pos_min: int,
    n_neg_min: int,
) -> list[dict]:
    """Countdown-flavored wrapper: grade by arithmetic-expression check."""
    return _filter_rollouts(
        prompts=prompts,
        env_responses=env_responses,
        extra_fields=extra_fields,
        completions_by_problem=completions_by_problem,
        n_pos_min=n_pos_min,
        n_neg_min=n_neg_min,
        grade_fn=lambda er, extracted: extracted is not None
        and _evaluate_and_verify_countdown(extracted, er.data.numbers, er.data.target),
        entry_extra_fn=lambda er: {
            "numbers": er.data.numbers,
            "target": er.data.target,
        },
    )


def filter_gsm8k_rollouts(
    *,
    prompts: list[str],
    env_responses: list[EnvResponse[MathState]],
    extra_fields: list[dict],
    completions_by_problem: list[list[str]],
    n_pos_min: int,
    n_neg_min: int,
) -> list[dict]:
    """GSM8K-flavored wrapper: grade by numeric equality.

    The gold answer string is also surfaced into the emitted entry via the
    caller's ``extra_fields[b]["equation"]`` (see ``_load_gsm8k_problems`` in
    ``generate_inverse_cot_rollouts_vllm``), so ``entry_extra_fn`` itself is
    a no-op here.
    """
    return _filter_rollouts(
        prompts=prompts,
        env_responses=env_responses,
        extra_fields=extra_fields,
        completions_by_problem=completions_by_problem,
        n_pos_min=n_pos_min,
        n_neg_min=n_neg_min,
        grade_fn=lambda er, extracted: gsm8k_match(extracted, er.data.answer),
        entry_extra_fn=lambda _er: {},
    )
