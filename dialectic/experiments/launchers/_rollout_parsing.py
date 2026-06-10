"""Shared parsing / grading logic for rollout launchers.

The vLLM rollout launcher calls into this helper after its generation step.
Each env (countdown, GSM8K, …) plugs in two callables:

- ``grade_fn(env_response, extracted) -> bool`` — decides ``is_correct``
  given the gold problem data and the string extracted from
  ``<answer>…</answer>``.
- ``entry_extra_fn(env_response) -> dict`` — emits env-specific fields
  baked into the kept JSONL entry (e.g. countdown's ``numbers`` / ``target``).

Everything else — parsing the completion into ``(cot, answer)``, counting
per-batch stats, building the output entry — is shared in
``_parse_rollouts``.

Note: every prompt is emitted, regardless of its correctness mix (even with
zero parseable completions). Rollout artifacts are COMPLETE by convention;
any filtering (mixed-correctness for q training, ``is_correct`` for SFT, …)
happens explicitly at load time in the consumer.
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


def _parse_rollouts(
    *,
    prompts: list[str],
    env_responses: list[EnvResponse],
    extra_fields: list[dict],
    completions_by_problem: list[list[str]],
    grade_fn: Callable[[EnvResponse, str | None], bool],
    entry_extra_fn: Callable[[EnvResponse], dict],
) -> list[dict]:
    """Parse and grade a batch of rollouts into JSONL-ready entries.

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
    grade_fn
        Returns whether the model's extracted ``<answer>…</answer>`` content
        is correct for the given problem.
    entry_extra_fn
        Returns env-specific fields to embed in the emitted JSONL entry.
    """
    counts = _BatchCounts()
    entries: list[dict] = []

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

        entries.append(
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
        f"{counts.unparsed}/{total} unparsed"
    )
    return entries


def parse_countdown_rollouts(
    *,
    prompts: list[str],
    env_responses: list[EnvResponse[Countdown]],
    extra_fields: list[dict],
    completions_by_problem: list[list[str]],
) -> list[dict]:
    """Countdown-flavored wrapper: grade by arithmetic-expression check."""
    return _parse_rollouts(
        prompts=prompts,
        env_responses=env_responses,
        extra_fields=extra_fields,
        completions_by_problem=completions_by_problem,
        grade_fn=lambda er, extracted: extracted is not None
        and _evaluate_and_verify_countdown(extracted, er.data.numbers, er.data.target),
        entry_extra_fn=lambda er: {
            "numbers": er.data.numbers,
            "target": er.data.target,
        },
    )


def parse_gsm8k_rollouts(
    *,
    prompts: list[str],
    env_responses: list[EnvResponse[MathState]],
    extra_fields: list[dict],
    completions_by_problem: list[list[str]],
) -> list[dict]:
    """GSM8K-flavored wrapper: grade by numeric equality.

    The gold answer string is also surfaced into the emitted entry via the
    caller's ``extra_fields[b]["equation"]`` (see ``_load_gsm8k_problems`` in
    ``generate_inverse_cot_rollouts``), so ``entry_extra_fn`` itself is
    a no-op here.
    """
    return _parse_rollouts(
        prompts=prompts,
        env_responses=env_responses,
        extra_fields=extra_fields,
        completions_by_problem=completions_by_problem,
        grade_fn=lambda er, extracted: gsm8k_match(extracted, er.data.answer),
        entry_extra_fn=lambda _er: {},
    )
