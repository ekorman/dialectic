import re
from typing import Callable, Protocol, Sequence

from dialectic.rl.env import Countdown
from dialectic.rl.types import QA, E, EnvResponse, RewardResult, T

RewardComponentFn = Callable[..., float]


class RewardFn(Protocol[T, E]):
    def __call__(
        self,
        *,
        env_response: EnvResponse[T],
        raw_model_output: str | None = None,
        extracted_model_output: E,
    ) -> RewardResult: ...


def weighted_reward(
    components: Sequence[tuple[str, float, RewardComponentFn]],
) -> RewardFn:
    """Create a reward function as a weighted sum of components."""

    def fn(
        *,
        env_response: EnvResponse,
        raw_model_output: str | None = None,
        extracted_model_output: object,
    ) -> RewardResult:
        values: dict[str, float] = {}
        total = 0.0
        for name, weight, component in components:
            value = component(
                env_response=env_response,
                raw_model_output=raw_model_output,
                extracted_model_output=extracted_model_output,
            )
            values[name] = value
            total += weight * value
        return RewardResult(total=total, components=values)

    return fn


# --- Countdown-specific ---


def _evaluate_and_verify_countdown(expr: str, numbers: list[int], target: int) -> bool:
    if not re.match(r"^[\d+\-*/()\s]+$", expr):
        return False

    used_numbers = sorted(int(n) for n in re.findall(r"\d+", expr))
    if used_numbers != sorted(numbers):
        return False

    try:
        result = eval(expr, {"__builtins__": {}}, {})
        return abs(result - target) < 1e-6
    except Exception:
        return False


def countdown_correct(
    *, env_response: EnvResponse[Countdown], extracted_model_output: str | None, **_
) -> float:
    if extracted_model_output is None:
        return 0.0
    try:
        return (
            1.0
            if _evaluate_and_verify_countdown(
                extracted_model_output,
                env_response.data.numbers,
                env_response.data.target,
            )
            else 0.0
        )
    except Exception:
        return 0.0


def countdown_hybrid_correct(
    *, env_response: EnvResponse[Countdown], raw_model_output: str | None, **_
) -> float:
    if not raw_model_output:
        return 0.0
    try:
        text = raw_model_output.strip()
        segments = text.split("|")
        last_segment = segments[-1].strip()
        if "=" not in last_segment:
            return 0.0
        expr, stated_result_str = last_segment.rsplit("=", 1)
        expr = expr.strip()
        stated_result_str = stated_result_str.strip()
        if not stated_result_str:
            return 0.0
        stated_result = float(stated_result_str)
        target = env_response.data.target
        if abs(stated_result - target) > 1e-6:
            return 0.0
        return (
            1.0
            if _evaluate_and_verify_countdown(expr, env_response.data.numbers, target)
            else 0.0
        )
    except Exception:
        return 0.0


# --- Generic (environment-agnostic) ---


def answer_tags(*, raw_model_output: str | None, **_) -> float:
    if not raw_model_output:
        return 0.0
    if (
        raw_model_output.count("<answer>") == 1
        and raw_model_output.count("</answer>") == 1
    ):
        return 1.0
    return 0.0


def think_tags(
    tag_name: str = "think", prefilled_open: bool = False
) -> RewardComponentFn:
    """Return a component fn that checks for thinking tags."""

    def fn(*, raw_model_output: str | None, **_) -> float:
        if not raw_model_output:
            return 0.0
        close_tag = f"</{tag_name}>"
        if prefilled_open:
            return 1.0 if raw_model_output.count(close_tag) == 1 else 0.0
        open_tag = f"<{tag_name}>"
        if (
            raw_model_output.count(open_tag) == 1
            and raw_model_output.count(close_tag) == 1
        ):
            return 1.0
        return 0.0

    return fn


def length_bonus(
    max_bonus: float = 0.05, normalize_length: int = 500
) -> RewardComponentFn:
    def fn(*, raw_model_output: str | None, **_) -> float:
        if raw_model_output is None:
            return 0.0
        return min(len(raw_model_output) / normalize_length, max_bonus)

    return fn


def arithmetic_correct(tolerance: float = 1e-6) -> RewardComponentFn:
    def fn(
        *, env_response: EnvResponse[QA[float]], extracted_model_output: float, **_
    ) -> float:
        return (
            1.0
            if abs(env_response.data.answer - extracted_model_output) < tolerance
            else 0.0
        )

    return fn
