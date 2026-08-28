import re
from typing import Callable, Protocol, Sequence

from dialectic.rl.env import Countdown, MathState
from dialectic.rl.types import E, EnvResponse, RewardResult, T

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


# --- GSM8K-specific ---


def gsm8k_correct(
    *, env_response: EnvResponse[MathState], extracted_model_output: str | None, **_
) -> float:
    if extracted_model_output is None:
        return 0.0
    try:
        predicted = float(extracted_model_output.replace(",", "").strip())
        target = float(env_response.data.answer)
    except (ValueError, TypeError):
        return 0.0
    return 1.0 if abs(predicted - target) < 1e-6 else 0.0


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
