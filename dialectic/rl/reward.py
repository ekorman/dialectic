import re
from abc import ABC, abstractmethod
from typing import Generic, Sequence

from dialectic.rl.env import Countdown
from dialectic.rl.types import QA, E, EnvResponse, RewardResult, T


class RewardFn(ABC, Generic[T, E]):
    @abstractmethod
    def __call__(
        self,
        *,
        env_response: EnvResponse[T],
        raw_model_output: str | None = None,
        extracted_model_output: E,
    ) -> RewardResult: ...


class RewardComponent(ABC, Generic[T, E]):
    """A single reward component."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for logging."""
        ...

    @property
    def group(self) -> str | None:
        """Optional group name for max() selection. None = additive."""
        return None

    @property
    def weight(self) -> float:
        """Weight applied to this component's value."""
        return 1.0

    @abstractmethod
    def __call__(
        self,
        *,
        env_response: EnvResponse[T],
        raw_model_output: str | None = None,
        extracted_model_output: E,
    ) -> float: ...


class CompositeRewardFn(RewardFn[T, E]):
    def __init__(self, components: Sequence[RewardComponent[T, E]]):
        self.components = list(components)

    def __call__(
        self,
        *,
        env_response: EnvResponse[T],
        raw_model_output: str | None = None,
        extracted_model_output: E,
    ) -> RewardResult:
        component_values: dict[str, float] = {}
        for comp in self.components:
            component_values[comp.name] = comp(
                env_response=env_response,
                raw_model_output=raw_model_output,
                extracted_model_output=extracted_model_output,
            )

        groups: dict[str, list[tuple[str, float, float]]] = {}
        additive: list[tuple[str, float, float]] = []

        for comp in self.components:
            value = component_values[comp.name]
            if comp.group is not None:
                groups.setdefault(comp.group, []).append(
                    (comp.name, value, comp.weight)
                )
            else:
                additive.append((comp.name, value, comp.weight))

        total = sum(value * weight for _, value, weight in additive)
        for group_components in groups.values():
            max_value, max_weight = max(
                ((v, w) for _, v, w in group_components),
                key=lambda x: x[0] * x[1],
            )
            total += max_value * max_weight

        return RewardResult(total=total, components=component_values)


class LengthBonusComponent(RewardComponent[T, E]):
    """Adds bonus based on output length. Task-agnostic."""

    def __init__(
        self,
        max_bonus: float = 0.05,
        normalize_length: int = 500,
        weight: float = 1.0,
    ):
        self._max_bonus = max_bonus
        self._normalize_length = normalize_length
        self._weight = weight

    @property
    def name(self) -> str:
        return "length_bonus"

    @property
    def weight(self) -> float:
        return self._weight

    def __call__(
        self,
        *,
        env_response: EnvResponse[T],
        raw_model_output: str | None = None,
        extracted_model_output: E,
    ) -> float:
        if raw_model_output is None:
            return 0.0
        return min(len(raw_model_output) / self._normalize_length, self._max_bonus)


def _evaluate_and_verify_countdown(expr: str, numbers: list[int], target: int) -> bool:
    used_numbers = [int(n) for n in re.findall(r"\d+", expr)]

    available = numbers.copy()
    for n in used_numbers:
        if n in available:
            available.remove(n)
        else:
            return False

    try:
        result = eval(expr, {"__builtins__": {}}, {})
        return abs(result - target) < 1e-6
    except Exception:
        return False


class CountdownCorrectComponent(RewardComponent[Countdown, str | None]):
    """1.0 if expression evaluates to target using valid numbers."""

    @property
    def name(self) -> str:
        return "correct"

    @property
    def group(self) -> str:
        return "accuracy"

    def __call__(
        self,
        *,
        env_response: EnvResponse[Countdown],
        raw_model_output: str | None = None,
        extracted_model_output: str | None,
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


class CountdownParseableComponent(RewardComponent[Countdown, str | None]):
    """0.3 if has <answer> tags with parseable expression (regardless of correctness)."""

    @property
    def name(self) -> str:
        return "parseable"

    @property
    def group(self) -> str:
        return "accuracy"

    def __call__(
        self,
        *,
        env_response: EnvResponse[Countdown],
        raw_model_output: str | None = None,
        extracted_model_output: str | None,
    ) -> float:
        if extracted_model_output is None:
            return 0.0
        try:
            eval(extracted_model_output, {"__builtins__": {}}, {})
            return 0.3
        except Exception:
            return 0.0


class CountdownAnswerTagsComponent(RewardComponent[Countdown, str | None]):
    """0.1 if has <answer> tags (even if unparseable)."""

    @property
    def name(self) -> str:
        return "answer_tags"

    @property
    def group(self) -> str:
        return "accuracy"

    def __call__(
        self,
        *,
        env_response: EnvResponse[Countdown],
        raw_model_output: str | None = None,
        extracted_model_output: str | None,
    ) -> float:
        return 0.1 if extracted_model_output is not None else 0.0


class CountdownThinkTagsComponent(RewardComponent[Countdown, str | None]):
    """0.05 if has <think> tags."""

    @property
    def name(self) -> str:
        return "think_tags"

    @property
    def group(self) -> str:
        return "accuracy"

    def __call__(
        self,
        *,
        env_response: EnvResponse[Countdown],
        raw_model_output: str | None = None,
        extracted_model_output: str | None,
    ) -> float:
        if (
            raw_model_output
            and "<think>" in raw_model_output
            and "</think>" in raw_model_output
        ):
            return 0.05
        return 0.0


class CountdownRewardFn(RewardFn[Countdown, str | None]):
    """Binary reward: 1.0 if equation is valid and equals target, else 0.0"""

    def __init__(self):
        self._composite = CompositeRewardFn([CountdownCorrectComponent()])

    def __call__(
        self,
        *,
        env_response: EnvResponse[Countdown],
        raw_model_output: str | None = None,
        extracted_model_output: str | None,
    ) -> RewardResult:
        return self._composite(
            env_response=env_response,
            raw_model_output=raw_model_output,
            extracted_model_output=extracted_model_output,
        )


class CountdownWithFormatRewardFn(RewardFn[Countdown, str | None]):
    """
    Reward with format shaping to ensure learning signal.

    Components in "accuracy" group: max() selects highest (1.0 > 0.3 > 0.1 > 0.05)
    Plus small length bonus (up to 0.05) to create variance between similar outputs.
    """

    def __init__(self):
        self._composite = CompositeRewardFn(
            [
                CountdownCorrectComponent(),
                CountdownParseableComponent(),
                CountdownAnswerTagsComponent(),
                CountdownThinkTagsComponent(),
            ]
        )

    def __call__(
        self,
        *,
        env_response: EnvResponse[Countdown],
        raw_model_output: str | None = None,
        extracted_model_output: str | None,
    ) -> RewardResult:
        return self._composite(
            env_response=env_response,
            raw_model_output=raw_model_output,
            extracted_model_output=extracted_model_output,
        )


class ArithmeticRewardFn(RewardFn[QA[float], float]):
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance

    def __call__(
        self,
        *,
        env_response: EnvResponse[QA[float]],
        raw_model_output: str | None = None,
        extracted_model_output: float,
    ) -> RewardResult:
        correct = (
            abs(env_response.data.answer - extracted_model_output) < self.tolerance
        )
        value = 1.0 if correct else 0.0
        return RewardResult(total=value, components={"correct": value})
