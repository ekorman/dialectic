import re
from abc import ABC, abstractmethod
from typing import Generic

from dialectic.rl.types import QA, Countdown, E, EnvResponse, T


class RewardFn(ABC, Generic[T, E]):
    # might need to include other things here, such as logits if doing KL in reward
    @abstractmethod
    def __call__(
        self,
        *,
        env_response: EnvResponse[T],
        raw_model_output: str | None = None,
        extracted_model_output: E,
    ) -> float: ...


class ArithmeticRewardFn(RewardFn[QA[float], float]):
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance

    def __call__(
        self,
        *,
        env_response: EnvResponse[QA[float]],
        raw_model_output: str | None = None,
        extracted_model_output: float,
    ) -> float:
        return (
            1.0
            if abs(env_response.data.answer - extracted_model_output) < self.tolerance
            else 0.0
        )


class CountdownRewardFn(RewardFn[Countdown, str | None]):
    """Binary reward: 1.0 if equation is valid and equals target, else 0.0"""

    def __call__(
        self,
        *,
        env_response: EnvResponse[Countdown],
        raw_model_output: str | None = None,
        extracted_model_output: str | None,
    ) -> float:
        if extracted_model_output is None:
            return 0.0

        target = env_response.data.target
        numbers = env_response.data.numbers

        try:
            return (
                1.0
                if self._evaluate_and_verify(extracted_model_output, numbers, target)
                else 0.0
            )
        except Exception:
            return 0.0

    def _evaluate_and_verify(self, expr: str, numbers: list[int], target: int) -> bool:
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


class CountdownWithFormatRewardFn(RewardFn[Countdown, str | None]):
    """
    Reward with format shaping to ensure learning signal.

    Base rewards:
    - 1.0: Correct answer
    - 0.3: Has <answer> tags with parseable expression (wrong result)
    - 0.1: Has <answer> tags (unparseable)
    - 0.05: Has <think> tags only
    - 0.0: No structure

    Plus small length bonus (up to 0.05) to create variance between similar outputs.
    """

    def __call__(
        self,
        *,
        env_response: EnvResponse[Countdown],
        raw_model_output: str | None = None,
        extracted_model_output: str | None,
    ) -> float:
        if raw_model_output is None:
            return 0.0

        target = env_response.data.target
        numbers = env_response.data.numbers

        base_reward = 0.0

        if extracted_model_output is not None:
            try:
                if self._evaluate_and_verify(extracted_model_output, numbers, target):
                    return 1.0
            except Exception:
                pass

            if self._is_parseable_expression(extracted_model_output):
                base_reward = 0.3
            else:
                base_reward = 0.1
        elif "<think>" in raw_model_output and "</think>" in raw_model_output:
            base_reward = 0.05

        length_bonus = min(len(raw_model_output) / 500.0, 0.05)
        return base_reward + length_bonus

    def _evaluate_and_verify(self, expr: str, numbers: list[int], target: int) -> bool:
        used_numbers = [int(n) for n in re.findall(r"\d+", expr)]

        available = numbers.copy()
        for n in used_numbers:
            if n in available:
                available.remove(n)
            else:
                return False

        result = eval(expr, {"__builtins__": {}}, {})
        return abs(result - target) < 1e-6

    def _is_parseable_expression(self, expr: str) -> bool:
        try:
            eval(expr, {"__builtins__": {}}, {})
            return True
        except Exception:
            return False
