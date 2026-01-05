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
