from abc import ABC, abstractmethod
from typing import Generic

from dialectic.rl.types import QA, E, EnvResponse, T


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
