from dataclasses import dataclass
from typing import Generic, TypeVar

A = TypeVar("A")
T = TypeVar("T")
E = TypeVar("E")


@dataclass
class EnvResponse(Generic[T]):
    is_done: bool
    data: T


@dataclass
class RewardResult:
    total: float
    components: dict[str, float]
