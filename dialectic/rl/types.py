from dataclasses import dataclass
from typing import Generic, TypeVar

A = TypeVar("A")
T = TypeVar("T")
R = TypeVar("R")
E = TypeVar("E")


@dataclass
class EnvResponse(Generic[T]):
    is_done: bool
    data: T


@dataclass
class QA[R]:
    question: str
    answer: R


@dataclass
class Countdown:
    prompt: str
    numbers: list[int]
    target: int


@dataclass
class RewardResult:
    total: float
    components: dict[str, float]
