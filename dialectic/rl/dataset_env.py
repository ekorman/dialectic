import random

from dialectic.rl.env import Env, EpisodeIsDoneError
from dialectic.rl.types import EnvResponse, T


class DatasetEnv(Env[T, None]):
    """Environment that serves problems from a pre-loaded dataset."""

    def __init__(
        self,
        problems: list[EnvResponse[T]],
        seed: int | None = None,
        label: str = "dataset",
    ):
        self.problems = problems
        self._seed = seed
        self._label = label
        self.rng = random.Random(seed)

    def reset(self, seed: int | None = None) -> EnvResponse[T]:
        if seed is not None:
            self.rng.seed(seed)
        return self.rng.choice(self.problems)

    def reseed(self) -> None:
        self.rng = random.Random(self._seed)

    def step(self, action: None):
        raise EpisodeIsDoneError

    def __str__(self) -> str:
        return f"{self._label}_n{len(self.problems)}"
