import json
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Generic

from dialectic.rl.types import A, EnvResponse, T


class EpisodeIsDoneError(RuntimeError):
    pass


class Env(ABC, Generic[T, A]):
    eval_mode: bool = False

    @abstractmethod
    def reseed(self) -> None: ...

    @abstractmethod
    def reset(self, seed: int | None = None) -> EnvResponse[T] | None: ...

    @abstractmethod
    def step(self, action: A) -> EnvResponse[T] | None: ...


class GSM8kEnv(Env["MathState", None]):
    """
    GSM8K grade-school math word problem environment.

    Serves problems from a GSM8K-format JSONL file (each line a JSON object
    with ``question`` and ``answer``, where ``answer`` ends with a
    ``#### <number>`` line). Single-step episodes.

    Parameters
    ----------
    path : str or Path
        Path to the GSM8K JSONL file.
    prompt_template : str
        Template applied to each question; must contain a ``{question}``
        placeholder.
    eval_mode : bool
        If True, problems are served sequentially; otherwise sampled at random.
    seed : int or None
        Random seed used when sampling problems (ignored in eval mode).
    """

    def __init__(
        self,
        *,
        path: str | Path,
        prompt_template: str,
        eval_mode: bool,
        seed: int | None = None,
    ):
        self.eval_mode = eval_mode
        self.path = path
        self.prompt_template = prompt_template
        with open(path) as f:
            self.data = [json.loads(line) for line in f]
        self._seed = seed
        if eval_mode:
            self._idx = 0
        else:
            self.rng = random.Random(seed)

    def reseed(self) -> None:
        if self.eval_mode:
            self._idx = 0
        else:
            self.rng = random.Random(self._seed)

    def __str__(self) -> str:
        return f"gsm8k_n{len(self.data)}"

    def _get_state(self, index: int) -> "MathState":
        q = self.data[index]["question"]
        a = self.data[index]["answer"]
        m = re.search(r"####\s*([^\n]+)", a)
        if m is None:
            raise RuntimeError(f"Error extracting answer from {a}")
        answer = m.group(1).strip().replace(",", "")
        return MathState(
            prompt=self.prompt_template.format(question=q),
            answer=answer,
            problem_type="gsm8k",
        )

    def reset(self, seed: int | None = None) -> EnvResponse["MathState"]:
        if self.eval_mode and seed is not None:
            raise ValueError("Should not pass a seed when in eval mode")

        if self.eval_mode:
            index = self._idx
            self._idx += 1
        else:
            if seed is not None:
                self.rng = random.Random(seed)
            index = self.rng.randint(0, len(self.data) - 1)

        return EnvResponse(is_done=True, data=self._get_state(index))

    def step(self, action: None):
        raise EpisodeIsDoneError


@dataclass
class CountdownStep:
    left: int
    op: str
    right: int
    result: int

    def format_step(self) -> str:
        return f"{self.left} {self.op} {self.right} = {self.result} |"


@dataclass
class Countdown:
    prompt: str
    numbers: list[int]
    target: int
    solution: list[CountdownStep] | None = None


def build_countdown_equation(
    numbers: list[int], steps: list[CountdownStep], target: int
) -> str:
    pool: list[tuple[int, str]] = [(n, str(n)) for n in numbers]

    for step in steps:
        left_idx = next(i for i, (v, _) in enumerate(pool) if v == step.left)
        _, left_expr = pool.pop(left_idx)

        right_idx = next(i for i, (v, _) in enumerate(pool) if v == step.right)
        _, right_expr = pool.pop(right_idx)

        if " " in left_expr:
            left_expr = f"({left_expr})"
        if " " in right_expr:
            right_expr = f"({right_expr})"

        combined = f"{left_expr} {step.op} {right_expr}"
        pool.append((step.result, combined))

    _, final_expr = pool[-1]
    return f"{final_expr} = {target}"


class CountdownEnv(Env[Countdown, None]):
    """
    Countdown game environment.

    Given N numbers and a target, find an arithmetic expression
    using each number at most once that equals the target.

    Parameters
    ----------
    n_larges : int or list[int]
        Number of large numbers to include. If a list, must be same length as
        n_total and n_ops; a configuration is sampled uniformly per problem.
    n_total : int or list[int]
        Total numbers to include. Same list constraint as n_larges.
    n_ops : int or list[int]
        Number of random operations used to generate the target. Same list
        constraint as n_larges.
    """

    SMALLS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2
    LARGES = [25, 50, 75, 100]

    @staticmethod
    def _valid_ops(a: int, b: int) -> list[tuple[str, int, int, int]]:
        ops: list[tuple[str, int, int, int]] = []
        ops.append(("+", a + b, a, b))
        ops.append(("*", a * b, a, b))

        if a > b:
            ops.append(("-", a - b, a, b))
        elif b > a:
            ops.append(("-", b - a, b, a))

        if b != 0 and a % b == 0:
            q = a // b
            if q > 0:
                ops.append(("/", q, a, b))
        if a != b and a != 0 and b % a == 0:
            q = b // a
            if q > 0:
                ops.append(("/", q, b, a))

        return ops

    def _generate_problem(self) -> tuple[list[int], int, list[CountdownStep]]:
        idx = self.rng.randrange(len(self._n_larges))
        n_larges = self._n_larges[idx]
        n_total = self._n_total[idx]
        n_ops = self._n_ops[idx]

        numbers = self.rng.sample(self.LARGES, n_larges) + self.rng.sample(
            self.SMALLS, n_total - n_larges
        )
        self.rng.shuffle(numbers)
        pool = numbers[:]
        steps: list[CountdownStep] = []

        for _ in range(n_ops):
            if len(pool) < 2:
                break
            i, j = self.rng.sample(range(len(pool)), 2)
            candidates = self._valid_ops(pool[i], pool[j])
            op, result, left, right = self.rng.choice(candidates)

            for idx in sorted((i, j), reverse=True):
                del pool[idx]

            pool.append(result)
            steps.append(CountdownStep(left=left, op=op, right=right, result=result))

        return numbers, pool[-1], steps

    def __init__(
        self,
        *,
        prompt_template: str = "Using the numbers {numbers}, create an equation that equals {target}. ",
        n_larges: int | list[int] = 2,
        n_total: int | list[int] = 6,
        n_ops: int | list[int] = 5,
        seed: int | None = None,
    ):
        self._n_larges = [n_larges] if isinstance(n_larges, int) else n_larges
        self._n_total = [n_total] if isinstance(n_total, int) else n_total
        self._n_ops = [n_ops] if isinstance(n_ops, int) else n_ops

        lengths = {len(self._n_larges), len(self._n_total), len(self._n_ops)}
        if len(lengths) != 1:
            raise ValueError(
                f"n_larges, n_total, and n_ops must have the same length, "
                f"got {len(self._n_larges)}, {len(self._n_total)}, {len(self._n_ops)}"
            )

        self.prompt_template = prompt_template
        self._seed = seed
        self.rng = random.Random(seed)

    def reseed(self) -> None:
        self.rng = random.Random(self._seed)

    def __str__(self) -> str:
        return f"countdown_ops{'_'.join(map(str, self._n_ops))}_n{'_'.join(map(str, self._n_total))}_lg{'_'.join(map(str, self._n_larges))}"

    def reset(self, seed: int | None = None) -> EnvResponse[Countdown]:
        if seed is not None:
            self.rng.seed(seed)

        numbers, target, steps = self._generate_problem()

        prompt = self.prompt_template.format(numbers=numbers, target=target)

        return EnvResponse(
            is_done=True,
            data=Countdown(
                prompt=prompt, numbers=numbers, target=target, solution=steps
            ),
        )

    def step(self, action: None):
        raise EpisodeIsDoneError


@dataclass
class MathState:
    prompt: str
    answer: str
    problem_type: str
