import json
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Literal

from dialectic.rl.math import generate_problem
from dialectic.rl.types import QA, A, EnvResponse, T


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


class ArithmeticEnv(Env[QA[float], None]):
    """
    Toy RL environment for simple arithmetic problems.

    Generates random arithmetic problems for validating RL algorithms
    with small LLMs. Single-step episodes.

    Parameters
    ----------
    min_value : int, optional
        Minimum value for operands. Default is 0.
    max_value : int, optional
        Maximum value for operands. Default is 100.
    operations : tuple[str, ...], optional
        Operations to use. Default is ("+", "-").
    num_operands : int, optional
        Number of operands in each problem. Default is 2.
    """

    def __init__(
        self,
        min_value: int = 0,
        max_value: int = 100,
        operations: tuple[str, ...] = ("+", "-"),
        num_operands: int = 2,
    ):
        self.min_value = min_value
        self.max_value = max_value
        self.operations = operations
        self.num_operands = num_operands
        self.rng = random.Random()

    def reseed(self) -> None:
        pass

    def _generate_problem(self) -> tuple[str, float]:
        """
        Generate a random arithmetic problem.

        Returns
        -------
        tuple[str, int | float]
            Question string and correct answer.
        """
        operands = [
            self.rng.randint(self.min_value, self.max_value)
            for _ in range(self.num_operands)
        ]
        ops = [self.rng.choice(self.operations) for _ in range(self.num_operands - 1)]

        expr_parts = [str(operands[0])]
        for i, op in enumerate(ops):
            expr_parts.append(op)
            expr_parts.append(str(operands[i + 1]))
        expression = " ".join(expr_parts)

        result: int | float = operands[0]
        for i, op in enumerate(ops):
            if op == "+":
                result += operands[i + 1]
            elif op == "-":
                result -= operands[i + 1]
            elif op == "*":
                result *= operands[i + 1]
            elif op == "/":
                result /= operands[i + 1]

        result = float(result)

        question = f"What is {expression}?"
        return question, result

    def reset(self, seed: int | None = None) -> EnvResponse[QA[float]]:
        self.rng.seed(seed)
        q, a = self._generate_problem()
        return EnvResponse(
            is_done=True,
            data=QA(question=q, answer=a),
        )

    def step(self, action: None):
        raise EpisodeIsDoneError


class GSM8kEnv(Env[QA[float], None]):
    def __init__(
        self,
        *,
        path: str | Path,
        eval_mode: bool,
    ):
        self.eval_mode = eval_mode
        self.path = path
        with open(path) as f:
            self.data = [json.loads(line) for line in f]
        if eval_mode:
            self._idx = 0
        else:
            self.rng = random.Random()

    def reseed(self) -> None:
        if self.eval_mode:
            self._idx = 0

    def _get_question_and_answer(self, index: int) -> QA[float]:
        q = self.data[index]["question"]
        a = self.data[index]["answer"]
        m = re.search(r"####\s*([^\n]+)", a)
        if m is None:
            raise RuntimeError(f"Error extracting answer from {a}")
        a = float(m.group(1).strip())
        return QA(question=q, answer=a)

    def reset(self, seed: int | None = None) -> EnvResponse[QA[float]]:
        if self.eval_mode and seed is not None:
            raise ValueError("Should not pass a seed when in eval mode")

        if self.eval_mode:
            index = self._idx
            self._idx += 1
        else:
            self.rng = random.Random(seed)
            index = self.rng.randint(0, len(self.data) - 1)

        qa = self._get_question_and_answer(index)
        return EnvResponse(is_done=True, data=qa)

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


class MathEnv(Env[MathState, None]):
    """
    Math problem environment.

    Generates random math problems using MathDatasetConfig.
    Single-step episodes.

    Parameters
    ----------
    config : MathDatasetConfig
        Problem mix and difficulty configuration.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        *,
        direct_arithmetic_prob: float,
        twostep_arithmetic_prob: float,
        word_problem_prob: float,
        number_properties_prob: float,
        difficulty: Literal["trivial", "easy", "medium"],
        seed: int | None = None,
    ):
        self.direct_arithmetic_prob = direct_arithmetic_prob
        self.twostep_arithmetic_prob = twostep_arithmetic_prob
        self.word_problem_prob = word_problem_prob
        self.number_properties_prob = number_properties_prob
        self.difficulty = difficulty

        self._seed = seed
        self.rng = random.Random(seed)

    def reseed(self) -> None:
        self.rng = random.Random(self._seed)

    def __str__(self) -> str:
        return f"math_{self.difficulty}"

    def reset(self, seed: int | None = None) -> EnvResponse[MathState]:
        if seed is not None:
            self.rng.seed(seed)
        problem = generate_problem(
            direct_arithmetic_prob=self.direct_arithmetic_prob,
            twostep_arithmetic_prob=self.twostep_arithmetic_prob,
            word_problem_prob=self.word_problem_prob,
            number_properties_prob=self.number_properties_prob,
            difficulty=self.difficulty,
            rng=self.rng,
        )
        return EnvResponse(
            is_done=True,
            data=MathState(
                prompt=problem.question,
                answer=str(problem.answer),
                problem_type=problem.problem_type,
            ),
        )

    def step(self, action: None):
        raise EpisodeIsDoneError
