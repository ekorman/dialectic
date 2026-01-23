import json
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Generic

from dialectic.rl.types import QA, A, EnvResponse, T


class EpisodeIsDoneError(RuntimeError):
    pass


class Env(ABC, Generic[T, A]):
    eval_mode: bool = False

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

    def _get_question_and_answer(self, index: int) -> QA[float]:
        q = self.data[index]["question"]
        a = self.data[index]["answer"]
        m = re.search(r"####\s*([^\n]+)", a)
        if m is None:
            raise RuntimeError(f"Error extracting answer from {a}")
        a = float(m.group(1).strip())
        return QA(question=q, answer=a)

    # maybe should change name from `reset` to something else (e.g. `new_episode`) since `reset` makes it
    # sound like all internal state will be reset which is not true.
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
class Countdown:
    prompt: str
    numbers: list[int]
    target: int


class CountdownEnv(Env[Countdown, None]):
    """
    Countdown game environment.

    Given N numbers and a target, find an arithmetic expression
    using each number at most once that equals the target.

    Parameters
    ----------
    num_operands : int
        Number of operands to use. Default is 4.
    min_number : int
        Minimum value for operands. Default is 1.
    max_number : int
        Maximum value for operands. Default is 25.
    """

    def __init__(
        self,
        prompt_template: str | None = None,
        num_operands: int = 4,
        min_number: int = 1,
        max_number: int = 25,
    ):
        self.num_operands = num_operands
        self.min_number = min_number
        self.max_number = max_number
        self.prompt_template = prompt_template or (
            "Using the numbers {numbers}, create an equation that equals {target}. "
            "You can use +, -, *, / and each number at most once. "
            "Show your reasoning in <reasoning></reasoning> tags. Please be concise and give just one solution."
            "Put your final equation in <answer></answer> tags. "
            "For example, if the equation is 3+5*2, respond with <reasoning>[detailed reasoning explanations]</reasoning><answer>3+5*2</answer>."
        )
        self.rng = random.Random()

    def reset(self, seed: int | None = None) -> EnvResponse[Countdown]:
        if seed is not None:
            self.rng.seed(seed)
        numbers = [
            self.rng.randint(self.min_number, self.max_number)
            for _ in range(self.num_operands)
        ]

        ops = self.rng.choices(["+", "-", "*"], k=self.num_operands - 1)
        expr = str(numbers[0])
        for number, op in zip(numbers[1:], ops):
            expr += f"{op}{number}"

        target = eval(expr)

        prompt = self.prompt_template.format(numbers=numbers, target=target)

        return EnvResponse(
            is_done=True,
            data=Countdown(prompt=prompt, numbers=numbers, target=target),
        )

    def step(self, action: None):
        raise EpisodeIsDoneError
