import random
from abc import ABC, abstractmethod
from typing import Generic

from dialectic.rl.types import QA, A, EnvResponse, T


class EpisodeIsDoneError(RuntimeError):
    pass


class Env(ABC, Generic[T, A]):
    # internally keep track of done or not and throw error if step is called...
    @abstractmethod
    def reset(self, seed: int | None = None) -> EnvResponse[T]: ...

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


# for some enviornments we should have an `eval` mode where reset runs iteratively through
# all possible episodes
