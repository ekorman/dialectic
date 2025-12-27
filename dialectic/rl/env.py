import json
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


@dataclass
class Space(Generic[ObsType]):
    """Minimal space class for gymnasium compatibility."""

    pass


@dataclass
class Discrete(Space[int]):
    """Discrete action/observation space."""

    n: int


@dataclass
class Box(Space):
    """Continuous space with shape."""

    shape: tuple[int, ...]


class Env(ABC, Generic[ObsType, ActType]):
    """
    Minimal abstract RL environment compatible with gymnasium interface.

    Attributes
    ----------
    observation_space : Space[ObsType]
        The observation space specification.
    action_space : Space[ActType]
        The action space specification.
    """

    observation_space: Space[ObsType]
    action_space: Space[ActType]

    @abstractmethod
    def reset(self, seed: int | None = None) -> tuple[ObsType, dict[str, Any]]:
        """
        Reset the environment.

        Parameters
        ----------
        seed : int or None, optional
            Random seed for reproducibility.

        Returns
        -------
        tuple[ObsType, dict[str, Any]]
            Initial observation and info dictionary.
        """
        ...

    @abstractmethod
    def step(
        self, action: ActType
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        """
        Take an action in the environment.

        Parameters
        ----------
        action : ActType
            The action to take.

        Returns
        -------
        tuple[ObsType, float, bool, bool, dict[str, Any]]
            Observation, reward, terminated, truncated, and info dictionary.
        """
        ...


class GSM8kEnv(Env[str, str]):
    """
    RL environment for GSM8k math word problems.

    Single-step episodes where the agent receives a math question and
    must provide the correct numerical answer.

    Parameters
    ----------
    data_path : str
        Path to JSONL file with questions and answers.
    shuffle : bool, optional
        Whether to shuffle the problem order. Default is True.

    Attributes
    ----------
    problems : list[dict]
        Loaded problem data.
    current_problem : dict or None
        The current problem being solved.
    """

    def __init__(self, data_path: str, shuffle: bool = True):
        self.problems = self._load_data(data_path)
        self.shuffle = shuffle
        self.indices: list[int] = []
        self.current_problem: dict | None = None
        self.observation_space = Box(shape=(0,))
        self.action_space = Discrete(n=0)

    def _load_data(self, path: str) -> list[dict]:
        """
        Load problems from a JSONL file.

        Parameters
        ----------
        path : str
            Path to the JSONL file.

        Returns
        -------
        list[dict]
            List of problem dictionaries.
        """
        problems = []
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    problems.append(json.loads(line))
        return problems

    def _extract_answer(self, text: str) -> str | None:
        """
        Extract numerical answer from text.

        Looks for answers after '####' delimiter or as trailing number.

        Parameters
        ----------
        text : str
            Text to extract answer from.

        Returns
        -------
        str or None
            Extracted numerical answer or None if not found.
        """
        match = re.search(r"####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)", text)
        if match:
            return match.group(1).replace(",", "")

        match = re.search(r"(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*$", text.strip())
        if match:
            return match.group(1).replace(",", "")

        return None

    def reset(self, seed: int | None = None) -> tuple[str, dict[str, Any]]:
        """
        Reset the environment with a new problem.

        Parameters
        ----------
        seed : int or None, optional
            Random seed for reproducibility.

        Returns
        -------
        tuple[str, dict[str, Any]]
            Question string and info dictionary with problem index.
        """
        if seed is not None:
            random.seed(seed)

        if not self.indices:
            self.indices = list(range(len(self.problems)))
            if self.shuffle:
                random.shuffle(self.indices)

        idx = self.indices.pop()
        self.current_problem = self.problems[idx]

        return self.current_problem["question"], {"index": idx}

    def step(self, action: str) -> tuple[str, float, bool, bool, dict[str, Any]]:
        """
        Evaluate the proposed answer.

        Parameters
        ----------
        action : str
            Solution string containing the proposed answer.

        Returns
        -------
        tuple[str, float, bool, bool, dict[str, Any]]
            Empty observation, reward (1.0 if correct, 0.0 otherwise),
            terminated (always True), truncated (always False), and info.

        Raises
        ------
        RuntimeError
            If called before reset().
        """
        if self.current_problem is None:
            raise RuntimeError("Must call reset() before step()")

        correct_answer = self._extract_answer(self.current_problem["answer"])
        proposed_answer = self._extract_answer(action)

        correct = proposed_answer is not None and proposed_answer == correct_answer
        reward = 1.0 if correct else 0.0

        info = {
            "correct_answer": correct_answer,
            "proposed_answer": proposed_answer,
            "correct": correct,
            "question": self.current_problem["question"],
            "full_solution": self.current_problem["answer"],
        }

        return "", reward, True, False, info


class ArithmeticEnv(Env[str, str]):
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

    Attributes
    ----------
    current_question : str or None
        The current question string.
    current_answer : int or float or None
        The correct answer to the current question.
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

        self.current_question: str | None = None
        self.current_answer: int | float | None = None

        self.observation_space = Box(shape=(0,))
        self.action_space = Discrete(n=0)

    def _generate_problem(self) -> tuple[str, int | float]:
        """
        Generate a random arithmetic problem.

        Returns
        -------
        tuple[str, int | float]
            Question string and correct answer.
        """
        operands = [
            random.randint(self.min_value, self.max_value)
            for _ in range(self.num_operands)
        ]
        ops = [random.choice(self.operations) for _ in range(self.num_operands - 1)]

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

        if isinstance(result, float) and result.is_integer():
            result = int(result)

        question = f"What is {expression}?"
        return question, result

    def _extract_answer(self, text: str) -> str | None:
        """
        Extract numerical answer from response text.

        Parameters
        ----------
        text : str
            Text to extract answer from.

        Returns
        -------
        str or None
            Extracted number or None if not found.
        """
        match = re.search(r"(-?\d+(?:\.\d+)?)", text)
        if match:
            return match.group(1)
        return None

    def reset(self, seed: int | None = None) -> tuple[str, dict[str, Any]]:
        """
        Reset the environment with a new problem.

        Parameters
        ----------
        seed : int or None, optional
            Random seed for reproducibility.

        Returns
        -------
        tuple[str, dict[str, Any]]
            Question string and info dictionary with correct answer.
        """
        if seed is not None:
            random.seed(seed)

        self.current_question, self.current_answer = self._generate_problem()
        return self.current_question, {"answer": self.current_answer}

    def step(self, action: str) -> tuple[str, float, bool, bool, dict[str, Any]]:
        """
        Evaluate the proposed answer.

        Parameters
        ----------
        action : str
            Response string containing the proposed answer.

        Returns
        -------
        tuple[str, float, bool, bool, dict[str, Any]]
            Empty observation, reward (1.0 if correct, 0.0 otherwise),
            terminated (always True), truncated (always False), and info.

        Raises
        ------
        RuntimeError
            If called before reset().
        """
        if self.current_question is None:
            raise RuntimeError("Must call reset() before step()")

        proposed = self._extract_answer(action)
        correct_str = str(self.current_answer)

        if proposed is not None:
            try:
                proposed_num = float(proposed)
                correct_num = float(correct_str)
                correct = abs(proposed_num - correct_num) < 1e-6
            except ValueError:
                correct = False
        else:
            correct = False

        reward = 1.0 if correct else 0.0

        info = {
            "correct_answer": self.current_answer,
            "proposed_answer": proposed,
            "correct": correct,
            "question": self.current_question,
        }

        return "", reward, True, False, info
