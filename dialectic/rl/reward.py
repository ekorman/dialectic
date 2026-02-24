import re
from typing import Callable, Protocol, Sequence

from dialectic.rl.env import Countdown, MazeState
from dialectic.rl.maze import bfs_distance, validate_path
from dialectic.rl.types import QA, E, EnvResponse, RewardResult, T

RewardComponentFn = Callable[..., float]


class RewardFn(Protocol[T, E]):
    def __call__(
        self,
        *,
        env_response: EnvResponse[T],
        raw_model_output: str | None = None,
        extracted_model_output: E,
    ) -> RewardResult: ...


def weighted_reward(
    components: Sequence[tuple[str, float, RewardComponentFn]],
) -> RewardFn:
    """Create a reward function as a weighted sum of components."""

    def fn(
        *,
        env_response: EnvResponse,
        raw_model_output: str | None = None,
        extracted_model_output: object,
    ) -> RewardResult:
        values: dict[str, float] = {}
        total = 0.0
        for name, weight, component in components:
            value = component(
                env_response=env_response,
                raw_model_output=raw_model_output,
                extracted_model_output=extracted_model_output,
            )
            values[name] = value
            total += weight * value
        return RewardResult(total=total, components=values)

    return fn


# --- Countdown-specific ---


def _evaluate_and_verify_countdown(expr: str, numbers: list[int], target: int) -> bool:
    if not re.match(r"^[\d+\-*/()\s]+$", expr):
        return False

    used_numbers = sorted(int(n) for n in re.findall(r"\d+", expr))
    if used_numbers != sorted(numbers):
        return False

    try:
        result = eval(expr, {"__builtins__": {}}, {})
        return abs(result - target) < 1e-6
    except Exception:
        return False


def countdown_correct(
    *, env_response: EnvResponse[Countdown], extracted_model_output: str | None, **_
) -> float:
    if extracted_model_output is None:
        return 0.0
    try:
        return (
            1.0
            if _evaluate_and_verify_countdown(
                extracted_model_output,
                env_response.data.numbers,
                env_response.data.target,
            )
            else 0.0
        )
    except Exception:
        return 0.0


# --- Generic (environment-agnostic) ---


def answer_tags(*, raw_model_output: str | None, **_) -> float:
    if not raw_model_output:
        return 0.0
    if (
        raw_model_output.count("<answer>") == 1
        and raw_model_output.count("</answer>") == 1
    ):
        return 1.0
    return 0.0


def think_tags(
    tag_name: str = "think", prefilled_open: bool = False
) -> RewardComponentFn:
    """Return a component fn that checks for thinking tags."""

    def fn(*, raw_model_output: str | None, **_) -> float:
        if not raw_model_output:
            return 0.0
        close_tag = f"</{tag_name}>"
        if prefilled_open:
            return 1.0 if raw_model_output.count(close_tag) == 1 else 0.0
        open_tag = f"<{tag_name}>"
        if (
            raw_model_output.count(open_tag) == 1
            and raw_model_output.count(close_tag) == 1
        ):
            return 1.0
        return 0.0

    return fn


def length_bonus(
    max_bonus: float = 0.05, normalize_length: int = 500
) -> RewardComponentFn:
    def fn(*, raw_model_output: str | None, **_) -> float:
        if raw_model_output is None:
            return 0.0
        return min(len(raw_model_output) / normalize_length, max_bonus)

    return fn


# --- Arithmetic ---


# --- Maze-specific ---


def maze_correct(
    *,
    env_response: EnvResponse[MazeState],
    extracted_model_output: list[str] | None,
    **_,
) -> float:
    if not extracted_model_output:
        return 0.0
    result = validate_path(env_response.data.maze, extracted_model_output)
    return 1.0 if result.reached_goal else 0.0


def maze_validity(
    *,
    env_response: EnvResponse[MazeState],
    extracted_model_output: list[str] | None,
    **_,
) -> float:
    if not extracted_model_output:
        return 0.0
    result = validate_path(env_response.data.maze, extracted_model_output)
    if result.total_moves == 0:
        return 0.0
    return result.valid_moves / result.total_moves


def maze_distance(
    *,
    env_response: EnvResponse[MazeState],
    extracted_model_output: list[str] | None,
    **_,
) -> float:
    """
    Proportional credit for getting closer to the goal.

    Returns 1.0 if at goal, 0.0 if at start or farther, linear in between.
    """
    maze = env_response.data.maze
    if not extracted_model_output:
        return 0.0
    result = validate_path(maze, extracted_model_output)
    if result.reached_goal:
        return 1.0
    start_dist = bfs_distance(maze.connections, maze.start, maze.goal)
    if start_dist is None or start_dist == 0:
        return 0.0
    final_dist = bfs_distance(maze.connections, result.final_pos, maze.goal)
    if final_dist is None:
        return 0.0
    return max(0.0, 1.0 - final_dist / start_dist)


# --- Arithmetic ---


def arithmetic_correct(tolerance: float = 1e-6) -> RewardComponentFn:
    def fn(
        *, env_response: EnvResponse[QA[float]], extracted_model_output: float, **_
    ) -> float:
        return (
            1.0
            if abs(env_response.data.answer - extracted_model_output) < tolerance
            else 0.0
        )

    return fn
