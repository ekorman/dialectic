import random
from collections import deque
from dataclasses import dataclass


@dataclass
class MazeConfig:
    """
    Parameters
    ----------
    height : int
        Number of rows (5-30).
    width : int
        Number of columns (5-30).
    min_solution_length : int or None
        Reject mazes with shorter optimal paths.
    max_solution_length : int or None
        Reject mazes with longer optimal paths.
    algorithm : str
        Generation algorithm. Currently only "dfs".
    openness : float
        Fraction of extra walls to remove after generation.
        0.0 = perfect maze, higher = more alternative paths.
    start_pos : str
        Start placement: "top_left", "random", "corners".
    goal_pos : str
        Goal placement: "bottom_right", "random", "farthest", "corners".
    seed : int or None
        Random seed for reproducibility.
    """

    height: int = 5
    width: int = 5
    min_solution_length: int | None = None
    max_solution_length: int | None = None
    algorithm: str = "dfs"
    openness: float = 0.0
    start_pos: str = "top_left"
    goal_pos: str = "bottom_right"
    seed: int | None = None


DIRECTIONS: dict[str, tuple[int, int]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

OPPOSITE: dict[str, str] = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}


@dataclass
class Maze:
    height: int
    width: int
    connections: dict[tuple[int, int], set[str]]
    start: tuple[int, int]
    goal: tuple[int, int]
    solution: list[str]
    solution_length: int


@dataclass
class PathResult:
    valid_moves: int
    total_moves: int
    reached_goal: bool
    final_pos: tuple[int, int]
    path_length: int


def _generate_dfs_maze(
    height: int, width: int, rng: random.Random
) -> dict[tuple[int, int], set[str]]:
    connections: dict[tuple[int, int], set[str]] = {
        (r, c): set() for r in range(height) for c in range(width)
    }
    visited: set[tuple[int, int]] = set()
    stack: list[tuple[int, int]] = []

    start = (rng.randint(0, height - 1), rng.randint(0, width - 1))
    visited.add(start)
    stack.append(start)

    while stack:
        current = stack[-1]
        neighbors = []
        for direction, (dr, dc) in DIRECTIONS.items():
            nr, nc = current[0] + dr, current[1] + dc
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in visited:
                neighbors.append((direction, (nr, nc)))

        if neighbors:
            direction, neighbor = rng.choice(neighbors)
            connections[current].add(direction)
            connections[neighbor].add(OPPOSITE[direction])
            visited.add(neighbor)
            stack.append(neighbor)
        else:
            stack.pop()

    return connections


def _apply_openness(
    connections: dict[tuple[int, int], set[str]],
    height: int,
    width: int,
    openness: float,
    rng: random.Random,
) -> None:
    walls: list[tuple[tuple[int, int], str]] = []
    for r in range(height):
        for c in range(width):
            for direction, (dr, dc) in DIRECTIONS.items():
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width:
                    if direction not in connections[(r, c)]:
                        walls.append(((r, c), direction))

    seen: set[frozenset[tuple[tuple[int, int], str]]] = set()
    unique_walls: list[tuple[tuple[int, int], str]] = []
    for cell, direction in walls:
        dr, dc = DIRECTIONS[direction]
        neighbor = (cell[0] + dr, cell[1] + dc)
        key = frozenset({(cell, direction), (neighbor, OPPOSITE[direction])})
        if key not in seen:
            seen.add(key)
            unique_walls.append((cell, direction))

    num_to_remove = int(len(unique_walls) * openness)
    if num_to_remove == 0:
        return

    rng.shuffle(unique_walls)
    for cell, direction in unique_walls[:num_to_remove]:
        dr, dc = DIRECTIONS[direction]
        neighbor = (cell[0] + dr, cell[1] + dc)
        connections[cell].add(direction)
        connections[neighbor].add(OPPOSITE[direction])


def _bfs_solve(
    connections: dict[tuple[int, int], set[str]],
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[str] | None:
    queue: deque[tuple[tuple[int, int], list[str]]] = deque([(start, [])])
    visited: set[tuple[int, int]] = {start}

    while queue:
        pos, path = queue.popleft()
        if pos == goal:
            return path
        for direction in sorted(connections[pos]):
            dr, dc = DIRECTIONS[direction]
            neighbor = (pos[0] + dr, pos[1] + dc)
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [direction]))

    return None


def _bfs_farthest(
    connections: dict[tuple[int, int], set[str]],
    start: tuple[int, int],
) -> tuple[int, int]:
    queue: deque[tuple[tuple[int, int], int]] = deque([(start, 0)])
    visited: set[tuple[int, int]] = {start}
    farthest = start
    max_dist = 0

    while queue:
        pos, dist = queue.popleft()
        if dist > max_dist:
            max_dist = dist
            farthest = pos
        for direction in sorted(connections[pos]):
            dr, dc = DIRECTIONS[direction]
            neighbor = (pos[0] + dr, pos[1] + dc)
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))

    return farthest


def _resolve_positions(
    config: MazeConfig,
    connections: dict[tuple[int, int], set[str]],
    rng: random.Random,
) -> tuple[tuple[int, int], tuple[int, int]]:
    h, w = config.height, config.width
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]

    if config.start_pos == "top_left":
        start = (0, 0)
    elif config.start_pos == "random":
        start = (rng.randint(0, h - 1), rng.randint(0, w - 1))
    elif config.start_pos == "corners":
        start = rng.choice(corners)
    else:
        raise ValueError(f"Unknown start_pos: {config.start_pos}")

    if config.goal_pos == "bottom_right":
        goal = (h - 1, w - 1)
    elif config.goal_pos == "random":
        goal = start
        while goal == start:
            goal = (rng.randint(0, h - 1), rng.randint(0, w - 1))
    elif config.goal_pos == "farthest":
        goal = _bfs_farthest(connections, start)
    elif config.goal_pos == "corners":
        remaining = [c for c in corners if c != start]
        goal = rng.choice(remaining)
    else:
        raise ValueError(f"Unknown goal_pos: {config.goal_pos}")

    return start, goal


def generate_maze(
    config: MazeConfig, rng: random.Random, *, max_attempts: int = 1000
) -> Maze:
    """
    Generate a solvable maze via randomized DFS with optional openness.

    Parameters
    ----------
    config : MazeConfig
        Maze generation parameters.
    rng : random.Random
        Random number generator instance.
    max_attempts : int
        Maximum rejection sampling attempts before raising.

    Returns
    -------
    Maze

    Raises
    ------
    RuntimeError
        If no valid maze is found within max_attempts.
    """
    if config.algorithm != "dfs":
        raise ValueError(f"Unsupported algorithm: {config.algorithm}")

    for _ in range(max_attempts):
        connections = _generate_dfs_maze(config.height, config.width, rng)
        if config.openness > 0:
            _apply_openness(
                connections, config.height, config.width, config.openness, rng
            )

        start, goal = _resolve_positions(config, connections, rng)
        solution = _bfs_solve(connections, start, goal)

        if solution is None:
            continue

        sol_len = len(solution)
        if (
            config.min_solution_length is not None
            and sol_len < config.min_solution_length
        ):
            continue
        if (
            config.max_solution_length is not None
            and sol_len > config.max_solution_length
        ):
            continue

        return Maze(
            height=config.height,
            width=config.width,
            connections=connections,
            start=start,
            goal=goal,
            solution=solution,
            solution_length=sol_len,
        )

    raise RuntimeError(
        f"Failed to generate maze satisfying constraints after {max_attempts} attempts"
    )


def tokenize_maze(maze: Maze) -> str:
    """
    Convert a maze to adjacency list text format.

    Parameters
    ----------
    maze : Maze

    Returns
    -------
    str
        Text representation with header and per-cell connection lines.
    """
    lines = [
        f"Maze {maze.height}x{maze.width} | Start: ({maze.start[0]},{maze.start[1]}) | Goal: ({maze.goal[0]},{maze.goal[1]})"
    ]
    for r in range(maze.height):
        for c in range(maze.width):
            dirs = sorted(maze.connections[(r, c)])
            if dirs:
                lines.append(f"({r},{c}) > {' '.join(dirs)}")
            else:
                lines.append(f"({r},{c}) >")
    lines.append("Path:")
    return "\n".join(lines)


def validate_path(maze: Maze, moves: list[str]) -> PathResult:
    """
    Simulate a sequence of moves through the maze.

    Invalid moves cause the agent to stay in place; execution continues.

    Parameters
    ----------
    maze : Maze
    moves : list[str]
        Sequence of directional moves.

    Returns
    -------
    PathResult
    """
    pos = maze.start
    valid_moves = 0
    visited = {pos}

    for move in moves:
        if move in maze.connections.get(pos, set()):
            dr, dc = DIRECTIONS[move]
            pos = (pos[0] + dr, pos[1] + dc)
            visited.add(pos)
            valid_moves += 1

    return PathResult(
        valid_moves=valid_moves,
        total_moves=len(moves),
        reached_goal=pos == maze.goal,
        final_pos=pos,
        path_length=len(visited),
    )


EASY = MazeConfig(
    height=5,
    width=5,
    algorithm="dfs",
    openness=0.0,
    start_pos="top_left",
    goal_pos="bottom_right",
)

MEDIUM = MazeConfig(
    height=7,
    width=7,
    algorithm="dfs",
    openness=0.15,
    start_pos="top_left",
    goal_pos="farthest",
    min_solution_length=12,
)

HARD = MazeConfig(
    height=10,
    width=10,
    algorithm="dfs",
    openness=0.3,
    start_pos="random",
    goal_pos="farthest",
    min_solution_length=25,
)

VERY_HARD = MazeConfig(
    height=15,
    width=15,
    algorithm="dfs",
    openness=0.3,
    start_pos="random",
    goal_pos="farthest",
    min_solution_length=40,
)
