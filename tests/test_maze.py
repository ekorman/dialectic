import random

import pytest

from dialectic.rl.env import MazeState
from dialectic.rl.extractors import extract_maze_moves
from dialectic.rl.maze import (
    EASY,
    HARD,
    MEDIUM,
    VERY_HARD,
    Maze,
    MazeConfig,
    bfs_distance,
    generate_maze,
    tokenize_maze,
    validate_path,
)
from dialectic.rl.reward import maze_correct, maze_distance, maze_validity
from dialectic.rl.types import EnvResponse


class TestGeneration:
    def test_maze_always_solvable(self):
        rng = random.Random(42)
        for _ in range(20):
            maze = generate_maze(MazeConfig(), rng)
            assert maze.solution is not None
            assert maze.solution_length > 0

    def test_correct_dimensions(self):
        rng = random.Random(0)
        for h, w in [(5, 5), (7, 10), (10, 7)]:
            maze = generate_maze(MazeConfig(height=h, width=w), rng)
            assert maze.height == h
            assert maze.width == w
            assert len(maze.connections) == h * w

    def test_solution_within_bounds(self):
        rng = random.Random(0)
        cfg = MazeConfig(
            height=7, width=7, min_solution_length=8, max_solution_length=20
        )
        for _ in range(10):
            maze = generate_maze(cfg, rng)
            assert 8 <= maze.solution_length <= 20

    def test_impossible_constraints_raises(self):
        rng = random.Random(0)
        cfg = MazeConfig(
            height=5, width=5, min_solution_length=100, max_solution_length=200
        )
        with pytest.raises(RuntimeError, match="Failed to generate"):
            generate_maze(cfg, rng, max_attempts=50)


class TestSeeding:
    def test_same_seed_same_maze(self):
        cfg = MazeConfig(height=7, width=7)
        m1 = generate_maze(cfg, random.Random(42))
        m2 = generate_maze(cfg, random.Random(42))
        assert m1.connections == m2.connections
        assert m1.start == m2.start
        assert m1.goal == m2.goal
        assert m1.solution == m2.solution

    def test_different_seed_different_maze(self):
        cfg = MazeConfig(height=7, width=7)
        m1 = generate_maze(cfg, random.Random(1))
        m2 = generate_maze(cfg, random.Random(2))
        assert m1.connections != m2.connections


class TestOpenness:
    def test_perfect_maze_single_path(self):
        rng = random.Random(42)
        cfg = MazeConfig(height=5, width=5, openness=0.0)
        maze = generate_maze(cfg, rng)
        total_connections = sum(len(dirs) for dirs in maze.connections.values())
        # perfect maze: exactly (h*w - 1) edges, each counted twice
        assert total_connections == (maze.height * maze.width - 1) * 2

    def test_openness_adds_connections(self):
        cfg = MazeConfig(height=7, width=7, openness=0.3)
        maze = generate_maze(cfg, random.Random(42))
        total_connections = sum(len(dirs) for dirs in maze.connections.values())
        perfect_connections = (maze.height * maze.width - 1) * 2
        assert total_connections > perfect_connections


class TestTokenization:
    def test_header_format(self):
        maze = generate_maze(MazeConfig(), random.Random(0))
        text = tokenize_maze(maze)
        lines = text.split("\n")
        assert lines[0].startswith("Maze 5x5")
        assert "Start:" in lines[0]
        assert "Goal:" in lines[0]
        assert lines[-1] == "Path:"

    def test_all_cells_present(self):
        cfg = MazeConfig(height=5, width=5)
        maze = generate_maze(cfg, random.Random(0))
        text = tokenize_maze(maze)
        for r in range(5):
            for c in range(5):
                assert f"({r},{c})" in text

    def test_connections_match(self):
        maze = generate_maze(MazeConfig(), random.Random(0))
        text = tokenize_maze(maze)
        for line in text.split("\n")[1:-1]:
            cell_str, dirs_str = line.split(" > ")
            r, c = map(int, cell_str.strip("()").split(","))
            dirs = set(dirs_str.split()) if dirs_str.strip() else set()
            assert dirs == maze.connections[(r, c)]


class TestPathValidation:
    def test_valid_solution_reaches_goal(self):
        maze = generate_maze(MazeConfig(), random.Random(0))
        result = validate_path(maze, maze.solution)
        assert result.reached_goal
        assert result.valid_moves == len(maze.solution)
        assert result.final_pos == maze.goal

    def test_empty_moves(self):
        maze = generate_maze(MazeConfig(), random.Random(0))
        result = validate_path(maze, [])
        assert not result.reached_goal
        assert result.valid_moves == 0
        assert result.total_moves == 0
        assert result.final_pos == maze.start

    def test_invalid_moves_stay_in_place(self):
        cfg = MazeConfig(height=5, width=5, openness=0.0)
        maze = generate_maze(cfg, random.Random(0))
        all_dirs = ["up", "down", "left", "right"]
        invalid_dirs = [d for d in all_dirs if d not in maze.connections[maze.start]]
        if invalid_dirs:
            result = validate_path(maze, invalid_dirs)
            assert result.valid_moves == 0
            assert result.final_pos == maze.start

    def test_mixed_valid_invalid(self):
        maze = generate_maze(MazeConfig(), random.Random(0))
        moves = maze.solution[:2] + ["up", "up", "up"] + maze.solution[2:]
        result = validate_path(maze, moves)
        assert result.total_moves == len(moves)
        assert result.valid_moves <= result.total_moves

    def test_path_length_counts_visited_cells(self):
        maze = generate_maze(MazeConfig(), random.Random(0))
        result = validate_path(maze, maze.solution)
        assert result.path_length >= 2  # at least start and goal


class TestReward:
    @pytest.fixture
    def maze(self) -> Maze:
        return generate_maze(MazeConfig(), random.Random(0))

    def _make_env_response(self, maze: Maze) -> EnvResponse[MazeState]:
        return EnvResponse(
            is_done=True,
            data=MazeState(prompt="", maze=maze),
        )

    def test_correct_reward_on_solution(self, maze: Maze):
        resp = self._make_env_response(maze)
        assert (
            maze_correct(env_response=resp, extracted_model_output=maze.solution) == 1.0
        )

    def test_correct_reward_on_bad_path(self, maze: Maze):
        resp = self._make_env_response(maze)
        assert maze_correct(env_response=resp, extracted_model_output=["up"]) == 0.0

    def test_correct_reward_on_none(self, maze: Maze):
        resp = self._make_env_response(maze)
        assert maze_correct(env_response=resp, extracted_model_output=None) == 0.0

    def test_validity_reward_all_valid(self, maze: Maze):
        resp = self._make_env_response(maze)
        v = maze_validity(env_response=resp, extracted_model_output=maze.solution)
        assert v == 1.0

    def test_validity_reward_none(self, maze: Maze):
        resp = self._make_env_response(maze)
        assert maze_validity(env_response=resp, extracted_model_output=None) == 0.0

    def test_validity_reward_empty(self, maze: Maze):
        resp = self._make_env_response(maze)
        assert maze_validity(env_response=resp, extracted_model_output=[]) == 0.0

    def test_validity_reward_partial(self, maze: Maze):
        valid_move = maze.solution[0]
        dr, dc = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}[
            valid_move
        ]
        next_pos = (maze.start[0] + dr, maze.start[1] + dc)
        bad_dirs = [
            d
            for d in ["up", "down", "left", "right"]
            if d not in maze.connections[next_pos]
        ]
        if bad_dirs:
            moves = [valid_move, bad_dirs[0]]
            resp = self._make_env_response(maze)
            v = maze_validity(env_response=resp, extracted_model_output=moves)
            assert v == 0.5

    def test_distance_reward_on_solution(self, maze: Maze):
        resp = self._make_env_response(maze)
        assert (
            maze_distance(env_response=resp, extracted_model_output=maze.solution)
            == 1.0
        )

    def test_distance_reward_on_none(self, maze: Maze):
        resp = self._make_env_response(maze)
        assert maze_distance(env_response=resp, extracted_model_output=None) == 0.0

    def test_distance_reward_no_movement(self, maze: Maze):
        resp = self._make_env_response(maze)
        bad_dirs = [
            d
            for d in ["up", "down", "left", "right"]
            if d not in maze.connections[maze.start]
        ]
        if bad_dirs:
            d = maze_distance(env_response=resp, extracted_model_output=[bad_dirs[0]])
            assert d == 0.0

    def test_distance_reward_partial_progress(self, maze: Maze):
        resp = self._make_env_response(maze)
        half = maze.solution[: len(maze.solution) // 2]
        d = maze_distance(env_response=resp, extracted_model_output=half)
        assert 0.0 < d < 1.0


class TestBfsDistance:
    def test_same_cell(self):
        maze = generate_maze(MazeConfig(), random.Random(0))
        assert bfs_distance(maze.connections, maze.start, maze.start) == 0

    def test_start_to_goal(self):
        maze = generate_maze(MazeConfig(), random.Random(0))
        dist = bfs_distance(maze.connections, maze.start, maze.goal)
        assert dist == maze.solution_length

    def test_symmetric(self):
        maze = generate_maze(MazeConfig(), random.Random(0))
        d1 = bfs_distance(maze.connections, maze.start, maze.goal)
        d2 = bfs_distance(maze.connections, maze.goal, maze.start)
        assert d1 == d2


class TestExtractor:
    def test_extract_simple(self):
        text = "<answer>right, down, left, up</answer>"
        assert extract_maze_moves(text) == ["right", "down", "left", "up"]

    def test_extract_single_char(self):
        text = "<answer>R D L U</answer>"
        assert extract_maze_moves(text) == ["right", "down", "left", "up"]

    def test_extract_mixed_case(self):
        text = "<answer>Right, DOWN, Left, UP</answer>"
        assert extract_maze_moves(text) == ["right", "down", "left", "up"]

    def test_extract_newlines(self):
        text = "<answer>right\ndown\nleft</answer>"
        assert extract_maze_moves(text) == ["right", "down", "left"]

    def test_no_answer_tags(self):
        assert extract_maze_moves("just some text") is None

    def test_empty_answer_tags(self):
        assert extract_maze_moves("<answer></answer>") is None

    def test_garbage_in_tags(self):
        assert extract_maze_moves("<answer>foo bar baz</answer>") is None

    def test_with_thinking(self):
        text = (
            "<think>I need to go right then down</think>\n<answer>right, down</answer>"
        )
        assert extract_maze_moves(text) == ["right", "down"]


class TestDifficultyPresets:
    @pytest.mark.parametrize("preset", [EASY, MEDIUM, HARD, VERY_HARD])
    def test_preset_generates(self, preset: MazeConfig):
        maze = generate_maze(preset, random.Random(42))
        assert maze.solution_length > 0
        if preset.min_solution_length is not None:
            assert maze.solution_length >= preset.min_solution_length


class TestStartGoalPlacement:
    def test_top_left_bottom_right(self):
        maze = generate_maze(MazeConfig(), random.Random(0))
        assert maze.start == (0, 0)
        assert maze.goal == (4, 4)

    def test_farthest(self):
        cfg = MazeConfig(height=7, width=7, start_pos="top_left", goal_pos="farthest")
        maze = generate_maze(cfg, random.Random(0))
        assert maze.start == (0, 0)
        assert maze.goal != (0, 0)

    def test_random_positions(self):
        cfg = MazeConfig(height=7, width=7, start_pos="random", goal_pos="random")
        maze = generate_maze(cfg, random.Random(0))
        assert maze.start != maze.goal

    def test_corners(self):
        cfg = MazeConfig(height=7, width=7, start_pos="corners", goal_pos="corners")
        corners = {(0, 0), (0, 6), (6, 0), (6, 6)}
        maze = generate_maze(cfg, random.Random(0))
        assert maze.start in corners
        assert maze.goal in corners
        assert maze.start != maze.goal
