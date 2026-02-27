# Maze Generator Spec

## Overview

A lightweight, self-contained maze generator for RL training. Produces random solvable mazes as tokenized text prompts with computed optimal solutions. No external dependencies beyond numpy.

## Difficulty Parameters

```python
@dataclass
class MazeConfig:
    # Grid dimensions
    height: int = 5          # number of rows (5-30)
    width: int = 5           # number of columns (5-30)

    # Path complexity
    min_solution_length: int = None   # reject mazes with shorter optimal paths
    max_solution_length: int = None   # reject mazes with longer optimal paths

    # Maze structure
    algorithm: str = "dfs"   # "dfs" (long winding paths) | "kruskal" (more open) | "wilson" (uniform random)
    openness: float = 0.0    # fraction of extra walls to remove after generation (0.0 = perfect maze, 0.5 = very open)
                             # higher openness = more alternative paths = harder planning (more choices)

    # Start/goal placement
    start_pos: str = "top_left"      # "top_left" | "random" | "corners" (random corner pair)
    goal_pos: str = "bottom_right"   # "bottom_right" | "random" | "farthest" (farthest from start)

    # Reproducibility
    seed: int = None
```

### How difficulty scales

**Easy** (good for initial training):
- 5×5 grid, DFS algorithm, openness=0.0 (perfect maze, exactly one path)
- Solution length ~8-15 moves
- Simple: model just needs to find the single existing path

**Medium**:
- 7×7 grid, DFS algorithm, openness=0.1-0.2
- Solution length ~15-30 moves
- Some route choices due to removed walls, but not overwhelming

**Hard**:
- 10×10+ grid, openness=0.3+, start/goal="farthest"
- Solution length 30+ moves
- Many alternative routes, requires genuine planning to find efficient path

**Key insight**: `openness` controls a meaningful difficulty axis. A perfect maze (openness=0) has exactly one path between any two cells — no planning needed, just follow the corridor. Adding openness creates loops and choices, requiring the model to actually evaluate alternatives. This is where soft tokens should shine.

## Maze Representation

### Internal representation

```python
class Maze:
    grid: np.ndarray          # (height, width) cell values: 0=open, 1=wall (if using wall grid)
    walls: dict               # {(r,c): set of directions with walls} - adjacency-based
    start: tuple[int, int]    # (row, col)
    goal: tuple[int, int]     # (row, col)
    solution: list[str]       # optimal path as list of moves: ["right", "right", "down", ...]
    solution_length: int      # number of moves in optimal path
```

### Tokenized text format (prompt)

Keep it simple. Use an adjacency/connection list format that's compact and unambiguous:

```
Maze 5x5 | Start: (0,0) | Goal: (4,4)
(0,0) > right down
(0,1) > left right
(0,2) > left down
(1,0) > down
(1,1) > right
...
Path:
```

Each line lists a cell and its valid moves (which neighbors are reachable — no wall in between). The model reads this, then generates moves.

**Alternative simpler format** (ASCII grid):

```
S . . # .
# # . # .
. . . . .
. # # # .
. . . . G
```

Where `S`=start, `G`=goal, `.`=open, `#`=wall. Simpler to parse visually but may be harder for the model to reason about connectivity.

**Recommendation**: Start with the adjacency list format. It explicitly encodes connectivity (which is what the model needs to reason about) rather than requiring the model to infer it from a grid. This is the format used by most successful maze-solving transformer work.

### Action tokens

The model outputs one of 4 tokens per hard token position:
- `<up>` or `U`
- `<down>` or `D`
- `<left>` or `L`
- `<right>` or `R`

Use single-character tokens (U/D/L/R) if we want to minimize tokenizer complexity, or special tokens if the base model tokenizer supports them.

## Generation Algorithm

### Core: Randomized DFS (default)

```
1. Start with grid where all walls are present
2. Pick a random starting cell, mark it visited
3. While there are unvisited cells reachable:
   a. From current cell, pick a random unvisited neighbor
   b. Remove the wall between current and neighbor
   c. Move to neighbor, mark visited
   d. If no unvisited neighbors, backtrack
4. Result: a "perfect" maze (exactly one path between any two cells)
```

### Post-processing: openness

After generating the perfect maze, randomly remove `openness * num_remaining_walls` additional walls. This creates loops and multiple paths.

### Solution computation

BFS from start to goal. Returns shortest path as list of moves. If `openness > 0`, there may be multiple shortest paths; BFS gives one.

### Rejection sampling

If `min_solution_length` or `max_solution_length` is set, regenerate mazes that don't satisfy the constraint. For reasonable parameter ranges, rejection rates should be low.

## API

```python
def generate_maze(config: MazeConfig) -> Maze:
    """Generate a single maze."""
    ...

def generate_dataset(config: MazeConfig, n: int) -> list[Maze]:
    """Generate n mazes, applying rejection sampling for solution length constraints."""
    ...

def tokenize_maze(maze: Maze, format: str = "adjacency") -> str:
    """Convert maze to text prompt."""
    ...

def validate_path(maze: Maze, moves: list[str]) -> dict:
    """
    Check a sequence of moves against the maze.
    Returns:
        {
            "valid": bool,           # all moves legal (no wall collisions)?
            "reached_goal": bool,    # did the path end at the goal?
            "num_valid_moves": int,  # how many moves were legal before first illegal move
            "num_moves": int,        # total moves attempted
            "final_pos": (int, int), # where the agent ended up
            "optimal_length": int,   # length of shortest path
            "path_length": int,      # length of agent's path
        }
    """
    ...

def compute_reward(maze: Maze, moves: list[str], config: RewardConfig) -> float:
    """
    Compute reward for a trajectory.
    """
    result = validate_path(maze, moves)
    reward = 0.0
    if result["reached_goal"]:
        reward += config.goal_reward           # e.g., 1.0
    if result["valid"]:
        reward += config.validity_reward       # e.g., 0.5
    # optional efficiency bonus
    if result["reached_goal"] and config.efficiency_weight > 0:
        ratio = result["optimal_length"] / max(result["path_length"], 1)
        reward += config.efficiency_weight * ratio  # e.g., 0.25 * (optimal/actual)
    return reward
```

## Difficulty Presets

```python
EASY = MazeConfig(height=5, width=5, algorithm="dfs", openness=0.0,
                  start_pos="top_left", goal_pos="bottom_right")

MEDIUM = MazeConfig(height=7, width=7, algorithm="dfs", openness=0.15,
                    start_pos="top_left", goal_pos="farthest",
                    min_solution_length=12)

HARD = MazeConfig(height=10, width=10, algorithm="dfs", openness=0.3,
                  start_pos="random", goal_pos="farthest",
                  min_solution_length=25)

VERY_HARD = MazeConfig(height=15, width=15, algorithm="dfs", openness=0.3,
                       start_pos="random", goal_pos="farthest",
                       min_solution_length=40)
```

## Curriculum Strategy

For training, gradually increase difficulty:
1. **Phase 1**: EASY mazes until model consistently solves >80%
2. **Phase 2**: Mix of EASY (30%) and MEDIUM (70%)
3. **Phase 3**: Mix of MEDIUM (30%) and HARD (70%)

This mirrors how the Countdown setup works with number count/size. The curriculum helps the model learn basic navigation before tackling harder planning.

## Token Budget Estimation

For the prompt (adjacency list format):
- 5×5 maze: ~25 cells × ~8 tokens per cell ≈ 200 tokens
- 7×7 maze: ~49 cells × ~8 tokens per cell ≈ 400 tokens
- 10×10 maze: ~100 cells × ~8 tokens per cell ≈ 800 tokens

For the reasoning trace (with 4:1 soft:hard interleaving):
- 5×5 maze, ~12 moves: 12 hard tokens + 48 soft tokens = 60 positions
- 7×7 maze, ~20 moves: 20 hard tokens + 80 soft tokens = 100 positions
- 10×10 maze, ~35 moves: 35 hard tokens + 140 soft tokens = 175 positions

Total context lengths are well within Qwen-2.5's capabilities at all difficulty levels.

## Testing / Sanity Checks

Before training, verify:
1. Generated mazes are always solvable (BFS finds a path)
2. Solution length distribution matches expectations for each difficulty level
3. Tokenized format round-trips correctly (can reconstruct maze from tokens)
4. `validate_path` correctly identifies valid/invalid moves
5. Reward function produces expected values for known paths
