import json
from typing import Callable

import extty

from dialectic.experiments.params import (
    CountdownParams,
    MazeRewardParams,
    RewardParams,
    TrainParams,
)
from dialectic.experiments.prompts import (
    MAZE_INTERNAL_REASONING_PROMPT,
    PromptCollection,
)
from dialectic.experiments.reward_fns import get_countdown_reward_fn, get_maze_reward_fn
from dialectic.llm.templates import Message
from dialectic.rl.env import Countdown, CountdownEnv, MathState, MazeEnv, MazeState
from dialectic.rl.extractors import extract_from_answer_tags, extract_maze_moves
from dialectic.rl.maze import MazeConfig
from dialectic.rl.types import EnvResponse


def get_state_to_str(
    *,
    format_messages: Callable[[list[Message], bool], str],
    system_prompt: str | None = None,
    assistant_prefill: str | None = None,
):
    def _state_to_str(data: Countdown | MazeState | MathState) -> str:
        msgs = []
        if system_prompt:
            msgs.append(Message(role="system", content=system_prompt))
        msgs.append(Message(role="user", content=data.prompt))
        ret = format_messages(msgs, True)
        if assistant_prefill:
            ret += assistant_prefill
        return ret

    return _state_to_str


def get_countdown_env_reward_fn_extractor_val_envs(
    train_params: TrainParams,
    reward_params: RewardParams,
    prompt_collection: PromptCollection,
    countdown_params: CountdownParams,
):
    n_ops = countdown_params.n_ops
    n_larges = countdown_params.n_larges
    n_total = countdown_params.n_total

    env = CountdownEnv(
        seed=train_params.seed,
        n_larges=n_larges,
        n_total=n_total,
        n_ops=n_ops,
        prompt_template=prompt_collection.env_prompt,
    )
    reward_fn = get_countdown_reward_fn(
        answer_tags_weight=reward_params.answer_tags_weight,
        think_tags_weight=reward_params.think_tags_weight,
    )
    extractor = extract_from_answer_tags

    n_ops_list = [n_ops] if isinstance(n_ops, int) else n_ops
    n_total_list = [n_total] if isinstance(n_total, int) else n_total
    n_larges_list = [n_larges] if isinstance(n_larges, int) else n_larges
    val_envs = [
        CountdownEnv(
            seed=2026 + i,
            n_larges=n_larges_list[i],
            n_total=n_total_list[i],
            n_ops=n_ops_list[i],
            prompt_template=prompt_collection.env_prompt,
        )
        for i in range(len(n_ops_list))
    ]
    return env, reward_fn, extractor, val_envs


def get_maze_env_reward_fn_extractor_val_envs(
    train_params: TrainParams,
    reward_params: RewardParams,
    maze_reward_params: MazeRewardParams,
    maze_config: MazeConfig,
    prompt_collection: PromptCollection,
):
    env = MazeEnv(
        config=maze_config,
        prompt_template=prompt_collection.env_prompt,
        seed=train_params.seed,
    )
    reward_fn = get_maze_reward_fn(
        answer_tags_weight=reward_params.answer_tags_weight,
        validity_weight=maze_reward_params.validity_weight,
        distance_weight=maze_reward_params.distance_weight,
        think_tags_weight=reward_params.think_tags_weight,
    )
    val_envs = [
        MazeEnv(
            config=maze_config,
            prompt_template=MAZE_INTERNAL_REASONING_PROMPT.env_prompt,
            seed=2026,
        )
    ]

    return env, reward_fn, extract_maze_moves, val_envs


def load_countdown_dataset_artifacts(
    artifact_names: list[str],
    prompt_template: str,
) -> list[tuple[EnvResponse[Countdown], dict]]:
    """Load countdown problems from one or more dataset artifacts.

    Returns list of (env_response, extra_fields) tuples, where extra_fields
    contains passthrough fields like 'split' and 'equation'.
    """
    all_problems: list[tuple[EnvResponse[Countdown], dict]] = []
    for name in artifact_names:
        data = extty.load_artifact(name)
        if not isinstance(data, bytes):
            raise ValueError(f"Expected bytes from artifact {name}, got {type(data)}")
        count = 0
        for line in data.decode().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            numbers = entry["numbers"]
            target = entry["target"]
            prompt = prompt_template.format(numbers=numbers, target=target)
            extra: dict = {}
            for key in ("split", "equation"):
                if key in entry:
                    extra[key] = entry[key]
            all_problems.append(
                (
                    EnvResponse(
                        is_done=True,
                        data=Countdown(
                            prompt=prompt,
                            numbers=numbers,
                            target=target,
                            solution=None,
                        ),
                    ),
                    extra,
                )
            )
            count += 1
        print(f"Loaded {count} problems from artifact '{name}'")
    print(f"Total: {len(all_problems)} problems from {len(artifact_names)} artifact(s)")
    return all_problems
