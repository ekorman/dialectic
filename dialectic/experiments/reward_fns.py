from dialectic.experiments.prompts import REASONING_TAG
from dialectic.rl.reward import (
    answer_tags,
    countdown_correct,
    maze_correct,
    maze_distance,
    maze_validity,
    scratch_tags,
    think_tags,
    weighted_reward,
)


def get_countdown_reward_fn(
    answer_tags_weight: float,
    think_tags_weight: float,
    scratch_tags_weight: float = 0.0,
):
    components = [
        ("correct", 1.0, countdown_correct),
        ("answer_tags", answer_tags_weight, answer_tags),
    ]
    if think_tags_weight > 0:
        components.append(
            (
                "think_tags",
                think_tags_weight,
                think_tags(REASONING_TAG, prefilled_open=True),
            )
        )
    if scratch_tags_weight > 0:
        components.append(("scratch_tags", scratch_tags_weight, scratch_tags))
    reward_fn = weighted_reward(components)
    return reward_fn


def get_maze_reward_fn(
    answer_tags_weight: float,
    validity_weight: float,
    distance_weight: float,
    think_tags_weight: float,
):
    components = [
        ("correct", 1.0, maze_correct),
        ("distance", distance_weight, maze_distance),
        ("validity", validity_weight, maze_validity),
        ("answer_tags", answer_tags_weight, answer_tags),
    ]
    if think_tags_weight > 0:
        components.append(
            (
                "think_tags",
                think_tags_weight,
                think_tags(REASONING_TAG, prefilled_open=True),
            )
        )
    return weighted_reward(components)
