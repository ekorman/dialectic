from dialectic.experiments.prompts import REASONING_TAG
from dialectic.rl.reward import (
    answer_tags,
    countdown_correct,
    think_tags,
    weighted_reward,
)


def get_countdown_reward_fn(answer_tags_weight: float, think_tags_weight: float):
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
    reward_fn = weighted_reward(components)
    return reward_fn
