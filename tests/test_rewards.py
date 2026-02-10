import pytest

from dialectic.rl.env import Countdown
from dialectic.rl.reward import (
    _evaluate_and_verify_countdown,
    answer_tags,
    countdown_correct,
    think_tags,
    weighted_reward,
)
from dialectic.rl.types import EnvResponse


class TestEvaluateAndVerifyCountdown:
    def test_correct_using_all_numbers(self):
        assert _evaluate_and_verify_countdown("(1 + 3) * 1", [1, 3, 1], 4)

    def test_correct_but_missing_numbers(self):
        assert not _evaluate_and_verify_countdown("1 + 3", [1, 3, 1], 4)

    def test_wrong_result(self):
        assert not _evaluate_and_verify_countdown("1 + 3 + 1", [1, 3, 1], 10)

    def test_uses_number_not_in_list(self):
        assert not _evaluate_and_verify_countdown("2 + 3", [1, 3], 5)

    def test_duplicate_usage(self):
        assert not _evaluate_and_verify_countdown("3 + 3", [3, 5], 6)

    def test_invalid_characters_rejected(self):
        assert not _evaluate_and_verify_countdown("__import__('os')", [1], 1)

    def test_decimal_rejected(self):
        assert not _evaluate_and_verify_countdown("1.5 + 2.5", [1, 5, 2, 5], 4)

    def test_division(self):
        assert _evaluate_and_verify_countdown("10 / 2", [10, 2], 5)

    def test_complex_expression(self):
        assert _evaluate_and_verify_countdown(
            "(25 + 75) * 2 - 50", [25, 75, 2, 50], 150
        )


def _make_env_response(numbers: list[int], target: int) -> EnvResponse[Countdown]:
    return EnvResponse(
        is_done=True,
        data=Countdown(prompt="", numbers=numbers, target=target),
    )


class TestCountdownCorrect:
    def test_correct_gives_reward(self):
        result = countdown_correct(
            env_response=_make_env_response([1, 3, 1], 4),
            extracted_model_output="(1 + 3) * 1",
        )
        assert result == 1.0

    def test_missing_numbers_gives_no_reward(self):
        result = countdown_correct(
            env_response=_make_env_response([1, 3, 1], 4),
            extracted_model_output="1 + 3",
        )
        assert result == 0.0

    def test_none_output_gives_no_reward(self):
        result = countdown_correct(
            env_response=_make_env_response([1, 3, 1], 4),
            extracted_model_output=None,
        )
        assert result == 0.0


class TestWeightedReward:
    def test_correct_answer(self):
        fn = weighted_reward([("correct", 1.0, countdown_correct)])
        result = fn(
            env_response=_make_env_response([1, 3, 1], 4),
            extracted_model_output="(1 + 3) * 1",
        )
        assert result.total == 1.0
        assert result.components["correct"] == 1.0

    def test_partial_numbers_no_reward(self):
        fn = weighted_reward([("correct", 1.0, countdown_correct)])
        result = fn(
            env_response=_make_env_response([1, 3, 1], 4),
            extracted_model_output="1 + 3",
        )
        assert result.total == 0.0
        assert result.components["correct"] == 0.0

    def test_weighted_sum(self):
        fn = weighted_reward(
            [
                ("correct", 1.0, countdown_correct),
                ("answer_tags", 0.1, answer_tags),
                ("think_tags", 0.05, think_tags("think")),
            ]
        )
        result = fn(
            env_response=_make_env_response([2, 3, 5], 10),
            raw_model_output="<think>thinking</think><answer>2 + 3 + 5</answer>",
            extracted_model_output="2 + 3 + 5",
        )
        assert result.components["correct"] == 1.0
        assert result.components["answer_tags"] == 1.0
        assert result.components["think_tags"] == 1.0
        assert result.total == pytest.approx(1.15)


class TestAnswerTags:
    def test_single_answer_block(self):
        result = answer_tags(
            raw_model_output="<think>reasoning</think><answer>1</answer>",
        )
        assert result == 1.0

    def test_multiple_answer_blocks_no_reward(self):
        result = answer_tags(
            raw_model_output="<answer>answer1</answer> <answer>answer2</answer>",
        )
        assert result == 0.0

    def test_no_answer_tags(self):
        result = answer_tags(
            raw_model_output="just some text",
        )
        assert result == 0.0

    def test_none_output(self):
        result = answer_tags(
            raw_model_output=None,
        )
        assert result == 0.0


class TestThinkTags:
    def test_single_think_block(self):
        fn = think_tags("think")
        result = fn(
            raw_model_output="<think>reasoning</think><answer>1</answer>",
        )
        assert result == 1.0

    def test_multiple_think_blocks_no_reward(self):
        fn = think_tags("think")
        result = fn(
            raw_model_output="<think>first</think><think>second</think><answer>1</answer>",
        )
        assert result == 0.0

    def test_no_think_tags(self):
        fn = think_tags("think")
        result = fn(
            raw_model_output="just an answer",
        )
        assert result == 0.0

    def test_prefilled_open_single_close(self):
        fn = think_tags("think", prefilled_open=True)
        result = fn(
            raw_model_output="reasoning</think><answer>1</answer>",
        )
        assert result == 1.0

    def test_prefilled_open_multiple_close_no_reward(self):
        fn = think_tags("think", prefilled_open=True)
        result = fn(
            raw_model_output="first</think>second</think><answer>1</answer>",
        )
        assert result == 0.0

    def test_prefilled_open_ignores_open_tags(self):
        fn = think_tags("think", prefilled_open=True)
        result = fn(
            raw_model_output="reasoning</think><think>extra</think>",
        )
        assert result == 0.0

    def test_none_output(self):
        fn = think_tags("think")
        result = fn(
            raw_model_output=None,
        )
        assert result == 0.0
