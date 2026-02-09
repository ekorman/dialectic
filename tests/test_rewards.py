from dialectic.rl.env import Countdown
from dialectic.rl.reward import (
    CountdownAnswerTagsComponent,
    CountdownCorrectComponent,
    CountdownRewardFn,
    CountdownThinkTagsComponent,
    _evaluate_and_verify_countdown,
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


class TestCountdownCorrectComponent:
    def test_correct_gives_reward(self):
        comp = CountdownCorrectComponent()
        result = comp(
            env_response=_make_env_response([1, 3, 1], 4),
            extracted_model_output="(1 + 3) * 1",
        )
        assert result == 1.0

    def test_missing_numbers_gives_no_reward(self):
        comp = CountdownCorrectComponent()
        result = comp(
            env_response=_make_env_response([1, 3, 1], 4),
            extracted_model_output="1 + 3",
        )
        assert result == 0.0

    def test_none_output_gives_no_reward(self):
        comp = CountdownCorrectComponent()
        result = comp(
            env_response=_make_env_response([1, 3, 1], 4),
            extracted_model_output=None,
        )
        assert result == 0.0


class TestCountdownRewardFn:
    def test_correct_answer(self):
        fn = CountdownRewardFn()
        result = fn(
            env_response=_make_env_response([1, 3, 1], 4),
            extracted_model_output="(1 + 3) * 1",
        )
        assert result.total == 1.0
        assert result.components["correct"] == 1.0

    def test_partial_numbers_no_reward(self):
        fn = CountdownRewardFn()
        result = fn(
            env_response=_make_env_response([1, 3, 1], 4),
            extracted_model_output="1 + 3",
        )
        assert result.total == 0.0
        assert result.components["correct"] == 0.0


_ENV = _make_env_response([1], 1)


class TestCountdownAnswerTagsComponent:
    def test_single_answer_block(self):
        comp = CountdownAnswerTagsComponent()
        result = comp(
            env_response=_ENV,
            raw_model_output="<think>reasoning</think><answer>1</answer>",
            extracted_model_output="1",
        )
        assert result == 0.1

    def test_multiple_answer_blocks_no_reward(self):
        comp = CountdownAnswerTagsComponent()
        result = comp(
            env_response=_ENV,
            raw_model_output="<answer>answer1</answer> <answer>answer2</answer>",
            extracted_model_output="answer1",
        )
        assert result == 0.0

    def test_no_answer_tags(self):
        comp = CountdownAnswerTagsComponent()
        result = comp(
            env_response=_ENV,
            raw_model_output="just some text",
            extracted_model_output=None,
        )
        assert result == 0.0

    def test_none_output(self):
        comp = CountdownAnswerTagsComponent()
        result = comp(
            env_response=_ENV,
            raw_model_output=None,
            extracted_model_output=None,
        )
        assert result == 0.0


class TestCountdownThinkTagsComponent:
    def test_single_think_block(self):
        comp = CountdownThinkTagsComponent("think")
        result = comp(
            env_response=_ENV,
            raw_model_output="<think>reasoning</think><answer>1</answer>",
            extracted_model_output=None,
        )
        assert result == 0.05

    def test_multiple_think_blocks_no_reward(self):
        comp = CountdownThinkTagsComponent("think")
        result = comp(
            env_response=_ENV,
            raw_model_output="<think>first</think><think>second</think><answer>1</answer>",
            extracted_model_output=None,
        )
        assert result == 0.0

    def test_no_think_tags(self):
        comp = CountdownThinkTagsComponent("think")
        result = comp(
            env_response=_ENV,
            raw_model_output="just an answer",
            extracted_model_output=None,
        )
        assert result == 0.0

    def test_prefilled_open_single_close(self):
        comp = CountdownThinkTagsComponent("think", prefilled_open=True)
        result = comp(
            env_response=_ENV,
            raw_model_output="reasoning</think><answer>1</answer>",
            extracted_model_output=None,
        )
        assert result == 0.05

    def test_prefilled_open_multiple_close_no_reward(self):
        comp = CountdownThinkTagsComponent("think", prefilled_open=True)
        result = comp(
            env_response=_ENV,
            raw_model_output="first</think>second</think><answer>1</answer>",
            extracted_model_output=None,
        )
        assert result == 0.0

    def test_prefilled_open_ignores_open_tags(self):
        comp = CountdownThinkTagsComponent("think", prefilled_open=True)
        result = comp(
            env_response=_ENV,
            raw_model_output="reasoning</think><think>extra</think>",
            extracted_model_output=None,
        )
        assert result == 0.0

    def test_none_output(self):
        comp = CountdownThinkTagsComponent("think")
        result = comp(
            env_response=_ENV,
            raw_model_output=None,
            extracted_model_output=None,
        )
        assert result == 0.0
