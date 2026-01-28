"""Tests for the evaluation module."""

import torch

from dialectic.rl.env import Countdown
from dialectic.rl.evaluate import EvaluationResult, evaluate
from dialectic.rl.extractors import extract_from_answer_tags
from dialectic.rl.reward import CountdownRewardFn
from dialectic.rl.train import RolloutBatch, generate_rollout_batch


def countdown_state_to_str(data: Countdown) -> str:
    return data.prompt


class TestEvaluate:
    def test_evaluate_runs_without_errors(self, tiny_model, tokenizer, env):
        """Evaluate function completes without errors."""
        result, _ = evaluate(
            net=tiny_model,
            env=env,
            reward_fn=CountdownRewardFn(),
            state_to_str=countdown_state_to_str,
            tokenizer=tokenizer,
            eos_token_id=151643,
            pad_token_id=151643,
            extractor=extract_from_answer_tags,
            max_tokens_generated=20,
            max_episodes=4,
            batch_size=2,
            group_size=2,
            temperature=1.0,
        )

        assert isinstance(result, EvaluationResult)

    def test_evaluate_returns_correct_episode_count(self, tiny_model, tokenizer, env):
        """Evaluate returns the correct number of episodes."""
        result, _ = evaluate(
            net=tiny_model,
            env=env,
            reward_fn=CountdownRewardFn(),
            state_to_str=countdown_state_to_str,
            tokenizer=tokenizer,
            eos_token_id=151643,
            pad_token_id=151643,
            extractor=extract_from_answer_tags,
            max_tokens_generated=20,
            max_episodes=6,
            batch_size=2,
            group_size=1,
            temperature=1.0,
        )

        assert result.n_episodes == 6

    def test_evaluate_result_fields(self, tiny_model, tokenizer, env):
        """EvaluationResult contains all expected fields with valid values."""
        result, _ = evaluate(
            net=tiny_model,
            env=env,
            reward_fn=CountdownRewardFn(),
            state_to_str=countdown_state_to_str,
            tokenizer=tokenizer,
            eos_token_id=151643,
            pad_token_id=151643,
            extractor=extract_from_answer_tags,
            max_tokens_generated=20,
            max_episodes=4,
            batch_size=2,
            group_size=2,
            temperature=1.0,
        )

        assert isinstance(result.reward_mean, float)
        assert isinstance(result.reward_std, float)
        assert isinstance(result.component_means, dict)

    def test_evaluate_greedy_sampling(self, tiny_model, tokenizer, env):
        """Evaluate works with greedy sampling (temperature=0)."""
        result, _ = evaluate(
            net=tiny_model,
            env=env,
            reward_fn=CountdownRewardFn(),
            state_to_str=countdown_state_to_str,
            tokenizer=tokenizer,
            eos_token_id=151643,
            pad_token_id=151643,
            extractor=extract_from_answer_tags,
            max_tokens_generated=20,
            max_episodes=2,
            batch_size=1,
            group_size=1,
            temperature=0.0,
        )

        assert result.n_episodes == 2
        assert not any(torch.isnan(torch.tensor([result.reward_mean])))


class TestGenerateRolloutBatch:
    def test_rollout_batch_structure(self, tiny_model, tokenizer, env):
        """generate_rollout_batch returns RolloutBatch with correct structure."""
        batch_size = 2
        group_size = 3

        rollout = generate_rollout_batch(
            net=tiny_model,
            env=env,
            reward_fn=CountdownRewardFn(),
            state_to_str=countdown_state_to_str,
            tokenizer=tokenizer,
            eos_token_id=151643,
            pad_token_id=151643,
            extractor=extract_from_answer_tags,
            batch_size=batch_size,
            group_size=group_size,
            temperature=1.0,
            max_tokens_generated=20,
        )

        assert isinstance(rollout, RolloutBatch)
        assert len(rollout.env_responses) == batch_size
        assert len(rollout.prompts) == batch_size
        assert len(rollout.output_strs) == group_size
        assert len(rollout.output_strs[0]) == batch_size
        assert len(rollout.reward_results) == group_size
        assert len(rollout.reward_results[0]) == batch_size
        assert rollout.rewards.shape == (group_size, batch_size)
        assert len(rollout.completion_token_ids) == group_size
        assert rollout.attention_mask.shape[0] == batch_size
        assert rollout.t_generation > 0
