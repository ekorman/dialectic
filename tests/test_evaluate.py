"""Tests for the evaluation module."""

import pytest
import torch
from tokenizers import Tokenizer

from dialectic.llm.qwen import Qwen
from dialectic.rl.env import CountdownEnv
from dialectic.rl.evaluate import EvaluationResult, evaluate
from dialectic.rl.extractors import extract_from_answer_tags
from dialectic.rl.reward import CountdownRewardFn
from dialectic.rl.train import RolloutBatch, generate_rollout_batch
from dialectic.rl.types import Countdown


def create_tiny_model(vocab_size: int = 151936) -> Qwen:
    return Qwen(
        d=32,
        vocab_size=vocab_size,
        n_decoder_layers=2,
        attn_head_d=16,
        attn_num_heads=4,
        attn_num_kv_heads=2,
        mlp_hidden_d=64,
    )


def countdown_state_to_str(data: Countdown) -> str:
    return data.prompt


class TestEvaluate:
    @pytest.fixture
    def tiny_model(self):
        return create_tiny_model()

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_file("qwen-tokenizer/tokenizer.json")

    @pytest.fixture
    def env(self):
        return CountdownEnv(
            num_operands=2,
            min_number=1,
            max_number=5,
        )

    def test_evaluate_runs_without_errors(self, tiny_model, tokenizer, env):
        """Evaluate function completes without errors."""
        result = evaluate(
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
        result = evaluate(
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
        result = evaluate(
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
        assert isinstance(result.all_rewards, list)
        assert len(result.all_rewards) == 4 * 2  # max_episodes * group_size

    def test_evaluate_greedy_sampling(self, tiny_model, tokenizer, env):
        """Evaluate works with greedy sampling (temperature=0)."""
        result = evaluate(
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
    @pytest.fixture
    def tiny_model(self):
        return create_tiny_model()

    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_file("qwen-tokenizer/tokenizer.json")

    @pytest.fixture
    def env(self):
        return CountdownEnv(
            num_operands=2,
            min_number=1,
            max_number=5,
        )

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
