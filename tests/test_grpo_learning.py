"""
Integration tests verifying GRPO can learn.

Tier 1: Mechanical verification with tiny model (~10 sec)
Tier 2: Learning verification with Qwen-0.6B (slower, requires GPU)

Run with:
    uv run pytest tests/test_grpo_learning.py -v -s
    uv run pytest tests/test_grpo_learning.py::TestGRPOMechanics -v  # Tier 1 only
    uv run pytest tests/test_grpo_learning.py::TestGRPOLearning -v -s  # Tier 2 only
"""

import math

import pytest
import torch
from tokenizers import Tokenizer

from dialectic.llm.qwen import Qwen
from dialectic.rl.env import CountdownEnv
from dialectic.rl.extractors import extract_from_answer_tags
from dialectic.rl.reward import CountdownRewardFn
from dialectic.rl.train import train_grpo
from dialectic.rl.types import Countdown


def create_tiny_model(vocab_size: int = 151936) -> Qwen:
    """Create a tiny model for fast testing."""
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
    """Convert Countdown data to prompt string."""
    return data.prompt


class TestGRPOMechanics:
    """Tier 1: Verify training mechanics work with tiny model."""

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

    def test_training_loop_completes(self, tiny_model, tokenizer, env):
        """Training loop runs without errors."""
        opt = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)

        metrics = train_grpo(
            net=tiny_model,
            opt=opt,
            env=env,
            reward_fn=CountdownRewardFn(),
            state_to_str=countdown_state_to_str,
            tokenizer=tokenizer,
            eos_token_id=151643,
            pad_token_id=151643,
            extractor=extract_from_answer_tags,
            beta=0.01,
            eps=0.2,
            mu=1,
            max_tokens_generated=20,
            max_episodes=4,
            update_ref_net_batch_cadence=5,
            batch_size=2,
            group_size=2,
            temperature=1.0,
        )

        assert len(metrics.losses) > 0
        assert len(metrics.mean_rewards) > 0

    def test_loss_is_finite(self, tiny_model, tokenizer, env):
        """Loss values are not NaN/Inf."""
        opt = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)

        metrics = train_grpo(
            net=tiny_model,
            opt=opt,
            env=env,
            reward_fn=CountdownRewardFn(),
            state_to_str=countdown_state_to_str,
            tokenizer=tokenizer,
            eos_token_id=151643,
            pad_token_id=151643,
            extractor=extract_from_answer_tags,
            beta=0.01,
            eps=0.2,
            mu=1,
            max_tokens_generated=20,
            max_episodes=4,
            update_ref_net_batch_cadence=5,
            batch_size=2,
            group_size=2,
            temperature=1.0,
        )

        for loss in metrics.losses:
            assert math.isfinite(loss), f"Loss is not finite: {loss}"

    def test_gradients_flow(self, tiny_model, tokenizer, env):
        """Model parameters are updated during training.

        Uses a length-based reward to ensure variance in rewards,
        which creates non-zero advantages and allows gradients to flow.
        """
        from dialectic.rl.reward import RewardFn
        from dialectic.rl.types import EnvResponse

        class LengthRewardFn(RewardFn[Countdown, str | None]):
            """Reward based on output length to ensure variance."""

            def __call__(
                self,
                *,
                env_response: EnvResponse[Countdown],
                raw_model_output: str | None = None,
                extracted_model_output: str | None,
            ) -> float:
                if raw_model_output is None:
                    return 0.0
                return len(raw_model_output) / 100.0

        opt = torch.optim.Adam(tiny_model.parameters(), lr=1e-2)

        params_before = {
            name: param.clone() for name, param in tiny_model.named_parameters()
        }

        train_grpo(
            net=tiny_model,
            opt=opt,
            env=env,
            reward_fn=LengthRewardFn(),
            state_to_str=countdown_state_to_str,
            tokenizer=tokenizer,
            eos_token_id=151643,
            pad_token_id=151643,
            extractor=extract_from_answer_tags,
            beta=0.01,
            eps=0.2,
            mu=1,
            max_tokens_generated=20,
            max_episodes=8,
            update_ref_net_batch_cadence=10,
            batch_size=2,
            group_size=2,
            temperature=1.0,
        )

        params_changed = False
        for name, param in tiny_model.named_parameters():
            if not torch.allclose(params_before[name], param, atol=1e-8):
                params_changed = True
                break

        assert params_changed, "No parameters changed during training"


class TestCountdownReward:
    """Test the countdown reward function."""

    def test_correct_answer(self):
        """Reward is 1.0 for correct answers."""
        from dialectic.rl.types import EnvResponse

        reward_fn = CountdownRewardFn()
        env_response = EnvResponse(
            is_done=True,
            data=Countdown(prompt="...", numbers=[2, 3, 5], target=10),
        )

        reward = reward_fn(
            env_response=env_response,
            raw_model_output="<answer>2 + 3 + 5</answer>",
            extracted_model_output="2 + 3 + 5",
        )
        assert reward == 1.0

    def test_wrong_answer(self):
        """Reward is 0.0 for wrong answers."""
        from dialectic.rl.types import EnvResponse

        reward_fn = CountdownRewardFn()
        env_response = EnvResponse(
            is_done=True,
            data=Countdown(prompt="...", numbers=[2, 3, 5], target=10),
        )

        reward = reward_fn(
            env_response=env_response,
            raw_model_output="<answer>2 + 3</answer>",
            extracted_model_output="2 + 3",
        )
        assert reward == 0.0

    def test_invalid_numbers(self):
        """Reward is 0.0 when using numbers not in the set."""
        from dialectic.rl.types import EnvResponse

        reward_fn = CountdownRewardFn()
        env_response = EnvResponse(
            is_done=True,
            data=Countdown(prompt="...", numbers=[2, 3, 5], target=10),
        )

        reward = reward_fn(
            env_response=env_response,
            raw_model_output="<answer>4 + 6</answer>",
            extracted_model_output="4 + 6",
        )
        assert reward == 0.0

    def test_no_answer(self):
        """Reward is 0.0 when no answer extracted."""
        from dialectic.rl.types import EnvResponse

        reward_fn = CountdownRewardFn()
        env_response = EnvResponse(
            is_done=True,
            data=Countdown(prompt="...", numbers=[2, 3, 5], target=10),
        )

        reward = reward_fn(
            env_response=env_response,
            raw_model_output="I don't know",
            extracted_model_output=None,
        )
        assert reward == 0.0


class TestAnswerExtractor:
    """Test the answer tag extractor."""

    def test_extracts_answer(self):
        assert extract_from_answer_tags("<answer>2 + 3</answer>") == "2 + 3"

    def test_extracts_multiline(self):
        text = "<answer>\n2 + 3 + 5\n</answer>"
        assert extract_from_answer_tags(text) == "2 + 3 + 5"

    def test_returns_none_if_no_tags(self):
        assert extract_from_answer_tags("no tags here") is None

    def test_with_think_tags(self):
        text = "<think>Let me think...</think><answer>42</answer>"
        assert extract_from_answer_tags(text) == "42"
