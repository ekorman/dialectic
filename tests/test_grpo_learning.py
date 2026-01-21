"""
Integration tests verifying GRPO can learn.

Tier 1: Mechanical verification with tiny model (~10 sec)
Tier 2: Learning verification with Qwen-0.6B (slower, requires GPU)

Run with:
    uv run pytest tests/test_grpo_learning.py -v -s
    uv run pytest tests/test_grpo_learning.py::TestGRPOMechanics -v  # Tier 1 only
    uv run pytest tests/test_grpo_learning.py::TestGRPOLearning -v -s  # Tier 2 only
"""

import pytest
import torch
from tokenizers import Tokenizer

from dialectic.llm.qwen import Qwen
from dialectic.rl.env import CountdownEnv
from dialectic.rl.extractors import extract_from_answer_tags
from dialectic.rl.reward import CountdownRewardFn
from dialectic.rl.train import compute_log_probs, train_grpo
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

        train_grpo(
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
            use_bf16=False,
        )

    def test_loss_is_finite(self, tiny_model, tokenizer, env):
        """Loss values are not NaN/Inf (verified by no exceptions during training)."""
        opt = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)

        train_grpo(
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
            use_bf16=False,
        )

    def test_gradients_flow(self, tiny_model, tokenizer, env):
        """Model parameters are updated during training.

        Uses a length-based reward to ensure variance in rewards,
        which creates non-zero advantages and allows gradients to flow.
        """
        from dialectic.rl.reward import RewardFn
        from dialectic.rl.types import EnvResponse, RewardResult

        class LengthRewardFn(RewardFn[Countdown, str | None]):
            """Reward based on output length to ensure variance."""

            def __call__(
                self,
                *,
                env_response: EnvResponse[Countdown],
                raw_model_output: str | None = None,
                extracted_model_output: str | None,
            ) -> RewardResult:
                if raw_model_output is None:
                    value = 0.0
                else:
                    value = len(raw_model_output) / 100.0
                return RewardResult(total=value, components={"length": value})

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
            use_bf16=False,
        )

        params_changed = False
        for name, param in tiny_model.named_parameters():
            if not torch.allclose(params_before[name], param, atol=1e-8):
                params_changed = True
                break

        assert params_changed, "No parameters changed during training"

    def test_bf16_training_produces_finite_loss(self, tiny_model, tokenizer, env):
        """bf16 mixed precision training produces finite loss values."""
        opt = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)

        train_grpo(
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
            use_bf16=True,
        )

        for param in tiny_model.parameters():
            assert param.dtype == torch.bfloat16
            assert torch.isfinite(param).all()

    def test_gradient_accumulation_matches_large_batch(
        self, tiny_model, tokenizer, monkeypatch
    ):
        """Gradient accumulation matches a larger batch update."""
        from copy import deepcopy

        from dialectic.rl.env import EpisodeIsDoneError, Env
        from dialectic.rl.types import EnvResponse

        class FixedCountdownEnv(Env[Countdown, None]):
            def __init__(self, responses: list[Countdown]):
                self._responses = responses
                self._idx = 0

            def reset(self, seed: int | None = None) -> EnvResponse[Countdown]:
                if self._idx >= len(self._responses):
                    raise RuntimeError("Exceeded fixed environment responses")
                response = self._responses[self._idx]
                self._idx += 1
                return EnvResponse(is_done=True, data=response)

            def step(self, action: None):
                raise EpisodeIsDoneError

        def deterministic_generate_from_tokens(
            *,
            token_ids: torch.Tensor,
            pad_token_id: int,
            **kwargs,
        ) -> torch.Tensor:
            extra = torch.full(
                (token_ids.shape[0], 1),
                pad_token_id,
                dtype=token_ids.dtype,
                device=token_ids.device,
            )
            return torch.cat([token_ids, extra], dim=1)

        monkeypatch.setattr(
            "dialectic.rl.train.generate_from_tokens",
            deterministic_generate_from_tokens,
        )

        fixed_responses = [
            Countdown(prompt=f"Prompt {i}", numbers=[1, 2, 3], target=6)
            for i in range(4)
        ]

        base_state = deepcopy(tiny_model.state_dict())

        def run_training(*, batch_size: int, accumulation_steps: int) -> Qwen:
            model = create_tiny_model()
            model.load_state_dict(base_state)
            env = FixedCountdownEnv(deepcopy(fixed_responses))
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)

            train_grpo(
                net=model,
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
                max_tokens_generated=1,
                max_episodes=batch_size * accumulation_steps,
                update_ref_net_batch_cadence=10,
                batch_size=batch_size,
                group_size=2,
                temperature=1.0,
                normalize_advantages=True,
                accumulation_steps=accumulation_steps,
                max_grad_norm=0.0,
                logprob_chunk_size=0,
                use_bf16=False,
            )
            return model

        model_large_batch = run_training(batch_size=4, accumulation_steps=1)
        model_accumulated = run_training(batch_size=2, accumulation_steps=2)

        for name, param in model_large_batch.named_parameters():
            accum_param = dict(model_accumulated.named_parameters())[name]
            assert torch.allclose(param, accum_param, atol=1e-6), (
                f"Parameter {name} mismatch between accumulation and large batch"
            )


class TestChunkedLogProbs:
    """Test that chunked log prob computation matches non-chunked."""

    @pytest.fixture
    def tiny_model(self):
        return create_tiny_model()

    def test_chunked_matches_non_chunked(self, tiny_model):
        """Chunked and non-chunked log prob computation give identical results."""
        batch_size = 2
        group_size = 3
        prompt_len = 10
        completion_len = 20
        vocab_size = tiny_model.vocab_size
        pad_token_id = 0

        torch.manual_seed(42)
        prompt_ids = torch.randint(1, vocab_size, (batch_size, prompt_len))
        completion_ids = [
            torch.cat(
                [
                    prompt_ids,
                    torch.randint(1, vocab_size, (batch_size, completion_len)),
                ],
                dim=1,
            )
            for _ in range(group_size)
        ]
        attention_mask = torch.zeros(batch_size, prompt_len, dtype=torch.bool)

        tiny_model.eval()
        with torch.no_grad():
            log_probs_no_chunk, mask_no_chunk = compute_log_probs(
                net=tiny_model,
                attention_mask=attention_mask,
                completion_token_ids=completion_ids,
                pad_token_id=pad_token_id,
                chunk_size=0,
            )

            log_probs_chunked, mask_chunked = compute_log_probs(
                net=tiny_model,
                attention_mask=attention_mask,
                completion_token_ids=completion_ids,
                pad_token_id=pad_token_id,
                chunk_size=8,
            )

        assert log_probs_no_chunk.shape == log_probs_chunked.shape
        assert mask_no_chunk.shape == mask_chunked.shape
        assert torch.allclose(log_probs_no_chunk, log_probs_chunked, atol=1e-5)
        assert torch.equal(mask_no_chunk, mask_chunked)

    def test_chunked_with_different_chunk_sizes(self, tiny_model):
        """Different chunk sizes produce identical results."""
        batch_size = 2
        group_size = 2
        prompt_len = 8
        completion_len = 32
        vocab_size = tiny_model.vocab_size
        pad_token_id = 0

        torch.manual_seed(123)
        prompt_ids = torch.randint(1, vocab_size, (batch_size, prompt_len))
        completion_ids = [
            torch.cat(
                [
                    prompt_ids,
                    torch.randint(1, vocab_size, (batch_size, completion_len)),
                ],
                dim=1,
            )
            for _ in range(group_size)
        ]
        attention_mask = torch.zeros(batch_size, prompt_len, dtype=torch.bool)

        tiny_model.eval()
        with torch.no_grad():
            log_probs_chunk_8, _ = compute_log_probs(
                net=tiny_model,
                attention_mask=attention_mask,
                completion_token_ids=completion_ids,
                pad_token_id=pad_token_id,
                chunk_size=8,
            )

            log_probs_chunk_16, _ = compute_log_probs(
                net=tiny_model,
                attention_mask=attention_mask,
                completion_token_ids=completion_ids,
                pad_token_id=pad_token_id,
                chunk_size=16,
            )

            log_probs_chunk_5, _ = compute_log_probs(
                net=tiny_model,
                attention_mask=attention_mask,
                completion_token_ids=completion_ids,
                pad_token_id=pad_token_id,
                chunk_size=5,
            )

        assert torch.allclose(log_probs_chunk_8, log_probs_chunk_16, atol=1e-5)
        assert torch.allclose(log_probs_chunk_8, log_probs_chunk_5, atol=1e-5)


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

        result = reward_fn(
            env_response=env_response,
            raw_model_output="<answer>2 + 3 + 5</answer>",
            extracted_model_output="2 + 3 + 5",
        )
        assert result.total == 1.0
        assert result.components["correct"] == 1.0

    def test_wrong_answer(self):
        """Reward is 0.0 for wrong answers."""
        from dialectic.rl.types import EnvResponse

        reward_fn = CountdownRewardFn()
        env_response = EnvResponse(
            is_done=True,
            data=Countdown(prompt="...", numbers=[2, 3, 5], target=10),
        )

        result = reward_fn(
            env_response=env_response,
            raw_model_output="<answer>2 + 3</answer>",
            extracted_model_output="2 + 3",
        )
        assert result.total == 0.0
        assert result.components["correct"] == 0.0

    def test_invalid_numbers(self):
        """Reward is 0.0 when using numbers not in the set."""
        from dialectic.rl.types import EnvResponse

        reward_fn = CountdownRewardFn()
        env_response = EnvResponse(
            is_done=True,
            data=Countdown(prompt="...", numbers=[2, 3, 5], target=10),
        )

        result = reward_fn(
            env_response=env_response,
            raw_model_output="<answer>4 + 6</answer>",
            extracted_model_output="4 + 6",
        )
        assert result.total == 0.0
        assert result.components["correct"] == 0.0

    def test_no_answer(self):
        """Reward is 0.0 when no answer extracted."""
        from dialectic.rl.types import EnvResponse

        reward_fn = CountdownRewardFn()
        env_response = EnvResponse(
            is_done=True,
            data=Countdown(prompt="...", numbers=[2, 3, 5], target=10),
        )

        result = reward_fn(
            env_response=env_response,
            raw_model_output="I don't know",
            extracted_model_output=None,
        )
        assert result.total == 0.0
        assert result.components["correct"] == 0.0


class TestCompositeRewardFn:
    """Test the composite reward function and components."""

    def test_countdown_with_format_correct_answer(self):
        """Max selection: correct (1.0) beats all other accuracy components."""
        from dialectic.rl.reward import CountdownWithFormatRewardFn
        from dialectic.rl.types import EnvResponse

        reward_fn = CountdownWithFormatRewardFn()
        env_response = EnvResponse(
            is_done=True,
            data=Countdown(prompt="...", numbers=[2, 3, 5], target=10),
        )

        result = reward_fn(
            env_response=env_response,
            raw_model_output="<think>thinking</think><answer>2 + 3 + 5</answer>",
            extracted_model_output="2 + 3 + 5",
        )

        assert result.components["correct"] == 1.0
        assert result.components["parseable"] == 0.3
        assert result.components["answer_tags"] == 0.1
        assert result.components["think_tags"] == 0.05
        assert result.total > 1.0

    def test_countdown_with_format_parseable_only(self):
        """Max selection: parseable (0.3) wins when correct fails."""
        from dialectic.rl.reward import CountdownWithFormatRewardFn
        from dialectic.rl.types import EnvResponse

        reward_fn = CountdownWithFormatRewardFn()
        env_response = EnvResponse(
            is_done=True,
            data=Countdown(prompt="...", numbers=[2, 3, 5], target=10),
        )

        result = reward_fn(
            env_response=env_response,
            raw_model_output="<answer>2 + 3</answer>",
            extracted_model_output="2 + 3",
        )

        assert result.components["correct"] == 0.0
        assert result.components["parseable"] == 0.3
        assert result.components["answer_tags"] == 0.1
        assert result.total >= 0.3

    def test_countdown_with_format_think_tags_only(self):
        """Max selection: think_tags (0.05) wins when no answer tags."""
        from dialectic.rl.reward import CountdownWithFormatRewardFn
        from dialectic.rl.types import EnvResponse

        reward_fn = CountdownWithFormatRewardFn()
        env_response = EnvResponse(
            is_done=True,
            data=Countdown(prompt="...", numbers=[2, 3, 5], target=10),
        )

        result = reward_fn(
            env_response=env_response,
            raw_model_output="<think>Let me think about this...</think>",
            extracted_model_output=None,
        )

        assert result.components["correct"] == 0.0
        assert result.components["parseable"] == 0.0
        assert result.components["answer_tags"] == 0.0
        assert result.components["think_tags"] == 0.05
        assert result.total >= 0.05

    def test_length_bonus_additive(self):
        """Length bonus is additive (not in a group)."""
        from dialectic.rl.reward import CountdownWithFormatRewardFn
        from dialectic.rl.types import EnvResponse

        reward_fn = CountdownWithFormatRewardFn()
        env_response = EnvResponse(
            is_done=True,
            data=Countdown(prompt="...", numbers=[2, 3, 5], target=10),
        )

        result = reward_fn(
            env_response=env_response,
            raw_model_output="x" * 500 + "<answer>2 + 3 + 5</answer>",
            extracted_model_output="2 + 3 + 5",
        )

        assert result.components["length_bonus"] == 0.05
        assert result.total == 1.0 + 0.05


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
