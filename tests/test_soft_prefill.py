import torch

from dialectic.llm.generate import generate_with_soft_prefill
from dialectic.rl.env import MathEnv, MathState
from dialectic.rl.evaluate import EvaluationResult, evaluate_soft_prefill
from dialectic.rl.train import (
    compute_soft_prefill_log_probs,
    create_sft_val_fn,
    train_internal_reasoning_single_step_sft,
)
from dialectic.training import run_validation

PAD_TOKEN_ID = 151643
EOS_TOKEN_ID = 151645
SOFT_BLOCK_SIZE = 2


def math_state_to_str(data: MathState) -> str:
    return data.prompt


math_env_kwargs = {
    "direct_arithmetic_prob": 0.20,
    "twostep_arithmetic_prob": 0.30,
    "word_problem_prob": 0.35,
    "number_properties_prob": 0.15,
    "difficulty": "easy",
}


def _get_math_env(seed: int = 42):
    return MathEnv(
        direct_arithmetic_prob=0.2,
        twostep_arithmetic_prob=0.3,
        word_problem_prob=0.35,
        number_properties_prob=0.15,
        difficulty="easy",
        seed=seed,
    )


class TestMathEnvReset:
    def test_returns_valid_state(self):
        env = _get_math_env(seed=42)
        response = env.reset()
        assert response.is_done
        assert isinstance(response.data, MathState)
        assert len(response.data.prompt) > 0
        assert len(response.data.answer) > 0
        assert len(response.data.problem_type) > 0

    def test_answer_is_integer_string(self):
        env = _get_math_env(seed=42)
        for _ in range(20):
            response = env.reset()
            int(response.data.answer)

    def test_reseed_produces_same_sequence(self):
        env = _get_math_env(seed=42)
        first = [env.reset().data.prompt for _ in range(5)]
        env.reseed()
        second = [env.reset().data.prompt for _ in range(5)]
        assert first == second

    def test_different_seeds_differ(self):
        env1 = _get_math_env(seed=42)
        env2 = _get_math_env(seed=99)
        r1 = env1.reset().data.prompt
        r2 = env2.reset().data.prompt
        assert r1 != r2

    def test_str_repr(self):
        env = _get_math_env(seed=42)
        assert str(env) == "math_easy"


class TestComputeSoftPrefillLogProbsShapes:
    def test_output_shapes(self, tiny_model):
        torch.manual_seed(42)
        tiny_model.train()

        B, L, A = 2, 6, 3
        prompt_ids = torch.randint(0, 100, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.bool)
        answer_ids = torch.randint(0, 100, (B, A))
        answer_lengths = torch.tensor([3, 2])

        log_probs, mask = compute_soft_prefill_log_probs(
            net=tiny_model,
            prompt_token_ids=prompt_ids,
            attention_mask=attention_mask,
            answer_token_ids=answer_ids,
            answer_lengths=answer_lengths,
            soft_block_size=SOFT_BLOCK_SIZE,
            pad_token_id=PAD_TOKEN_ID,
            use_bf16=False,
        )

        assert log_probs.shape == (B, A)
        assert mask.shape == (B, A)
        assert mask[0].sum() == 3
        assert mask[1].sum() == 2
        assert (log_probs[mask] <= 0).all()

    def test_single_token_answer(self, tiny_model):
        torch.manual_seed(42)
        tiny_model.train()

        B, L, A = 2, 6, 1
        prompt_ids = torch.randint(0, 100, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.bool)
        answer_ids = torch.randint(0, 100, (B, A))
        answer_lengths = torch.tensor([1, 1])

        log_probs, mask = compute_soft_prefill_log_probs(
            net=tiny_model,
            prompt_token_ids=prompt_ids,
            attention_mask=attention_mask,
            answer_token_ids=answer_ids,
            answer_lengths=answer_lengths,
            soft_block_size=SOFT_BLOCK_SIZE,
            pad_token_id=PAD_TOKEN_ID,
            use_bf16=False,
        )

        assert log_probs.shape == (B, A)
        assert mask.all()

    def test_left_padded_prompt(self, tiny_model):
        torch.manual_seed(42)
        tiny_model.train()

        B, L, A = 2, 8, 3
        prompt_ids = torch.full((B, L), PAD_TOKEN_ID, dtype=torch.long)
        attention_mask = torch.zeros(B, L, dtype=torch.bool)
        prompt_ids[0, 2:] = torch.randint(0, 100, (L - 2,))
        attention_mask[0, 2:] = True
        prompt_ids[1, 4:] = torch.randint(0, 100, (L - 4,))
        attention_mask[1, 4:] = True

        answer_ids = torch.randint(0, 100, (B, A))
        answer_lengths = torch.tensor([3, 2])

        log_probs, mask = compute_soft_prefill_log_probs(
            net=tiny_model,
            prompt_token_ids=prompt_ids,
            attention_mask=attention_mask,
            answer_token_ids=answer_ids,
            answer_lengths=answer_lengths,
            soft_block_size=SOFT_BLOCK_SIZE,
            pad_token_id=PAD_TOKEN_ID,
            use_bf16=False,
        )

        assert log_probs.shape == (B, A)
        assert torch.isfinite(log_probs).all()


class TestSoftPrefillGradFlows:
    def test_grad_flows_to_model_weights(self, tiny_model):
        torch.manual_seed(42)
        tiny_model.train()

        B, L, A = 1, 4, 2
        prompt_ids = torch.randint(0, 100, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.bool)
        answer_ids = torch.randint(0, 100, (B, A))
        answer_lengths = torch.tensor([2])

        log_probs, mask = compute_soft_prefill_log_probs(
            net=tiny_model,
            prompt_token_ids=prompt_ids,
            attention_mask=attention_mask,
            answer_token_ids=answer_ids,
            answer_lengths=answer_lengths,
            soft_block_size=SOFT_BLOCK_SIZE,
            pad_token_id=PAD_TOKEN_ID,
            use_bf16=False,
        )

        loss = -(log_probs * mask).sum()
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in tiny_model.parameters()
        )
        assert has_grad

    def test_grad_flows_to_soft_projection(self, tiny_model_with_soft_projection):
        torch.manual_seed(42)
        model = tiny_model_with_soft_projection
        with torch.no_grad():
            model.soft_projection_alpha.fill_(0.01)
        model.train()

        B, L, A = 1, 4, 2
        prompt_ids = torch.randint(0, 100, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.bool)
        answer_ids = torch.randint(0, 100, (B, A))
        answer_lengths = torch.tensor([2])

        log_probs, mask = compute_soft_prefill_log_probs(
            net=model,
            prompt_token_ids=prompt_ids,
            attention_mask=attention_mask,
            answer_token_ids=answer_ids,
            answer_lengths=answer_lengths,
            soft_block_size=SOFT_BLOCK_SIZE,
            pad_token_id=PAD_TOKEN_ID,
            use_bf16=False,
        )

        loss = -(log_probs * mask).sum()
        loss.backward()

        assert model.soft_projection_alpha.grad is not None
        assert model.soft_projection_alpha.grad.abs().item() > 0

    def test_bptt_window_reduces_grad(self, tiny_model):
        torch.manual_seed(42)
        tiny_model.train()

        soft_block_size = 4
        B, L, A = 1, 4, 2
        prompt_ids = torch.randint(0, 100, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.bool)
        answer_ids = torch.randint(0, 100, (B, A))
        answer_lengths = torch.tensor([2])

        tiny_model.zero_grad()
        lp_full, mask = compute_soft_prefill_log_probs(
            net=tiny_model,
            prompt_token_ids=prompt_ids,
            attention_mask=attention_mask,
            answer_token_ids=answer_ids,
            answer_lengths=answer_lengths,
            soft_block_size=soft_block_size,
            pad_token_id=PAD_TOKEN_ID,
            use_bf16=False,
            soft_bptt_window=soft_block_size,
        )
        (-(lp_full * mask).sum()).backward()
        grad_norm_full = (
            sum(
                p.grad.norm().item() ** 2
                for p in tiny_model.parameters()
                if p.grad is not None
            )
            ** 0.5
        )

        tiny_model.zero_grad()
        lp_window, _ = compute_soft_prefill_log_probs(
            net=tiny_model,
            prompt_token_ids=prompt_ids,
            attention_mask=attention_mask,
            answer_token_ids=answer_ids,
            answer_lengths=answer_lengths,
            soft_block_size=soft_block_size,
            pad_token_id=PAD_TOKEN_ID,
            use_bf16=False,
            soft_bptt_window=1,
        )
        (-(lp_window * mask).sum()).backward()
        grad_norm_window = (
            sum(
                p.grad.norm().item() ** 2
                for p in tiny_model.parameters()
                if p.grad is not None
            )
            ** 0.5
        )

        torch.testing.assert_close(lp_full, lp_window, atol=1e-5, rtol=1e-5)
        assert grad_norm_window <= grad_norm_full * 1.01


class TestGenerateWithSoftPrefill:
    def test_output_shape(self, tiny_model):
        torch.manual_seed(42)
        tiny_model.eval()

        B, L = 2, 8
        token_ids = torch.randint(0, 100, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.bool)
        max_new = 10

        out = generate_with_soft_prefill(
            net=tiny_model,
            token_ids=token_ids,
            attention_mask=attention_mask,
            soft_block_size=SOFT_BLOCK_SIZE,
            max_new_tokens=max_new,
            eos_token_id=EOS_TOKEN_ID,
            pad_token_id=PAD_TOKEN_ID,
            temperature=0.0,
        )

        assert out.token_ids.shape == (B, max_new)
        assert out.lengths.shape == (B,)
        assert (out.lengths <= max_new).all()
        assert (out.lengths >= 1).all()

    def test_greedy_deterministic(self, tiny_model):
        torch.manual_seed(42)
        tiny_model.eval()

        B, L = 2, 8
        token_ids = torch.randint(0, 100, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.bool)

        out1 = generate_with_soft_prefill(
            net=tiny_model,
            token_ids=token_ids,
            attention_mask=attention_mask,
            soft_block_size=SOFT_BLOCK_SIZE,
            max_new_tokens=5,
            eos_token_id=EOS_TOKEN_ID,
            pad_token_id=PAD_TOKEN_ID,
            temperature=0.0,
        )
        out2 = generate_with_soft_prefill(
            net=tiny_model,
            token_ids=token_ids,
            attention_mask=attention_mask,
            soft_block_size=SOFT_BLOCK_SIZE,
            max_new_tokens=5,
            eos_token_id=EOS_TOKEN_ID,
            pad_token_id=PAD_TOKEN_ID,
            temperature=0.0,
        )

        assert (out1.token_ids == out2.token_ids).all()
        assert (out1.lengths == out2.lengths).all()


class TestTrainMathSft:
    def test_step_runs_and_changes_params(self, tiny_model, tokenizer):
        torch.manual_seed(42)

        env = _get_math_env()
        opt = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)

        params_before = {
            name: param.clone() for name, param in tiny_model.named_parameters()
        }

        train_internal_reasoning_single_step_sft(
            net=tiny_model,
            opt=opt,
            env=env,
            state_to_str=math_state_to_str,
            tokenizer=tokenizer,
            pad_token_id=PAD_TOKEN_ID,
            eos_token_id=EOS_TOKEN_ID,
            soft_block_size=2,
            soft_bptt_window=2,
            max_answer_tokens=8,
            max_episodes=4,
            batch_size=2,
            accumulation_steps=1,
            max_grad_norm=1.0,
            normalize_by_sequence_length=True,
            use_bf16=False,
            val_episodes=0,
            val_envs=[],
            val_batch_size=0,
        )

        params_changed = False
        for name, param in tiny_model.named_parameters():
            if not torch.allclose(params_before[name], param, atol=1e-8):
                params_changed = True
                break
        assert params_changed


class TestEvaluateSoftPrefill:
    def test_evaluate_soft_prefill_runs(self, tiny_model, tokenizer):
        torch.manual_seed(42)
        tiny_model.eval()

        env = _get_math_env()

        result, examples = evaluate_soft_prefill(
            net=tiny_model,
            env=env,
            state_to_str=math_state_to_str,
            answer_extractor=lambda data: data.answer,
            tokenizer=tokenizer,
            eos_token_id=EOS_TOKEN_ID,
            pad_token_id=PAD_TOKEN_ID,
            soft_block_size=SOFT_BLOCK_SIZE,
            max_new_tokens=8,
            max_episodes=4,
            batch_size=2,
        )

        assert isinstance(result, EvaluationResult)
        assert result.n_episodes == 4
        assert 0.0 <= result.reward_mean <= 1.0
        assert result.reward_std >= 0.0
        assert "correct" in result.component_means
        assert len(examples) == 4
        for ex in examples:
            assert len(ex.responses) == 1
            assert len(ex.rewards) == 1
            assert "correct" in ex.rewards[0]

    def test_run_validation_uses_soft_prefill(self, tiny_model, tokenizer):
        """Verify run_validation with soft_prefill val_fn uses evaluate_soft_prefill,
        producing only 'correct' reward component (not 'answer_tags')."""
        torch.manual_seed(42)

        val_envs = [_get_math_env(seed=2026)]
        val_envs = [
            MathEnv(
                direct_arithmetic_prob=1.0,
                twostep_arithmetic_prob=0.0,
                word_problem_prob=0.0,
                number_properties_prob=0.0,
                difficulty="trivial",
                seed=2026,
            )
        ]

        val_fn = create_sft_val_fn(
            net=tiny_model,
            state_to_str=math_state_to_str,
            tokenizer=tokenizer,
            eos_token_id=EOS_TOKEN_ID,
            pad_token_id=PAD_TOKEN_ID,
            soft_block_size=SOFT_BLOCK_SIZE,
            max_answer_tokens=8,
            use_bf16=False,
            val_episodes=4,
            val_batch_size=2,
        )

        metrics = run_validation(val_envs=val_envs, val_fn=val_fn)

        assert "val/math_trivial/reward_mean" in metrics
        assert "val/math_trivial/reward/correct" in metrics
        assert "answer_tags" not in str(metrics)
