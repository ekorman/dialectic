import pytest
import torch

from dialectic.experiments.launchers import extract_separator_token_id
from dialectic.experiments.launchers.multi_step_hybrid_reasoning_sft import (
    train_hybrid_reasoning_sft_countdown,
)
from dialectic.experiments.params import (
    CountdownParams,
    HybridReasoningParams,
    MultiStepSFTParams,
    TrainParams,
)
from dialectic.llm.components.kv_cache import KVCache
from dialectic.llm.generate import generate_variable_length_internal_reasoning_tokens
from dialectic.llm.registry import MODEL_REGISTRY
from dialectic.rl.env import (
    Countdown,
    CountdownEnv,
    CountdownStep,
    build_countdown_equation,
)
from dialectic.rl.reward import countdown_hybrid_correct, weighted_reward
from dialectic.rl.train import (
    compute_variable_length_internal_reasoning_log_probs,
    make_variable_length_per_cycle_backward_callback,
    make_variable_length_sft_per_cycle_backward_callback,
    stack_and_pad_variable_length_internal_reasoning,
)
from dialectic.rl.types import EnvResponse

EOS_TOKEN_ID = 151645
PAD_TOKEN_ID = 151643
SOFT_BLOCK_SIZE = 2
MAX_CYCLES = 5
MAX_TOKENS_PER_CYCLE = 10


class TestCountdownEnvSolution:
    def test_solution_present(self):
        env = CountdownEnv(seed=42)
        resp = env.reset()
        assert resp.data.solution is not None
        assert len(resp.data.solution) > 0

    def test_solution_steps_are_arithmetically_correct(self):
        env = CountdownEnv(seed=42)
        for _ in range(20):
            resp = env.reset()
            for step in resp.data.solution:
                if step.op == "+":
                    assert step.left + step.right == step.result
                elif step.op == "-":
                    assert step.left - step.right == step.result
                elif step.op == "*":
                    assert step.left * step.right == step.result
                elif step.op == "/":
                    assert step.left // step.right == step.result
                    assert step.left % step.right == 0

    def test_final_result_equals_target(self):
        env = CountdownEnv(seed=42)
        for _ in range(20):
            resp = env.reset()
            assert resp.data.solution is not None
            assert len(resp.data.solution) > 0
            assert resp.data.solution[-1].result == resp.data.target

    def test_solution_length_matches_n_ops(self):
        env = CountdownEnv(n_ops=3, n_total=6, n_larges=2, seed=42)
        for _ in range(20):
            resp = env.reset()
            assert len(resp.data.solution) <= 3

    def test_seed_reproducibility_includes_solution(self):
        env1 = CountdownEnv(seed=42)
        env2 = CountdownEnv(seed=42)
        for _ in range(5):
            r1 = env1.reset()
            r2 = env2.reset()
            assert len(r1.data.solution) == len(r2.data.solution)
            for s1, s2 in zip(r1.data.solution, r2.data.solution):
                assert s1.left == s2.left
                assert s1.op == s2.op
                assert s1.right == s2.right
                assert s1.result == s2.result

    def test_countdown_step_dataclass(self):
        step = CountdownStep(left=75, op="-", right=2, result=73)
        assert step.left == 75
        assert step.op == "-"
        assert step.right == 2
        assert step.result == 73

    def test_valid_ops_returns_4_tuples(self):
        ops = CountdownEnv._valid_ops(10, 3)
        for op_str, result, left, right in ops:
            assert isinstance(op_str, str)
            assert isinstance(result, int)
            assert isinstance(left, int)
            assert isinstance(right, int)

    def test_valid_ops_subtraction_ordering(self):
        ops = CountdownEnv._valid_ops(10, 3)
        sub_ops = [(o, r, l, rr) for o, r, l, rr in ops if o == "-"]
        assert len(sub_ops) == 1
        _, result, left, right = sub_ops[0]
        assert left == 10
        assert right == 3
        assert result == 7

    def test_valid_ops_division_ordering(self):
        ops = CountdownEnv._valid_ops(12, 4)
        div_ops = [(o, r, l, rr) for o, r, l, rr in ops if o == "/"]
        assert len(div_ops) == 1
        _, result, left, right = div_ops[0]
        assert left == 12
        assert right == 4
        assert result == 3


class TestBuildCountdownEquation:
    def test_single_step(self):
        steps = [CountdownStep(left=75, op="-", right=2, result=73)]
        eq = build_countdown_equation([75, 2], steps, 73)
        assert eq == "75 - 2 = 73"

    def test_two_steps(self):
        steps = [
            CountdownStep(left=75, op="-", right=2, result=73),
            CountdownStep(left=73, op="+", right=25, result=98),
        ]
        eq = build_countdown_equation([75, 2, 25], steps, 98)
        assert eq == "(75 - 2) + 25 = 98"

    def test_three_steps(self):
        steps = [
            CountdownStep(left=10, op="+", right=5, result=15),
            CountdownStep(left=15, op="*", right=3, result=45),
            CountdownStep(left=45, op="-", right=2, result=43),
        ]
        eq = build_countdown_equation([10, 5, 3, 2], steps, 43)
        assert eq == "((10 + 5) * 3) - 2 = 43"

    def test_bare_numbers_not_parenthesized(self):
        steps = [CountdownStep(left=4, op="*", right=5, result=20)]
        eq = build_countdown_equation([4, 5], steps, 20)
        assert eq == "4 * 5 = 20"

    def test_from_env(self):
        env = CountdownEnv(seed=42)
        for _ in range(20):
            resp = env.reset()
            assert resp.data.solution is not None
            eq = build_countdown_equation(
                resp.data.numbers, resp.data.solution, resp.data.target
            )
            assert eq.endswith(f"= {resp.data.target}")
            expr = eq.split("=")[0].strip()
            result = eval(expr, {"__builtins__": {}}, {})
            assert result == resp.data.target


class TestCountdownHybridCorrectReward:
    def _make_env_response(
        self, numbers: list[int], target: int
    ) -> EnvResponse[Countdown]:
        return EnvResponse(
            is_done=True,
            data=Countdown(prompt="", numbers=numbers, target=target),
        )

    def test_equation_correct(self):
        result = countdown_hybrid_correct(
            env_response=self._make_env_response([1, 2, 3], 6),
            raw_model_output="1 + 2 = 3 | 3 + 3 = 6 | (1 + 2) + 3 = 6",
        )
        assert result == 1.0

    def test_equation_wrong_target(self):
        result = countdown_hybrid_correct(
            env_response=self._make_env_response([1, 2, 3], 6),
            raw_model_output="1 + 2 = 3 | (1 + 2) * 3 = 5",
        )
        assert result == 0.0

    def test_wrong_arithmetic_right_target(self):
        result = countdown_hybrid_correct(
            env_response=self._make_env_response([75, 8, 2], 602),
            raw_model_output="8 * 2 = 602 | 75 = 8 | 2 |(8 * 2) * 602 = 602",
        )
        assert result == 0.0

    def test_wrong_numbers_used(self):
        result = countdown_hybrid_correct(
            env_response=self._make_env_response([10, 5], 15),
            raw_model_output="7 + 8 = 15",
        )
        assert result == 0.0

    def test_last_segment_after_pipe(self):
        result = countdown_hybrid_correct(
            env_response=self._make_env_response([10, 5], 15),
            raw_model_output="10 + 5 = 15",
        )
        assert result == 1.0

    def test_none_output(self):
        result = countdown_hybrid_correct(
            env_response=self._make_env_response([1], 1),
            raw_model_output=None,
        )
        assert result == 0.0

    def test_empty_output(self):
        result = countdown_hybrid_correct(
            env_response=self._make_env_response([1], 1),
            raw_model_output="",
        )
        assert result == 0.0

    def test_no_numbers_in_output(self):
        result = countdown_hybrid_correct(
            env_response=self._make_env_response([1], 1),
            raw_model_output="no numbers here",
        )
        assert result == 0.0

    def test_stated_result_doesnt_match_evaluation(self):
        """Expression evaluates to target but stated result is wrong."""
        result = countdown_hybrid_correct(
            env_response=self._make_env_response([100, 6, 1], 94),
            raw_model_output="100 - 6 = 94 | 1 * 94 = 91 * (100 - 6) = 994 | 1 * (100 - 6) = 994",
        )
        assert result == 0.0

    def test_correct_expr_wrong_stated_result(self):
        """Last segment: expr is correct but stated number after = is wrong."""
        result = countdown_hybrid_correct(
            env_response=self._make_env_response([10, 5], 15),
            raw_model_output="10 + 5 = 99",
        )
        assert result == 0.0

    def test_correct_with_matching_stated_result(self):
        result = countdown_hybrid_correct(
            env_response=self._make_env_response([10, 5], 15),
            raw_model_output="10 + 5 = 15",
        )
        assert result == 1.0

    def test_weighted_reward_integration(self):
        fn = weighted_reward([("correct", 1.0, countdown_hybrid_correct)])
        output = "75 - 2 = 73"
        result = fn(
            env_response=self._make_env_response([75, 2], 73),
            raw_model_output=output,
            extracted_model_output=output,
        )
        assert result.total == 1.0
        assert result.components["correct"] == 1.0


class TestGenerateVariableLengthInternalReasoningTokens:
    def test_output_shapes(self, tiny_model):
        torch.manual_seed(42)
        B, L = 2, 10
        token_ids = torch.randint(0, 100, (B, L))

        out = generate_variable_length_internal_reasoning_tokens(
            net=tiny_model,
            token_ids=token_ids,
            soft_block_size=SOFT_BLOCK_SIZE,
            max_cycles=MAX_CYCLES,
            max_tokens_per_cycle=MAX_TOKENS_PER_CYCLE,
            separator_token_id=999,
            done_token_id=EOS_TOKEN_ID,
            pad_token_id=PAD_TOKEN_ID,
            temperature=1.0,
        )

        assert out.hard_token_ids.shape == (B, MAX_CYCLES, MAX_TOKENS_PER_CYCLE)
        assert out.hard_token_lengths.shape == (B, MAX_CYCLES)
        assert out.n_cycles.shape == (B,)
        assert (out.n_cycles >= 1).all()
        assert (out.n_cycles <= MAX_CYCLES).all()

    def test_eos_terminates_generation(self, tiny_model):
        torch.manual_seed(42)
        B, L = 2, 8
        token_ids = torch.randint(0, 100, (B, L))

        out = generate_variable_length_internal_reasoning_tokens(
            net=tiny_model,
            token_ids=token_ids,
            soft_block_size=SOFT_BLOCK_SIZE,
            max_cycles=MAX_CYCLES,
            max_tokens_per_cycle=MAX_TOKENS_PER_CYCLE,
            separator_token_id=999,
            done_token_id=EOS_TOKEN_ID,
            pad_token_id=PAD_TOKEN_ID,
            temperature=1.0,
        )

        for b in range(B):
            nc = out.n_cycles[b].item()
            if nc < MAX_CYCLES:
                last_cycle = nc - 1
                tlen = out.hard_token_lengths[b, last_cycle].item()
                last_token = out.hard_token_ids[b, last_cycle, tlen - 1].item()
                assert last_token == EOS_TOKEN_ID

    def test_padded_positions_are_pad(self, tiny_model):
        torch.manual_seed(42)
        B, L = 2, 8
        token_ids = torch.randint(0, 100, (B, L))

        out = generate_variable_length_internal_reasoning_tokens(
            net=tiny_model,
            token_ids=token_ids,
            soft_block_size=SOFT_BLOCK_SIZE,
            max_cycles=MAX_CYCLES,
            max_tokens_per_cycle=MAX_TOKENS_PER_CYCLE,
            separator_token_id=999,
            done_token_id=EOS_TOKEN_ID,
            pad_token_id=PAD_TOKEN_ID,
            temperature=1.0,
        )

        for b in range(B):
            nc = out.n_cycles[b].item()
            for c in range(nc, MAX_CYCLES):
                assert (out.hard_token_ids[b, c] == PAD_TOKEN_ID).all()
            for c in range(nc):
                tlen = out.hard_token_lengths[b, c].item()
                if tlen < MAX_TOKENS_PER_CYCLE:
                    assert (out.hard_token_ids[b, c, tlen:] == PAD_TOKEN_ID).all()

    def test_hard_token_lengths_positive_for_active_cycles(self, tiny_model):
        torch.manual_seed(42)
        B, L = 3, 6
        token_ids = torch.randint(0, 100, (B, L))

        out = generate_variable_length_internal_reasoning_tokens(
            net=tiny_model,
            token_ids=token_ids,
            soft_block_size=SOFT_BLOCK_SIZE,
            max_cycles=MAX_CYCLES,
            max_tokens_per_cycle=MAX_TOKENS_PER_CYCLE,
            separator_token_id=999,
            done_token_id=EOS_TOKEN_ID,
            pad_token_id=PAD_TOKEN_ID,
            temperature=1.0,
        )

        for b in range(B):
            nc = out.n_cycles[b].item()
            for c in range(nc):
                assert out.hard_token_lengths[b, c].item() > 0


class TestComputeVariableLengthInternalReasoningLogProbs:
    def test_output_shape(self, tiny_model):
        torch.manual_seed(42)
        tiny_model.train()

        B, G, C, T_max = 2, 1, 3, 5
        L = 6
        prompt_ids = torch.randint(0, 100, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.bool)
        hard_ids = torch.randint(0, 100, (B, G, C, T_max))
        hard_lengths = torch.tensor([[[3, 4, 2]], [[5, 3, 0]]])
        n_cycles = torch.tensor([[3], [2]])

        log_probs, mask = compute_variable_length_internal_reasoning_log_probs(
            net=tiny_model,
            prompt_token_ids=prompt_ids,
            attention_mask=attention_mask,
            hard_token_ids=hard_ids,
            hard_token_lengths=hard_lengths,
            n_cycles=n_cycles,
            soft_block_size=SOFT_BLOCK_SIZE,
        )

        assert log_probs.shape == (B, G, C)
        assert mask.shape == (B, G, C)

    def test_completion_mask_matches_n_cycles(self, tiny_model):
        torch.manual_seed(42)
        tiny_model.train()

        B, G, C, T_max = 2, 1, 4, 5
        L = 6
        prompt_ids = torch.randint(0, 100, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.bool)
        hard_ids = torch.randint(0, 100, (B, G, C, T_max))
        hard_lengths = torch.tensor([[[3, 4, 2, 1]], [[5, 3, 0, 0]]])
        n_cycles = torch.tensor([[4], [2]])

        _, mask = compute_variable_length_internal_reasoning_log_probs(
            net=tiny_model,
            prompt_token_ids=prompt_ids,
            attention_mask=attention_mask,
            hard_token_ids=hard_ids,
            hard_token_lengths=hard_lengths,
            n_cycles=n_cycles,
            soft_block_size=SOFT_BLOCK_SIZE,
        )

        for b in range(B):
            nc = n_cycles[b, 0].item()
            assert mask[b, 0, :nc].all()
            if nc < C:
                assert not mask[b, 0, nc:].any()

    def test_log_probs_are_negative(self, tiny_model):
        torch.manual_seed(42)
        tiny_model.train()

        B, G, C, T_max = 2, 1, 3, 4
        L = 6
        prompt_ids = torch.randint(0, 100, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.bool)
        hard_ids = torch.randint(0, 100, (B, G, C, T_max))
        hard_lengths = torch.tensor([[[3, 2, 4]], [[4, 3, 2]]])
        n_cycles = torch.tensor([[3], [3]])

        log_probs, mask = compute_variable_length_internal_reasoning_log_probs(
            net=tiny_model,
            prompt_token_ids=prompt_ids,
            attention_mask=attention_mask,
            hard_token_ids=hard_ids,
            hard_token_lengths=hard_lengths,
            n_cycles=n_cycles,
            soft_block_size=SOFT_BLOCK_SIZE,
        )

        assert (log_probs[mask] <= 0).all()

    def test_grad_flows_through_soft_block(self, tiny_model):
        torch.manual_seed(42)
        tiny_model.train()

        B, G, C, T_max = 1, 1, 2, 3
        L = 4
        prompt_ids = torch.randint(0, 100, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.bool)
        hard_ids = torch.randint(0, 100, (B, G, C, T_max))
        hard_lengths = torch.tensor([[[3, 2]]])
        n_cycles = torch.tensor([[2]])

        log_probs, mask = compute_variable_length_internal_reasoning_log_probs(
            net=tiny_model,
            prompt_token_ids=prompt_ids,
            attention_mask=attention_mask,
            hard_token_ids=hard_ids,
            hard_token_lengths=hard_lengths,
            n_cycles=n_cycles,
            soft_block_size=SOFT_BLOCK_SIZE,
        )

        loss = (log_probs * mask).sum()
        loss.backward()

        has_grad = False
        for _, p in tiny_model.named_parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad

    def test_per_cycle_backward_frees_memory(self, tiny_model):
        torch.manual_seed(42)
        tiny_model.train()

        B, G, C, T_max = 2, 1, 3, 4
        L = 6
        prompt_ids = torch.randint(0, 100, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.bool)
        hard_ids = torch.randint(0, 100, (B, G, C, T_max))
        hard_lengths = torch.tensor([[[3, 2, 4]], [[4, 3, 2]]])
        n_cycles = torch.tensor([[3], [3]])

        cycle_indices = torch.arange(C).unsqueeze(0).expand(B, C)
        completion_mask = cycle_indices < n_cycles[:, 0].unsqueeze(-1)

        tiny_model.zero_grad()
        callback = make_variable_length_sft_per_cycle_backward_callback(
            B=B,
            completion_mask=completion_mask,
            hard_token_lengths=hard_lengths[:, 0],
            n_cycles=n_cycles[:, 0],
            normalize_by_sequence_length=True,
            loss_scale=1.0,
        )

        log_probs, _ = compute_variable_length_internal_reasoning_log_probs(
            net=tiny_model,
            prompt_token_ids=prompt_ids,
            attention_mask=attention_mask,
            hard_token_ids=hard_ids,
            hard_token_lengths=hard_lengths,
            n_cycles=n_cycles,
            soft_block_size=SOFT_BLOCK_SIZE,
            cycle_callback=callback,
        )

        assert not log_probs.requires_grad

        has_grad = False
        for _, p in tiny_model.named_parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad

    def test_grad_magnitude_reasonable(self, tiny_model):
        torch.manual_seed(42)
        tiny_model.train()

        B, G, C, T_max = 1, 1, 2, 3
        L = 4
        prompt_ids = torch.randint(0, 100, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.bool)
        hard_ids = torch.randint(0, 100, (B, G, C, T_max))
        hard_lengths = torch.tensor([[[3, 2]]])
        n_cycles = torch.tensor([[2]])

        log_probs, mask = compute_variable_length_internal_reasoning_log_probs(
            net=tiny_model,
            prompt_token_ids=prompt_ids,
            attention_mask=attention_mask,
            hard_token_ids=hard_ids,
            hard_token_lengths=hard_lengths,
            n_cycles=n_cycles,
            soft_block_size=SOFT_BLOCK_SIZE,
        )

        loss = (log_probs * mask).sum()
        loss.backward()

        for name, p in tiny_model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Non-finite grad in {name}"
                assert not torch.isnan(p.grad).any(), f"NaN grad in {name}"

    def test_bptt_window(self, tiny_model):
        torch.manual_seed(42)
        tiny_model.train()

        B, G, C, T_max = 1, 1, 2, 3
        L = 4
        prompt_ids = torch.randint(0, 100, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.bool)
        hard_ids = torch.randint(0, 100, (B, G, C, T_max))
        hard_lengths = torch.tensor([[[3, 2]]])
        n_cycles = torch.tensor([[2]])

        log_probs, mask = compute_variable_length_internal_reasoning_log_probs(
            net=tiny_model,
            prompt_token_ids=prompt_ids,
            attention_mask=attention_mask,
            hard_token_ids=hard_ids,
            hard_token_lengths=hard_lengths,
            n_cycles=n_cycles,
            soft_block_size=SOFT_BLOCK_SIZE,
            soft_bptt_window=1,
        )

        loss = (log_probs * mask).sum()
        loss.backward()

        has_grad = False
        for _, p in tiny_model.named_parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad


class TestMakeVariableLengthSFTPerCycleBackwardCallback:
    def test_normalizes_by_total_tokens(self):
        B = 2
        completion_mask = torch.tensor([[True, True, True], [True, True, False]])
        hard_token_lengths = torch.tensor([[3, 4, 2], [5, 3, 0]])
        n_cycles = torch.tensor([3, 2])

        callback = make_variable_length_sft_per_cycle_backward_callback(
            B=B,
            completion_mask=completion_mask,
            hard_token_lengths=hard_token_lengths,
            n_cycles=n_cycles,
            normalize_by_sequence_length=True,
            loss_scale=1.0,
        )

        lp = torch.tensor([-1.0, -2.0], requires_grad=True)
        result = callback(lp, 0)
        assert not result.requires_grad

    def test_no_normalization(self):
        B = 2
        completion_mask = torch.tensor([[True, True, True], [True, True, False]])
        hard_token_lengths = torch.tensor([[3, 4, 2], [5, 3, 0]])
        n_cycles = torch.tensor([3, 2])

        callback = make_variable_length_sft_per_cycle_backward_callback(
            B=B,
            completion_mask=completion_mask,
            hard_token_lengths=hard_token_lengths,
            n_cycles=n_cycles,
            normalize_by_sequence_length=False,
            loss_scale=1.0,
        )

        lp = torch.tensor([-1.0, -2.0], requires_grad=True)
        result = callback(lp, 0)
        assert not result.requires_grad


class TestValidOps:
    def test_no_duplicate_division_when_equal(self):
        """When a == b, _valid_ops should not return two identical division entries."""
        ops = CountdownEnv._valid_ops(5, 5)
        div_ops = [o for o in ops if o[0] == "/"]
        assert len(div_ops) <= 1

    def test_division_only_valid_direction(self):
        """When a != b, only the direction that divides evenly should appear."""
        ops = CountdownEnv._valid_ops(6, 3)
        div_ops = [o for o in ops if o[0] == "/"]
        div_results = sorted([o[1] for o in div_ops])
        assert div_results == [2]

    def test_division_6_and_2(self):
        ops = CountdownEnv._valid_ops(6, 2)
        div_ops = [o for o in ops if o[0] == "/"]
        assert len(div_ops) == 1
        assert div_ops[0] == ("/", 3, 6, 2)


class TestBuildCountdownEquationDuplicateValues:
    def test_duplicate_numbers_in_pool(self):
        """When pool has duplicate values, build_countdown_equation should
        still produce a correct equation by matching the right operands."""
        steps = [
            CountdownStep(left=3, op="+", right=3, result=6),
        ]
        result = build_countdown_equation([3, 3], steps, 6)
        assert result == "3 + 3 = 6"

    def test_duplicate_intermediate_values(self):
        """If an intermediate result equals an original number, the equation
        should still be correct."""
        steps = [
            CountdownStep(left=5, op="-", right=3, result=2),
            CountdownStep(left=2, op="+", right=2, result=4),
        ]
        result = build_countdown_equation([5, 3, 2], steps, 4)
        assert "= 4" in result


class TestKVCacheSizingVariableLength:
    def test_kv_cache_accounts_for_logit_forward_pass(self, tiny_model, monkeypatch):
        """Each cycle uses soft_block_size + 1 + max_tokens_per_cycle KV slots.

        Verify at runtime by capturing the max_seq_len passed to KVCache.
        """

        captured_max_seq_lens: list[int] = []
        orig_init = KVCache.__init__

        def patched_init(self, *args, **kwargs):
            captured_max_seq_lens.append(kwargs.get("max_seq_len") or args[0])
            orig_init(self, *args, **kwargs)

        monkeypatch.setattr(KVCache, "__init__", patched_init)

        soft_block_size = 2
        max_cycles = 2
        max_tokens_per_cycle = 3
        B = 1
        device = next(tiny_model.parameters()).device

        prompt_ids = torch.randint(0, 100, (B, 4), device=device)
        attn_mask = torch.ones(B, 4, dtype=torch.bool, device=device)
        L = prompt_ids.shape[1]

        generate_variable_length_internal_reasoning_tokens(
            net=tiny_model,
            token_ids=prompt_ids,
            attention_mask=attn_mask,
            done_token_id=2,
            pad_token_id=0,
            soft_block_size=soft_block_size,
            max_cycles=max_cycles,
            max_tokens_per_cycle=max_tokens_per_cycle,
        )

        expected = L + max_cycles * (soft_block_size + 1 + max_tokens_per_cycle)
        assert any(s == expected for s in captured_max_seq_lens), (
            f"Expected KVCache max_seq_len={expected}, got {captured_max_seq_lens}"
        )


class TestValEnvsListLengthValidation:
    def test_mismatched_list_lengths_raises(self):
        """CountdownEnv must reject mismatched list lengths for config params."""
        with pytest.raises(ValueError):
            CountdownEnv(
                n_ops=[3, 4, 5],
                n_total=6,
                n_larges=2,
            )

    def test_launcher_val_envs_mixed_scalar_and_list(self):
        """Launcher must detect mismatched list lengths for countdown val envs.

        When n_ops is a list of 2 but n_total/n_larges are scalars (→ lists of 1),
        the launcher should raise ValueError, not silently produce wrong val envs.
        """

        with pytest.raises(ValueError, match="same length"):
            train_hybrid_reasoning_sft_countdown(
                train_params=TrainParams(
                    model_name="qwen3-0.6b",
                    lr=1e-4,
                    max_episodes=1,
                    batch_size=1,
                    accumulation_steps=1,
                    max_grad_norm=1.0,
                    max_tokens_generated=100,
                    compile_model=False,
                    use_bf16=False,
                    seed=42,
                    temperature=1.0,
                    val_batch_size=1,
                    val_episodes=1,
                    val_freq=1,
                ),
                multistep_sft_params=MultiStepSFTParams(
                    normalize_by_sequence_length=True
                ),
                hybrid_reasoning_params=HybridReasoningParams(
                    soft_block_size=2,
                    soft_bptt_window=2,
                    max_cycles=5,
                    max_tokens_per_cycle=5,
                ),
                countdown_params=CountdownParams(
                    n_ops=[3, 4],
                    n_total=6,
                    n_larges=2,
                ),
            )


class TestKVCacheSizingLogProbs:
    def test_log_prob_kv_cache_formula(self, tiny_model, monkeypatch):
        """Verify compute_variable_length_internal_reasoning_log_probs allocates
        KV cache with soft_block_size + T_max slots per cycle at runtime."""

        captured_max_seq_lens: list[int] = []
        orig_init = KVCache.__init__

        def patched_init(self, *args, **kwargs):
            captured_max_seq_lens.append(kwargs.get("max_seq_len") or args[0])
            orig_init(self, *args, **kwargs)

        monkeypatch.setattr(KVCache, "__init__", patched_init)

        B, G, C, T_max = 1, 1, 2, 3
        soft_block_size = 2
        device = next(tiny_model.parameters()).device
        prompt_ids = torch.randint(0, 100, (B, 4), device=device)
        attn_mask = torch.ones(B, 4, dtype=torch.bool, device=device)
        hard_ids = torch.randint(0, 100, (B, G, C, T_max), device=device)
        hard_lengths = torch.full((B, G, C), T_max, dtype=torch.long, device=device)
        n_cycles = torch.full((B, G), C, dtype=torch.long, device=device)

        with torch.no_grad():
            compute_variable_length_internal_reasoning_log_probs(
                net=tiny_model,
                prompt_token_ids=prompt_ids,
                attention_mask=attn_mask,
                hard_token_ids=hard_ids,
                hard_token_lengths=hard_lengths,
                n_cycles=n_cycles,
                soft_block_size=soft_block_size,
            )

        L = prompt_ids.shape[1]
        expected = L + C * (soft_block_size + T_max)
        assert any(s == expected for s in captured_max_seq_lens), (
            f"Expected KVCache max_seq_len={expected}, got {captured_max_seq_lens}"
        )


class TestSeparatorTokenExtraction:
    def test_separator_matches_training_format(self):
        """The separator token extracted by the launcher must match the last
        token of CountdownStep.format_step() when encoded."""

        model_info = MODEL_REGISTRY["qwen3-0.6b"]
        tokenizer = model_info.load_tokenizer()

        sep_id = extract_separator_token_id(tokenizer)

        step = CountdownStep(left=75, op="-", right=2, result=73)
        ids = tokenizer.encode(step.format_step(), add_special_tokens=False).ids
        assert ids[-1] == sep_id


class TestStackAndPadVariableLengthInternalReasoning:
    def test_output_shapes(self):
        B, G = 2, 3
        C_vals = [4, 3, 5]
        T_vals = [6, 8, 7]
        pad_id = 0

        hard_token_ids = [
            torch.randint(1, 100, (B, C_vals[g], T_vals[g])) for g in range(G)
        ]
        hard_token_lengths = [
            torch.randint(1, T_vals[g] + 1, (B, C_vals[g])) for g in range(G)
        ]
        n_cycles = [torch.tensor([C_vals[g], C_vals[g] - 1]) for g in range(G)]

        stacked_ids, stacked_lengths, stacked_n = (
            stack_and_pad_variable_length_internal_reasoning(
                hard_token_ids, hard_token_lengths, n_cycles, pad_id
            )
        )

        max_c = max(C_vals)
        max_t = max(T_vals)
        assert stacked_ids.shape == (B, G, max_c, max_t)
        assert stacked_lengths.shape == (B, G, max_c)
        assert stacked_n.shape == (B, G)

    def test_padding_fills_with_pad_token(self):
        B = 1
        pad_id = -1
        ids_0 = torch.ones(B, 2, 3, dtype=torch.long) * 10
        ids_1 = torch.ones(B, 3, 5, dtype=torch.long) * 20
        lengths_0 = torch.tensor([[2, 1]])
        lengths_1 = torch.tensor([[3, 4, 2]])
        n_0 = torch.tensor([2])
        n_1 = torch.tensor([3])

        stacked_ids, stacked_lengths, stacked_n = (
            stack_and_pad_variable_length_internal_reasoning(
                [ids_0, ids_1], [lengths_0, lengths_1], [n_0, n_1], pad_id
            )
        )

        assert stacked_ids[0, 0, :2, :3].eq(10).all()
        assert stacked_ids[0, 0, 2:, :].eq(pad_id).all()
        assert stacked_ids[0, 0, :, 3:].eq(pad_id).all()

        assert stacked_ids[0, 1, :3, :5].eq(20).all()

    def test_n_cycles_preserved(self):
        B, G = 2, 2
        pad_id = 0
        ids = [torch.randint(1, 50, (B, 3, 4)) for _ in range(G)]
        lengths = [torch.tensor([[3, 2, 1], [4, 3, 2]]) for _ in range(G)]
        nc = [torch.tensor([3, 2]), torch.tensor([1, 3])]

        _, _, stacked_n = stack_and_pad_variable_length_internal_reasoning(
            ids, lengths, nc, pad_id
        )

        assert stacked_n[0, 0] == 3
        assert stacked_n[0, 1] == 1
        assert stacked_n[1, 0] == 2
        assert stacked_n[1, 1] == 3


class TestMakeVariableLengthPerCycleBackwardCallback:
    def test_callback_detaches_output(self):
        B, G, C = 2, 2, 3
        advs = torch.randn(B, G, 1)
        completion_mask = torch.ones(B, G, C, dtype=torch.bool)
        hard_token_lengths = torch.tensor(
            [[[3, 4, 2], [5, 3, 1]], [[4, 2, 3], [6, 1, 2]]]
        )
        n_cycles = torch.tensor([[3, 3], [3, 3]])

        callback = make_variable_length_per_cycle_backward_callback(
            B=B,
            G=G,
            advs=advs,
            completion_mask=completion_mask,
            hard_token_lengths=hard_token_lengths,
            n_cycles=n_cycles,
            old_log_probs=None,
            ref_log_probs=None,
            beta=0.0,
            eps=None,
            normalize_by_sequence_length=True,
            loss_scale=1.0,
        )

        lp = torch.randn(B, G, requires_grad=True)
        result = callback(lp, 0)
        assert not result.requires_grad

    def test_normalize_by_sequence_length_values(self):
        B, G, C = 1, 1, 2
        advs = torch.ones(B, G, 1)
        completion_mask = torch.ones(B, G, C, dtype=torch.bool)
        hard_token_lengths = torch.tensor([[[5, 3]]])
        n_cycles = torch.tensor([[2]])

        callback_norm = make_variable_length_per_cycle_backward_callback(
            B=B,
            G=G,
            advs=advs,
            completion_mask=completion_mask,
            hard_token_lengths=hard_token_lengths,
            n_cycles=n_cycles,
            old_log_probs=None,
            ref_log_probs=None,
            beta=0.0,
            eps=None,
            normalize_by_sequence_length=True,
            loss_scale=1.0,
        )
        callback_no_norm = make_variable_length_per_cycle_backward_callback(
            B=B,
            G=G,
            advs=advs,
            completion_mask=completion_mask,
            hard_token_lengths=hard_token_lengths,
            n_cycles=n_cycles,
            old_log_probs=None,
            ref_log_probs=None,
            beta=0.0,
            eps=None,
            normalize_by_sequence_length=False,
            loss_scale=1.0,
        )

        lp_norm = torch.tensor([[-1.0]], requires_grad=True)
        lp_no_norm = torch.tensor([[-1.0]], requires_grad=True)

        callback_norm(lp_norm, 0)
        callback_no_norm(lp_no_norm, 0)

        grad_norm = lp_norm.grad.item()
        grad_no_norm = lp_no_norm.grad.item()
        assert abs(grad_norm) < abs(grad_no_norm)

    def test_clipped_objective_with_eps(self):
        B, G, C = 1, 1, 2
        advs = torch.ones(B, G, 1)
        completion_mask = torch.ones(B, G, C, dtype=torch.bool)
        hard_token_lengths = torch.tensor([[[3, 3]]])
        n_cycles = torch.tensor([[2]])
        old_log_probs = torch.tensor([[[-1.0, -1.0]]])

        callback = make_variable_length_per_cycle_backward_callback(
            B=B,
            G=G,
            advs=advs,
            completion_mask=completion_mask,
            hard_token_lengths=hard_token_lengths,
            n_cycles=n_cycles,
            old_log_probs=old_log_probs,
            ref_log_probs=None,
            beta=0.0,
            eps=0.2,
            normalize_by_sequence_length=False,
            loss_scale=1.0,
        )

        lp = torch.tensor([[-1.0]], requires_grad=True)
        result = callback(lp, 0)
        assert not result.requires_grad

    def test_masks_invalid_cycles(self):
        B, G = 1, 1
        advs = torch.ones(B, G, 1)
        completion_mask = torch.tensor([[[True, False, False]]])
        hard_token_lengths = torch.tensor([[[5, 0, 0]]])
        n_cycles = torch.tensor([[1]])

        callback = make_variable_length_per_cycle_backward_callback(
            B=B,
            G=G,
            advs=advs,
            completion_mask=completion_mask,
            hard_token_lengths=hard_token_lengths,
            n_cycles=n_cycles,
            old_log_probs=None,
            ref_log_probs=None,
            beta=0.0,
            eps=None,
            normalize_by_sequence_length=False,
            loss_scale=1.0,
        )

        lp = torch.tensor([[0.0]], requires_grad=True)
        callback(lp, 1)
        assert lp.grad.item() == 0.0


SEPARATOR_TOKEN_ID = 999
ARITHMETIC_TOKEN_IDS = list(range(15, 25)) + [7, 8, 9, 10, 12, 14, 28, 198, 220]
VALID_COUNTDOWN_TOKEN_IDS = ARITHMETIC_TOKEN_IDS + [SEPARATOR_TOKEN_ID, EOS_TOKEN_ID]


class TestConstrainedDecodingVariableLength:
    def test_all_hard_tokens_in_allowlist(self, tiny_model):
        torch.manual_seed(42)
        B, L = 3, 8
        token_ids = torch.randint(0, 100, (B, L))

        out = generate_variable_length_internal_reasoning_tokens(
            net=tiny_model,
            token_ids=token_ids,
            soft_block_size=SOFT_BLOCK_SIZE,
            max_cycles=MAX_CYCLES,
            max_tokens_per_cycle=MAX_TOKENS_PER_CYCLE,
            separator_token_id=SEPARATOR_TOKEN_ID,
            done_token_id=EOS_TOKEN_ID,
            pad_token_id=PAD_TOKEN_ID,
            temperature=1.0,
            valid_hard_token_ids=VALID_COUNTDOWN_TOKEN_IDS,
        )

        allowed = set(VALID_COUNTDOWN_TOKEN_IDS) | {PAD_TOKEN_ID}
        for b in range(B):
            nc = out.n_cycles[b].item()
            for c in range(nc):
                tlen = out.hard_token_lengths[b, c].item()
                for t in range(tlen):
                    tok = out.hard_token_ids[b, c, t].item()
                    assert tok in allowed, (
                        f"Token {tok} at b={b} c={c} t={t} not in allowlist"
                    )

    def test_unconstrained_produces_out_of_set_tokens(self, tiny_model):
        torch.manual_seed(42)
        B, L = 3, 8
        token_ids = torch.randint(0, 100, (B, L))

        out = generate_variable_length_internal_reasoning_tokens(
            net=tiny_model,
            token_ids=token_ids,
            soft_block_size=SOFT_BLOCK_SIZE,
            max_cycles=MAX_CYCLES,
            max_tokens_per_cycle=MAX_TOKENS_PER_CYCLE,
            separator_token_id=SEPARATOR_TOKEN_ID,
            done_token_id=EOS_TOKEN_ID,
            pad_token_id=PAD_TOKEN_ID,
            temperature=1.0,
        )

        allowed = set(VALID_COUNTDOWN_TOKEN_IDS) | {PAD_TOKEN_ID}
        all_tokens = set()
        for b in range(B):
            nc = out.n_cycles[b].item()
            for c in range(nc):
                tlen = out.hard_token_lengths[b, c].item()
                for t in range(tlen):
                    all_tokens.add(out.hard_token_ids[b, c, t].item())
        assert not all_tokens.issubset(allowed), (
            "Unconstrained generation should produce tokens outside the arithmetic set"
        )

    def test_constrained_output_shapes_unchanged(self, tiny_model):
        torch.manual_seed(42)
        B, L = 2, 10
        token_ids = torch.randint(0, 100, (B, L))

        out = generate_variable_length_internal_reasoning_tokens(
            net=tiny_model,
            token_ids=token_ids,
            soft_block_size=SOFT_BLOCK_SIZE,
            max_cycles=MAX_CYCLES,
            max_tokens_per_cycle=MAX_TOKENS_PER_CYCLE,
            separator_token_id=SEPARATOR_TOKEN_ID,
            done_token_id=EOS_TOKEN_ID,
            pad_token_id=PAD_TOKEN_ID,
            temperature=1.0,
            valid_hard_token_ids=VALID_COUNTDOWN_TOKEN_IDS,
        )

        assert out.hard_token_ids.shape == (B, MAX_CYCLES, MAX_TOKENS_PER_CYCLE)
        assert out.hard_token_lengths.shape == (B, MAX_CYCLES)
        assert out.n_cycles.shape == (B,)

    def test_constrained_log_probs_valid_mask(self, tiny_model):
        torch.manual_seed(42)
        tiny_model.train()

        B, G, C, T_max = 2, 1, 3, 4
        L = 6
        prompt_ids = torch.randint(0, 100, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.bool)
        hard_ids = torch.randint(0, len(ARITHMETIC_TOKEN_IDS), (B, G, C, T_max))
        for i in range(B):
            for g in range(G):
                for c in range(C):
                    for t in range(T_max):
                        hard_ids[i, g, c, t] = ARITHMETIC_TOKEN_IDS[
                            hard_ids[i, g, c, t].item()
                        ]
        hard_lengths = torch.tensor([[[3, 2, 4]], [[4, 3, 2]]])
        n_cycles = torch.tensor([[3], [3]])

        log_probs, mask = compute_variable_length_internal_reasoning_log_probs(
            net=tiny_model,
            prompt_token_ids=prompt_ids,
            attention_mask=attention_mask,
            hard_token_ids=hard_ids,
            hard_token_lengths=hard_lengths,
            n_cycles=n_cycles,
            soft_block_size=SOFT_BLOCK_SIZE,
            valid_hard_token_ids=VALID_COUNTDOWN_TOKEN_IDS,
        )

        assert log_probs.shape == (B, G, C)
        assert (log_probs[mask] <= 0).all()
        assert torch.isfinite(log_probs[mask]).all()

    def test_constrained_rollout_tokens_in_allowlist(self, tiny_model, tokenizer, env):
        from dialectic.rl.rollout import (
            generate_variable_length_internal_reasoning_rollout_batch,
        )

        model_info = MODEL_REGISTRY["qwen3-0.6b"]
        sep_id = extract_separator_token_id(tokenizer)

        from dialectic.experiments.launchers.hybrid_reasoning_grpo import (
            _get_valid_countdown_hard_token_ids,
        )

        valid_ids = _get_valid_countdown_hard_token_ids(
            tokenizer=tokenizer,
            separator_token_id=sep_id,
            eos_token_id=model_info.eos_token_id,
        )

        rollout = generate_variable_length_internal_reasoning_rollout_batch(
            net=tiny_model,
            env=env,
            reward_fn=weighted_reward([("correct", 1.0, countdown_hybrid_correct)]),
            state_to_str=lambda data: data.prompt,
            tokenizer=tokenizer,
            eos_token_id=model_info.eos_token_id,
            pad_token_id=model_info.pad_token_id,
            separator_token_id=sep_id,
            batch_size=2,
            temperature=1.0,
            soft_block_size=SOFT_BLOCK_SIZE,
            max_cycles=MAX_CYCLES,
            max_tokens_per_cycle=MAX_TOKENS_PER_CYCLE,
            valid_hard_token_ids=valid_ids,
        )

        allowed = set(valid_ids) | {model_info.pad_token_id}
        for b in range(2):
            nc = rollout.n_cycles[b].item()
            for c in range(nc):
                tlen = rollout.hard_token_lengths[b, c].item()
                for t in range(tlen):
                    tok = rollout.hard_token_ids[b, c, t].item()
                    assert tok in allowed


class TestGetValidCountdownHardTokenIds:
    def test_includes_digit_tokens(self, tokenizer):
        from dialectic.experiments.launchers.hybrid_reasoning_grpo import (
            _get_valid_countdown_hard_token_ids,
        )

        sep_id = extract_separator_token_id(tokenizer)
        valid_ids = _get_valid_countdown_hard_token_ids(
            tokenizer=tokenizer,
            separator_token_id=sep_id,
            eos_token_id=EOS_TOKEN_ID,
        )

        for digit in "0123456789":
            token_id = tokenizer.encode(digit, add_special_tokens=False).ids[0]
            assert token_id in valid_ids

    def test_includes_operator_tokens(self, tokenizer):
        from dialectic.experiments.launchers.hybrid_reasoning_grpo import (
            _get_valid_countdown_hard_token_ids,
        )

        sep_id = extract_separator_token_id(tokenizer)
        valid_ids = _get_valid_countdown_hard_token_ids(
            tokenizer=tokenizer,
            separator_token_id=sep_id,
            eos_token_id=EOS_TOKEN_ID,
        )

        for op in ["+", "-", "*", "/", "=", "(", ")"]:
            token_id = tokenizer.encode(op, add_special_tokens=False).ids[0]
            assert token_id in valid_ids

    def test_includes_separator_and_eos(self, tokenizer):
        from dialectic.experiments.launchers.hybrid_reasoning_grpo import (
            _get_valid_countdown_hard_token_ids,
        )

        sep_id = extract_separator_token_id(tokenizer)
        valid_ids = _get_valid_countdown_hard_token_ids(
            tokenizer=tokenizer,
            separator_token_id=sep_id,
            eos_token_id=EOS_TOKEN_ID,
        )

        assert sep_id in valid_ids
        assert EOS_TOKEN_ID in valid_ids

    def test_excludes_alpha_tokens(self, tokenizer):
        from dialectic.experiments.launchers.hybrid_reasoning_grpo import (
            _get_valid_countdown_hard_token_ids,
        )

        sep_id = extract_separator_token_id(tokenizer)
        valid_ids = _get_valid_countdown_hard_token_ids(
            tokenizer=tokenizer,
            separator_token_id=sep_id,
            eos_token_id=EOS_TOKEN_ID,
        )
        valid_set = set(valid_ids)

        for word in ["hello", "the", "print"]:
            token_id = tokenizer.encode(word, add_special_tokens=False).ids[0]
            assert token_id not in valid_set

    def test_is_sorted(self, tokenizer):
        from dialectic.experiments.launchers.hybrid_reasoning_grpo import (
            _get_valid_countdown_hard_token_ids,
        )

        sep_id = extract_separator_token_id(tokenizer)
        valid_ids = _get_valid_countdown_hard_token_ids(
            tokenizer=tokenizer,
            separator_token_id=sep_id,
            eos_token_id=EOS_TOKEN_ID,
        )

        assert valid_ids == sorted(valid_ids)

    def test_vocab_reduction(self, tokenizer):
        from dialectic.experiments.launchers.hybrid_reasoning_grpo import (
            _get_valid_countdown_hard_token_ids,
        )

        sep_id = extract_separator_token_id(tokenizer)
        valid_ids = _get_valid_countdown_hard_token_ids(
            tokenizer=tokenizer,
            separator_token_id=sep_id,
            eos_token_id=EOS_TOKEN_ID,
        )

        full_vocab_size = len(tokenizer.get_vocab())
        assert len(valid_ids) < full_vocab_size / 10


class TestGroupedVariableLengthRolloutBatch:
    def test_rollout_shapes_and_grouping(self, tiny_model, tokenizer, env):
        from dialectic.rl.rollout import (
            generate_grouped_variable_length_internal_reasoning_rollout_batch,
        )

        B, G = 2, 3
        max_cycles = 4
        max_tokens_per_cycle = 8
        soft_block_size = 2

        model_info = MODEL_REGISTRY["qwen3-0.6b"]
        sep_id = extract_separator_token_id(tokenizer)

        def state_to_str(data):
            return data.prompt

        reward_fn = weighted_reward([("correct", 1.0, countdown_hybrid_correct)])

        rollout = generate_grouped_variable_length_internal_reasoning_rollout_batch(
            net=tiny_model,
            env=env,
            reward_fn=reward_fn,
            state_to_str=state_to_str,
            tokenizer=tokenizer,
            eos_token_id=model_info.eos_token_id,
            pad_token_id=model_info.pad_token_id,
            separator_token_id=sep_id,
            batch_size=B,
            group_size=G,
            temperature=1.0,
            soft_block_size=soft_block_size,
            max_cycles=max_cycles,
            max_tokens_per_cycle=max_tokens_per_cycle,
        )

        assert len(rollout.hard_token_ids) == G
        assert len(rollout.hard_token_lengths) == G
        assert len(rollout.n_cycles) == G
        assert len(rollout.output_strs) == G
        assert len(rollout.reward_results) == G

        for g in range(G):
            assert rollout.hard_token_ids[g].shape[0] == B
            assert rollout.hard_token_ids[g].shape[1] == max_cycles
            assert rollout.hard_token_ids[g].shape[2] == max_tokens_per_cycle
            assert rollout.hard_token_lengths[g].shape == (B, max_cycles)
            assert rollout.n_cycles[g].shape == (B,)
            assert len(rollout.output_strs[g]) == B
            assert len(rollout.reward_results[g]) == B

        assert rollout.rewards.shape == (G, B)
        assert rollout.prompt_token_ids.shape[0] == B
        assert rollout.attention_mask.shape[0] == B
        assert len(rollout.env_responses) == B
        assert len(rollout.prompts) == B
