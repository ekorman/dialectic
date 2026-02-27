import torch

from dialectic.llm.components import GradSafeKVCache
from dialectic.llm.generate import generate_internal_reasoning_tokens
from dialectic.rl.env import MazeEnv, MazeState
from dialectic.rl.maze import MazeConfig
from dialectic.rl.reward import maze_correct, maze_distance, weighted_reward
from dialectic.rl.train import (
    compute_grpo_loss,
    compute_internal_reasoning_log_probs,
    grpo_advantage,
    make_per_cycle_backward_callback,
    make_sft_per_cycle_backward_callback,
    stack_and_pad_internal_reasoning,
    train_internal_reasoning_grpo,
    train_internal_reasoning_sft,
)

DIRECTION_TOKENS = [100, 200, 300, 400]
EOS_TOKEN_ID = 151645
PAD_TOKEN_ID = 151643
VALID_HARD_TOKEN_IDS = DIRECTION_TOKENS + [EOS_TOKEN_ID]
SOFT_BLOCK_SIZE = 2
MAX_CYCLES = 5


def maze_state_to_str(data: MazeState) -> str:
    return data.prompt


class TestGenerateInternalReasoningTokens:
    def test_output_shape(self, tiny_model):
        torch.manual_seed(42)
        B, L = 2, 10
        token_ids = torch.randint(0, 100, (B, L))

        out = generate_internal_reasoning_tokens(
            net=tiny_model,
            token_ids=token_ids,
            soft_block_size=SOFT_BLOCK_SIZE,
            max_cycles=MAX_CYCLES,
            valid_hard_token_ids=VALID_HARD_TOKEN_IDS,
            done_token_id=EOS_TOKEN_ID,
            pad_token_id=PAD_TOKEN_ID,
            temperature=1.0,
        )

        assert out.hard_token_ids.shape == (B, MAX_CYCLES)
        assert out.hard_log_probs.shape == (B, MAX_CYCLES)
        assert out.n_cycles.shape == (B,)
        assert (out.n_cycles <= MAX_CYCLES).all()
        assert (out.n_cycles >= 1).all()

    def test_valid_tokens_only(self, tiny_model):
        torch.manual_seed(42)
        B, L = 3, 8
        token_ids = torch.randint(0, 100, (B, L))

        out = generate_internal_reasoning_tokens(
            net=tiny_model,
            token_ids=token_ids,
            soft_block_size=SOFT_BLOCK_SIZE,
            max_cycles=MAX_CYCLES,
            valid_hard_token_ids=VALID_HARD_TOKEN_IDS,
            done_token_id=EOS_TOKEN_ID,
            pad_token_id=PAD_TOKEN_ID,
            temperature=1.0,
        )

        valid_set = set(VALID_HARD_TOKEN_IDS + [PAD_TOKEN_ID])
        for b in range(B):
            for c in range(MAX_CYCLES):
                assert out.hard_token_ids[b, c].item() in valid_set

    def test_done_stops_generation(self, tiny_model):
        torch.manual_seed(42)
        B, L = 2, 8
        token_ids = torch.randint(0, 100, (B, L))

        out = generate_internal_reasoning_tokens(
            net=tiny_model,
            token_ids=token_ids,
            soft_block_size=SOFT_BLOCK_SIZE,
            max_cycles=MAX_CYCLES,
            valid_hard_token_ids=VALID_HARD_TOKEN_IDS,
            done_token_id=EOS_TOKEN_ID,
            pad_token_id=PAD_TOKEN_ID,
            temperature=1.0,
        )

        for b in range(B):
            nc = out.n_cycles[b].item()
            if nc < MAX_CYCLES:
                assert out.hard_token_ids[b, nc - 1].item() == EOS_TOKEN_ID
                for c in range(nc, MAX_CYCLES):
                    assert out.hard_token_ids[b, c].item() == PAD_TOKEN_ID

    def test_no_valid_mask_uses_full_vocab(self, tiny_model):
        torch.manual_seed(42)
        B, L = 2, 8
        token_ids = torch.randint(0, 100, (B, L))

        out = generate_internal_reasoning_tokens(
            net=tiny_model,
            token_ids=token_ids,
            soft_block_size=SOFT_BLOCK_SIZE,
            max_cycles=MAX_CYCLES,
            valid_hard_token_ids=None,
            done_token_id=EOS_TOKEN_ID,
            pad_token_id=PAD_TOKEN_ID,
            temperature=1.0,
        )

        assert out.hard_token_ids.shape == (B, MAX_CYCLES)


class TestStackAndPadInternalReasoning:
    def test_shapes(self):
        B, G = 2, 3
        C1, C2, C3 = 5, 3, 4
        pad = 0

        ids = [
            torch.randint(1, 10, (B, C1)),
            torch.randint(1, 10, (B, C2)),
            torch.randint(1, 10, (B, C3)),
        ]
        ncs = [
            torch.tensor([5, 3]),
            torch.tensor([2, 3]),
            torch.tensor([4, 1]),
        ]

        stacked_ids, stacked_n = stack_and_pad_internal_reasoning(ids, ncs, pad)

        assert stacked_ids.shape == (B, G, C1)
        assert stacked_n.shape == (B, G)
        torch.testing.assert_close(stacked_ids[:, 0, :C1], ids[0])
        torch.testing.assert_close(stacked_ids[:, 1, :C2], ids[1])
        assert stacked_ids[:, 1, C2:].eq(pad).all()


class TestComputeInternalReasoningLogProbs:
    def test_output_shape(self, tiny_model):
        torch.manual_seed(42)
        tiny_model.train()

        B, G, C = 2, 2, 4
        L = 6
        prompt_ids = torch.randint(0, 100, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.bool)
        hard_ids = torch.randint(0, len(VALID_HARD_TOKEN_IDS), (B, G, C))
        for b in range(B):
            for g in range(G):
                for c in range(C):
                    hard_ids[b, g, c] = VALID_HARD_TOKEN_IDS[hard_ids[b, g, c].item()]
        n_cycles = torch.full((B, G), C, dtype=torch.long)

        log_probs, mask = compute_internal_reasoning_log_probs(
            net=tiny_model,
            prompt_token_ids=prompt_ids,
            attention_mask=attention_mask,
            hard_token_ids=hard_ids,
            n_cycles=n_cycles,
            valid_hard_token_ids=VALID_HARD_TOKEN_IDS,
            soft_block_size=SOFT_BLOCK_SIZE,
            pad_token_id=PAD_TOKEN_ID,
        )

        assert log_probs.shape == (B, G, C)
        assert mask.shape == (B, G, C)
        assert (log_probs[mask] <= 0).all()

    def test_completion_mask_matches_n_cycles(self, tiny_model):
        torch.manual_seed(42)
        tiny_model.train()

        B, G, C = 2, 2, 5
        L = 6
        prompt_ids = torch.randint(0, 100, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.bool)
        hard_ids = torch.full((B, G, C), VALID_HARD_TOKEN_IDS[0], dtype=torch.long)
        n_cycles = torch.tensor([[3, 5], [2, 4]])

        log_probs, mask = compute_internal_reasoning_log_probs(
            net=tiny_model,
            prompt_token_ids=prompt_ids,
            attention_mask=attention_mask,
            hard_token_ids=hard_ids,
            n_cycles=n_cycles,
            valid_hard_token_ids=VALID_HARD_TOKEN_IDS,
            soft_block_size=SOFT_BLOCK_SIZE,
            pad_token_id=PAD_TOKEN_ID,
        )

        for b in range(B):
            for g in range(G):
                nc = n_cycles[b, g].item()
                assert mask[b, g, :nc].all()
                if nc < C:
                    assert not mask[b, g, nc:].any()


class TestGradSafeKVCacheEquivalence:
    def test_forward_matches_no_cache(self, tiny_model):
        """GradSafeKVCache token-by-token produces same logits as full-sequence."""
        torch.manual_seed(42)
        tiny_model.eval()

        B, L = 2, 8
        token_ids = torch.randint(0, 100, (B, L))
        embeds = tiny_model.embed_tokens(token_ids)

        h_full = tiny_model(embeds, return_hidden_states=True)
        logits_full = tiny_model.lm_head(h_full)

        grad_caches = [GradSafeKVCache() for _ in tiny_model.layers]
        hidden_list = []
        for i in range(L):
            h = tiny_model(
                embeds[:, i : i + 1],
                kv_caches=grad_caches,
                return_hidden_states=True,
            )
            hidden_list.append(h)
        logits_cached = tiny_model.lm_head(torch.cat(hidden_list, dim=1))

        torch.testing.assert_close(logits_full, logits_cached, atol=1e-5, rtol=1e-5)

    def test_gradients_match_no_cache(self, tiny_model):
        """Gradients through GradSafeKVCache match full-sequence gradients."""
        torch.manual_seed(42)
        tiny_model.train()

        B, L = 2, 6
        token_ids = torch.randint(0, 100, (B, L))

        embeds_a = tiny_model.embed_tokens(token_ids)
        h_full = tiny_model(embeds_a, return_hidden_states=True)
        loss_full = tiny_model.lm_head(h_full).sum()
        tiny_model.zero_grad()
        loss_full.backward()
        grads_full = {
            name: p.grad.clone()
            for name, p in tiny_model.named_parameters()
            if p.grad is not None
        }

        tiny_model.zero_grad()
        embeds_b = tiny_model.embed_tokens(token_ids)
        grad_caches = [GradSafeKVCache() for _ in tiny_model.layers]
        hidden_list = []
        for i in range(L):
            h = tiny_model(
                embeds_b[:, i : i + 1],
                kv_caches=grad_caches,
                return_hidden_states=True,
            )
            hidden_list.append(h)
        loss_cached = tiny_model.lm_head(torch.cat(hidden_list, dim=1)).sum()
        loss_cached.backward()
        grads_cached = {
            name: p.grad.clone()
            for name, p in tiny_model.named_parameters()
            if p.grad is not None
        }

        for name in grads_full:
            assert name in grads_cached, f"Missing gradient for {name}"
            torch.testing.assert_close(
                grads_full[name], grads_cached[name], atol=2e-4, rtol=2e-4
            )


class TestPerCycleBackward:
    def test_gradients_match_all_at_once(self, tiny_model):
        """Per-cycle backward produces same gradients as all-at-once backward."""
        torch.manual_seed(42)
        tiny_model.train()

        B, G, C = 2, 2, 3
        L = 6
        prompt_ids = torch.randint(0, 100, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.bool)
        hard_ids = torch.full((B, G, C), VALID_HARD_TOKEN_IDS[0], dtype=torch.long)
        n_cycles = torch.tensor([[2, 3], [3, 1]])
        cycle_indices = torch.arange(C).unsqueeze(0).unsqueeze(0).expand(B, G, C)
        completion_mask = cycle_indices < n_cycles.unsqueeze(-1)
        advs = torch.randn(B, G, 1)

        # All-at-once backward
        tiny_model.zero_grad()
        log_probs, _ = compute_internal_reasoning_log_probs(
            net=tiny_model,
            prompt_token_ids=prompt_ids,
            attention_mask=attention_mask,
            hard_token_ids=hard_ids,
            n_cycles=n_cycles,
            valid_hard_token_ids=VALID_HARD_TOKEN_IDS,
            soft_block_size=SOFT_BLOCK_SIZE,
            pad_token_id=PAD_TOKEN_ID,
        )
        loss, _, _ = compute_grpo_loss(
            log_probs=log_probs,
            old_log_probs=None,
            ref_log_probs=None,
            completion_mask=completion_mask,
            advs=advs,
            beta=0.0,
            eps=None,
            normalize_by_sequence_length=True,
        )
        loss.backward()
        grads_all_at_once = {
            name: p.grad.clone()
            for name, p in tiny_model.named_parameters()
            if p.grad is not None
        }

        # Per-cycle backward: the callback calls backward() inside the loop,
        # so gradients are already accumulated when the function returns.
        tiny_model.zero_grad()
        callback = make_per_cycle_backward_callback(
            B=B,
            G=G,
            advs=advs,
            completion_mask=completion_mask,
            old_log_probs=None,
            ref_log_probs=None,
            beta=0.0,
            eps=None,
            normalize_by_sequence_length=True,
            loss_scale=1.0,
        )
        compute_internal_reasoning_log_probs(
            net=tiny_model,
            prompt_token_ids=prompt_ids,
            attention_mask=attention_mask,
            hard_token_ids=hard_ids,
            n_cycles=n_cycles,
            valid_hard_token_ids=VALID_HARD_TOKEN_IDS,
            soft_block_size=SOFT_BLOCK_SIZE,
            pad_token_id=PAD_TOKEN_ID,
            cycle_callback=callback,
        )
        grads_per_cycle = {
            name: p.grad.clone()
            for name, p in tiny_model.named_parameters()
            if p.grad is not None
        }

        for name in grads_all_at_once:
            assert name in grads_per_cycle, f"Missing gradient for {name}"
            torch.testing.assert_close(
                grads_all_at_once[name], grads_per_cycle[name], atol=2e-4, rtol=2e-4
            )


class TestKVCacheAndAutograd:
    def test_no_autograd_error(self, tiny_model):
        """Verify that compute_internal_reasoning_log_probs with KV cache
        doesn't raise RuntimeError from in-place ops during backward."""
        torch.manual_seed(42)
        tiny_model.train()

        B, G, C = 1, 1, 3
        L = 4
        prompt_ids = torch.randint(0, 100, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.bool)
        hard_ids = torch.full((B, G, C), VALID_HARD_TOKEN_IDS[0], dtype=torch.long)
        n_cycles = torch.full((B, G), C, dtype=torch.long)

        log_probs, mask = compute_internal_reasoning_log_probs(
            net=tiny_model,
            prompt_token_ids=prompt_ids,
            attention_mask=attention_mask,
            hard_token_ids=hard_ids,
            n_cycles=n_cycles,
            valid_hard_token_ids=VALID_HARD_TOKEN_IDS,
            soft_block_size=SOFT_BLOCK_SIZE,
            pad_token_id=PAD_TOKEN_ID,
        )

        loss = (log_probs * mask).sum()
        loss.backward()

    def test_grad_flows_through_soft_block(self, tiny_model):
        """Verify model parameters have non-zero gradients after backward."""
        torch.manual_seed(42)
        tiny_model.train()

        B, G, C = 1, 1, 2
        L = 4
        prompt_ids = torch.randint(0, 100, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.bool)
        hard_ids = torch.full((B, G, C), VALID_HARD_TOKEN_IDS[0], dtype=torch.long)
        n_cycles = torch.full((B, G), C, dtype=torch.long)

        log_probs, mask = compute_internal_reasoning_log_probs(
            net=tiny_model,
            prompt_token_ids=prompt_ids,
            attention_mask=attention_mask,
            hard_token_ids=hard_ids,
            n_cycles=n_cycles,
            valid_hard_token_ids=VALID_HARD_TOKEN_IDS,
            soft_block_size=SOFT_BLOCK_SIZE,
            pad_token_id=PAD_TOKEN_ID,
        )

        loss = (log_probs * mask).sum()
        loss.backward()

        for _, p in tiny_model.named_parameters():
            assert p.grad is not None and p.grad.abs().sum() > 0

        for layer in tiny_model.layers:
            assert layer.self_attn.q_proj.weight.grad is not None
            assert layer.self_attn.q_proj.weight.grad.abs().sum() > 0

    def test_grad_truncated_at_cycle_boundary(self, tiny_model):
        """Verify gradients from cycle 2 don't flow to cycle 1's soft block."""
        torch.manual_seed(42)
        tiny_model.train()

        B, G, C = 1, 1, 2
        L = 4
        prompt_ids = torch.randint(0, 100, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.bool)
        hard_ids = torch.full((B, G, C), VALID_HARD_TOKEN_IDS[0], dtype=torch.long)

        n_cycles_single = torch.tensor([[1]])
        log_probs_c1, mask_c1 = compute_internal_reasoning_log_probs(
            net=tiny_model,
            prompt_token_ids=prompt_ids,
            attention_mask=attention_mask,
            hard_token_ids=hard_ids,
            n_cycles=n_cycles_single,
            valid_hard_token_ids=VALID_HARD_TOKEN_IDS,
            soft_block_size=SOFT_BLOCK_SIZE,
            pad_token_id=PAD_TOKEN_ID,
        )
        loss_c1 = log_probs_c1[:, :, 0].sum()
        tiny_model.zero_grad()
        loss_c1.backward()
        grad_c1_only = {
            name: p.grad.clone()
            for name, p in tiny_model.named_parameters()
            if p.grad is not None
        }

        n_cycles_two = torch.tensor([[2]])
        log_probs_c2, mask_c2 = compute_internal_reasoning_log_probs(
            net=tiny_model,
            prompt_token_ids=prompt_ids,
            attention_mask=attention_mask,
            hard_token_ids=hard_ids,
            n_cycles=n_cycles_two,
            valid_hard_token_ids=VALID_HARD_TOKEN_IDS,
            soft_block_size=SOFT_BLOCK_SIZE,
            pad_token_id=PAD_TOKEN_ID,
        )
        loss_c2_only = log_probs_c2[:, :, 1].sum()
        tiny_model.zero_grad()
        loss_c2_only.backward()
        grad_c2_only = {
            name: p.grad.clone()
            for name, p in tiny_model.named_parameters()
            if p.grad is not None
        }

        n_cycles_both = torch.tensor([[2]])
        log_probs_both, _ = compute_internal_reasoning_log_probs(
            net=tiny_model,
            prompt_token_ids=prompt_ids,
            attention_mask=attention_mask,
            hard_token_ids=hard_ids,
            n_cycles=n_cycles_both,
            valid_hard_token_ids=VALID_HARD_TOKEN_IDS,
            soft_block_size=SOFT_BLOCK_SIZE,
            pad_token_id=PAD_TOKEN_ID,
        )
        loss_both = log_probs_both[:, :, 0].sum() + log_probs_both[:, :, 1].sum()
        tiny_model.zero_grad()
        loss_both.backward()
        grad_both = {
            name: p.grad.clone()
            for name, p in tiny_model.named_parameters()
            if p.grad is not None
        }

        for name in grad_c1_only:
            if name in grad_c2_only and name in grad_both:
                expected_sum = grad_c1_only[name] + grad_c2_only[name]
                torch.testing.assert_close(
                    grad_both[name], expected_sum, atol=1e-4, rtol=1e-4
                )

    def test_grad_magnitude_reasonable(self, tiny_model):
        """Check gradients are finite, non-NaN, and have reasonable magnitude."""
        torch.manual_seed(42)
        tiny_model.train()

        B, G, C = 1, 1, 3
        L = 4
        prompt_ids = torch.randint(0, 100, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.bool)
        hard_ids = torch.full((B, G, C), VALID_HARD_TOKEN_IDS[0], dtype=torch.long)
        n_cycles = torch.full((B, G), C, dtype=torch.long)

        log_probs, mask = compute_internal_reasoning_log_probs(
            net=tiny_model,
            prompt_token_ids=prompt_ids,
            attention_mask=attention_mask,
            hard_token_ids=hard_ids,
            n_cycles=n_cycles,
            valid_hard_token_ids=VALID_HARD_TOKEN_IDS,
            soft_block_size=SOFT_BLOCK_SIZE,
            pad_token_id=PAD_TOKEN_ID,
        )

        loss = (log_probs * mask).sum()
        loss.backward()

        for name, p in tiny_model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Non-finite grad in {name}"
                assert not torch.isnan(p.grad).any(), f"NaN grad in {name}"
                grad_norm = p.grad.norm().item()
                assert grad_norm < 1e6, f"Exploding grad in {name}: norm={grad_norm}"


class TestTrainInternalReasoningGrpo:
    def test_training_loop_completes(self, tiny_model, tokenizer):
        torch.manual_seed(42)
        env = MazeEnv(config=MazeConfig(height=3, width=3), seed=42)
        opt = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)

        direction_words = ["up", "down", "left", "right"]
        valid_ids = []
        move_map: dict[int, str] = {}
        for word in direction_words:
            ids = tokenizer.encode(word, add_special_tokens=False).ids
            assert len(ids) == 1
            tid = ids[0]
            valid_ids.append(tid)
            move_map[tid] = word
        valid_ids.append(151645)

        reward_fn = weighted_reward(
            [
                ("correct", 1.0, maze_correct),
                ("distance", 0.5, maze_distance),
            ]
        )

        train_internal_reasoning_grpo(
            net=tiny_model,
            opt=opt,
            env=env,
            reward_fn=reward_fn,
            state_to_str=maze_state_to_str,
            tokenizer=tokenizer,
            eos_token_id=151645,
            pad_token_id=151643,
            extractor=lambda moves: moves if moves else None,
            move_id_to_name=move_map,
            valid_hard_token_ids=valid_ids,
            soft_block_size=2,
            max_cycles=5,
            beta=0.0,
            eps=None,
            mu=1,
            max_episodes=4,
            update_ref_net_batch_cadence=10,
            batch_size=2,
            group_size=2,
            temperature=1.0,
            use_bf16=False,
            advantage_fn=grpo_advantage,
            normalize_by_sequence_length=True,
        )

    def test_optimizer_step_changes_params(self, tiny_model, tokenizer):
        torch.manual_seed(42)
        env = MazeEnv(config=MazeConfig(height=3, width=3), seed=42)
        opt = torch.optim.Adam(tiny_model.parameters(), lr=1e-2)

        direction_words = ["up", "down", "left", "right"]
        valid_ids = []
        move_map: dict[int, str] = {}
        for word in direction_words:
            ids = tokenizer.encode(word, add_special_tokens=False).ids
            tid = ids[0]
            valid_ids.append(tid)
            move_map[tid] = word
        valid_ids.append(151645)

        from dialectic.rl.types import RewardResult

        rng = torch.Generator().manual_seed(42)

        def random_reward_fn(
            *, env_response, raw_model_output=None, extracted_model_output
        ) -> RewardResult:
            value = torch.rand((), generator=rng).item()
            return RewardResult(total=value, components={"random": value})

        params_before = {
            name: param.clone() for name, param in tiny_model.named_parameters()
        }

        train_internal_reasoning_grpo(
            net=tiny_model,
            opt=opt,
            env=env,
            reward_fn=random_reward_fn,
            state_to_str=maze_state_to_str,
            tokenizer=tokenizer,
            eos_token_id=151645,
            pad_token_id=151643,
            extractor=lambda moves: moves if moves else None,
            move_id_to_name=move_map,
            valid_hard_token_ids=valid_ids,
            soft_block_size=2,
            max_cycles=5,
            beta=0.0,
            eps=None,
            mu=1,
            max_episodes=8,
            update_ref_net_batch_cadence=10,
            batch_size=2,
            group_size=2,
            temperature=1.0,
            use_bf16=False,
            advantage_fn=grpo_advantage,
            normalize_by_sequence_length=True,
        )

        params_changed = False
        for name, param in tiny_model.named_parameters():
            if not torch.allclose(params_before[name], param, atol=1e-8):
                params_changed = True
                break

        assert params_changed, "No parameters changed during training"


class TestSoftProjection:
    def test_applied_to_soft_tokens_not_hard(self, tiny_model_with_soft_projection):
        """Projection is called exactly soft_block_size times per cycle (soft only, not hard)."""
        torch.manual_seed(42)
        model = tiny_model_with_soft_projection

        call_count = 0
        orig_apply = model.apply_soft_projection

        def counting_apply(h):
            nonlocal call_count
            call_count += 1
            return orig_apply(h)

        model.apply_soft_projection = counting_apply

        B, L = 1, 6
        token_ids = torch.randint(0, 100, (B, L))
        out = generate_internal_reasoning_tokens(
            net=model,
            token_ids=token_ids,
            soft_block_size=SOFT_BLOCK_SIZE,
            max_cycles=MAX_CYCLES,
            valid_hard_token_ids=VALID_HARD_TOKEN_IDS,
            done_token_id=EOS_TOKEN_ID,
            pad_token_id=PAD_TOKEN_ID,
            temperature=1.0,
        )

        completed_cycles = out.n_cycles[0].item()
        expected_calls = SOFT_BLOCK_SIZE * completed_cycles
        assert call_count == expected_calls, (
            f"apply_soft_projection called {call_count} times, "
            f"expected {expected_calls} ({SOFT_BLOCK_SIZE} soft * {completed_cycles} cycles)"
        )

    def test_zero_alpha_matches_no_projection(
        self, tiny_model, tiny_model_with_soft_projection
    ):
        """With alpha=0, outputs match model without projection."""
        torch.manual_seed(42)
        B, L = 2, 8
        token_ids = torch.randint(0, 100, (B, L))

        out_no_proj = generate_internal_reasoning_tokens(
            net=tiny_model,
            token_ids=token_ids,
            soft_block_size=SOFT_BLOCK_SIZE,
            max_cycles=MAX_CYCLES,
            valid_hard_token_ids=VALID_HARD_TOKEN_IDS,
            done_token_id=EOS_TOKEN_ID,
            pad_token_id=PAD_TOKEN_ID,
            temperature=1.0,
        )

        torch.manual_seed(42)
        out_with_proj = generate_internal_reasoning_tokens(
            net=tiny_model_with_soft_projection,
            token_ids=token_ids,
            soft_block_size=SOFT_BLOCK_SIZE,
            max_cycles=MAX_CYCLES,
            valid_hard_token_ids=VALID_HARD_TOKEN_IDS,
            done_token_id=EOS_TOKEN_ID,
            pad_token_id=PAD_TOKEN_ID,
            temperature=1.0,
        )

        torch.testing.assert_close(
            out_with_proj.hard_log_probs,
            out_no_proj.hard_log_probs,
            atol=1e-4,
            rtol=1e-4,
        )
        assert (out_with_proj.hard_token_ids == out_no_proj.hard_token_ids).all()

    def test_alpha_receives_gradients_at_zero(self, tiny_model_with_soft_projection):
        """With alpha=0, alpha gets gradients so it can grow."""
        torch.manual_seed(42)
        model = tiny_model_with_soft_projection
        model.train()

        B, G, C = 1, 1, 3
        L = 4
        prompt_ids = torch.randint(0, 100, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.bool)
        hard_ids = torch.full((B, G, C), VALID_HARD_TOKEN_IDS[0], dtype=torch.long)
        n_cycles = torch.full((B, G), C, dtype=torch.long)

        log_probs, mask = compute_internal_reasoning_log_probs(
            net=model,
            prompt_token_ids=prompt_ids,
            attention_mask=attention_mask,
            hard_token_ids=hard_ids,
            n_cycles=n_cycles,
            valid_hard_token_ids=VALID_HARD_TOKEN_IDS,
            soft_block_size=SOFT_BLOCK_SIZE,
            pad_token_id=PAD_TOKEN_ID,
        )

        loss = (log_probs * mask).sum()
        loss.backward()

        assert model.soft_projection_alpha is not None
        assert model.soft_projection_alpha.grad is not None
        assert model.soft_projection_alpha.grad.abs().item() > 0

    def test_projection_weight_receives_gradients_when_alpha_nonzero(
        self, tiny_model_with_soft_projection
    ):
        """With alpha != 0, projection weights also receive gradients."""
        torch.manual_seed(42)
        model = tiny_model_with_soft_projection
        model.soft_projection_alpha.data.fill_(1.0)
        model.train()

        B, G, C = 1, 1, 3
        L = 4
        prompt_ids = torch.randint(0, 100, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.bool)
        hard_ids = torch.full((B, G, C), VALID_HARD_TOKEN_IDS[0], dtype=torch.long)
        n_cycles = torch.full((B, G), C, dtype=torch.long)

        log_probs, mask = compute_internal_reasoning_log_probs(
            net=model,
            prompt_token_ids=prompt_ids,
            attention_mask=attention_mask,
            hard_token_ids=hard_ids,
            n_cycles=n_cycles,
            valid_hard_token_ids=VALID_HARD_TOKEN_IDS,
            soft_block_size=SOFT_BLOCK_SIZE,
            pad_token_id=PAD_TOKEN_ID,
        )

        loss = (log_probs * mask).sum()
        loss.backward()

        assert model.soft_projection is not None
        assert model.soft_projection.weight.grad is not None
        assert model.soft_projection.weight.grad.abs().sum() > 0


class TestSoftBpttWindow:
    def test_full_window_matches_default(self, tiny_model):
        """soft_bptt_window=soft_block_size should match default (None)."""
        torch.manual_seed(42)
        tiny_model.train()

        B, G, C = 1, 1, 2
        L = 4
        prompt_ids = torch.randint(0, 100, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.bool)
        hard_ids = torch.full((B, G, C), VALID_HARD_TOKEN_IDS[0], dtype=torch.long)
        n_cycles = torch.full((B, G), C, dtype=torch.long)

        tiny_model.zero_grad()
        lp_default, _ = compute_internal_reasoning_log_probs(
            net=tiny_model,
            prompt_token_ids=prompt_ids,
            attention_mask=attention_mask,
            hard_token_ids=hard_ids,
            n_cycles=n_cycles,
            valid_hard_token_ids=VALID_HARD_TOKEN_IDS,
            soft_block_size=SOFT_BLOCK_SIZE,
            pad_token_id=PAD_TOKEN_ID,
        )
        lp_default.sum().backward()
        grads_default = {
            n: p.grad.clone()
            for n, p in tiny_model.named_parameters()
            if p.grad is not None
        }

        tiny_model.zero_grad()
        lp_full, _ = compute_internal_reasoning_log_probs(
            net=tiny_model,
            prompt_token_ids=prompt_ids,
            attention_mask=attention_mask,
            hard_token_ids=hard_ids,
            n_cycles=n_cycles,
            valid_hard_token_ids=VALID_HARD_TOKEN_IDS,
            soft_block_size=SOFT_BLOCK_SIZE,
            pad_token_id=PAD_TOKEN_ID,
            soft_bptt_window=SOFT_BLOCK_SIZE,
        )
        lp_full.sum().backward()
        grads_full = {
            n: p.grad.clone()
            for n, p in tiny_model.named_parameters()
            if p.grad is not None
        }

        torch.testing.assert_close(lp_default, lp_full, atol=1e-5, rtol=1e-5)
        for name in grads_default:
            torch.testing.assert_close(
                grads_default[name], grads_full[name], atol=1e-5, rtol=1e-5
            )

    def test_window_reduces_grad_norm(self, tiny_model):
        """Smaller bptt window should produce smaller or equal gradient norms."""
        torch.manual_seed(42)
        tiny_model.train()

        soft_block_size = 4
        B, G, C = 1, 1, 2
        L = 4
        prompt_ids = torch.randint(0, 100, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.bool)
        hard_ids = torch.full((B, G, C), VALID_HARD_TOKEN_IDS[0], dtype=torch.long)
        n_cycles = torch.full((B, G), C, dtype=torch.long)

        tiny_model.zero_grad()
        lp_full, _ = compute_internal_reasoning_log_probs(
            net=tiny_model,
            prompt_token_ids=prompt_ids,
            attention_mask=attention_mask,
            hard_token_ids=hard_ids,
            n_cycles=n_cycles,
            valid_hard_token_ids=VALID_HARD_TOKEN_IDS,
            soft_block_size=soft_block_size,
            pad_token_id=PAD_TOKEN_ID,
            soft_bptt_window=soft_block_size,
        )
        lp_full.sum().backward()
        grad_norm_full = (
            sum(
                p.grad.norm().item() ** 2
                for p in tiny_model.parameters()
                if p.grad is not None
            )
            ** 0.5
        )

        tiny_model.zero_grad()
        lp_window, _ = compute_internal_reasoning_log_probs(
            net=tiny_model,
            prompt_token_ids=prompt_ids,
            attention_mask=attention_mask,
            hard_token_ids=hard_ids,
            n_cycles=n_cycles,
            valid_hard_token_ids=VALID_HARD_TOKEN_IDS,
            soft_block_size=soft_block_size,
            pad_token_id=PAD_TOKEN_ID,
            soft_bptt_window=1,
        )
        lp_window.sum().backward()
        grad_norm_window = (
            sum(
                p.grad.norm().item() ** 2
                for p in tiny_model.parameters()
                if p.grad is not None
            )
            ** 0.5
        )

        # Forward values should be identical (same computation, just different grad)
        torch.testing.assert_close(lp_full, lp_window, atol=1e-5, rtol=1e-5)

        # Windowed grad norm should be smaller (fewer layers of backprop)
        assert grad_norm_window <= grad_norm_full * 1.01, (
            f"Window grad norm {grad_norm_window} > full grad norm {grad_norm_full}"
        )

    def test_window_zero_raises_or_no_grad(self, tiny_model):
        """soft_bptt_window=0 means no soft tokens get gradients; only hard token pass."""
        torch.manual_seed(42)
        tiny_model.train()

        B, G, C = 1, 1, 2
        L = 4
        prompt_ids = torch.randint(0, 100, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.bool)
        hard_ids = torch.full((B, G, C), VALID_HARD_TOKEN_IDS[0], dtype=torch.long)
        n_cycles = torch.full((B, G), C, dtype=torch.long)

        tiny_model.zero_grad()
        lp, _ = compute_internal_reasoning_log_probs(
            net=tiny_model,
            prompt_token_ids=prompt_ids,
            attention_mask=attention_mask,
            hard_token_ids=hard_ids,
            n_cycles=n_cycles,
            valid_hard_token_ids=VALID_HARD_TOKEN_IDS,
            soft_block_size=SOFT_BLOCK_SIZE,
            pad_token_id=PAD_TOKEN_ID,
            soft_bptt_window=0,
        )
        lp.sum().backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in tiny_model.parameters()
        )
        assert has_grad, "Should still have gradients from the hard token forward pass"


class TestSFT:
    def _get_move_mappings(self, tokenizer):
        direction_words = ["up", "down", "left", "right"]
        valid_ids: list[int] = []
        move_id_to_name: dict[int, str] = {}
        move_name_to_id: dict[str, int] = {}
        for word in direction_words:
            ids = tokenizer.encode(word, add_special_tokens=False).ids
            assert len(ids) == 1
            tid = ids[0]
            valid_ids.append(tid)
            move_id_to_name[tid] = word
            move_name_to_id[word] = tid
        valid_ids.append(EOS_TOKEN_ID)
        return valid_ids, move_id_to_name, move_name_to_id

    def test_sft_callback_loss_finite(self):
        B, C = 2, 4
        completion_mask = torch.zeros(B, 1, C, dtype=torch.bool)
        completion_mask[0, 0, :3] = True
        completion_mask[1, 0, :2] = True

        lp_param = torch.randn(B, requires_grad=True)

        callback = make_sft_per_cycle_backward_callback(
            B=B,
            completion_mask=completion_mask,
            normalize_by_sequence_length=True,
            loss_scale=1.0,
        )

        result = callback(lp_param, 0)
        assert result.requires_grad is False
        assert lp_param.grad is not None
        assert torch.isfinite(lp_param.grad).all()

    def test_sft_eos_appended(self, tokenizer):
        valid_ids, move_id_to_name, move_name_to_id = self._get_move_mappings(tokenizer)

        solution = ["right", "down", "right"]
        move_ids = [move_name_to_id[m] for m in solution]
        move_ids.append(EOS_TOKEN_ID)

        assert len(move_ids) == 4
        assert move_ids[-1] == EOS_TOKEN_ID
        for m_id in move_ids[:-1]:
            assert m_id in move_name_to_id.values()

    def test_sft_step_runs(self, tiny_model, tokenizer):
        torch.manual_seed(42)
        valid_ids, move_id_to_name, move_name_to_id = self._get_move_mappings(tokenizer)
        env = MazeEnv(config=MazeConfig(height=3, width=3), seed=42)
        opt = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)

        params_before = {
            name: param.clone() for name, param in tiny_model.named_parameters()
        }

        train_internal_reasoning_sft(
            net=tiny_model,
            opt=opt,
            env=env,
            state_to_str=maze_state_to_str,
            tokenizer=tokenizer,
            pad_token_id=PAD_TOKEN_ID,
            eos_token_id=EOS_TOKEN_ID,
            move_name_to_id=move_name_to_id,
            valid_hard_token_ids=valid_ids,
            move_id_to_name=move_id_to_name,
            soft_block_size=2,
            max_cycles=10,
            max_episodes=4,
            batch_size=2,
            accumulation_steps=1,
            max_grad_norm=1.0,
            normalize_by_sequence_length=True,
            use_bf16=False,
        )

        params_changed = False
        for name, param in tiny_model.named_parameters():
            if not torch.allclose(params_before[name], param, atol=1e-8):
                params_changed = True
                break
        assert params_changed, "No parameters changed during SFT training"


class TestThinkTokens:
    def test_think_token_generation_shape(self, tiny_model):
        torch.manual_seed(42)
        B, L = 2, 10
        token_ids = torch.randint(0, 100, (B, L))
        think_id = 50

        out = generate_internal_reasoning_tokens(
            net=tiny_model,
            token_ids=token_ids,
            soft_block_size=SOFT_BLOCK_SIZE,
            max_cycles=MAX_CYCLES,
            valid_hard_token_ids=VALID_HARD_TOKEN_IDS,
            done_token_id=EOS_TOKEN_ID,
            pad_token_id=PAD_TOKEN_ID,
            temperature=1.0,
            think_token_id=think_id,
        )

        assert out.hard_token_ids.shape == (B, MAX_CYCLES)
        assert out.hard_log_probs.shape == (B, MAX_CYCLES)
        assert out.n_cycles.shape == (B,)
        assert (out.n_cycles <= MAX_CYCLES).all()

        valid_set = set(VALID_HARD_TOKEN_IDS + [PAD_TOKEN_ID])
        for b in range(B):
            for c in range(MAX_CYCLES):
                assert out.hard_token_ids[b, c].item() in valid_set

    def test_think_token_log_probs_shape(self, tiny_model):
        torch.manual_seed(42)
        tiny_model.train()

        B, G, C = 2, 2, 4
        L = 6
        think_id = 50
        prompt_ids = torch.randint(0, 100, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.bool)
        hard_ids = torch.full((B, G, C), VALID_HARD_TOKEN_IDS[0], dtype=torch.long)
        n_cycles = torch.full((B, G), C, dtype=torch.long)

        log_probs, mask = compute_internal_reasoning_log_probs(
            net=tiny_model,
            prompt_token_ids=prompt_ids,
            attention_mask=attention_mask,
            hard_token_ids=hard_ids,
            n_cycles=n_cycles,
            valid_hard_token_ids=VALID_HARD_TOKEN_IDS,
            soft_block_size=SOFT_BLOCK_SIZE,
            pad_token_id=PAD_TOKEN_ID,
            think_token_id=think_id,
        )

        assert log_probs.shape == (B, G, C)
        assert mask.shape == (B, G, C)
        assert (log_probs[mask] <= 0).all()

    def test_think_token_grad_flows(self, tiny_model):
        torch.manual_seed(42)
        tiny_model.train()

        B, G, C = 1, 1, 2
        L = 4
        think_id = 50
        prompt_ids = torch.randint(0, 100, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.bool)
        hard_ids = torch.full((B, G, C), VALID_HARD_TOKEN_IDS[0], dtype=torch.long)
        n_cycles = torch.full((B, G), C, dtype=torch.long)

        log_probs, mask = compute_internal_reasoning_log_probs(
            net=tiny_model,
            prompt_token_ids=prompt_ids,
            attention_mask=attention_mask,
            hard_token_ids=hard_ids,
            n_cycles=n_cycles,
            valid_hard_token_ids=VALID_HARD_TOKEN_IDS,
            soft_block_size=SOFT_BLOCK_SIZE,
            pad_token_id=PAD_TOKEN_ID,
            think_token_id=think_id,
        )

        loss = (log_probs * mask).sum()
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in tiny_model.parameters()
        )
        assert has_grad, "No gradients flowed through think token path"

    def test_think_token_differs_from_soft(self, tiny_model):
        torch.manual_seed(42)
        tiny_model.train()

        B, G, C = 1, 1, 3
        L = 4
        think_id = 50
        prompt_ids = torch.randint(0, 100, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.bool)
        hard_ids = torch.full((B, G, C), VALID_HARD_TOKEN_IDS[0], dtype=torch.long)
        n_cycles = torch.full((B, G), C, dtype=torch.long)

        lp_soft, _ = compute_internal_reasoning_log_probs(
            net=tiny_model,
            prompt_token_ids=prompt_ids,
            attention_mask=attention_mask,
            hard_token_ids=hard_ids,
            n_cycles=n_cycles,
            valid_hard_token_ids=VALID_HARD_TOKEN_IDS,
            soft_block_size=SOFT_BLOCK_SIZE,
            pad_token_id=PAD_TOKEN_ID,
        )

        lp_think, _ = compute_internal_reasoning_log_probs(
            net=tiny_model,
            prompt_token_ids=prompt_ids,
            attention_mask=attention_mask,
            hard_token_ids=hard_ids,
            n_cycles=n_cycles,
            valid_hard_token_ids=VALID_HARD_TOKEN_IDS,
            soft_block_size=SOFT_BLOCK_SIZE,
            pad_token_id=PAD_TOKEN_ID,
            think_token_id=think_id,
        )

        assert not torch.allclose(lp_soft, lp_think, atol=1e-5), (
            "Think token and soft token log probs should differ"
        )
