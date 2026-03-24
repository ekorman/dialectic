import torch

from dialectic.llm.generate import (
    NoiseReasoningGeneratorOutput,
    generate_noise_reasoning_tokens,
)
from dialectic.rl.env import Countdown, CountdownEnv
from dialectic.rl.extractors import extract_from_answer_tags
from dialectic.rl.grammar import (
    build_countdown_cycle_grammar_factory,
)
from dialectic.rl.reward import countdown_correct, weighted_reward
from dialectic.rl.rollout import generate_noise_reasoning_rollout_batch
from dialectic.rl.train import (
    compute_noise_reasoning_log_probs,
    stack_and_pad_noise_reasoning,
    train_noise_reasoning_grpo,
)

EOS_TOKEN_ID = 151645
PAD_TOKEN_ID = 151643


def countdown_state_to_str(data: Countdown) -> str:
    return data.prompt


# ---------------------------------------------------------------------------
# Grammar tests
# ---------------------------------------------------------------------------


class TestCountdownCycleGrammar:
    def test_scratch_route(self, tokenizer):
        factory = build_countdown_cycle_grammar_factory(tokenizer)
        g = factory(allow_answer=True)
        text = "<SCRATCH> 5 + 3 = 8 </SCRATCH>"
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        for tok in ids:
            assert tok in g.valid_token_ids()
            g.advance(tok)
        assert g.is_complete()
        assert not g.is_terminal()

    def test_answer_route(self, tokenizer):
        factory = build_countdown_cycle_grammar_factory(tokenizer)
        g = factory(allow_answer=True)
        text = "<answer> 5 + 3 </answer>"
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        for tok in ids:
            assert tok in g.valid_token_ids()
            g.advance(tok)
        assert g.is_complete()
        assert g.is_terminal()

    def test_answer_with_parens(self, tokenizer):
        factory = build_countdown_cycle_grammar_factory(tokenizer)
        g = factory(allow_answer=True)
        text = "<answer> (5 + 3) * 2 = 16 </answer>"
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        for tok in ids:
            assert tok in g.valid_token_ids()
            g.advance(tok)
        assert g.is_terminal()

    def test_answer_blocked(self, tokenizer):
        factory = build_countdown_cycle_grammar_factory(tokenizer)
        g = factory(allow_answer=False)
        g.advance(27)  # <
        valid = g.valid_token_ids()
        assert 9217 not in valid  # answer token blocked
        assert 37309 in valid  # SCR allowed

    def test_force_answer(self, tokenizer):
        factory = build_countdown_cycle_grammar_factory(tokenizer)
        g = factory(allow_answer=True, force_answer=True)
        g.advance(27)  # <
        valid = g.valid_token_ids()
        assert 37309 not in valid  # SCR blocked
        assert 9217 in valid  # answer forced

    def test_multidigit_scratch(self, tokenizer):
        factory = build_countdown_cycle_grammar_factory(tokenizer)
        g = factory(allow_answer=False)
        text = "<SCRATCH> 100 * 25 = 2500 </SCRATCH>"
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        for tok in ids:
            assert tok in g.valid_token_ids()
            g.advance(tok)
        assert g.is_complete()
        assert not g.is_terminal()

    def test_all_operators(self, tokenizer):
        factory = build_countdown_cycle_grammar_factory(tokenizer)
        for op in ["+", "-", "*", "/"]:
            g = factory(allow_answer=False)
            text = f"<SCRATCH> 3 {op} 2 = 1 </SCRATCH>"
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            for tok in ids:
                assert tok in g.valid_token_ids()
                g.advance(tok)
            assert g.is_complete()

    def test_not_complete_mid_sequence(self, tokenizer):
        factory = build_countdown_cycle_grammar_factory(tokenizer)
        g = factory(allow_answer=False)
        text = "<SCRATCH> 5 +"
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        for tok in ids:
            g.advance(tok)
        assert not g.is_complete()


# ---------------------------------------------------------------------------
# Generator tests
# ---------------------------------------------------------------------------


class TestGenerateNoiseReasoningTokens:
    def test_output_shapes(self, tiny_model, tokenizer):
        torch.manual_seed(42)
        tiny_model.eval()
        factory = build_countdown_cycle_grammar_factory(tokenizer)
        B, k = 2, 4

        output = generate_noise_reasoning_tokens(
            net=tiny_model,
            token_ids=torch.randint(0, 1000, (B, 10)),
            grammar_factory=factory,
            n_noise_per_cycle=k,
            noise_std=1.0,
            max_cycles=5,
            min_cycles=1,
            max_tokens_per_cycle=15,
            temperature=1.0,
        )

        assert isinstance(output, NoiseReasoningGeneratorOutput)
        assert output.hard_token_ids.shape[0] == B
        assert output.hard_token_ids.shape[1] == 5
        assert output.hard_token_ids.shape[2] == 15
        assert output.hard_token_lengths.shape == (B, 5)
        assert output.n_cycles.shape == (B,)
        assert output.noise_vectors.shape == (B, 5, k, tiny_model.d)
        assert output.cycle_is_terminal.shape == (B, 5)

    def test_last_cycle_forced_answer(self, tiny_model, tokenizer):
        torch.manual_seed(42)
        tiny_model.eval()
        factory = build_countdown_cycle_grammar_factory(tokenizer)

        output = generate_noise_reasoning_tokens(
            net=tiny_model,
            token_ids=torch.randint(0, 1000, (1, 8)),
            grammar_factory=factory,
            n_noise_per_cycle=4,
            noise_std=1.0,
            max_cycles=3,
            min_cycles=1,
            max_tokens_per_cycle=25,
            temperature=1.0,
        )

        nc = output.n_cycles[0].item()
        last_hl = output.hard_token_lengths[0, nc - 1].item()
        last_toks = output.hard_token_ids[0, nc - 1, :last_hl].tolist()
        decoded = tokenizer.decode(last_toks)
        assert "<answer>" in decoded or last_toks[1] == 9217

    def test_min_cycles_prevents_early_answer(self, tiny_model, tokenizer):
        torch.manual_seed(42)
        tiny_model.eval()
        factory = build_countdown_cycle_grammar_factory(tokenizer)

        output = generate_noise_reasoning_tokens(
            net=tiny_model,
            token_ids=torch.randint(0, 1000, (1, 8)),
            grammar_factory=factory,
            n_noise_per_cycle=4,
            noise_std=1.0,
            max_cycles=5,
            min_cycles=3,
            max_tokens_per_cycle=15,
            temperature=1.0,
        )

        nc = output.n_cycles[0].item()
        assert nc >= 3

    def test_noise_vectors_stored(self, tiny_model, tokenizer):
        torch.manual_seed(42)
        tiny_model.eval()
        factory = build_countdown_cycle_grammar_factory(tokenizer)

        output = generate_noise_reasoning_tokens(
            net=tiny_model,
            token_ids=torch.randint(0, 1000, (1, 8)),
            grammar_factory=factory,
            n_noise_per_cycle=4,
            noise_std=1.0,
            max_cycles=3,
            min_cycles=1,
            max_tokens_per_cycle=15,
            temperature=1.0,
        )

        nc = output.n_cycles[0].item()
        for c in range(nc):
            assert output.noise_vectors[0, c].abs().sum() > 0

    def test_different_noise_different_output(self, tiny_model, tokenizer):
        tiny_model.eval()
        factory = build_countdown_cycle_grammar_factory(tokenizer)
        prompt = torch.randint(0, 1000, (1, 8))

        torch.manual_seed(1)
        out1 = generate_noise_reasoning_tokens(
            net=tiny_model,
            token_ids=prompt,
            grammar_factory=factory,
            n_noise_per_cycle=4,
            noise_std=1.0,
            max_cycles=3,
            min_cycles=1,
            max_tokens_per_cycle=15,
            temperature=1.0,
        )

        torch.manual_seed(2)
        out2 = generate_noise_reasoning_tokens(
            net=tiny_model,
            token_ids=prompt,
            grammar_factory=factory,
            n_noise_per_cycle=4,
            noise_std=1.0,
            max_cycles=3,
            min_cycles=1,
            max_tokens_per_cycle=15,
            temperature=1.0,
        )

        assert not torch.equal(out1.noise_vectors, out2.noise_vectors)


# ---------------------------------------------------------------------------
# Stacking tests
# ---------------------------------------------------------------------------


class TestStackAndPadNoiseReasoning:
    def test_shapes(self):
        B, C, T, k, D = 2, 3, 5, 4, 16
        hard_ids = [torch.randint(0, 100, (B, C, T)) for _ in range(2)]
        hard_lengths = [torch.randint(1, T, (B, C)) for _ in range(2)]
        n_cycles = [torch.tensor([2, 3]) for _ in range(2)]
        noise = [torch.randn(B, C, k, D) for _ in range(2)]

        s_ids, s_len, s_nc, s_noise = stack_and_pad_noise_reasoning(
            hard_ids,
            hard_lengths,
            n_cycles,
            noise,
            pad_token_id=0,
        )

        assert s_ids.shape == (B, 2, C, T)
        assert s_len.shape == (B, 2, C)
        assert s_nc.shape == (B, 2)
        assert s_noise.shape == (B, 2, C, k, D)


# ---------------------------------------------------------------------------
# Log prob computation tests
# ---------------------------------------------------------------------------


class TestComputeNoiseReasoningLogProbs:
    def _make_inputs(self, net, B=1, G=1, C=2, T_max=5, k=4):
        prompt = torch.randint(0, 1000, (B, 8))
        attn_mask = torch.ones(B, 8, dtype=torch.bool)
        hard_ids = torch.randint(0, 1000, (B, G, C, T_max))
        hard_lengths = torch.full((B, G, C), 3, dtype=torch.long)
        n_cycles = torch.full((B, G), C, dtype=torch.long)
        noise = torch.randn(B, G, C, k, net.d)
        return prompt, attn_mask, hard_ids, hard_lengths, n_cycles, noise

    def test_output_shapes(self, tiny_model):
        torch.manual_seed(42)
        tiny_model.train()
        B, G, C = 2, 2, 3
        prompt, attn_mask, hard_ids, hard_lengths, n_cycles, noise = self._make_inputs(
            tiny_model, B=B, G=G, C=C
        )

        log_probs, mask = compute_noise_reasoning_log_probs(
            net=tiny_model,
            prompt_token_ids=prompt,
            attention_mask=attn_mask,
            hard_token_ids=hard_ids,
            hard_token_lengths=hard_lengths,
            n_cycles=n_cycles,
            noise_vectors=noise,
        )

        assert log_probs.shape == (B, G, C)
        assert mask.shape == (B, G, C)
        assert mask.all()

    def test_completion_mask_respects_n_cycles(self, tiny_model):
        torch.manual_seed(42)
        tiny_model.eval()
        B, G, C = 1, 1, 4
        prompt, attn_mask, hard_ids, hard_lengths, n_cycles, noise = self._make_inputs(
            tiny_model, B=B, G=G, C=C
        )
        n_cycles[:] = 2

        with torch.no_grad():
            _, mask = compute_noise_reasoning_log_probs(
                net=tiny_model,
                prompt_token_ids=prompt,
                attention_mask=attn_mask,
                hard_token_ids=hard_ids,
                hard_token_lengths=hard_lengths,
                n_cycles=n_cycles,
                noise_vectors=noise,
            )

        assert mask[0, 0, 0] and mask[0, 0, 1]
        assert not mask[0, 0, 2] and not mask[0, 0, 3]

    def test_gradient_flows(self, tiny_model):
        torch.manual_seed(42)
        tiny_model.train()
        prompt, attn_mask, hard_ids, hard_lengths, n_cycles, noise = self._make_inputs(
            tiny_model
        )

        log_probs, mask = compute_noise_reasoning_log_probs(
            net=tiny_model,
            prompt_token_ids=prompt,
            attention_mask=attn_mask,
            hard_token_ids=hard_ids,
            hard_token_lengths=hard_lengths,
            n_cycles=n_cycles,
            noise_vectors=noise,
        )

        assert log_probs.grad_fn is not None

        loss = (log_probs * mask).sum()
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in tiny_model.parameters()
        )
        assert has_grad, "No gradients flowed through noise reasoning log probs"

    def test_log_probs_negative(self, tiny_model):
        torch.manual_seed(42)
        tiny_model.eval()
        prompt, attn_mask, hard_ids, hard_lengths, n_cycles, noise = self._make_inputs(
            tiny_model
        )

        with torch.no_grad():
            log_probs, mask = compute_noise_reasoning_log_probs(
                net=tiny_model,
                prompt_token_ids=prompt,
                attention_mask=attn_mask,
                hard_token_ids=hard_ids,
                hard_token_lengths=hard_lengths,
                n_cycles=n_cycles,
                noise_vectors=noise,
            )

        assert (log_probs[mask] <= 0).all()

    def test_different_noise_different_log_probs(self, tiny_model):
        torch.manual_seed(42)
        tiny_model.eval()
        prompt, attn_mask, hard_ids, hard_lengths, n_cycles, noise1 = self._make_inputs(
            tiny_model
        )
        noise2 = torch.randn_like(noise1) * 2.0

        with torch.no_grad():
            lp1, _ = compute_noise_reasoning_log_probs(
                net=tiny_model,
                prompt_token_ids=prompt,
                attention_mask=attn_mask,
                hard_token_ids=hard_ids,
                hard_token_lengths=hard_lengths,
                n_cycles=n_cycles,
                noise_vectors=noise1,
            )
            lp2, _ = compute_noise_reasoning_log_probs(
                net=tiny_model,
                prompt_token_ids=prompt,
                attention_mask=attn_mask,
                hard_token_ids=hard_ids,
                hard_token_lengths=hard_lengths,
                n_cycles=n_cycles,
                noise_vectors=noise2,
            )

        assert not torch.allclose(lp1, lp2)


# ---------------------------------------------------------------------------
# Training loop tests
# ---------------------------------------------------------------------------


class TestTrainNoiseReasoningGrpo:
    def test_training_completes(self, tiny_model, tokenizer):
        torch.manual_seed(42)
        env = CountdownEnv(seed=42, n_ops=2, n_total=3, n_larges=0)
        factory = build_countdown_cycle_grammar_factory(tokenizer)
        opt = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)

        train_noise_reasoning_grpo(
            net=tiny_model,
            opt=opt,
            env=env,
            reward_fn=weighted_reward([("correct", 1.0, countdown_correct)]),
            state_to_str=countdown_state_to_str,
            tokenizer=tokenizer,
            extractor=extract_from_answer_tags,
            eos_token_id=EOS_TOKEN_ID,
            pad_token_id=PAD_TOKEN_ID,
            grammar_factory=factory,
            n_noise_per_cycle=4,
            noise_std=1.0,
            temperature=1.0,
            max_cycles=3,
            min_cycles=1,
            max_tokens_per_cycle=15,
            beta=0.0,
            eps=None,
            max_episodes=2,
            update_ref_net_batch_cadence=None,
            batch_size=1,
            group_size=2,
            accumulation_steps=1,
            max_grad_norm=1.0,
        )


# ---------------------------------------------------------------------------
# Rollout tests
# ---------------------------------------------------------------------------


class TestGenerateNoiseReasoningRolloutBatch:
    def test_rollout_shapes(self, tiny_model, tokenizer):
        torch.manual_seed(42)
        env = CountdownEnv(seed=42, n_ops=2, n_total=3, n_larges=0)
        factory = build_countdown_cycle_grammar_factory(tokenizer)
        B, G = 2, 3

        rollout = generate_noise_reasoning_rollout_batch(
            net=tiny_model,
            env=env,
            reward_fn=weighted_reward([("correct", 1.0, countdown_correct)]),
            state_to_str=countdown_state_to_str,
            tokenizer=tokenizer,
            extractor=extract_from_answer_tags,
            eos_token_id=EOS_TOKEN_ID,
            pad_token_id=PAD_TOKEN_ID,
            grammar_factory=factory,
            batch_size=B,
            group_size=G,
            n_noise_per_cycle=4,
            noise_std=1.0,
            temperature=1.0,
            max_cycles=3,
            min_cycles=1,
            max_tokens_per_cycle=15,
        )

        assert len(rollout.hard_token_ids) == G
        assert len(rollout.noise_vectors) == G
        assert rollout.rewards.shape == (G, B)
        assert len(rollout.output_strs) == G
        assert len(rollout.output_strs[0]) == B
        assert rollout.prompt_token_ids.shape[0] == B
        for g in range(G):
            assert rollout.hard_token_ids[g].shape[0] == B
            assert rollout.noise_vectors[g].shape[0] == B
            assert rollout.noise_vectors[g].shape[2] == 4
            assert rollout.noise_vectors[g].shape[3] == tiny_model.d
