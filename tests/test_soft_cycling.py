import torch

from dialectic.llm.generate import (
    SoftCyclingGeneratorOutput,
    generate_soft_cycling_tokens,
)
from dialectic.rl.env import Countdown, CountdownEnv
from dialectic.rl.extractors import extract_from_answer_tags
from dialectic.rl.grammar import build_countdown_grammar_specs
from dialectic.rl.reward import countdown_correct, weighted_reward
from dialectic.rl.rollout import generate_soft_cycling_rollout_batch
from dialectic.rl.train import (
    compute_soft_cycling_log_probs,
    make_variable_length_per_cycle_backward_callback,
    stack_and_pad_soft_cycling,
    train_soft_cycling_grpo,
)

EOS_TOKEN_ID = 151645
PAD_TOKEN_ID = 151643


def countdown_state_to_str(data: Countdown) -> str:
    return data.prompt


# ---------------------------------------------------------------------------
# Grammar tests
# ---------------------------------------------------------------------------


class TestCountdownStepGrammar:
    def test_valid_sequence(self, tokenizer):
        specs = build_countdown_grammar_specs(tokenizer)
        g = specs[0].grammar_factory()
        text = " 5 + 2 = 7 </SCRATCH>"
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        for tok in ids:
            assert tok in g.valid_token_ids()
            g.advance(tok)
        assert g.is_complete()

    def test_multidigit_sequence(self, tokenizer):
        specs = build_countdown_grammar_specs(tokenizer)
        g = specs[0].grammar_factory()
        text = " 100 - 25 = 75 </SCRATCH>"
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        for tok in ids:
            assert tok in g.valid_token_ids()
            g.advance(tok)
        assert g.is_complete()

    def test_all_operators(self, tokenizer):
        specs = build_countdown_grammar_specs(tokenizer)
        for op_str in ["+", "-", "*", "/"]:
            g = specs[0].grammar_factory()
            text = f" 3 {op_str} 2 = 1 </SCRATCH>"
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            for tok in ids:
                assert tok in g.valid_token_ids()
                g.advance(tok)
            assert g.is_complete()

    def test_reset(self, tokenizer):
        specs = build_countdown_grammar_specs(tokenizer)
        g = specs[0].grammar_factory()
        text = " 5 + 2 = 7 </SCRATCH>"
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        for tok in ids:
            g.advance(tok)
        assert g.is_complete()
        g.reset()
        assert not g.is_complete()
        assert 220 in g.valid_token_ids()

    def test_not_complete_mid_sequence(self, tokenizer):
        specs = build_countdown_grammar_specs(tokenizer)
        g = specs[0].grammar_factory()
        text = " 5 + 2"
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        for tok in ids:
            g.advance(tok)
        assert not g.is_complete()


class TestCountdownAnswerGrammar:
    def test_simple_expression(self, tokenizer):
        specs = build_countdown_grammar_specs(tokenizer)
        g = specs[1].grammar_factory()
        text = " 5 * 3 - 2 </answer>"
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        for tok in ids:
            assert tok in g.valid_token_ids()
            g.advance(tok)
        assert g.is_complete()

    def test_expression_with_parens(self, tokenizer):
        specs = build_countdown_grammar_specs(tokenizer)
        g = specs[1].grammar_factory()
        text = " 5 * (2 + 3) = 25 </answer>"
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        for tok in ids:
            assert tok in g.valid_token_ids(), f"Token {tok} not valid"
            g.advance(tok)
        assert g.is_complete()

    def test_not_complete_without_close(self, tokenizer):
        specs = build_countdown_grammar_specs(tokenizer)
        g = specs[1].grammar_factory()
        text = " 5 * 3"
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        for tok in ids:
            g.advance(tok)
        assert not g.is_complete()


class TestGrammarSpecs:
    def test_trigger_patterns(self, tokenizer):
        specs = build_countdown_grammar_specs(tokenizer)
        assert len(specs) == 2
        assert not specs[0].is_terminal
        assert specs[1].is_terminal

        scratch_ids = tokenizer.encode("<SCRATCH>", add_special_tokens=False).ids
        answer_ids = tokenizer.encode("<answer>", add_special_tokens=False).ids
        assert specs[0].trigger_token_ids == tuple(scratch_ids)
        assert specs[1].trigger_token_ids == tuple(answer_ids)


# ---------------------------------------------------------------------------
# Generator tests
# ---------------------------------------------------------------------------


class TestGenerateSoftCyclingTokens:
    def test_output_shapes(self, tiny_model, tokenizer):
        torch.manual_seed(42)
        tiny_model.eval()
        specs = build_countdown_grammar_specs(tokenizer)
        B = 2
        prompt = torch.randint(0, 1000, (B, 10))

        output = generate_soft_cycling_tokens(
            net=tiny_model,
            token_ids=prompt,
            grammar_specs=specs,
            max_tokens_generated=50,
            max_cycles=5,
            max_tokens_per_cycle=10,
            temperature=1.0,
            min_soft_steps=3,
        )

        assert isinstance(output, SoftCyclingGeneratorOutput)
        assert output.hard_token_ids.shape[0] == B
        assert output.hard_token_ids.shape[1] == 5
        assert output.hard_token_ids.shape[2] == 10
        assert output.hard_token_lengths.shape == (B, 5)
        assert output.n_cycles.shape == (B,)
        assert output.soft_lengths.shape == (B, 5)
        assert output.shadow_ids.shape[0] == B

    def test_max_tokens_respected(self, tiny_model, tokenizer):
        torch.manual_seed(42)
        tiny_model.eval()
        specs = build_countdown_grammar_specs(tokenizer)
        max_tok = 20

        output = generate_soft_cycling_tokens(
            net=tiny_model,
            token_ids=torch.randint(0, 1000, (1, 5)),
            grammar_specs=specs,
            max_tokens_generated=max_tok,
            max_cycles=10,
            max_tokens_per_cycle=10,
            temperature=1.0,
        )

        assert output.shadow_ids.shape[1] <= max_tok

    def test_gumbel_mode(self, tiny_model, tokenizer):
        torch.manual_seed(42)
        tiny_model.eval()
        specs = build_countdown_grammar_specs(tokenizer)

        output = generate_soft_cycling_tokens(
            net=tiny_model,
            token_ids=torch.randint(0, 1000, (1, 5)),
            grammar_specs=specs,
            max_tokens_generated=30,
            max_cycles=3,
            max_tokens_per_cycle=10,
            temperature=1.0,
            use_gumbel=True,
        )

        assert isinstance(output, SoftCyclingGeneratorOutput)
        assert output.shadow_ids.shape[0] == 1

    def test_soft_lengths_consistent_with_n_cycles(self, tiny_model, tokenizer):
        torch.manual_seed(42)
        tiny_model.eval()
        specs = build_countdown_grammar_specs(tokenizer)

        output = generate_soft_cycling_tokens(
            net=tiny_model,
            token_ids=torch.randint(0, 1000, (2, 8)),
            grammar_specs=specs,
            max_tokens_generated=40,
            max_cycles=5,
            max_tokens_per_cycle=10,
            temperature=1.0,
            min_soft_steps=2,
        )

        for b in range(2):
            nc = output.n_cycles[b].item()
            for c in range(nc):
                assert output.soft_lengths[b, c] > 0


# ---------------------------------------------------------------------------
# Stacking / padding tests
# ---------------------------------------------------------------------------


class TestStackAndPadSoftCycling:
    def test_shapes(self):
        B, C, T = 2, 3, 5
        hard_ids = [torch.randint(0, 100, (B, C, T)) for _ in range(2)]
        hard_lengths = [torch.randint(1, T, (B, C)) for _ in range(2)]
        n_cycles = [torch.tensor([2, 3]) for _ in range(2)]
        soft_lengths = [torch.randint(1, 10, (B, C)) for _ in range(2)]

        s_ids, s_len, s_nc, s_soft = stack_and_pad_soft_cycling(
            hard_ids,
            hard_lengths,
            n_cycles,
            soft_lengths,
            pad_token_id=0,
        )

        assert s_ids.shape == (B, 2, C, T)
        assert s_len.shape == (B, 2, C)
        assert s_nc.shape == (B, 2)
        assert s_soft.shape == (B, 2, C)

    def test_values_preserved(self):
        hard_ids = [
            torch.tensor([[[10, 20, 0], [30, 0, 0]]]),
            torch.tensor([[[40, 50, 60], [70, 80, 0]]]),
        ]
        hard_lengths = [torch.tensor([[2, 1]]), torch.tensor([[3, 2]])]
        n_cycles = [torch.tensor([2]), torch.tensor([2])]
        soft_lengths = [torch.tensor([[5, 3]]), torch.tensor([[4, 6]])]

        s_ids, s_len, s_nc, s_soft = stack_and_pad_soft_cycling(
            hard_ids,
            hard_lengths,
            n_cycles,
            soft_lengths,
            pad_token_id=0,
        )

        assert s_ids[0, 0, 0, 0] == 10
        assert s_ids[0, 1, 0, 0] == 40
        assert s_soft[0, 0, 0] == 5
        assert s_soft[0, 1, 1] == 6


# ---------------------------------------------------------------------------
# BPTT log prob tests
# ---------------------------------------------------------------------------


class TestComputeSoftCyclingLogProbs:
    def _make_inputs(self, B=1, G=1, C=2, T_max=3, soft_len=4, prompt_len=8):
        prompt = torch.randint(0, 1000, (B, prompt_len))
        attn_mask = torch.ones(B, prompt_len, dtype=torch.bool)
        hard_ids = torch.randint(0, 1000, (B, G, C, T_max))
        hard_lengths = torch.full((B, G, C), 2, dtype=torch.long)
        n_cycles = torch.full((B, G), C, dtype=torch.long)
        soft_lengths = torch.full((B, G, C), soft_len, dtype=torch.long)
        return prompt, attn_mask, hard_ids, hard_lengths, n_cycles, soft_lengths

    def test_output_shapes(self, tiny_model):
        torch.manual_seed(42)
        tiny_model.train()
        B, G, C = 2, 2, 3
        prompt, attn_mask, hard_ids, hard_lengths, n_cycles, soft_lengths = (
            self._make_inputs(B=B, G=G, C=C)
        )

        log_probs, mask = compute_soft_cycling_log_probs(
            net=tiny_model,
            prompt_token_ids=prompt,
            attention_mask=attn_mask,
            hard_token_ids=hard_ids,
            hard_token_lengths=hard_lengths,
            n_cycles=n_cycles,
            soft_lengths=soft_lengths,
            temperature=1.0,
        )

        assert log_probs.shape == (B, G, C)
        assert mask.shape == (B, G, C)
        assert mask.all()

    def test_completion_mask_respects_n_cycles(self, tiny_model):
        torch.manual_seed(42)
        tiny_model.eval()
        B, G, C = 1, 1, 4
        prompt, attn_mask, hard_ids, hard_lengths, n_cycles, soft_lengths = (
            self._make_inputs(B=B, G=G, C=C)
        )
        n_cycles[:] = 2

        with torch.no_grad():
            _, mask = compute_soft_cycling_log_probs(
                net=tiny_model,
                prompt_token_ids=prompt,
                attention_mask=attn_mask,
                hard_token_ids=hard_ids,
                hard_token_lengths=hard_lengths,
                n_cycles=n_cycles,
                soft_lengths=soft_lengths,
                temperature=1.0,
            )

        assert mask[0, 0, 0] and mask[0, 0, 1]
        assert not mask[0, 0, 2] and not mask[0, 0, 3]

    def test_gradient_flows_softmax(self, tiny_model):
        torch.manual_seed(42)
        tiny_model.train()
        prompt, attn_mask, hard_ids, hard_lengths, n_cycles, soft_lengths = (
            self._make_inputs()
        )

        log_probs, mask = compute_soft_cycling_log_probs(
            net=tiny_model,
            prompt_token_ids=prompt,
            attention_mask=attn_mask,
            hard_token_ids=hard_ids,
            hard_token_lengths=hard_lengths,
            n_cycles=n_cycles,
            soft_lengths=soft_lengths,
            temperature=1.0,
            use_gumbel=False,
        )

        loss = (log_probs * mask).sum()
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in tiny_model.parameters()
        )
        assert has_grad, "No gradients flowed with softmax"

    def test_gradient_flows_gumbel(self, tiny_model):
        torch.manual_seed(42)
        tiny_model.train()
        prompt, attn_mask, hard_ids, hard_lengths, n_cycles, soft_lengths = (
            self._make_inputs()
        )

        log_probs, mask = compute_soft_cycling_log_probs(
            net=tiny_model,
            prompt_token_ids=prompt,
            attention_mask=attn_mask,
            hard_token_ids=hard_ids,
            hard_token_lengths=hard_lengths,
            n_cycles=n_cycles,
            soft_lengths=soft_lengths,
            temperature=1.0,
            use_gumbel=True,
        )

        loss = (log_probs * mask).sum()
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in tiny_model.parameters()
        )
        assert has_grad, "No gradients flowed with gumbel softmax"

    def test_bptt_window(self, tiny_model):
        torch.manual_seed(42)
        tiny_model.train()
        prompt, attn_mask, hard_ids, hard_lengths, n_cycles, soft_lengths = (
            self._make_inputs(soft_len=6)
        )

        log_probs, mask = compute_soft_cycling_log_probs(
            net=tiny_model,
            prompt_token_ids=prompt,
            attention_mask=attn_mask,
            hard_token_ids=hard_ids,
            hard_token_lengths=hard_lengths,
            n_cycles=n_cycles,
            soft_lengths=soft_lengths,
            temperature=1.0,
            soft_bptt_window=2,
        )

        loss = (log_probs * mask).sum()
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in tiny_model.parameters()
        )
        assert has_grad, "No gradients flowed with BPTT window"

    def test_per_cycle_backward_callback(self, tiny_model):
        torch.manual_seed(42)
        tiny_model.train()
        B, G, C = 1, 1, 2
        prompt, attn_mask, hard_ids, hard_lengths, n_cycles, soft_lengths = (
            self._make_inputs(B=B, G=G, C=C)
        )

        advs = torch.ones(B, G, 1)
        completion_mask = torch.ones(B, G, C, dtype=torch.bool)

        callback = make_variable_length_per_cycle_backward_callback(
            B=B,
            G=G,
            advs=advs,
            completion_mask=completion_mask,
            hard_token_lengths=hard_lengths,
            n_cycles=n_cycles,
            old_log_probs=None,
            ref_log_probs=None,
            beta=0.0,
            eps=None,
            normalize_by_sequence_length=True,
            loss_scale=1.0,
        )

        log_probs, mask = compute_soft_cycling_log_probs(
            net=tiny_model,
            prompt_token_ids=prompt,
            attention_mask=attn_mask,
            hard_token_ids=hard_ids,
            hard_token_lengths=hard_lengths,
            n_cycles=n_cycles,
            soft_lengths=soft_lengths,
            temperature=1.0,
            cycle_callback=callback,
        )

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in tiny_model.parameters()
        )
        assert has_grad, "Per-cycle backward did not produce gradients"

    def test_zero_hard_tokens_no_crash(self, tiny_model):
        torch.manual_seed(42)
        tiny_model.train()
        B, G, C = 1, 1, 1
        prompt = torch.randint(0, 1000, (B, 8))
        attn_mask = torch.ones(B, 8, dtype=torch.bool)
        hard_ids = torch.randint(0, 1000, (B, G, C, 3))
        hard_lengths = torch.zeros(B, G, C, dtype=torch.long)
        n_cycles = torch.ones(B, G, dtype=torch.long)
        soft_lengths = torch.full((B, G, C), 5, dtype=torch.long)

        log_probs, mask = compute_soft_cycling_log_probs(
            net=tiny_model,
            prompt_token_ids=prompt,
            attention_mask=attn_mask,
            hard_token_ids=hard_ids,
            hard_token_lengths=hard_lengths,
            n_cycles=n_cycles,
            soft_lengths=soft_lengths,
            temperature=1.0,
        )

        loss = (log_probs * mask).sum()
        loss.backward()


# ---------------------------------------------------------------------------
# Training loop tests
# ---------------------------------------------------------------------------


class TestTrainSoftCyclingGrpo:
    def test_training_loop_completes(self, tiny_model, tokenizer):
        torch.manual_seed(42)
        env = CountdownEnv(seed=42, n_ops=2, n_total=3, n_larges=0)
        specs = build_countdown_grammar_specs(tokenizer)
        opt = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)

        train_soft_cycling_grpo(
            net=tiny_model,
            opt=opt,
            env=env,
            reward_fn=weighted_reward([("correct", 1.0, countdown_correct)]),
            state_to_str=countdown_state_to_str,
            tokenizer=tokenizer,
            extractor=extract_from_answer_tags,
            eos_token_id=EOS_TOKEN_ID,
            pad_token_id=PAD_TOKEN_ID,
            grammar_specs=specs,
            temperature=1.0,
            max_tokens_generated=30,
            max_cycles=3,
            max_tokens_per_cycle=10,
            beta=0.0,
            eps=None,
            max_episodes=2,
            update_ref_net_batch_cadence=None,
            batch_size=2,
            group_size=2,
            min_soft_steps=2,
            accumulation_steps=1,
            max_grad_norm=1.0,
        )

    def test_training_with_kl(self, tiny_model, tokenizer):
        torch.manual_seed(42)
        env = CountdownEnv(seed=42, n_ops=2, n_total=3, n_larges=0)
        specs = build_countdown_grammar_specs(tokenizer)
        opt = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)

        train_soft_cycling_grpo(
            net=tiny_model,
            opt=opt,
            env=env,
            reward_fn=weighted_reward([("correct", 1.0, countdown_correct)]),
            state_to_str=countdown_state_to_str,
            tokenizer=tokenizer,
            extractor=extract_from_answer_tags,
            eos_token_id=EOS_TOKEN_ID,
            pad_token_id=PAD_TOKEN_ID,
            grammar_specs=specs,
            temperature=1.0,
            max_tokens_generated=30,
            max_cycles=3,
            max_tokens_per_cycle=10,
            beta=0.01,
            eps=None,
            max_episodes=2,
            update_ref_net_batch_cadence=5,
            batch_size=2,
            group_size=2,
            min_soft_steps=2,
            accumulation_steps=1,
            max_grad_norm=1.0,
        )

    def test_training_with_gumbel(self, tiny_model, tokenizer):
        torch.manual_seed(42)
        env = CountdownEnv(seed=42, n_ops=2, n_total=3, n_larges=0)
        specs = build_countdown_grammar_specs(tokenizer)
        opt = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)

        train_soft_cycling_grpo(
            net=tiny_model,
            opt=opt,
            env=env,
            reward_fn=weighted_reward([("correct", 1.0, countdown_correct)]),
            state_to_str=countdown_state_to_str,
            tokenizer=tokenizer,
            extractor=extract_from_answer_tags,
            eos_token_id=EOS_TOKEN_ID,
            pad_token_id=PAD_TOKEN_ID,
            grammar_specs=specs,
            temperature=1.0,
            max_tokens_generated=30,
            max_cycles=3,
            max_tokens_per_cycle=10,
            beta=0.0,
            eps=None,
            max_episodes=2,
            update_ref_net_batch_cadence=None,
            batch_size=2,
            group_size=2,
            use_gumbel=True,
            min_soft_steps=2,
            accumulation_steps=1,
            max_grad_norm=1.0,
        )


# ---------------------------------------------------------------------------
# Rollout tests
# ---------------------------------------------------------------------------


class TestGenerateSoftCyclingRolloutBatch:
    def test_rollout_shapes(self, tiny_model, tokenizer):
        torch.manual_seed(42)
        env = CountdownEnv(seed=42, n_ops=2, n_total=3, n_larges=0)
        specs = build_countdown_grammar_specs(tokenizer)
        B, G = 2, 3

        rollout = generate_soft_cycling_rollout_batch(
            net=tiny_model,
            env=env,
            reward_fn=weighted_reward([("correct", 1.0, countdown_correct)]),
            state_to_str=countdown_state_to_str,
            tokenizer=tokenizer,
            extractor=extract_from_answer_tags,
            eos_token_id=EOS_TOKEN_ID,
            pad_token_id=PAD_TOKEN_ID,
            grammar_specs=specs,
            batch_size=B,
            group_size=G,
            temperature=1.0,
            max_tokens_generated=30,
            max_cycles=3,
            max_tokens_per_cycle=10,
            min_soft_steps=2,
        )

        assert len(rollout.hard_token_ids) == G
        assert len(rollout.soft_lengths) == G
        assert rollout.rewards.shape == (G, B)
        assert len(rollout.output_strs) == G
        assert len(rollout.output_strs[0]) == B
        assert rollout.prompt_token_ids.shape[0] == B
        assert rollout.attention_mask.shape[0] == B
        for g in range(G):
            assert rollout.hard_token_ids[g].shape[0] == B
            assert rollout.n_cycles[g].shape == (B,)
