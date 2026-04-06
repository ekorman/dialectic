import torch
from torch.nn.functional import scaled_dot_product_attention

from dialectic.experiments.launchers.inverse_cot import (
    PreTokenizedCompletion,
    PreTokenizedPrompt,
    _build_contrastive_batch,
    _compute_contrastive_loss,
    _compute_nll_loss,
    _parse_cot_and_answer,
)
from dialectic.llm.components.attention import attention
from dialectic.llm.generate import generate_hard_tokens
from dialectic.llm.inverse_cot import InverseCotModel, create_prefix_lm_mask
from dialectic.llm.qwen import create_qwen

TINY_QWEN_KWARGS = dict(
    d=32,
    vocab_size=256,
    n_decoder_layers=2,
    attn_head_d=16,
    attn_num_heads=4,
    attn_num_kv_heads=2,
    mlp_hidden_d=64,
    tie_weights=True,
    rope_base_value=10000,
)


def _make_p():
    torch.manual_seed(42)
    return create_qwen(**TINY_QWEN_KWARGS)


# ---------- prefix-LM mask ----------


class TestPrefixLmMask:
    def test_shape(self):
        mask = create_prefix_lm_mask(
            torch.tensor([3, 5]), seq_len=8, device=torch.device("cpu")
        )
        assert mask.shape == (2, 1, 8, 8)
        assert mask.dtype == torch.bool

    def test_prefix_bidirectional(self):
        mask = create_prefix_lm_mask(
            torch.tensor([4]), seq_len=8, device=torch.device("cpu")
        )
        m = mask[0, 0]
        assert m[:4, :4].all(), "prefix tokens should attend to all prefix tokens"

    def test_prefix_cannot_see_cot(self):
        mask = create_prefix_lm_mask(
            torch.tensor([4]), seq_len=8, device=torch.device("cpu")
        )
        m = mask[0, 0]
        assert not m[:4, 4:].any(), "prefix tokens should not attend to CoT tokens"

    def test_cot_sees_all_prefix(self):
        mask = create_prefix_lm_mask(
            torch.tensor([4]), seq_len=8, device=torch.device("cpu")
        )
        m = mask[0, 0]
        assert m[4:, :4].all(), "CoT tokens should attend to all prefix tokens"

    def test_cot_causal(self):
        mask = create_prefix_lm_mask(
            torch.tensor([4]), seq_len=8, device=torch.device("cpu")
        )
        m = mask[0, 0]
        cot_cot = m[4:, 4:]
        for i in range(cot_cot.shape[0]):
            for j in range(cot_cot.shape[1]):
                assert cot_cot[i, j] == (j <= i), f"cot[{i},{j}] should be {j <= i}"

    def test_varying_prefix_lengths(self):
        mask = create_prefix_lm_mask(
            torch.tensor([2, 6]), seq_len=8, device=torch.device("cpu")
        )
        # sample 0: prefix=2, cot=6
        m0 = mask[0, 0]
        assert m0[:2, :2].all()
        assert not m0[:2, 2:].any()
        assert m0[2:, :2].all()
        # sample 1: prefix=6, cot=2
        m1 = mask[1, 0]
        assert m1[:6, :6].all()
        assert not m1[:6, 6:].any()
        assert m1[6:, :6].all()

    def test_all_prefix(self):
        mask = create_prefix_lm_mask(
            torch.tensor([8]), seq_len=8, device=torch.device("cpu")
        )
        m = mask[0, 0]
        assert m.all(), "when everything is prefix, all attention should be allowed"

    def test_all_cot(self):
        mask = create_prefix_lm_mask(
            torch.tensor([0]), seq_len=8, device=torch.device("cpu")
        )
        m = mask[0, 0]
        expected = torch.ones(8, 8, dtype=torch.bool).tril()
        assert (m == expected).all(), "when prefix_len=0, should be standard causal"


# ---------- 4D attention mask ----------


class TestAttention4dMask:
    def test_2d_mask_unchanged(self):
        """2D masks should produce identical results to before the change."""
        B, NH, L, D = 2, 4, 6, 8
        q = torch.randn(B, NH, L, D)
        k = torch.randn(B, NH, L, D)
        v = torch.randn(B, NH, L, D)
        mask_2d = torch.ones(B, L, dtype=torch.bool)
        mask_2d[0, :2] = False  # some padding

        result = attention(q, k, v, causal=False, attention_mask=mask_2d)
        expected = scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask_2d.view(B, 1, 1, L),
            is_causal=False,
            enable_gqa=True,
        )
        torch.testing.assert_close(result, expected)

    def test_4d_mask_passthrough(self):
        """4D masks should be passed directly to SDPA."""
        B, NH, L, D = 2, 4, 6, 8
        q = torch.randn(B, NH, L, D)
        k = torch.randn(B, NH, L, D)
        v = torch.randn(B, NH, L, D)
        mask_4d = torch.ones(B, 1, L, L, dtype=torch.bool).tril()

        result = attention(q, k, v, causal=False, attention_mask=mask_4d)
        expected = scaled_dot_product_attention(
            q, k, v, attn_mask=mask_4d, is_causal=False, enable_gqa=True
        )
        torch.testing.assert_close(result, expected)

    def test_4d_equivalent_to_causal(self):
        """A 4D lower-triangular mask should produce the same result as causal=True."""
        B, NH, L, D = 1, 4, 6, 8
        q = torch.randn(B, NH, L, D)
        k = torch.randn(B, NH, L, D)
        v = torch.randn(B, NH, L, D)

        causal_result = attention(q, k, v, causal=True)
        mask_4d = torch.ones(B, 1, L, L, dtype=torch.bool).tril()
        mask_result = attention(q, k, v, causal=False, attention_mask=mask_4d)
        torch.testing.assert_close(causal_result, mask_result)


# ---------- InverseCotModel weight sharing & freezing ----------


class TestInverseCotModelWeights:
    def test_embed_tokens_shared(self):
        p = _make_p()
        q = InverseCotModel(p)
        assert q.embed_tokens is p.embed_tokens

    def test_mlp_shared(self):
        p = _make_p()
        q = InverseCotModel(p)
        for i in range(len(p.layers)):
            assert q.layers[i].mlp is p.layers[i].mlp

    def test_pre_mlp_norm_shared(self):
        p = _make_p()
        q = InverseCotModel(p)
        for i in range(len(p.layers)):
            assert q.layers[i].pre_mlp_norm is p.layers[i].post_attention_layernorm

    def test_frozen_params(self):
        p = _make_p()
        q = InverseCotModel(p)
        for param in q.embed_tokens.parameters():
            assert not param.requires_grad
        for layer in q.layers:
            for param in layer.mlp.parameters():
                assert not param.requires_grad
            for param in layer.pre_mlp_norm.parameters():
                assert not param.requires_grad

    def test_trainable_params(self):
        p = _make_p()
        q = InverseCotModel(p)
        for layer in q.layers:
            for param in layer.self_attn.parameters():
                assert param.requires_grad
            for param in layer.post_mlp_norm.parameters():
                assert param.requires_grad
        for param in q.norm.parameters():
            assert param.requires_grad
        for param in q.lm_head.parameters():
            assert param.requires_grad

    def test_lm_head_is_fresh(self):
        """q's lm_head should NOT be tied to p's embed_tokens."""
        p = _make_p()
        q = InverseCotModel(p)
        assert q.lm_head.weight is not p.embed_tokens.weight

    def test_gradient_does_not_flow_to_frozen(self):
        p = _make_p()
        q = InverseCotModel(p)
        input_ids = torch.randint(0, TINY_QWEN_KWARGS["vocab_size"], (2, 8))
        logits = q(input_ids, return_all_logits=True)
        logits.sum().backward()
        for param in q.embed_tokens.parameters():
            assert param.grad is None
        for layer in q.layers:
            for param in layer.mlp.parameters():
                assert param.grad is None
        assert q.lm_head.weight.grad is not None


# ---------- reversed layer ordering ----------


class TestReversedLayerOrder:
    def test_mlp_runs_before_attention(self):
        """Verify MLP executes before attention by tracking call order."""
        p = _make_p()
        q = InverseCotModel(p)
        layer = q.layers[0]

        call_order = []
        orig_mlp_forward = layer.mlp.forward
        orig_attn_forward = layer.self_attn.forward

        def mlp_hook(*args, **kwargs):
            call_order.append("mlp")
            return orig_mlp_forward(*args, **kwargs)

        def attn_hook(*args, **kwargs):
            call_order.append("attn")
            return orig_attn_forward(*args, **kwargs)

        layer.mlp.forward = mlp_hook
        layer.self_attn.forward = attn_hook

        x = torch.randn(1, 4, TINY_QWEN_KWARGS["d"])
        layer(x)

        assert call_order == ["mlp", "attn"]


# ---------- InverseCotModel forward ----------


class TestInverseCotModelForward:
    def test_output_shape(self):
        p = _make_p()
        q = InverseCotModel(p)
        B, L = 2, 10
        input_ids = torch.randint(0, TINY_QWEN_KWARGS["vocab_size"], (B, L))
        prefix_lengths = torch.tensor([4, 6])
        mask = create_prefix_lm_mask(prefix_lengths, L, input_ids.device)

        logits = q(input_ids, attention_mask=mask, return_all_logits=True)
        assert logits.shape == (B, L, TINY_QWEN_KWARGS["vocab_size"])

    def test_last_token_only(self):
        p = _make_p()
        q = InverseCotModel(p)
        input_ids = torch.randint(0, TINY_QWEN_KWARGS["vocab_size"], (2, 10))
        logits = q(input_ids, return_all_logits=False)
        assert logits.shape == (2, 1, TINY_QWEN_KWARGS["vocab_size"])

    def test_return_hidden_states(self):
        p = _make_p()
        q = InverseCotModel(p)
        input_ids = torch.randint(0, TINY_QWEN_KWARGS["vocab_size"], (2, 10))
        hidden = q(input_ids, return_hidden_states=True)
        assert hidden.shape == (2, 10, TINY_QWEN_KWARGS["d"])


# ---------- generation ----------


class TestInverseCotGeneration:
    def test_kv_cache_generation(self):
        p = _make_p()
        q = InverseCotModel(p)
        prefix = torch.randint(0, TINY_QWEN_KWARGS["vocab_size"], (1, 6))

        output = generate_hard_tokens(
            net=q,
            token_ids=prefix,
            sampling_strategy="greedy",
            eos_token_id=TINY_QWEN_KWARGS["vocab_size"] - 1,
            pad_token_id=TINY_QWEN_KWARGS["vocab_size"] - 2,
            max_tokens_generated=10,
            use_kv_cache=True,
        )
        assert output.tokens.shape[0] == 1
        assert output.tokens.shape[1] >= prefix.shape[1]
        assert output.tokens.shape[1] <= prefix.shape[1] + 10

    def test_batch_generation(self):
        p = _make_p()
        q = InverseCotModel(p)
        B = 3
        prefix = torch.randint(0, TINY_QWEN_KWARGS["vocab_size"], (B, 6))

        output = generate_hard_tokens(
            net=q,
            token_ids=prefix,
            sampling_strategy="greedy",
            eos_token_id=TINY_QWEN_KWARGS["vocab_size"] - 1,
            pad_token_id=TINY_QWEN_KWARGS["vocab_size"] - 2,
            max_tokens_generated=10,
            use_kv_cache=True,
        )
        assert output.tokens.shape[0] == B


# ---------- _parse_cot_and_answer ----------


class TestParseCotAndAnswer:
    def test_think_tags(self):
        text = "<think>\n100 + 50 = 150\n</think>\n\n(100 + 50)"
        result = _parse_cot_and_answer(text)
        assert result is not None
        cot, answer = result
        assert cot == "100 + 50 = 150"
        assert answer == "(100 + 50)"

    def test_think_tags_with_answer_tags(self):
        text = "<think>\nreasoning\n</think>\n\n<answer>42</answer>"
        result = _parse_cot_and_answer(text)
        assert result is not None
        cot, answer = result
        assert cot == "reasoning"
        assert answer == "<answer>42</answer>"

    def test_think_tags_empty_cot(self):
        text = "<think>\n\n</think>\n\n42"
        assert _parse_cot_and_answer(text) is None

    def test_think_tags_empty_answer(self):
        text = "<think>\nreasoning\n</think>"
        assert _parse_cot_and_answer(text) is None

    def test_answer_tags_fallback(self):
        text = "Let me think step by step\n<answer>42 + 8</answer>"
        result = _parse_cot_and_answer(text)
        assert result is not None
        cot, answer = result
        assert cot == "Let me think step by step\n"
        assert answer == "<answer>42 + 8</answer>"

    def test_no_tags(self):
        assert _parse_cot_and_answer("just some text without tags") is None

    def test_answer_tags_empty_cot(self):
        assert _parse_cot_and_answer("<answer>42</answer>") is None

    def test_multiline_think(self):
        text = "<think>\nstep 1\nstep 2\nstep 3\n</think>\n\nexpr"
        result = _parse_cot_and_answer(text)
        assert result is not None
        cot, answer = result
        assert "step 1" in cot
        assert "step 3" in cot
        assert answer == "expr"


# ---------- NLL loss ----------


class TestEosTokenInTrainingData:
    def test_eos_token_present_in_input_ids(self, tokenizer):
        """The actual EOS token ID must appear in the training sequence."""
        eos_token_id = 151645  # Qwen <|im_end|>

        prefix_str = "some prompt <answer>42</answer>"
        cot_str = "Let me think step by step"

        prefix_enc = tokenizer.encode(prefix_str)
        cot_enc = tokenizer.encode(cot_str)
        ids = prefix_enc.ids + cot_enc.ids + [eos_token_id]

        assert ids[-1] == eos_token_id

    def test_string_roundtrip_loses_eos(self, tokenizer):
        """Demonstrate that decode→encode round-trip does NOT preserve the
        special EOS token, which is why we append the ID directly."""
        eos_token_id = 151645
        eos_str = tokenizer.decode([eos_token_id])
        cot_with_eos = "Let me think" + eos_str
        re_encoded = tokenizer.encode(cot_with_eos)
        assert eos_token_id not in re_encoded.ids


class TestNllLoss:
    def test_zero_loss_when_mask_empty(self):
        p = _make_p()
        q = InverseCotModel(p)
        B, L = 2, 10
        input_ids = torch.randint(0, TINY_QWEN_KWARGS["vocab_size"], (B, L))
        prefix_lengths = torch.tensor([L, L])  # all prefix, no CoT
        loss_mask = torch.zeros(B, L, dtype=torch.bool)

        loss, nll, _ = _compute_nll_loss(q, input_ids, prefix_lengths, loss_mask, True)
        assert loss.item() == 0.0

    def test_loss_only_on_masked_positions(self):
        """Loss should differ when mask changes, even with same input."""
        p = _make_p()
        q = InverseCotModel(p)
        B, L = 2, 10
        input_ids = torch.randint(0, TINY_QWEN_KWARGS["vocab_size"], (B, L))
        prefix_lengths = torch.tensor([4, 4])

        mask1 = torch.zeros(B, L, dtype=torch.bool)
        mask1[:, 4:7] = True
        mask2 = torch.zeros(B, L, dtype=torch.bool)
        mask2[:, 7:] = True

        _, nll1, _ = _compute_nll_loss(q, input_ids, prefix_lengths, mask1, True)
        _, nll2, _ = _compute_nll_loss(q, input_ids, prefix_lengths, mask2, True)
        assert nll1 != nll2

    def test_normalize_by_sequence_length(self):
        """Per-sequence normalization should differ from global average."""
        p = _make_p()
        q = InverseCotModel(p)
        B, L = 2, 12
        input_ids = torch.randint(0, TINY_QWEN_KWARGS["vocab_size"], (B, L))
        prefix_lengths = torch.tensor([3, 8])

        positions = torch.arange(L).unsqueeze(0)
        loss_mask = positions >= prefix_lengths.unsqueeze(1)

        _, nll_norm, _ = _compute_nll_loss(
            q, input_ids, prefix_lengths, loss_mask, True
        )
        _, nll_global, _ = _compute_nll_loss(
            q, input_ids, prefix_lengths, loss_mask, False
        )
        # with different CoT lengths (9 vs 4), these should generally differ
        assert nll_norm != nll_global

    def test_loss_is_differentiable(self):
        p = _make_p()
        q = InverseCotModel(p)
        B, L = 2, 10
        input_ids = torch.randint(0, TINY_QWEN_KWARGS["vocab_size"], (B, L))
        prefix_lengths = torch.tensor([4, 4])
        loss_mask = torch.zeros(B, L, dtype=torch.bool)
        loss_mask[:, 4:] = True

        loss, _, _ = _compute_nll_loss(q, input_ids, prefix_lengths, loss_mask, True)
        loss.backward()
        assert q.lm_head.weight.grad is not None


# ---------- contrastive loss ----------


def _make_prompts(n_prompts=2, group_size=4, n_correct=2):
    """Create synthetic PreTokenizedPrompt data for testing."""
    prompts = []
    for i in range(n_prompts):
        completions = []
        for g in range(group_size):
            completions.append(
                PreTokenizedCompletion(
                    answer_ids=[10 + i, 20 + g],
                    cot_ids=[30 + g, 40 + g, 50 + g],
                    is_correct=g < n_correct,
                )
            )
        prompts.append(
            PreTokenizedPrompt(prompt_ids=[1, 2, 3], completions=completions)
        )
    return prompts


class TestContrastiveLoss:
    def test_build_batch_shapes(self):
        prompts = _make_prompts(n_prompts=2, group_size=4)
        input_ids, prefix_lengths, loss_mask, is_correct, group_sizes = (
            _build_contrastive_batch(
                prompts, eos_token_id=99, pad_token_id=0, device=torch.device("cpu")
            )
        )
        assert input_ids.shape[0] == 8  # 2 prompts * 4 completions
        assert prefix_lengths.shape == (8,)
        assert loss_mask.shape == input_ids.shape
        assert is_correct.shape == (8,)
        assert group_sizes == [4, 4]

    def test_build_batch_correctness_mask(self):
        prompts = _make_prompts(n_prompts=1, group_size=4, n_correct=2)
        _, _, _, is_correct, _ = _build_contrastive_batch(
            prompts, eos_token_id=99, pad_token_id=0, device=torch.device("cpu")
        )
        assert is_correct.tolist() == [True, True, False, False]

    def test_contrastive_loss_all_positive(self):
        """When all examples are positive, contrastive loss should be 0."""
        p = _make_p()
        q = InverseCotModel(p)
        prompts = _make_prompts(n_prompts=2, group_size=4, n_correct=4)
        input_ids, prefix_lengths, loss_mask, is_correct, group_sizes = (
            _build_contrastive_batch(
                prompts, eos_token_id=99, pad_token_id=0, device=torch.device("cpu")
            )
        )
        _, metrics = _compute_contrastive_loss(
            q,
            input_ids,
            prefix_lengths,
            loss_mask,
            is_correct,
            group_sizes,
            contrastive_weight=1.0,
        )
        assert metrics["train/contrastive_loss"] == 0.0

    def test_contrastive_loss_with_negatives(self):
        """Contrastive loss should be > 0 when there are negatives."""
        p = _make_p()
        q = InverseCotModel(p)
        prompts = _make_prompts(n_prompts=2, group_size=4, n_correct=2)
        input_ids, prefix_lengths, loss_mask, is_correct, group_sizes = (
            _build_contrastive_batch(
                prompts, eos_token_id=99, pad_token_id=0, device=torch.device("cpu")
            )
        )
        _, metrics = _compute_contrastive_loss(
            q,
            input_ids,
            prefix_lengths,
            loss_mask,
            is_correct,
            group_sizes,
            contrastive_weight=1.0,
        )
        assert metrics["train/contrastive_loss"] > 0.0

    def test_contrastive_loss_is_differentiable(self):
        p = _make_p()
        q = InverseCotModel(p)
        prompts = _make_prompts(n_prompts=2, group_size=4, n_correct=2)
        input_ids, prefix_lengths, loss_mask, is_correct, group_sizes = (
            _build_contrastive_batch(
                prompts, eos_token_id=99, pad_token_id=0, device=torch.device("cpu")
            )
        )
        loss, _ = _compute_contrastive_loss(
            q,
            input_ids,
            prefix_lengths,
            loss_mask,
            is_correct,
            group_sizes,
            contrastive_weight=1.0,
        )
        loss.backward()
        assert q.lm_head.weight.grad is not None

    def test_contrastive_weight_zero_matches_nll_only(self):
        """With contrastive_weight=0, contrastive loss shouldn't contribute."""
        p = _make_p()
        q = InverseCotModel(p)
        prompts = _make_prompts(n_prompts=2, group_size=4, n_correct=2)
        input_ids, prefix_lengths, loss_mask, is_correct, group_sizes = (
            _build_contrastive_batch(
                prompts, eos_token_id=99, pad_token_id=0, device=torch.device("cpu")
            )
        )
        loss_w0, m0 = _compute_contrastive_loss(
            q,
            input_ids,
            prefix_lengths,
            loss_mask,
            is_correct,
            group_sizes,
            contrastive_weight=0.0,
        )
        assert m0["train/loss"] == m0["train/nll"]


# ---------- JSONL round-trip ----------


class TestJsonlRoundTrip:
    def test_write_and_load(self, tokenizer, monkeypatch):
        import json

        from dialectic.experiments.launchers import inverse_cot

        entries = [
            {
                "prompt_str": "Using [1, 2, 3], reach 6",
                "numbers": [1, 2, 3],
                "target": 6,
                "completions": [
                    {
                        "cot": "1+2=3, 3+3=6",
                        "answer": "<answer>1+2+3</answer>",
                        "is_correct": True,
                    },
                    {
                        "cot": "1*2=2",
                        "answer": "<answer>1*2</answer>",
                        "is_correct": False,
                    },
                ],
            }
        ]
        jsonl_bytes = "\n".join(json.dumps(e) for e in entries).encode()
        monkeypatch.setattr(
            inverse_cot.extty, "load_artifact", lambda name: jsonl_bytes
        )

        loaded = inverse_cot._load_rollout_artifact("test-artifact", tokenizer)
        assert len(loaded) == 1
        assert len(loaded[0].completions) == 2
        assert loaded[0].completions[0].is_correct is True
        assert loaded[0].completions[1].is_correct is False
        assert len(loaded[0].prompt_ids) > 0
        assert len(loaded[0].completions[0].cot_ids) > 0
