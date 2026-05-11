import torch
from torch.nn.functional import scaled_dot_product_attention

from dialectic.llm.components.attention import attention
from dialectic.llm.inverse_cot import create_prefix_lm_mask
from dialectic.llm.qwen import create_qwen
from dialectic.rl.extractors import parse_cot_and_answer
from dialectic.rl.inverse_cot_data import (
    PreTokenizedCompletion,
    PreTokenizedPrompt,
    build_shuffled_batch,
    load_rollout_artifacts,
    subsample_completions,
)
from dialectic.rl.inverse_cot_eval import expressions_match

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


# ---------- parse_cot_and_answer ----------


class TestParseCotAndAnswer:
    def test_think_tags(self):
        text = "<think>\n100 + 50 = 150\n</think>\n\n(100 + 50)"
        result = parse_cot_and_answer(text)
        assert result is not None
        cot, answer = result
        assert cot == "100 + 50 = 150"
        assert answer == "(100 + 50)"

    def test_think_tags_with_answer_tags(self):
        text = "<think>\nreasoning\n</think>\n\n<answer>42</answer>"
        result = parse_cot_and_answer(text)
        assert result is not None
        cot, answer = result
        assert cot == "reasoning"
        assert answer == "<answer>42</answer>"

    def test_think_tags_empty_cot(self):
        text = "<think>\n\n</think>\n\n42"
        assert parse_cot_and_answer(text) is None

    def test_think_tags_empty_answer(self):
        text = "<think>\nreasoning\n</think>"
        assert parse_cot_and_answer(text) is None

    def test_answer_tags_fallback(self):
        text = "Let me think step by step\n<answer>42 + 8</answer>"
        result = parse_cot_and_answer(text)
        assert result is not None
        cot, answer = result
        assert cot == "Let me think step by step\n"
        assert answer == "<answer>42 + 8</answer>"

    def test_no_tags(self):
        assert parse_cot_and_answer("just some text without tags") is None

    def test_answer_tags_empty_cot(self):
        assert parse_cot_and_answer("<answer>42</answer>") is None

    def test_multiline_think(self):
        text = "<think>\nstep 1\nstep 2\nstep 3\n</think>\n\nexpr"
        result = parse_cot_and_answer(text)
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


class TestJsonlRoundTrip:
    def test_write_and_load(self, tokenizer, monkeypatch):
        import json

        from dialectic.rl import inverse_cot_data

        entries = [
            {
                "prompt_str": "Using [1, 2, 3], reach 6",
                "numbers": [1, 2, 3],
                "target": 6,
                "split": "train",
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
            inverse_cot_data.extty, "load_artifact", lambda name, **kwargs: jsonl_bytes
        )

        by_split = load_rollout_artifacts(["test-artifact"], tokenizer)
        assert "train" in by_split
        loaded = by_split["train"]
        assert len(loaded) == 1
        assert len(loaded[0].completions) == 2
        assert loaded[0].completions[0].is_correct is True
        assert loaded[0].completions[1].is_correct is False
        assert len(loaded[0].prompt_ids) > 0
        assert len(loaded[0].completions[0].cot_ids) > 0

    def test_equation_loaded_and_stripped(self, tokenizer, monkeypatch):
        import json

        from dialectic.rl import inverse_cot_data

        # Each prompt needs at least one correct AND one incorrect
        # completion to pass `load_rollout_artifacts`'s mixed-correctness
        # invariant (which `generate_inverse_cot_rollouts` already enforces
        # at rollout generation time via `--n-pos-min 1 --n-neg-min 1`).
        entries = [
            {
                "prompt_str": "solve this",
                "numbers": [1, 2],
                "target": 3,
                "split": "train",
                "equation": "(1 + 2) = 3",
                "completions": [
                    {
                        "cot": "add them",
                        "answer": "<answer>1+2</answer>",
                        "is_correct": True,
                    },
                    {
                        "cot": "subtract them",
                        "answer": "<answer>2-1</answer>",
                        "is_correct": False,
                    },
                ],
            },
            {
                "prompt_str": "solve that",
                "numbers": [4, 5],
                "target": 20,
                "split": "train",
                "completions": [
                    {
                        "cot": "multiply",
                        "answer": "<answer>4*5</answer>",
                        "is_correct": True,
                    },
                    {
                        "cot": "add",
                        "answer": "<answer>4+5</answer>",
                        "is_correct": False,
                    },
                ],
            },
        ]
        jsonl_bytes = "\n".join(json.dumps(e) for e in entries).encode()
        monkeypatch.setattr(
            inverse_cot_data.extty, "load_artifact", lambda name, **kwargs: jsonl_bytes
        )

        by_split = load_rollout_artifacts(["test"], tokenizer)
        loaded = by_split["train"]
        assert loaded[0].equation == "(1 + 2)"
        assert loaded[1].equation is None


class TestSubsampleCompletionsRng:
    def _prompt(self, k: int) -> PreTokenizedPrompt:
        completions = [
            PreTokenizedCompletion(
                answer_ids=[10 + i],
                cot_ids=[20 + i, 21 + i],
                is_correct=(i % 2 == 0),
            )
            for i in range(k)
        ]
        return PreTokenizedPrompt(prompt_ids=[1, 2, 3], completions=completions)

    def test_explicit_rng_is_reproducible(self):
        import random as _random

        prompt = self._prompt(16)
        # Two independent calls with freshly-seeded rngs should produce
        # byte-identical selections. This is the property val relies on
        # to make its dataset deterministic across calls.
        rng1 = _random.Random(42)
        a = subsample_completions(prompt, k=8, rng=rng1)
        rng2 = _random.Random(42)
        b = subsample_completions(prompt, k=8, rng=rng2)
        assert [c.answer_ids for c in a.completions] == [
            c.answer_ids for c in b.completions
        ]

    def test_default_rng_uses_global_random(self):
        import random as _random

        prompt = self._prompt(16)
        # With the default (rng=None), `subsample_completions` uses the
        # global `random` module. Seeding it before each call gives the
        # same result, proving the default path still taps global state.
        _random.seed(123)
        a = subsample_completions(prompt, k=8)
        _random.seed(123)
        b = subsample_completions(prompt, k=8)
        assert [c.answer_ids for c in a.completions] == [
            c.answer_ids for c in b.completions
        ]

    def test_explicit_rng_independent_of_global_state(self):
        import random as _random

        prompt = self._prompt(16)
        rng = _random.Random(7)
        a = subsample_completions(prompt, k=8, rng=rng)
        # Advance the global random state between calls — shouldn't
        # affect an explicit rng instance.
        _random.random()
        _random.random()
        rng2 = _random.Random(7)
        b = subsample_completions(prompt, k=8, rng=rng2)
        assert [c.answer_ids for c in a.completions] == [
            c.answer_ids for c in b.completions
        ]

    def test_returns_exactly_k_with_duplicate_completions(self):
        # Regression: two rollouts with byte-identical (answer_ids, cot_ids,
        # is_correct) triggered dataclass value-equality in the old
        # `c not in selected` path, shrinking the "remaining" pool and
        # yielding fewer than k completions. Uniform-K assertion downstream
        # in `build_infonce_batch` then blew up with mismatched group sizes.
        import random as _random

        dup_correct = PreTokenizedCompletion(
            answer_ids=[10], cot_ids=[20, 21], is_correct=True
        )
        completions = [
            dup_correct,
            dup_correct,  # byte-identical duplicate of the first
            PreTokenizedCompletion(answer_ids=[11], cot_ids=[22, 23], is_correct=True),
            PreTokenizedCompletion(answer_ids=[12], cot_ids=[24, 25], is_correct=False),
            PreTokenizedCompletion(answer_ids=[13], cot_ids=[26, 27], is_correct=False),
            PreTokenizedCompletion(answer_ids=[14], cot_ids=[28, 29], is_correct=False),
            PreTokenizedCompletion(answer_ids=[15], cot_ids=[30, 31], is_correct=False),
            PreTokenizedCompletion(answer_ids=[16], cot_ids=[32, 33], is_correct=False),
        ]
        prompt = PreTokenizedPrompt(prompt_ids=[1, 2, 3], completions=completions)
        assert len(prompt.completions) == 8

        # Run many trials to exercise all random draws; each must return
        # exactly k completions regardless of which duplicate gets picked.
        rng = _random.Random(0)
        for _ in range(50):
            out = subsample_completions(prompt, k=8, rng=rng)
            assert len(out.completions) == 8, (
                f"expected 8 completions, got {len(out.completions)}"
            )
            # And the mixed-correctness guarantee still holds.
            assert any(c.is_correct for c in out.completions)
            assert any(not c.is_correct for c in out.completions)


class TestMaxCotTokensFilter:
    def _jsonl_entry(
        self,
        prompt_idx: int,
        correct_cot_len: int,
        incorrect_cot_len: int,
    ) -> dict:
        # Build a fake rollout with exactly one correct + one incorrect
        # completion, where each CoT is a string that tokenizes to roughly
        # the requested number of tokens. Using "word " × N gives one
        # token per repeat for the Qwen3 tokenizer in practice.
        return {
            "prompt_str": f"prompt {prompt_idx}",
            "numbers": [1, 2],
            "target": 3,
            "split": "train",
            "completions": [
                {
                    "cot": "word " * correct_cot_len,
                    "answer": "<answer>a</answer>",
                    "is_correct": True,
                },
                {
                    "cot": "word " * incorrect_cot_len,
                    "answer": "<answer>b</answer>",
                    "is_correct": False,
                },
            ],
        }

    def test_max_cot_tokens_drops_long_completions(self, tokenizer, monkeypatch):
        import json as _json

        from dialectic.rl import inverse_cot_data

        entries = [
            # first prompt: both completions short → kept
            self._jsonl_entry(0, correct_cot_len=10, incorrect_cot_len=10),
            # second prompt: correct is long → dropped; only incorrect
            # remains → prompt loses mixed-correctness → dropped
            self._jsonl_entry(1, correct_cot_len=500, incorrect_cot_len=10),
            # third prompt: both short again → kept
            self._jsonl_entry(2, correct_cot_len=15, incorrect_cot_len=20),
        ]
        jsonl_bytes = "\n".join(_json.dumps(e) for e in entries).encode()
        monkeypatch.setattr(
            inverse_cot_data.extty, "load_artifact", lambda name, **kwargs: jsonl_bytes
        )

        by_split = load_rollout_artifacts(
            ["test"],
            tokenizer,
            max_cot_tokens=100,
            min_completions_per_prompt=2,
        )
        loaded = by_split["train"]
        # 3 prompts loaded → 1 dropped for losing mixed correctness
        assert len(loaded) == 2
        prompt_strs = {tokenizer.decode(p.prompt_ids).strip() for p in loaded}
        assert "prompt 0" in " ".join(prompt_strs)
        assert "prompt 2" in " ".join(prompt_strs)
        assert "prompt 1" not in " ".join(prompt_strs)

    def test_max_cot_tokens_none_keeps_everything(self, tokenizer, monkeypatch):
        import json as _json

        from dialectic.rl import inverse_cot_data

        entries = [self._jsonl_entry(0, 10, 500)]
        jsonl_bytes = "\n".join(_json.dumps(e) for e in entries).encode()
        monkeypatch.setattr(
            inverse_cot_data.extty, "load_artifact", lambda name, **kwargs: jsonl_bytes
        )

        by_split = load_rollout_artifacts(["test"], tokenizer, max_cot_tokens=None)
        assert len(by_split["train"]) == 1
        assert len(by_split["train"][0].completions) == 2

    def test_min_completions_per_prompt_drops_small_prompts(
        self, tokenizer, monkeypatch
    ):
        import json as _json

        from dialectic.rl import inverse_cot_data

        entries = [
            # prompt with only 2 completions, but we'll require 4 → dropped
            self._jsonl_entry(0, 10, 10),
        ]
        jsonl_bytes = "\n".join(_json.dumps(e) for e in entries).encode()
        monkeypatch.setattr(
            inverse_cot_data.extty, "load_artifact", lambda name, **kwargs: jsonl_bytes
        )

        by_split = load_rollout_artifacts(
            ["test"], tokenizer, min_completions_per_prompt=4
        )
        assert by_split.get("train", []) == []


# ---------- build_shuffled_batch ----------


class TestBuildShuffledBatch:
    def test_shapes(self):
        prompts = _make_prompts(n_prompts=2, group_size=3, n_correct=1)
        input_ids, prefix_lengths, loss_mask = build_shuffled_batch(
            prompts, eos_token_id=99, pad_token_id=0, device=torch.device("cpu")
        )
        assert input_ids.shape[0] == 6  # 2 * 3
        assert prefix_lengths.shape == (6,)
        assert loss_mask.shape == input_ids.shape

    def test_answers_are_rolled(self):
        prompts = [
            PreTokenizedPrompt(
                prompt_ids=[1, 2],
                completions=[
                    PreTokenizedCompletion(
                        answer_ids=[10], cot_ids=[20], is_correct=True
                    ),
                    PreTokenizedCompletion(
                        answer_ids=[11], cot_ids=[21], is_correct=False
                    ),
                ],
            )
        ]
        input_ids, prefix_lengths, _ = build_shuffled_batch(
            prompts, eos_token_id=99, pad_token_id=0, device=torch.device("cpu")
        )
        # original: [1,2,10,20,99] and [1,2,11,21,99]
        # shuffled answers: sample 0 gets answer [11], sample 1 gets answer [10]
        assert input_ids[0].tolist()[:4] == [1, 2, 11, 20]
        assert input_ids[1].tolist()[:4] == [1, 2, 10, 21]


# ---------- expressions_match ----------


class TestExpressionsMatch:
    def test_equal_expressions(self):
        assert expressions_match("1+2", "3")

    def test_equivalent_expressions(self):
        assert expressions_match("(25*6)+9+9", "100+50+18")

    def test_unequal_expressions(self):
        assert not expressions_match("1+2", "4")

    def test_malformed_pred(self):
        assert not expressions_match("not_math", "3")

    def test_malformed_gt(self):
        assert not expressions_match("3", "not_math")

    def test_both_malformed(self):
        assert not expressions_match("abc", "def")

    def test_division(self):
        assert expressions_match("10/2", "5")

    def test_float_tolerance(self):
        assert expressions_match("1/3*3", "1")
