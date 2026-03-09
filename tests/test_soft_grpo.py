import torch

from dialectic.rl.env import Countdown, CountdownEnv
from dialectic.rl.extractors import extract_from_answer_tags
from dialectic.rl.reward import countdown_correct, weighted_reward
from dialectic.rl.train import (
    compute_soft_log_probs,
    grpo_advantage,
    stack_and_pad_soft,
    train_soft_grpo,
)
from dialectic.rl.types import RewardResult


def countdown_state_to_str(data: Countdown) -> str:
    return data.prompt


class TestStackAndPadSoft:
    def test_shapes_and_padding(self):
        B, D = 2, 4
        pad_token_id = 0
        e1 = torch.randn(B, 5, D)
        e2 = torch.randn(B, 3, D)
        s1 = torch.randint(1, 10, (B, 5))
        s2 = torch.randint(1, 10, (B, 3))
        m1 = torch.ones(B, 5, dtype=torch.bool)
        m2 = torch.zeros(B, 3, dtype=torch.bool)

        se, ss, sm, npm = stack_and_pad_soft(
            embeddings=[e1, e2],
            shadow_ids=[s1, s2],
            hard_masks=[m1, m2],
            pad_token_id=pad_token_id,
        )

        assert se.shape == (B, 2, 5, D)
        assert ss.shape == (B, 2, 5)
        assert sm.shape == (B, 2, 5)
        assert npm.shape == (B, 2, 5)

        torch.testing.assert_close(se[:, 0, :5], e1)
        torch.testing.assert_close(se[:, 1, :3], e2)
        assert se[:, 1, 3:].eq(0).all()

        torch.testing.assert_close(ss[:, 0, :5], s1)
        torch.testing.assert_close(ss[:, 1, :3], s2)
        assert ss[:, 1, 3:].eq(pad_token_id).all()

        assert sm[:, 0, :5].all()
        assert not sm[:, 1, :3].any()
        assert sm[:, 1, 3:].all()

        assert npm[:, 0, :5].all()
        assert npm[:, 1, :3].all()
        assert not npm[:, 1, 3:].any()


class TestComputeSoftLogProbs:
    def test_chunked_matches_non_chunked(self, tiny_model):
        torch.manual_seed(42)
        tiny_model.eval()

        B, G = 2, 2
        prompt_len = 6
        completion_len = 8
        total_len = prompt_len + completion_len
        V = tiny_model.vocab_size
        D = tiny_model.d
        noise_std = 0.1
        temperature = 1.0
        pad_token_id = 0

        W = tiny_model.embed_tokens.weight

        completion_embeddings = []
        completion_shadow_ids = []
        hard_masks = []
        for _ in range(G):
            prompt_ids = torch.randint(1, V, (B, prompt_len))
            gen_soft = torch.randn(B, completion_len, V).softmax(-1)
            full_shadow_ids = torch.cat([prompt_ids, gen_soft.argmax(-1)], dim=1)

            prompt_emb = W[prompt_ids].float()
            gen_emb = (gen_soft.float() @ W.float()) + torch.randn(
                B, completion_len, D
            ) * noise_std
            full_embeddings = torch.cat([prompt_emb, gen_emb], dim=1)
            completion_embeddings.append(full_embeddings)
            completion_shadow_ids.append(full_shadow_ids)

            mask = torch.ones(B, total_len, dtype=torch.bool)
            mask[:, prompt_len:] = False
            hard_masks.append(mask)

        attention_mask = torch.ones(B, prompt_len, dtype=torch.bool)

        with torch.no_grad():
            lp_full, mask_full = compute_soft_log_probs(
                net=tiny_model,
                attention_mask=attention_mask,
                completion_embeddings=completion_embeddings,
                completion_shadow_ids=completion_shadow_ids,
                hard_tokens_mask=hard_masks,
                noise_std=noise_std,
                temperature=temperature,
                pad_token_id=pad_token_id,
                chunk_size=0,
                normalize_soft_pdf_by_dim=True,
            )

            lp_chunked, mask_chunked = compute_soft_log_probs(
                net=tiny_model,
                attention_mask=attention_mask,
                completion_embeddings=completion_embeddings,
                completion_shadow_ids=completion_shadow_ids,
                hard_tokens_mask=hard_masks,
                noise_std=noise_std,
                temperature=temperature,
                pad_token_id=pad_token_id,
                chunk_size=4,
                normalize_soft_pdf_by_dim=True,
            )

        assert lp_full.shape == (B, G, completion_len)
        assert lp_full.shape == lp_chunked.shape
        torch.testing.assert_close(lp_full, lp_chunked, atol=1e-4, rtol=1e-4)
        assert torch.equal(mask_full, mask_chunked)

    def test_hard_positions_get_cross_entropy(self, tiny_model):
        torch.manual_seed(42)
        tiny_model.eval()

        B, G = 1, 1
        prompt_len = 4
        completion_len = 4
        total_len = prompt_len + completion_len
        V = tiny_model.vocab_size
        D = tiny_model.d
        noise_std = 0.1
        pad_token_id = 0

        W = tiny_model.embed_tokens.weight

        prompt_ids = torch.randint(1, V, (B, prompt_len))
        gen_ids = torch.randint(1, V, (B, completion_len))
        full_shadow_ids = torch.cat([prompt_ids, gen_ids], dim=1)

        prompt_emb = W[prompt_ids].float()
        gen_emb = W[gen_ids].float() + torch.randn(B, completion_len, D) * noise_std
        full_embeddings = torch.cat([prompt_emb, gen_emb], dim=1)

        all_hard_mask = torch.ones(B, total_len, dtype=torch.bool)
        attention_mask = torch.ones(B, prompt_len, dtype=torch.bool)

        with torch.no_grad():
            lp, _ = compute_soft_log_probs(
                net=tiny_model,
                attention_mask=attention_mask,
                completion_embeddings=[full_embeddings],
                completion_shadow_ids=[full_shadow_ids],
                hard_tokens_mask=[all_hard_mask],
                noise_std=noise_std,
                temperature=1.0,
                pad_token_id=pad_token_id,
                chunk_size=0,
                normalize_soft_pdf_by_dim=False,
            )

        assert lp.shape == (B, G, completion_len)
        assert (lp <= 0).all()

    def test_soft_positions_get_gaussian(self, tiny_model):
        torch.manual_seed(42)
        tiny_model.eval()

        B, G = 1, 1
        prompt_len = 4
        completion_len = 4
        total_len = prompt_len + completion_len
        V = tiny_model.vocab_size
        D = tiny_model.d
        noise_std = 0.1
        pad_token_id = 0

        W = tiny_model.embed_tokens.weight

        prompt_ids = torch.randint(1, V, (B, prompt_len))
        gen_soft = torch.randn(B, completion_len, V).softmax(-1)
        full_shadow_ids = torch.cat([prompt_ids, gen_soft.argmax(-1)], dim=1)

        noise = torch.randn(B, completion_len, D) * noise_std
        prompt_emb = W[prompt_ids].float()
        gen_emb = (gen_soft.float() @ W.float()) + noise.float()
        full_embeddings = torch.cat([prompt_emb, gen_emb], dim=1)

        all_soft_mask = torch.zeros(B, total_len, dtype=torch.bool)
        all_soft_mask[:, :prompt_len] = True
        attention_mask = torch.ones(B, prompt_len, dtype=torch.bool)

        with torch.no_grad():
            lp, _ = compute_soft_log_probs(
                net=tiny_model,
                attention_mask=attention_mask,
                completion_embeddings=[full_embeddings],
                completion_shadow_ids=[full_shadow_ids],
                hard_tokens_mask=[all_soft_mask],
                noise_std=noise_std,
                temperature=1.0,
                pad_token_id=pad_token_id,
                chunk_size=0,
                normalize_soft_pdf_by_dim=True,
            )

        assert lp.shape == (B, G, completion_len)

        with torch.no_grad():
            stacked_emb = full_embeddings.unsqueeze(1)
            flat = stacked_emb.view(B, total_len, D)
            full_attn_mask = torch.ones(B, total_len, dtype=torch.bool)
            full_attn_mask[:, :prompt_len] = attention_mask
            logits = tiny_model(
                flat,
                return_all_logits=True,
                attention_mask=full_attn_mask,
            )
            logits_comp = logits[:, prompt_len - 1 : -1]
            comp_embeddings = flat[:, prompt_len:]
            e_action = comp_embeddings.float()
            mu_new = torch.softmax(logits_comp / 1.0, dim=-1) @ W.float()
            expected = -0.5 * ((e_action - mu_new) ** 2).mean(-1) / (noise_std**2)
            expected = expected.view(B, G, completion_len)

        torch.testing.assert_close(lp, expected)

    def test_gradient_flows_through_soft_log_probs(self, tiny_model):
        tiny_model.train()

        B = 1
        prompt_len = 4
        completion_len = 4
        total_len = prompt_len + completion_len
        V = tiny_model.vocab_size
        D = tiny_model.d
        noise_std = 0.1
        pad_token_id = 0

        W = tiny_model.embed_tokens.weight

        prompt_ids = torch.randint(1, V, (B, prompt_len))
        gen_soft = torch.randn(B, completion_len, V).softmax(-1)
        full_shadow_ids = torch.cat([prompt_ids, gen_soft.argmax(-1)], dim=1)

        noise = torch.randn(B, completion_len, D) * noise_std
        prompt_emb = W[prompt_ids].float().detach()
        gen_emb = (gen_soft.float() @ W.float()).detach() + noise.float()
        full_embeddings = torch.cat([prompt_emb, gen_emb], dim=1)

        mask = torch.zeros(B, total_len, dtype=torch.bool)
        mask[:, :prompt_len] = True
        attention_mask = torch.ones(B, prompt_len, dtype=torch.bool)

        lp, _ = compute_soft_log_probs(
            net=tiny_model,
            attention_mask=attention_mask,
            completion_embeddings=[full_embeddings],
            completion_shadow_ids=[full_shadow_ids],
            hard_tokens_mask=[mask],
            noise_std=noise_std,
            temperature=1.0,
            pad_token_id=pad_token_id,
            chunk_size=0,
            normalize_soft_pdf_by_dim=False,
        )

        loss = lp.sum()
        loss.backward()

        has_grad = False
        for p in tiny_model.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No gradients flowed to model parameters"


class TestTrainSoftGrpo:
    def _test_training_loop_completes(self, tiny_model, tokenizer, eps):
        env = CountdownEnv()
        opt = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)

        train_soft_grpo(
            net=tiny_model,
            opt=opt,
            env=env,
            reward_fn=weighted_reward([("correct", 1.0, countdown_correct)]),
            state_to_str=countdown_state_to_str,
            tokenizer=tokenizer,
            eos_token_id=151643,
            pad_token_id=151643,
            extractor=extract_from_answer_tags,
            beta=0.01,
            eps=eps,
            mu=1,
            accumulation_steps=1,
            max_grad_norm=1.0,
            max_tokens_generated=20,
            max_episodes=4,
            update_ref_net_batch_cadence=5,
            batch_size=2,
            group_size=2,
            temperature=1.0,
            noise_std=0.1,
            use_bf16=False,
            advantage_fn=grpo_advantage,
            normalize_by_sequence_length=True,
            normalize_soft_pdf_by_dim=False,
            val_episodes=0,
            val_envs=[],
            val_batch_size=0,
        )

    def test_training_loop_completes_eps_not_none(self, tiny_model, tokenizer):
        self._test_training_loop_completes(tiny_model, tokenizer, eps=0.2)

    def test_training_loop_completes_eps_none(self, tiny_model, tokenizer):
        self._test_training_loop_completes(tiny_model, tokenizer, eps=None)

    def test_gradients_flow(self, tiny_model, tokenizer):
        torch.manual_seed(123)
        env = CountdownEnv(seed=123)

        rng = torch.Generator().manual_seed(123)

        def length_reward_fn(
            *, env_response, raw_model_output=None, extracted_model_output
        ) -> RewardResult:
            if not raw_model_output:
                value = 0.0
            else:
                value = sum(ord(c) for c in raw_model_output) / 10000.0
            value += torch.rand((), generator=rng).item() * 1e-3
            return RewardResult(total=value, components={"length": value})

        opt = torch.optim.Adam(tiny_model.parameters(), lr=1e-2)

        params_before = {
            name: param.clone() for name, param in tiny_model.named_parameters()
        }

        train_soft_grpo(
            net=tiny_model,
            opt=opt,
            env=env,
            reward_fn=length_reward_fn,
            state_to_str=countdown_state_to_str,
            tokenizer=tokenizer,
            eos_token_id=999999,
            pad_token_id=151643,
            extractor=extract_from_answer_tags,
            beta=0.01,
            eps=0.2,
            mu=1,
            accumulation_steps=1,
            max_grad_norm=1,
            max_tokens_generated=20,
            max_episodes=8,
            update_ref_net_batch_cadence=10,
            batch_size=2,
            group_size=2,
            temperature=1.0,
            noise_std=0.1,
            use_bf16=False,
            advantage_fn=grpo_advantage,
            normalize_by_sequence_length=True,
            normalize_soft_pdf_by_dim=False,
            val_episodes=0,
            val_envs=[],
            val_batch_size=0,
        )

        params_changed = False
        for name, param in tiny_model.named_parameters():
            if not torch.allclose(params_before[name], param, atol=1e-8):
                params_changed = True
                break

        assert params_changed, "No parameters changed during training"
