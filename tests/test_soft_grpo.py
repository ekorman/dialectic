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
        B, V, D = 2, 10, 4
        pad_token_id = 0
        t1 = torch.randn(B, 5, V)
        t2 = torch.randn(B, 3, V)
        n1 = torch.randn(B, 5, D)
        n2 = torch.randn(B, 3, D)
        m1 = torch.ones(B, 5, dtype=torch.bool)
        m2 = torch.zeros(B, 3, dtype=torch.bool)

        st, sn, sm, npm = stack_and_pad_soft(
            tokens=[t1, t2],
            noise=[n1, n2],
            hard_masks=[m1, m2],
            pad_token_id=pad_token_id,
        )

        assert st.shape == (B, 2, 5, V)
        assert sn.shape == (B, 2, 5, D)
        assert sm.shape == (B, 2, 5)
        assert npm.shape == (B, 2, 5)

        torch.testing.assert_close(st[:, 0, :5], t1)
        torch.testing.assert_close(st[:, 1, :3], t2)
        assert st[:, 1, 3:].argmax(-1).eq(pad_token_id).all()

        torch.testing.assert_close(sn[:, 0, :5], n1)
        torch.testing.assert_close(sn[:, 1, :3], n2)
        assert sn[:, 1, 3:].eq(0).all()

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

        prompt_ids = torch.randint(1, V, (B, prompt_len))
        prompt_onehot = torch.nn.functional.one_hot(prompt_ids, V).float()

        completion_tokens = []
        completion_noise = []
        hard_masks = []
        for _ in range(G):
            gen_soft = torch.randn(B, completion_len, V).softmax(-1)
            full_tokens = torch.cat([prompt_onehot, gen_soft], dim=1)
            completion_tokens.append(full_tokens)

            full_noise = torch.randn(B, total_len, D) * noise_std
            completion_noise.append(full_noise)

            mask = torch.ones(B, total_len, dtype=torch.bool)
            mask[:, prompt_len:] = False
            hard_masks.append(mask)

        attention_mask = torch.ones(B, prompt_len, dtype=torch.bool)

        with torch.no_grad():
            lp_full, mask_full = compute_soft_log_probs(
                net=tiny_model,
                attention_mask=attention_mask,
                completion_tokens=completion_tokens,
                completion_noise=completion_noise,
                hard_tokens_mask=hard_masks,
                noise_std=noise_std,
                temperature=temperature,
                pad_token_id=pad_token_id,
                chunk_size=0,
            )

            lp_chunked, mask_chunked = compute_soft_log_probs(
                net=tiny_model,
                attention_mask=attention_mask,
                completion_tokens=completion_tokens,
                completion_noise=completion_noise,
                hard_tokens_mask=hard_masks,
                noise_std=noise_std,
                temperature=temperature,
                pad_token_id=pad_token_id,
                chunk_size=4,
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

        prompt_ids = torch.randint(1, V, (B, prompt_len))
        prompt_onehot = torch.nn.functional.one_hot(prompt_ids, V).float()

        gen_ids = torch.randint(1, V, (B, completion_len))
        gen_onehot = torch.nn.functional.one_hot(gen_ids, V).float()
        full_tokens = torch.cat([prompt_onehot, gen_onehot], dim=1)
        full_noise = torch.randn(B, total_len, D) * noise_std

        all_hard_mask = torch.ones(B, total_len, dtype=torch.bool)
        attention_mask = torch.ones(B, prompt_len, dtype=torch.bool)

        with torch.no_grad():
            lp, _ = compute_soft_log_probs(
                net=tiny_model,
                attention_mask=attention_mask,
                completion_tokens=[full_tokens],
                completion_noise=[full_noise],
                hard_tokens_mask=[all_hard_mask],
                noise_std=noise_std,
                temperature=1.0,
                pad_token_id=pad_token_id,
                chunk_size=0,
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

        prompt_ids = torch.randint(1, V, (B, prompt_len))
        prompt_onehot = torch.nn.functional.one_hot(prompt_ids, V).float()

        gen_soft = torch.randn(B, completion_len, V).softmax(-1)
        full_tokens = torch.cat([prompt_onehot, gen_soft], dim=1)
        full_noise = torch.randn(B, total_len, D) * noise_std

        all_soft_mask = torch.zeros(B, total_len, dtype=torch.bool)
        all_soft_mask[:, :prompt_len] = True
        attention_mask = torch.ones(B, prompt_len, dtype=torch.bool)

        with torch.no_grad():
            lp, _ = compute_soft_log_probs(
                net=tiny_model,
                attention_mask=attention_mask,
                completion_tokens=[full_tokens],
                completion_noise=[full_noise],
                hard_tokens_mask=[all_soft_mask],
                noise_std=noise_std,
                temperature=1.0,
                pad_token_id=pad_token_id,
                chunk_size=0,
            )

        assert lp.shape == (B, G, completion_len)

        with torch.no_grad():
            W = tiny_model.embed_tokens.weight
            stacked = full_tokens.unsqueeze(1)  # [B, 1, L, V]
            flat = stacked.view(B, total_len, V)
            noise_flat = full_noise.unsqueeze(1).view(B, total_len, D)
            full_attn_mask = torch.ones(B, total_len, dtype=torch.bool)
            full_attn_mask[:, :prompt_len] = attention_mask
            logits = tiny_model(
                flat,
                return_all_logits=True,
                attention_mask=full_attn_mask,
                soft_token_noise=noise_flat,
            )
            logits_comp = logits[:, prompt_len - 1 : -1]
            comp_tokens = flat[:, prompt_len:]
            comp_noise = noise_flat[:, prompt_len:]
            e_action = comp_tokens @ W + comp_noise
            mu_new = torch.softmax(logits_comp / 1.0, dim=-1) @ W
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

        prompt_ids = torch.randint(1, V, (B, prompt_len))
        prompt_onehot = torch.nn.functional.one_hot(prompt_ids, V).float()

        gen_soft = torch.randn(B, completion_len, V).softmax(-1)
        full_tokens = torch.cat([prompt_onehot, gen_soft], dim=1)
        full_noise = torch.randn(B, total_len, D) * noise_std

        mask = torch.zeros(B, total_len, dtype=torch.bool)
        mask[:, :prompt_len] = True
        attention_mask = torch.ones(B, prompt_len, dtype=torch.bool)

        lp, _ = compute_soft_log_probs(
            net=tiny_model,
            attention_mask=attention_mask,
            completion_tokens=[full_tokens],
            completion_noise=[full_noise],
            hard_tokens_mask=[mask],
            noise_std=noise_std,
            temperature=1.0,
            pad_token_id=pad_token_id,
            chunk_size=0,
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
    def test_training_loop_completes(self, tiny_model, tokenizer):
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
            eps=0.2,
            mu=1,
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
        )

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
                # Use a reward that's unlikely to tie across group samples.
                value = sum(ord(c) for c in raw_model_output) / 10000.0
            # Deterministic jitter to prevent identical rewards within a group.
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
        )

        params_changed = False
        for name, param in tiny_model.named_parameters():
            if not torch.allclose(params_before[name], param, atol=1e-8):
                params_changed = True
                break

        assert params_changed, "No parameters changed during training"
