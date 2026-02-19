"""Tests for validation evaluation during training."""

import extty

from dialectic.rl.env import Countdown, CountdownEnv
from dialectic.rl.extractors import extract_from_answer_tags
from dialectic.rl.reward import countdown_correct, weighted_reward
from dialectic.rl.train import ValidationConfig, run_validation


def countdown_state_to_str(data: Countdown) -> str:
    return data.prompt


class TestCountdownEnvStr:
    def test_scalar_config(self):
        env = CountdownEnv(n_ops=3, n_total=4, n_larges=1)
        assert str(env) == "countdown_ops3_n4_lg1"

    def test_list_config(self):
        env = CountdownEnv(n_ops=[3, 5], n_total=[4, 6], n_larges=[1, 2])
        assert str(env) == "countdown_ops3_5_n4_6_lg1_2"

    def test_default_config(self):
        env = CountdownEnv()
        assert str(env) == "countdown_ops5_n6_lg2"


class TestRunValidation:
    def _make_val_config(self, tokenizer, envs, **kwargs):
        reward_fn = weighted_reward([("correct", 1.0, countdown_correct)])
        defaults = dict(
            envs=envs,
            reward_fn=reward_fn,
            state_to_str=countdown_state_to_str,
            extractor=extract_from_answer_tags,
            tokenizer=tokenizer,
            eos_token_id=151643,
            pad_token_id=151643,
            max_episodes=4,
            batch_size=2,
            max_tokens_generated=20,
            use_bf16=False,
        )
        defaults.update(kwargs)
        return ValidationConfig(**defaults)

    def test_single_env_returns_expected_keys(self, tiny_model, tokenizer):
        env = CountdownEnv(n_ops=3, n_total=4, n_larges=1, seed=42)
        val_config = self._make_val_config(tokenizer, [env])

        metrics = run_validation(net=tiny_model, val_config=val_config)

        label = str(env)
        assert f"val/{label}/reward_mean" in metrics
        assert f"val/{label}/reward_std" in metrics
        assert f"val/{label}/reward/correct" in metrics
        assert "val/reward_mean" in metrics

    def test_multiple_envs_returns_per_env_metrics(self, tiny_model, tokenizer):
        env1 = CountdownEnv(n_ops=3, n_total=4, n_larges=1, seed=100)
        env2 = CountdownEnv(n_ops=5, n_total=6, n_larges=2, seed=200)
        val_config = self._make_val_config(tokenizer, [env1, env2])

        metrics = run_validation(net=tiny_model, val_config=val_config)

        for env in [env1, env2]:
            label = str(env)
            assert f"val/{label}/reward_mean" in metrics
            assert f"val/{label}/reward_std" in metrics

        assert "val/reward_mean" in metrics

    def test_aggregate_reward_is_mean_of_per_env(self, tiny_model, tokenizer):
        env1 = CountdownEnv(n_ops=3, n_total=4, n_larges=1, seed=100)
        env2 = CountdownEnv(n_ops=5, n_total=6, n_larges=2, seed=200)
        val_config = self._make_val_config(tokenizer, [env1, env2])

        metrics = run_validation(net=tiny_model, val_config=val_config)

        per_env_means = [metrics[f"val/{str(env)}/reward_mean"] for env in [env1, env2]]
        expected = sum(per_env_means) / len(per_env_means)
        assert metrics["val/reward_mean"] == expected

    def test_restores_training_mode(self, tiny_model, tokenizer):
        env = CountdownEnv(n_ops=3, n_total=4, n_larges=1, seed=42)
        val_config = self._make_val_config(tokenizer, [env])

        tiny_model.train()
        assert tiny_model.training

        run_validation(net=tiny_model, val_config=val_config)
        assert tiny_model.training

    def test_preserves_eval_mode(self, tiny_model, tokenizer):
        env = CountdownEnv(n_ops=3, n_total=4, n_larges=1, seed=42)
        val_config = self._make_val_config(tokenizer, [env])

        tiny_model.eval()
        assert not tiny_model.training

        run_validation(net=tiny_model, val_config=val_config)
        assert not tiny_model.training

    def test_deterministic_with_same_seeds(self, tiny_model, tokenizer):
        env1 = CountdownEnv(n_ops=3, n_total=4, n_larges=1, seed=42)
        env2 = CountdownEnv(n_ops=3, n_total=4, n_larges=1, seed=42)
        val_config1 = self._make_val_config(tokenizer, [env1])
        val_config2 = self._make_val_config(tokenizer, [env2])

        metrics1 = run_validation(net=tiny_model, val_config=val_config1)
        metrics2 = run_validation(net=tiny_model, val_config=val_config2)

        label = str(env1)
        assert (
            metrics1[f"val/{label}/reward_mean"] == metrics2[f"val/{label}/reward_mean"]
        )

    def test_no_grad_during_validation(self, tiny_model, tokenizer):
        env = CountdownEnv(n_ops=3, n_total=4, n_larges=1, seed=42)
        val_config = self._make_val_config(tokenizer, [env])

        tiny_model.train()
        run_validation(net=tiny_model, val_config=val_config)

        for param in tiny_model.parameters():
            assert param.grad is None

    def test_examples_have_rewards(self, tiny_model, tokenizer):
        env = CountdownEnv(n_ops=3, n_total=4, n_larges=1, seed=42)
        val_config = self._make_val_config(tokenizer, [env])

        metrics = run_validation(net=tiny_model, val_config=val_config)

        label = str(env)
        example = metrics[f"val/{label}/example"]
        assert isinstance(example, extty.BatchExample)
        assert len(example.prompts) > 0
        assert len(example.rewards) == len(example.prompts)
        for reward in example.rewards:
            assert isinstance(reward, list)
            assert all(isinstance(r, dict) for r in reward)
            assert all("correct" in r for r in reward)
