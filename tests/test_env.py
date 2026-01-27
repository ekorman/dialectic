import json
import tempfile

import pytest

from dialectic.rl.env import ArithmeticEnv, CountdownEnv, EpisodeIsDoneError, GSM8kEnv
from dialectic.rl.types import EnvResponse


@pytest.fixture
def sample_data() -> str:
    """Create a temporary JSONL file with sample GSM8k data."""
    data = [
        {
            "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
            "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72",
        },
        {
            "question": "If a train travels 60 miles in 1 hour, how far will it travel in 3 hours?",
            "answer": "The train travels 60 * 3 = <<60*3=180>>180 miles.\n#### 180",
        },
        {
            "question": "A baker has 24 cupcakes. She gives away 8. How many does she have left?",
            "answer": "24 - 8 = <<24-8=16>>16 cupcakes.\n#### 16",
        },
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
        return f.name


@pytest.fixture
def env(sample_data: str) -> GSM8kEnv:
    """Create a GSM8kEnv with sample data."""
    return GSM8kEnv(path=sample_data, eval_mode=True)


class TestGSM8kEnv:
    def test_load_data(self, env: GSM8kEnv):
        """Test that data is loaded correctly."""
        assert len(env.data) == 3
        assert "question" in env.data[0]
        assert "answer" in env.data[0]

    def test_reset_returns_question(self, env: GSM8kEnv):
        """Test that reset returns a question string."""
        resp = env.reset()
        assert isinstance(resp, EnvResponse)

    def test_cycles_through_all_problems(self, env: GSM8kEnv):
        """Test that environment cycles through all problems."""
        seen_questions = set()

        for _ in range(3):
            resp = env.reset()
            seen_questions.add(resp.data.question)

        assert len(seen_questions) == 3

    def test_seed_reproducibility(self, sample_data: str):
        """Test that same seed produces same sequence."""
        env1 = GSM8kEnv(path=sample_data, eval_mode=False)
        env2 = GSM8kEnv(path=sample_data, eval_mode=False)

        resp1 = env1.reset(seed=123)
        resp2 = env2.reset(seed=123)

        assert resp1 == resp2


class TestArithmeticEnv:
    def test_reset_returns_question(self):
        """Test that reset returns a question string."""
        env = ArithmeticEnv()
        resp = env.reset()
        q = resp.data.question

        assert q.startswith("What is")
        assert q.endswith("?")

    def test_arithmetic_env(self):
        env = ArithmeticEnv()
        s = env.reset()
        assert s.is_done
        assert s.data.question.startswith("What is")
        assert isinstance(s.data.answer, float)

        with pytest.raises(EpisodeIsDoneError):
            env.step(None)


class TestCountdownEnv:
    def test_reset_returns_valid_response(self):
        env = CountdownEnv()
        resp = env.reset()

        assert resp.is_done
        assert isinstance(resp.data.numbers, list)
        assert len(resp.data.numbers) == 4
        assert isinstance(resp.data.target, int)
        assert "prompt" in resp.data.__dict__

    def test_custom_num_operands(self):
        env = CountdownEnv(num_operands=6)
        resp = env.reset()

        assert len(resp.data.numbers) == 6

    def test_number_range(self):
        env = CountdownEnv(min_number=10, max_number=20)
        resp = env.reset()

        for num in resp.data.numbers:
            assert 10 <= num <= 20

    def test_seed_reproducibility_at_reset(self):
        env1 = CountdownEnv()
        env2 = CountdownEnv()

        resp1 = env1.reset(seed=42)
        resp2 = env2.reset(seed=42)

        assert resp1.data.numbers == resp2.data.numbers
        assert resp1.data.target == resp2.data.target

    def test_seed_reproducibility_at_init(self):
        env1 = CountdownEnv(seed=42)
        env2 = CountdownEnv(seed=42)

        env1_responses = [env1.reset() for _ in range(3)]
        env2_responses = [env2.reset() for _ in range(3)]

        for resp1, resp2 in zip(env1_responses, env2_responses):
            assert resp1.data.numbers == resp2.data.numbers
            assert resp1.data.target == resp2.data.target

    def test_step_raises_error(self):
        env = CountdownEnv()
        env.reset()

        with pytest.raises(EpisodeIsDoneError):
            env.step(None)
