import json
import tempfile
from pathlib import Path

import pytest

from dialectic.rl.env import ArithmeticEnv, EpisodeIsDoneError, GSM8kEnv


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
    return GSM8kEnv(sample_data, shuffle=False)


class TestGSM8kEnv:
    def test_load_data(self, env: GSM8kEnv):
        """Test that data is loaded correctly."""
        assert len(env.problems) == 3
        assert "question" in env.problems[0]
        assert "answer" in env.problems[0]

    def test_reset_returns_question(self, env: GSM8kEnv):
        """Test that reset returns a question string."""
        obs, info = env.reset(seed=42)
        assert isinstance(obs, str)
        assert len(obs) > 0
        assert "index" in info

    def test_step_correct_answer(self, env: GSM8kEnv):
        """Test that correct answers get reward 1.0."""
        env.reset(seed=42)
        # Find which problem we got and answer correctly
        correct_answer = env._extract_answer(env.current_problem["answer"])

        obs, reward, terminated, truncated, info = env.step(correct_answer)

        assert reward == 1.0
        assert terminated is True
        assert truncated is False
        assert info["correct"] is True
        assert info["correct_answer"] == correct_answer

    def test_step_incorrect_answer(self, env: GSM8kEnv):
        """Test that incorrect answers get reward 0.0."""
        env.reset(seed=42)

        obs, reward, terminated, truncated, info = env.step("999999")

        assert reward == 0.0
        assert terminated is True
        assert info["correct"] is False

    def test_step_without_reset_raises(self, sample_data: str):
        """Test that stepping without reset raises an error."""
        env = GSM8kEnv(sample_data)
        with pytest.raises(RuntimeError, match="Must call reset"):
            env.step("#### 42")

    def test_extract_answer_with_delimiter(self, env: GSM8kEnv):
        """Test answer extraction with #### delimiter."""
        assert env._extract_answer("The answer is #### 42") == "42"
        assert env._extract_answer("#### 123") == "123"
        assert env._extract_answer("Some work\n#### 999") == "999"

    def test_extract_answer_trailing_number(self, env: GSM8kEnv):
        """Test answer extraction with trailing number."""
        assert env._extract_answer("The answer is 42") == "42"
        assert env._extract_answer("Result: 123") == "123"

    def test_extract_answer_with_commas(self, env: GSM8kEnv):
        """Test answer extraction handles comma-separated numbers."""
        assert env._extract_answer("#### 1,000") == "1000"
        assert env._extract_answer("#### 1,234,567") == "1234567"

    def test_extract_answer_negative(self, env: GSM8kEnv):
        """Test answer extraction handles negative numbers."""
        assert env._extract_answer("#### -42") == "-42"
        assert env._extract_answer("The answer is -100") == "-100"

    def test_extract_answer_decimal(self, env: GSM8kEnv):
        """Test answer extraction handles decimals."""
        assert env._extract_answer("#### 3.14") == "3.14"
        assert env._extract_answer("The result is 2.5") == "2.5"

    def test_extract_answer_no_number(self, env: GSM8kEnv):
        """Test answer extraction returns None when no number found."""
        assert env._extract_answer("no numbers here") is None
        assert env._extract_answer("") is None

    def test_cycles_through_all_problems(self, env: GSM8kEnv):
        """Test that environment cycles through all problems."""
        seen_questions = set()

        for _ in range(3):
            obs, _ = env.reset()
            seen_questions.add(obs)
            env.step("#### 0")  # dummy answer

        assert len(seen_questions) == 3

    def test_refills_after_exhaustion(self, env: GSM8kEnv):
        """Test that indices refill after all problems are used."""
        # Use all problems
        for _ in range(3):
            env.reset()
            env.step("#### 0")

        # Should be able to continue
        obs, _ = env.reset()
        assert isinstance(obs, str)
        assert len(obs) > 0

    def test_seed_reproducibility(self, sample_data: str):
        """Test that same seed produces same sequence."""
        env1 = GSM8kEnv(sample_data, shuffle=True)
        env2 = GSM8kEnv(sample_data, shuffle=True)

        obs1, _ = env1.reset(seed=123)
        obs2, _ = env2.reset(seed=123)

        assert obs1 == obs2


def test_cleanup(sample_data: str):
    """Clean up temporary file."""
    Path(sample_data).unlink(missing_ok=True)


class TestArithmeticEnv:
    def test_reset_returns_question(self):
        """Test that reset returns a question string."""
        env = ArithmeticEnv()
        obs, info = env.reset(seed=42)

        assert isinstance(obs, str)
        assert obs.startswith("What is")
        assert obs.endswith("?")
        assert "answer" in info

    def test_step_correct_answer(self):
        """Test that correct answers get reward 1.0."""
        env = ArithmeticEnv()
        _, info = env.reset(seed=42)
        correct_answer = info["answer"]

        obs, reward, terminated, truncated, info = env.step(str(correct_answer))

        assert reward == 1.0
        assert terminated is True
        assert truncated is False
        assert info["correct"] is True

    def test_step_incorrect_answer(self):
        """Test that incorrect answers get reward 0.0."""
        env = ArithmeticEnv(min_value=1, max_value=10)
        env.reset(seed=42)

        # Use an answer that's definitely wrong
        obs, reward, terminated, truncated, info = env.step("999999")

        assert reward == 0.0
        assert terminated is True
        assert info["correct"] is False

    def test_step_without_reset_raises(self):
        """Test that stepping without reset raises an error."""
        env = ArithmeticEnv()
        with pytest.raises(RuntimeError, match="Must call reset"):
            env.step("42")

    def test_addition_only(self):
        """Test environment with only addition."""
        env = ArithmeticEnv(min_value=1, max_value=10, operations=("+",))
        obs, info = env.reset(seed=42)

        assert "+" in obs
        assert "-" not in obs
        assert "*" not in obs

    def test_subtraction_only(self):
        """Test environment with only subtraction."""
        env = ArithmeticEnv(min_value=1, max_value=10, operations=("-",))
        obs, info = env.reset(seed=42)

        assert "-" in obs
        assert "+" not in obs

    def test_multiplication(self):
        """Test environment with multiplication."""
        env = ArithmeticEnv(min_value=1, max_value=10, operations=("*",))
        obs, info = env.reset(seed=42)

        assert "*" in obs

    def test_multiple_operands(self):
        """Test environment with more than 2 operands."""
        env = ArithmeticEnv(
            min_value=1, max_value=10, operations=("+",), num_operands=3
        )
        obs, info = env.reset(seed=42)

        # Should have 2 plus signs for 3 operands
        assert obs.count("+") == 2

    def test_seed_reproducibility(self):
        """Test that same seed produces same problem."""
        env1 = ArithmeticEnv()
        env2 = ArithmeticEnv()

        obs1, info1 = env1.reset(seed=123)
        obs2, info2 = env2.reset(seed=123)

        assert obs1 == obs2
        assert info1["answer"] == info2["answer"]

    def test_answer_as_string(self):
        """Test that answers can be passed as strings."""
        env = ArithmeticEnv()
        _, info = env.reset(seed=42)
        correct = info["answer"]

        # Answer passed as string
        _, reward, _, _, _ = env.step(str(correct))
        assert reward == 1.0

    def test_negative_results(self):
        """Test that negative results are handled correctly."""
        env = ArithmeticEnv(min_value=0, max_value=10, operations=("-",))

        # Generate problems until we get a negative result
        for seed in range(100):
            _, info = env.reset(seed=seed)
            if info["answer"] < 0:
                # Test that we can answer correctly
                _, reward, _, _, _ = env.step(str(info["answer"]))
                assert reward == 1.0
                break

    def test_value_range(self):
        """Test that operands are within specified range."""
        env = ArithmeticEnv(min_value=10, max_value=20, operations=("+",))

        for seed in range(10):
            obs, _ = env.reset(seed=seed)
            # Extract numbers from "What is X + Y?"
            numbers = [int(n) for n in obs.replace("?", "").split() if n.isdigit()]
            for n in numbers:
                assert 10 <= n <= 20


def test_arithmetic_env():
    env = ArithmeticEnv()
    s = env.reset()
    assert s.is_done
    assert s.data.question.startswith("What is")
    assert isinstance(s.data.answer, float)

    with pytest.raises(EpisodeIsDoneError):
        env.step()
