"""
Procedural Math Problem Generator for SFT Training

Generates math problems at 1-2 operation difficulty across 4 types:
1. Direct arithmetic
2. Two-step arithmetic
3. Simple word problems (GSM8K-style patterns)
4. Number properties

All problems have verified ground truth answers.
All answers are integers (no fractions/decimals) to keep output simple.
"""

import math
import random
from dataclasses import dataclass, field


@dataclass
class MathProblem:
    question: str
    answer: int
    problem_type: str
    num_operations: int
    difficulty: str  # "trivial", "easy", "medium"
    metadata: dict = field(default_factory=dict)


# ============================================================
# Type 1: Direct Arithmetic
# ============================================================


def generate_direct_arithmetic(
    rng: random.Random, difficulty: str = "easy"
) -> MathProblem:
    """Single operation arithmetic: a op b = ?"""

    if difficulty == "trivial":
        ranges = {"add": (1, 50), "sub": (1, 50), "mul": (1, 12), "div_max": 50}
    elif difficulty == "easy":
        ranges = {"add": (1, 200), "sub": (1, 200), "mul": (1, 30), "div_max": 200}
    else:  # medium
        ranges = {"add": (10, 1000), "sub": (10, 1000), "mul": (2, 100), "div_max": 500}

    op = rng.choice(["+", "-", "*", "/"])

    if op == "+":
        lo, hi = ranges["add"]
        a, b = rng.randint(lo, hi), rng.randint(lo, hi)
        answer = a + b
        expr = f"{a} + {b}"

    elif op == "-":
        lo, hi = ranges["sub"]
        a, b = rng.randint(lo, hi), rng.randint(lo, hi)
        if a < b:
            a, b = b, a  # keep answer non-negative
        answer = a - b
        expr = f"{a} - {b}"

    elif op == "*":
        lo, hi = ranges["mul"]
        a, b = rng.randint(lo, hi), rng.randint(lo, hi)
        answer = a * b
        expr = f"{a} * {b}"

    else:  # division
        divisor = rng.randint(2, 20)
        quotient = rng.randint(1, ranges["div_max"] // max(divisor, 1))
        a = divisor * quotient  # ensure clean division
        answer = quotient
        expr = f"{a} / {divisor}"

    # Randomly phrase it
    phrasing = rng.choice(
        [
            f"What is {expr}?",
            f"Calculate {expr}.",
            f"Compute {expr}.",
            f"{expr} = ?",
        ]
    )

    return MathProblem(
        question=phrasing,
        answer=answer,
        problem_type="direct_arithmetic",
        num_operations=1,
        difficulty=difficulty,
        metadata={"expression": expr, "operator": op},
    )


# ============================================================
# Type 2: Two-Step Arithmetic
# ============================================================


def generate_twostep_arithmetic(
    rng: random.Random, difficulty: str = "easy"
) -> MathProblem:
    """Two operations: (a op1 b) op2 c = ?"""

    if difficulty == "easy":
        num_range = (1, 30)
    else:
        num_range = (2, 100)

    # Keep trying until we get a valid (integer, reasonable) answer
    for _ in range(50):
        a = rng.randint(*num_range)
        b = rng.randint(*num_range)
        c = rng.randint(*num_range)

        ops = ["+", "-", "*"]
        op1 = rng.choice(ops)
        op2 = rng.choice(ops)

        # Compute inner
        if op1 == "+":
            inner = a + b
        elif op1 == "-":
            inner = a - b
        else:
            inner = a * b

        # Compute outer
        if op2 == "+":
            answer = inner + c
        elif op2 == "-":
            answer = inner - c
        else:
            answer = inner * c

        # Reject if answer is negative, zero, or unreasonably large
        if answer < 0 or answer > 100000:
            continue

        expr = f"({a} {op1} {b}) {op2} {c}"

        phrasing = rng.choice(
            [
                f"What is {expr}?",
                f"Calculate {expr}.",
                f"Compute: first find {a} {op1} {b}, then {op2} {c}.",
                f"If you compute {a} {op1} {b} and then {op2} {c}, what do you get?",
            ]
        )

        return MathProblem(
            question=phrasing,
            answer=answer,
            problem_type="twostep_arithmetic",
            num_operations=2,
            difficulty=difficulty,
            metadata={"expression": expr, "op1": op1, "op2": op2},
        )

    # Fallback: simple addition chain
    a, b, c = rng.randint(1, 50), rng.randint(1, 50), rng.randint(1, 50)
    return MathProblem(
        question=f"What is {a} + {b} + {c}?",
        answer=a + b + c,
        problem_type="twostep_arithmetic",
        num_operations=2,
        difficulty=difficulty,
        metadata={"expression": f"{a} + {b} + {c}", "op1": "+", "op2": "+"},
    )


# ============================================================
# Type 3: Word Problems (GSM8K-style)
# ============================================================


def _generate_shopping_problem(rng: random.Random) -> MathProblem:
    """Buy items, compute cost or change."""
    items = [
        "apples",
        "oranges",
        "books",
        "pens",
        "tickets",
        "cookies",
        "pencils",
        "notebooks",
        "bottles of water",
        "sandwiches",
    ]
    names = [
        "Alice",
        "Bob",
        "Carlos",
        "Diana",
        "Emma",
        "Frank",
        "Grace",
        "Henry",
        "Ivy",
        "Jake",
    ]

    item = rng.choice(items)
    name = rng.choice(names)
    price = rng.randint(2, 15)
    quantity = rng.randint(2, 10)
    total_cost = price * quantity

    variant = rng.choice(["total", "change"])

    if variant == "total":
        question = (
            f"{name} buys {quantity} {item} at ${price} each. "
            f"How much does {name} spend in total?"
        )
        answer = total_cost
        metadata = {"variant": "total_cost"}
    else:
        valid_bills = [b for b in [20, 50, 100, 200] if b > total_cost]
        if not valid_bills:
            # Total too high for change variant, switch to total
            question = (
                f"{name} buys {quantity} {item} at ${price} each. "
                f"How much does {name} spend in total?"
            )
            return MathProblem(
                question=question,
                answer=total_cost,
                problem_type="word_problem",
                num_operations=1,
                difficulty="easy",
                metadata={"variant": "total_cost"},
            )
        bill = rng.choice(valid_bills)
        question = (
            f"{name} buys {quantity} {item} at ${price} each and pays with "
            f"a ${bill} bill. How much change does {name} get?"
        )
        answer = bill - total_cost
        metadata = {"variant": "change"}

    return MathProblem(
        question=question,
        answer=answer,
        problem_type="word_problem",
        num_operations=2 if variant == "change" else 1,
        difficulty="easy",
        metadata=metadata,
    )


def _generate_distribution_problem(rng: random.Random) -> MathProblem:
    """Split items among people, possibly with remainder used."""
    names = ["Alice", "Bob", "Carlos", "Diana", "Emma", "Frank"]
    items = ["stickers", "cards", "marbles", "candies", "toys", "coins"]

    name = rng.choice(names)
    item = rng.choice(items)
    num_friends = rng.randint(2, 8)

    # Ensure clean division
    per_person = rng.randint(2, 15)
    total = per_person * num_friends

    variant = rng.choice(["per_person", "with_remainder"])

    if variant == "per_person":
        question = (
            f"{name} has {total} {item} and wants to share them equally "
            f"among {num_friends} friends. How many {item} does each friend get?"
        )
        answer = per_person
    else:
        kept = rng.randint(1, 10)
        total_with_kept = total + kept
        question = (
            f"{name} has {total_with_kept} {item}. After keeping {kept} for "
            f"themselves, {name} splits the rest equally among {num_friends} "
            f"friends. How many does each friend get?"
        )
        answer = per_person

    return MathProblem(
        question=question,
        answer=answer,
        problem_type="word_problem",
        num_operations=2 if variant == "with_remainder" else 1,
        difficulty="easy",
        metadata={"variant": variant},
    )


def _generate_comparison_problem(rng: random.Random) -> MathProblem:
    """Compare quantities, compute difference or ratio."""
    names = ["Alice", "Bob", "Carlos", "Diana"]
    items = ["books", "points", "coins", "stickers", "miles"]

    n1, n2 = rng.sample(names, 2)
    item = rng.choice(items)

    variant = rng.choice(["difference", "multiple", "combined"])

    if variant == "difference":
        a = rng.randint(10, 100)
        b = rng.randint(10, 100)
        if a < b:
            a, b = b, a
        question = (
            f"{n1} has {a} {item} and {n2} has {b} {item}. "
            f"How many more {item} does {n1} have than {n2}?"
        )
        answer = a - b

    elif variant == "multiple":
        b = rng.randint(3, 20)
        multiplier = rng.randint(2, 5)
        a = b * multiplier
        question = (
            f"{n1} has {a} {item}. {n2} has {b} {item}. "
            f"How many times more {item} does {n1} have compared to {n2}?"
        )
        answer = multiplier

    else:  # combined
        a = rng.randint(5, 50)
        b = rng.randint(5, 50)
        question = (
            f"{n1} has {a} {item} and {n2} has {b} {item}. "
            f"How many {item} do they have combined?"
        )
        answer = a + b

    return MathProblem(
        question=question,
        answer=answer,
        problem_type="word_problem",
        num_operations=1,
        difficulty="easy",
        metadata={"variant": variant},
    )


def _generate_rate_problem(rng: random.Random) -> MathProblem:
    """Rate * time = total, or total / rate = time."""
    scenarios = [
        ("reads {rate} pages per hour", "pages", "hours"),
        ("drives {rate} miles per hour", "miles", "hours"),
        ("earns ${rate} per hour", "dollars", "hours"),
        ("makes {rate} cookies per batch", "cookies", "batches"),
        ("scores {rate} points per game", "points", "games"),
        ("types {rate} words per minute", "words", "minutes"),
    ]

    template, unit, time_unit = rng.choice(scenarios)
    names = ["Alice", "Bob", "Carlos", "Diana", "Emma"]
    name = rng.choice(names)

    rate = rng.randint(3, 50)
    time = rng.randint(2, 12)
    total = rate * time

    variant = rng.choice(["find_total", "find_time"])

    if variant == "find_total":
        action = template.format(rate=rate)
        question = f"{name} {action}. How many {unit} after {time} {time_unit}?"
        answer = total
    else:
        action = template.format(rate=rate)
        question = (
            f"{name} {action}. How many {time_unit} does it take "
            f"to reach {total} {unit}?"
        )
        answer = time

    return MathProblem(
        question=question,
        answer=answer,
        problem_type="word_problem",
        num_operations=1,
        difficulty="easy",
        metadata={"variant": variant, "rate": rate, "time": time},
    )


def _generate_multi_step_word_problem(rng: random.Random) -> MathProblem:
    """Two-step word problems closer to GSM8K difficulty."""
    variant = rng.choice(["earn_spend", "collect_distribute", "growth"])

    names = ["Alice", "Bob", "Carlos", "Diana", "Emma", "Frank"]
    name = rng.choice(names)

    if variant == "earn_spend":
        hourly = rng.randint(8, 25)
        hours = rng.randint(3, 10)
        spent = rng.randint(10, hourly * hours - 5)
        earned = hourly * hours
        answer = earned - spent
        question = (
            f"{name} earns ${hourly} per hour and works {hours} hours. "
            f"After work, {name} spends ${spent} on groceries. "
            f"How much money does {name} have left?"
        )

    elif variant == "collect_distribute":
        days = rng.randint(3, 7)
        per_day = rng.randint(5, 20)
        total_collected = days * per_day
        gave_away = rng.randint(1, total_collected // 2)
        answer = total_collected - gave_away
        question = (
            f"{name} collects {per_day} seashells each day for {days} days. "
            f"Then {name} gives {gave_away} seashells to a friend. "
            f"How many seashells does {name} have now?"
        )

    else:  # growth
        initial = rng.randint(10, 100)
        added = rng.randint(5, 30)
        multiplier = rng.randint(2, 4)
        after_add = initial + added
        answer = after_add * multiplier
        question = (
            f"{name} starts with {initial} points. After a bonus round, "
            f"{name} gains {added} more points. In the final round, "
            f"{name}'s score is multiplied by {multiplier}. "
            f"What is {name}'s final score?"
        )

    return MathProblem(
        question=question,
        answer=answer,
        problem_type="word_problem",
        num_operations=2,
        difficulty="medium",
        metadata={"variant": variant},
    )


def generate_word_problem(rng: random.Random, difficulty: str = "easy") -> MathProblem:
    """Generate a random word problem from available templates."""
    if difficulty == "medium":
        generators = [_generate_multi_step_word_problem]
    else:
        generators = [
            _generate_shopping_problem,
            _generate_distribution_problem,
            _generate_comparison_problem,
            _generate_rate_problem,
        ]

    gen = rng.choice(generators)
    return gen(rng)


# ============================================================
# Type 4: Number Properties
# ============================================================


def generate_number_properties(
    rng: random.Random, difficulty: str = "easy"
) -> MathProblem:
    """Questions about remainders, factors, divisibility, squares, etc."""

    if difficulty == "trivial":
        num_range = (2, 50)
    elif difficulty == "easy":
        num_range = (2, 200)
    else:
        num_range = (10, 500)

    variant = rng.choice(
        ["remainder", "largest_factor", "square", "is_divisible", "gcd"]
    )

    if variant == "remainder":
        divisor = rng.randint(3, 15)
        number = rng.randint(num_range[0], num_range[1])
        answer = number % divisor
        question = f"What is the remainder when {number} is divided by {divisor}?"

    elif variant == "largest_factor":
        # Pick a composite number and find its largest factor below some threshold
        threshold = rng.randint(5, 15)
        number = rng.randint(threshold + 1, 200)
        factors = [i for i in range(2, min(threshold + 1, number)) if number % i == 0]
        if not factors:
            # Fallback
            number = 36
            threshold = 10
            factors = [i for i in range(2, threshold + 1) if 36 % i == 0]
        answer = max(factors)
        question = (
            f"What is the largest factor of {number} that is less than {threshold + 1}?"
        )

    elif variant == "square":
        base = rng.randint(2, 20 if difficulty == "easy" else 30)
        answer = base * base
        question = rng.choice(
            [
                f"What is {base} squared?",
                f"What is {base} * {base}?",
                f"Compute {base}^2.",
            ]
        )

    elif variant == "is_divisible":
        divisor = rng.choice([2, 3, 4, 5, 6, 7, 8, 9])
        # Generate a number that may or may not be divisible
        if rng.random() < 0.5:
            quotient = rng.randint(2, 50)
            number = divisor * quotient
            answer = 1  # yes, divisible (we use 1 for yes, 0 for no)
        else:
            number = rng.randint(num_range[0], num_range[1])
            while number % divisor == 0:
                number = rng.randint(num_range[0], num_range[1])
            answer = 0
        question = f"Is {number} divisible by {divisor}? Answer 1 for yes, 0 for no."

    else:  # gcd
        a = rng.randint(6, 100)
        b = rng.randint(6, 100)

        answer = math.gcd(a, b)
        question = f"What is the greatest common divisor of {a} and {b}?"

    return MathProblem(
        question=question,
        answer=answer,
        problem_type="number_properties",
        num_operations=1,
        difficulty=difficulty,
        metadata={"variant": variant},
    )


@dataclass
class MathDatasetConfig:
    """Configuration for the math problem mix."""

    mix: dict = field(
        default_factory=lambda: {
            "direct_arithmetic": 0.20,
            "twostep_arithmetic": 0.30,
            "word_problem": 0.35,
            "number_properties": 0.15,
        }
    )
    difficulty: str = "easy"


GENERATORS = {
    "direct_arithmetic": generate_direct_arithmetic,
    "twostep_arithmetic": generate_twostep_arithmetic,
    "word_problem": generate_word_problem,
    "number_properties": generate_number_properties,
}


def generate_problem(config: MathDatasetConfig, rng: random.Random) -> MathProblem:
    """Generate a single problem according to the mix distribution."""
    r = rng.random()
    cumulative = 0.0
    for problem_type, weight in config.mix.items():
        cumulative += weight
        if r < cumulative:
            generator = GENERATORS[problem_type]
            return generator(rng, difficulty=config.difficulty)

    # Fallback to last type
    last_type = list(config.mix.keys())[-1]
    return GENERATORS[last_type](rng, difficulty=config.difficulty)


def generate_dataset(config: MathDatasetConfig, n: int) -> list[MathProblem]:
    """Generate n problems according to config."""
    rng = random.Random(config.seed)
    problems = []
    for _ in range(n):
        problems.append(generate_problem(config, rng))
    return problems


def format_for_sft(problem: MathProblem, include_answer: bool = True) -> dict:
    """
    Format a problem for SFT training.
    Returns prompt and answer as strings.
    """
    prompt = f"Question: {problem.question}\nAnswer:"
    answer = str(problem.answer)

    return {
        "prompt": prompt,
        "answer": answer,
        "full": f"{prompt} {answer}" if include_answer else prompt,
        "problem_type": problem.problem_type,
        "num_operations": problem.num_operations,
    }


TRIVIAL_CONFIG = MathDatasetConfig(
    mix={
        "direct_arithmetic": 0.50,
        "twostep_arithmetic": 0.00,
        "word_problem": 0.25,
        "number_properties": 0.25,
    },
    difficulty="trivial",
)

EASY_CONFIG = MathDatasetConfig(
    mix={
        "direct_arithmetic": 0.20,
        "twostep_arithmetic": 0.30,
        "word_problem": 0.35,
        "number_properties": 0.15,
    },
    difficulty="easy",
)

MEDIUM_CONFIG = MathDatasetConfig(
    mix={
        "direct_arithmetic": 0.15,
        "twostep_arithmetic": 0.30,
        "word_problem": 0.40,
        "number_properties": 0.15,
    },
    difficulty="medium",
)
