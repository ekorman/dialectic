import re


def parse_cot_and_answer(text: str) -> tuple[str, str] | None:
    """Split model output into (CoT, answer).

    Tries <think>...</think> format first (CoT is inside think tags,
    answer is everything after). Falls back to splitting at <answer> tags.
    Returns None if neither format is found.
    """
    think_match = re.search(r"<think>(.*?)</think>(.*)", text, re.DOTALL)
    if think_match is not None:
        cot = think_match.group(1).strip()
        answer = think_match.group(2).strip()
        if cot and answer:
            return cot, answer

    answer_match = re.search(r"<answer>.*?</answer>", text, re.DOTALL)
    if answer_match is not None:
        cot = text[: answer_match.start()]
        answer = answer_match.group(0)
        if cot.strip():
            return cot, answer

    return None


def extract_from_boxed(text: str) -> str:
    """
    Extract content from within \boxed{...} in LaTeX format.
    """
    pattern = r"\\boxed\{([^}]*)\}"
    matches = re.findall(pattern, text)
    if matches:
        return matches[0].strip()
    else:
        raise ValueError("No boxed content found in the text.")


def extract_from_answer_tags(text: str) -> str | None:
    """Extract content from <answer>...</answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None
