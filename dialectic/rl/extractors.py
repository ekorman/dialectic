import re


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
