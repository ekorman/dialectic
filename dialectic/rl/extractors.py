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
