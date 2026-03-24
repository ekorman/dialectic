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


def extract_from_a_line(text: str) -> str | None:
    """Extract the expression from the last line starting with 'A '.

    For the simplified grammar format where answer lines are ``A expr``.
    """
    for line in reversed(text.split("\n")):
        line = line.strip()
        if line.startswith("A "):
            return line[2:].strip()
    return None


def extract_countdown_answer(text: str) -> str | None:
    """Extract countdown answer from either tag or simplified format."""
    result = extract_from_answer_tags(text)
    if result is not None:
        return result
    return extract_from_a_line(text)


_MOVE_ALIASES: dict[str, str] = {
    "u": "up",
    "d": "down",
    "l": "left",
    "r": "right",
    "up": "up",
    "down": "down",
    "left": "left",
    "right": "right",
}


def extract_maze_moves(text: str) -> list[str] | None:
    """
    Extract maze move sequence from <answer> tags.

    Parses comma/space/newline separated directional tokens
    (up/down/left/right or U/D/L/R, case-insensitive).

    Parameters
    ----------
    text : str
        Raw model output.

    Returns
    -------
    list[str] or None
        List of normalized moves, or None if no answer tags or no valid tokens.
    """
    inner = extract_from_answer_tags(text)
    if inner is None:
        return None

    tokens = re.split(r"[,\s]+", inner.lower())
    moves = [_MOVE_ALIASES[t] for t in tokens if t in _MOVE_ALIASES]
    return moves if moves else None
