from tokenizers import Tokenizer

from dialectic.rl.env import CountdownStep


def extract_separator_token_id(tokenizer: Tokenizer) -> int:
    """Extract the separator token from a sample training step.

    Derives the token from the same format used in training
    (``CountdownStep.format_step()``) so it is guaranteed to match.
    """
    sample = CountdownStep(left=1, op="+", right=2, result=3).format_step()
    ids = tokenizer.encode(sample, add_special_tokens=False).ids
    return ids[-1]
