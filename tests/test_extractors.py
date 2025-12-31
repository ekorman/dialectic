import pytest

from dialectic.rl.extractors import extract_from_boxed


def test_extract_from_boxed():
    text = "The result is \\boxed{42} which is the answer."
    extracted = extract_from_boxed(text)
    assert extracted == "42"

    text_no_boxed = "There is no boxed content here."
    with pytest.raises(ValueError) as excinfo:
        extract_from_boxed(text_no_boxed)

    assert "No boxed content" in str(excinfo.value)
