from dataclasses import dataclass
from typing import Literal

from tokenizers import Encoding, Tokenizer
from transformers import AutoTokenizer


@dataclass
class Message:
    role: Literal["user"]
    content: str


def get_input_text_from_messages(
    messages: list[Message], add_generation_prompt: bool
) -> str:
    # TODO: implement other types of messages
    t = ""
    for message in messages:
        if message.role == "user":
            t += f"<|im_start|>user\n{message.content}<|im_end|>\n"
        else:
            raise ValueError
    if add_generation_prompt:
        t += "<|im_start|>assistant\n"

    return t


def test_tokenizer():
    auto_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    tokenizer: Tokenizer = Tokenizer.from_file("qwen-tokenizer/tokenizer.json")

    prompt = "Give me a short introduction to large language model."
    messages = [{"role": "user", "content": prompt}]
    text = auto_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,  # Switches between thinking and non-thinking modes. Default is True.
    )

    assert text == get_input_text_from_messages(
        messages=[Message(**m) for m in messages], add_generation_prompt=True
    )

    x: Encoding = tokenizer.encode_batch([text])  # or could do .encode(text)
    y = auto_tokenizer([text], return_tensors="pt")

    assert y.tokens() == x[0].tokens
    assert y.input_ids.tolist()[0] == x[0].ids
