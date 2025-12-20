from dataclasses import dataclass
from typing import Literal


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
