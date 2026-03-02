from dataclasses import dataclass
from typing import Callable

from tokenizers import Tokenizer

from dialectic.artifacts import Artifact, get_artifact
from dialectic.llm.llama import (
    LLAMA_32_TOKENIZER,
    load_llama_32_1b_instruct,
    load_llama_32_3b_instruct,
)
from dialectic.llm.qwen import load_qwen3_06b, load_qwen3_17b
from dialectic.llm.templates import (
    Message,
    get_llama_input_text_from_messages,
    get_qwen_input_text_from_messages,
)


@dataclass
class ModelInfo:
    net_factory: Callable
    tokenizer: str | Artifact
    eos_token_id: int
    pad_token_id: int
    format_messages: Callable[[list[Message], bool], str]

    def load_net(self, **kwargs):
        return self.net_factory(**kwargs)

    def load_tokenizer(self) -> Tokenizer:
        if isinstance(self.tokenizer, str):
            return Tokenizer.from_pretrained(self.tokenizer)
        return Tokenizer.from_file(str(get_artifact(self.tokenizer)[0]))


MODEL_REGISTRY: dict[str, ModelInfo] = {
    "qwen3-0.6b": ModelInfo(
        net_factory=lambda **kw: load_qwen3_06b(True, **kw),
        tokenizer="Qwen/Qwen3-0.6B",
        eos_token_id=151645,
        pad_token_id=151643,
        format_messages=lambda msgs, gen: get_qwen_input_text_from_messages(
            msgs, gen, enable_thinking=False
        ),
    ),
    "qwen3-1.7b": ModelInfo(
        net_factory=lambda **kw: load_qwen3_17b(True, **kw),
        tokenizer="Qwen/Qwen3-1.7B",
        eos_token_id=151645,
        pad_token_id=151643,
        format_messages=lambda msgs, gen: get_qwen_input_text_from_messages(
            msgs, gen, enable_thinking=False
        ),
    ),
    "llama-3.2-1b-instruct": ModelInfo(
        net_factory=lambda **kw: load_llama_32_1b_instruct(True, **kw),
        tokenizer=LLAMA_32_TOKENIZER,
        eos_token_id=128009,
        pad_token_id=128009,
        format_messages=lambda msgs, gen: get_llama_input_text_from_messages(msgs, gen),
    ),
    "llama-3.2-3b-instruct": ModelInfo(
        net_factory=lambda **kw: load_llama_32_3b_instruct(True, **kw),
        tokenizer=LLAMA_32_TOKENIZER,
        eos_token_id=128009,
        pad_token_id=128009,
        format_messages=lambda msgs, gen: get_llama_input_text_from_messages(msgs, gen),
    ),
}
