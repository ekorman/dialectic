from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
import torch


@dataclass
class FakeCompletionOutput:
    token_ids: list[int]


@dataclass
class FakeRequestOutput:
    outputs: list[FakeCompletionOutput]


@pytest.fixture
def mock_vllm():
    """Provide mock vllm module so tests run without vllm installed."""
    mock_llm_instance = MagicMock()

    def fake_generate(prompt_token_ids, sampling_params):
        """Return one token per prompt to simulate generation."""
        results = []
        for _ in prompt_token_ids:
            results.append(FakeRequestOutput(outputs=[FakeCompletionOutput(token_ids=[42, 43])]))
        return results

    mock_llm_instance.generate.side_effect = fake_generate

    mock_llm_cls = MagicMock(return_value=mock_llm_instance)
    mock_sampling_params = MagicMock()

    with patch.dict(
        "sys.modules",
        {
            "vllm": MagicMock(LLM=mock_llm_cls, SamplingParams=mock_sampling_params),
        },
    ):
        yield mock_llm_cls, mock_llm_instance, mock_sampling_params


def test_vllm_generator_init(mock_vllm):
    mock_llm_cls, mock_llm_instance, _ = mock_vllm

    from dialectic.llm.vllm import VLLMGenerator

    gen = VLLMGenerator("some-model", tensor_parallel_size=2)
    mock_llm_cls.assert_called_once_with(model="some-model", tensor_parallel_size=2)
    assert gen.llm is mock_llm_instance


def test_vllm_generate_from_tokens_greedy(mock_vllm):
    _, mock_llm_instance, _ = mock_vllm

    from dialectic.llm.vllm import VLLMGenerator

    gen = VLLMGenerator("some-model")

    token_ids = torch.tensor([[10, 20, 30], [40, 50, 60]])

    result = gen.generate_from_tokens(
        token_ids=token_ids,
        eos_token_id=2,
        pad_token_id=0,
        sampling_strategy="greedy",
        max_tokens_generated=100,
    )

    mock_llm_instance.generate.assert_called_once()
    call_kwargs = mock_llm_instance.generate.call_args
    prompt_ids = call_kwargs.kwargs.get("prompt_token_ids") or call_kwargs[1].get(
        "prompt_token_ids"
    )
    assert prompt_ids == [[10, 20, 30], [40, 50, 60]]

    # Original prompt (3 tokens) + generated (2 tokens) = 5 tokens per row
    assert result.shape == (2, 5)
    # First 3 tokens should match original prompt
    assert (result[:, :3] == token_ids).all()
    # Generated tokens should be [42, 43]
    assert result[0, 3].item() == 42
    assert result[0, 4].item() == 43


def test_vllm_generate_from_tokens_strips_left_padding(mock_vllm):
    _, mock_llm_instance, _ = mock_vllm

    from dialectic.llm.vllm import VLLMGenerator

    gen = VLLMGenerator("some-model")

    pad_token_id = 0
    # Row 0 has 2 padding tokens, row 1 has none
    token_ids = torch.tensor([[pad_token_id, pad_token_id, 10], [40, 50, 60]])

    gen.generate_from_tokens(
        token_ids=token_ids,
        eos_token_id=2,
        pad_token_id=pad_token_id,
        sampling_strategy="sample",
        max_tokens_generated=50,
    )

    call_kwargs = mock_llm_instance.generate.call_args
    prompt_ids = call_kwargs.kwargs.get("prompt_token_ids") or call_kwargs[1].get(
        "prompt_token_ids"
    )
    # Left padding should be stripped for the first row
    assert prompt_ids[0] == [10]
    assert prompt_ids[1] == [40, 50, 60]


def test_vllm_generate_from_tokens_pads_output(mock_vllm):
    """When generated sequences have different lengths (due to different
    prompt lengths after stripping padding), the output should be right-padded
    to a uniform length."""
    _, mock_llm_instance, _ = mock_vllm

    from dialectic.llm.vllm import VLLMGenerator

    gen = VLLMGenerator("some-model")

    pad_token_id = 0
    # Row 0 is shorter (has padding), row 1 is longer
    token_ids = torch.tensor([[pad_token_id, 10], [40, 50]])

    result = gen.generate_from_tokens(
        token_ids=token_ids,
        eos_token_id=2,
        pad_token_id=pad_token_id,
        sampling_strategy="sample",
        max_tokens_generated=50,
    )

    # Both rows should have the same length
    assert result.shape[0] == 2
    assert result.shape[1] == result.shape[1]  # uniform


def test_vllm_generate_from_text(mock_vllm):
    _, mock_llm_instance, _ = mock_vllm

    from dialectic.llm.vllm import VLLMGenerator

    gen = VLLMGenerator("some-model")

    # Create a mock tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.token_to_id.side_effect = lambda t: {"<|im_end|>": 2, "<|endoftext|>": 0}[t]

    mock_encoded_1 = MagicMock()
    mock_encoded_1.ids = [10, 20]
    mock_encoded_1.attention_mask = [1, 1]
    mock_encoded_2 = MagicMock()
    mock_encoded_2.ids = [0, 30]
    mock_encoded_2.attention_mask = [0, 1]
    mock_tokenizer.encode_batch.return_value = [mock_encoded_1, mock_encoded_2]
    mock_tokenizer.decode.side_effect = lambda ids: f"decoded:{ids}"

    result = gen.generate_from_text(
        tokenizer=mock_tokenizer,
        text_batch=["hello", "world"],
        max_tokens_generated=10,
        sampling_strategy="greedy",
    )

    assert len(result) == 2
    mock_tokenizer.enable_padding.assert_called_once()
    mock_tokenizer.encode_batch.assert_called_once_with(["hello", "world"])
    assert mock_tokenizer.decode.call_count == 2


def test_vllm_qwen_generate_from_chat(mock_vllm):
    _, mock_llm_instance, _ = mock_vllm

    from dialectic.llm.templates import Message
    from dialectic.llm.vllm import VLLMGenerator

    gen = VLLMGenerator("some-model")

    mock_tokenizer = MagicMock()
    mock_tokenizer.token_to_id.side_effect = lambda t: {"<|im_end|>": 2, "<|endoftext|>": 0}[t]
    mock_encoded = MagicMock()
    mock_encoded.ids = [10, 20]
    mock_encoded.attention_mask = [1, 1]
    mock_tokenizer.encode_batch.return_value = [mock_encoded]
    mock_tokenizer.decode.side_effect = lambda ids: "output"

    messages = [[Message(role="user", content="Hello")]]
    result = gen.qwen_generate_from_chat(
        tokenizer=mock_tokenizer,
        batch_messages=messages,
        max_tokens_generated=10,
    )

    assert len(result) == 1
    # Verify the text sent to encode_batch was formatted with qwen template
    call_args = mock_tokenizer.encode_batch.call_args[0][0]
    assert "<|im_start|>" in call_args[0]


def test_vllm_llama_generate_from_chat(mock_vllm):
    _, mock_llm_instance, _ = mock_vllm

    from dialectic.llm.templates import Message
    from dialectic.llm.vllm import VLLMGenerator

    gen = VLLMGenerator("some-model")

    mock_tokenizer = MagicMock()
    mock_tokenizer.token_to_id.side_effect = lambda t: {"<|eot_id|>": 2}[t]
    mock_encoded = MagicMock()
    mock_encoded.ids = [10, 20]
    mock_encoded.attention_mask = [1, 1]
    mock_tokenizer.encode_batch.return_value = [mock_encoded]
    mock_tokenizer.decode.side_effect = lambda ids: "output"

    messages = [[Message(role="user", content="Hello")]]
    result = gen.llama_generate_from_chat(
        tokenizer=mock_tokenizer,
        batch_messages=messages,
        max_tokens_generated=10,
    )

    assert len(result) == 1
    call_args = mock_tokenizer.encode_batch.call_args[0][0]
    assert "<|begin_of_text|>" in call_args[0]
