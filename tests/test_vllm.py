from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import torch

from dialectic.llm.qwen import create_qwen


@dataclass
class FakeCompletionOutput:
    token_ids: list[int]


@dataclass
class FakeRequestOutput:
    outputs: list[FakeCompletionOutput]


def _make_tiny_model():
    return create_qwen(
        d=32,
        vocab_size=500,
        n_decoder_layers=2,
        attn_head_d=16,
        attn_num_heads=4,
        attn_num_kv_heads=2,
        mlp_hidden_d=64,
    )


def _make_mock_vllm():
    """Return (mock_llm_cls, mock_llm_instance, mock_sampling_params, patcher)."""
    mock_llm_instance = MagicMock()

    def fake_generate(prompt_token_ids, sampling_params):
        results = []
        for _ in prompt_token_ids:
            results.append(
                FakeRequestOutput(outputs=[FakeCompletionOutput(token_ids=[42, 43])])
            )
        return results

    mock_llm_instance.generate.side_effect = fake_generate
    mock_llm_cls = MagicMock(return_value=mock_llm_instance)
    mock_sampling_params = MagicMock()
    mock_save_file = MagicMock()

    patcher = patch.dict(
        "sys.modules",
        {
            "vllm": MagicMock(LLM=mock_llm_cls, SamplingParams=mock_sampling_params),
            "safetensors": MagicMock(),
            "safetensors.torch": MagicMock(save_file=mock_save_file),
        },
    )
    return mock_llm_cls, mock_llm_instance, mock_sampling_params, mock_save_file, patcher


def test_vllm_generator_init():
    mock_llm_cls, mock_llm_instance, _, mock_save_file, patcher = _make_mock_vllm()
    with patcher:
        from dialectic.llm.vllm import VLLMGenerator

        model = _make_tiny_model()
        gen = VLLMGenerator(model, model_type="qwen3", tensor_parallel_size=2)

        # vLLM LLM should be called with a temp directory path + extra kwargs
        assert mock_llm_cls.call_count == 1
        call_kwargs = mock_llm_cls.call_args
        assert "tensor_parallel_size" in call_kwargs.kwargs
        assert call_kwargs.kwargs["tensor_parallel_size"] == 2

        # Weights should have been saved via safetensors
        assert mock_save_file.call_count == 1
        saved_state_dict = mock_save_file.call_args[0][0]
        # Keys should be in HF format (model.* prefix except lm_head)
        assert "model.embed_tokens.weight" in saved_state_dict
        assert "model.layers.0.self_attn.q_proj.weight" in saved_state_dict
        assert "lm_head.weight" in saved_state_dict

        assert gen.llm is mock_llm_instance


def test_vllm_config_generation():
    _, _, _, _, patcher = _make_mock_vllm()
    with patcher:
        from dialectic.llm.vllm import _make_hf_config

        model = _make_tiny_model()
        config = _make_hf_config(model, "qwen3")

        assert config["architectures"] == ["Qwen3ForCausalLM"]
        assert config["model_type"] == "qwen3"
        assert config["hidden_size"] == 32
        assert config["num_hidden_layers"] == 2
        assert config["num_attention_heads"] == 4
        assert config["num_key_value_heads"] == 2
        assert config["head_dim"] == 16
        assert config["intermediate_size"] == 64
        assert config["vocab_size"] == 500


def test_vllm_config_generation_llama():
    _, _, _, _, patcher = _make_mock_vllm()
    with patcher:
        from dialectic.llm.vllm import _make_hf_config

        model = _make_tiny_model()
        config = _make_hf_config(model, "llama")

        assert config["architectures"] == ["LlamaForCausalLM"]
        assert config["model_type"] == "llama"
        assert "rope_scaling" in config


def test_vllm_generate_from_tokens_greedy():
    _, mock_llm_instance, _, _, patcher = _make_mock_vllm()
    with patcher:
        from dialectic.llm.vllm import VLLMGenerator

        model = _make_tiny_model()
        gen = VLLMGenerator(model, model_type="qwen3")

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
        assert (result[:, :3] == token_ids).all()
        assert result[0, 3].item() == 42
        assert result[0, 4].item() == 43


def test_vllm_generate_from_tokens_strips_left_padding():
    _, mock_llm_instance, _, _, patcher = _make_mock_vllm()
    with patcher:
        from dialectic.llm.vllm import VLLMGenerator

        model = _make_tiny_model()
        gen = VLLMGenerator(model, model_type="qwen3")

        pad_token_id = 0
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
        assert prompt_ids[0] == [10]
        assert prompt_ids[1] == [40, 50, 60]


def test_vllm_generate_from_tokens_pads_output():
    _, _, _, _, patcher = _make_mock_vllm()
    with patcher:
        from dialectic.llm.vllm import VLLMGenerator

        model = _make_tiny_model()
        gen = VLLMGenerator(model, model_type="qwen3")

        pad_token_id = 0
        token_ids = torch.tensor([[pad_token_id, 10], [40, 50]])

        result = gen.generate_from_tokens(
            token_ids=token_ids,
            eos_token_id=2,
            pad_token_id=pad_token_id,
            sampling_strategy="sample",
            max_tokens_generated=50,
        )

        assert result.shape[0] == 2
        assert result.shape[1] == result.shape[1]  # uniform


def test_vllm_generate_from_text():
    _, _, _, _, patcher = _make_mock_vllm()
    with patcher:
        from dialectic.llm.vllm import VLLMGenerator

        model = _make_tiny_model()
        gen = VLLMGenerator(model, model_type="qwen3")

        mock_tokenizer = MagicMock()
        mock_tokenizer.token_to_id.side_effect = lambda t: {
            "<|im_end|>": 2,
            "<|endoftext|>": 0,
        }[t]
        mock_encoded_1 = MagicMock()
        mock_encoded_1.ids = [10, 20]
        mock_encoded_2 = MagicMock()
        mock_encoded_2.ids = [0, 30]
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


def test_vllm_qwen_generate_from_chat():
    _, _, _, _, patcher = _make_mock_vllm()
    with patcher:
        from dialectic.llm.templates import Message
        from dialectic.llm.vllm import VLLMGenerator

        model = _make_tiny_model()
        gen = VLLMGenerator(model, model_type="qwen3")

        mock_tokenizer = MagicMock()
        mock_tokenizer.token_to_id.side_effect = lambda t: {
            "<|im_end|>": 2,
            "<|endoftext|>": 0,
        }[t]
        mock_encoded = MagicMock()
        mock_encoded.ids = [10, 20]
        mock_tokenizer.encode_batch.return_value = [mock_encoded]
        mock_tokenizer.decode.side_effect = lambda ids: "output"

        messages = [[Message(role="user", content="Hello")]]
        result = gen.qwen_generate_from_chat(
            tokenizer=mock_tokenizer,
            batch_messages=messages,
            max_tokens_generated=10,
        )

        assert len(result) == 1
        call_args = mock_tokenizer.encode_batch.call_args[0][0]
        assert "<|im_start|>" in call_args[0]


def test_vllm_llama_generate_from_chat():
    _, _, _, _, patcher = _make_mock_vllm()
    with patcher:
        from dialectic.llm.templates import Message
        from dialectic.llm.vllm import VLLMGenerator

        model = _make_tiny_model()
        gen = VLLMGenerator(model, model_type="qwen3")

        mock_tokenizer = MagicMock()
        mock_tokenizer.token_to_id.side_effect = lambda t: {"<|eot_id|>": 2}[t]
        mock_encoded = MagicMock()
        mock_encoded.ids = [10, 20]
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
