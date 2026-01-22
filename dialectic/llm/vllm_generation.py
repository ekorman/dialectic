"""vLLM-based generation for faster inference during RL training.

This module provides vLLM integration for the generation phase of RL training,
offering faster throughput compared to the standard PyTorch implementation.
"""

from typing import TYPE_CHECKING

import torch
from jaxtyping import Int
from torch import Tensor

if TYPE_CHECKING:
    from vllm import LLM, SamplingParams


class VLLMGenerator:
    """Wrapper for vLLM-based generation compatible with GRPO training.

    This class manages a vLLM LLM instance for efficient batched generation,
    providing an interface compatible with the existing generate_from_tokens function.
    """

    def __init__(
        self,
        model_name_or_path: str,
        *,
        tokenizer_path: str | None = None,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
    ):
        """Initialize vLLM generator.

        Parameters
        ----------
        model_name_or_path
            Path to model weights or HuggingFace model name.
        tokenizer_path
            Path to tokenizer. If None, uses the model's default tokenizer.
        gpu_memory_utilization
            Fraction of GPU memory to use (0.0-1.0).
        max_model_len
            Maximum sequence length. If None, uses model's default.
        tensor_parallel_size
            Number of GPUs to use for tensor parallelism.
        dtype
            Data type for model weights ("auto", "half", "float16", "bfloat16", "float32").
        """
        try:
            from vllm import LLM
        except ImportError:
            raise ImportError(
                "vLLM is required for VLLMGenerator. "
                "Install with: pip install vllm"
            )

        self.llm = LLM(
            model=model_name_or_path,
            tokenizer=tokenizer_path,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            enforce_eager=False,  # Use CUDA graphs for better performance
            trust_remote_code=True,
        )

    @torch.inference_mode()
    def generate_from_tokens(
        self,
        token_ids: Int[Tensor, "B L"],
        eos_token_id: int,
        pad_token_id: int,
        max_tokens_generated: int,
        temperature: float = 1.0,
        sampling_strategy: str = "sample",
    ) -> Int[Tensor, "B L_new"]:
        """Generate completions using vLLM.

        Parameters
        ----------
        token_ids
            Input token IDs of shape [batch_size, seq_len].
        eos_token_id
            Token ID for end-of-sequence.
        pad_token_id
            Token ID for padding.
        max_tokens_generated
            Maximum number of new tokens to generate.
        temperature
            Sampling temperature (only used when sampling_strategy="sample").
        sampling_strategy
            Either "greedy" or "sample".

        Returns
        -------
        torch.Tensor
            Generated token IDs including the prompt, shape [batch_size, seq_len + generated_len].
        """
        from vllm import SamplingParams

        batch_size = token_ids.shape[0]
        device = token_ids.device

        # Convert token IDs to lists for vLLM
        prompt_token_ids = token_ids.tolist()

        # Configure sampling parameters
        if sampling_strategy == "greedy":
            sampling_params = SamplingParams(
                max_tokens=max_tokens_generated,
                temperature=0.0,  # Greedy decoding
                stop_token_ids=[eos_token_id],
            )
        else:  # "sample"
            sampling_params = SamplingParams(
                max_tokens=max_tokens_generated,
                temperature=temperature,
                stop_token_ids=[eos_token_id],
            )

        # Generate using vLLM
        outputs = self.llm.generate(
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        # Convert outputs back to tensor format
        # vLLM returns outputs in the same order as inputs
        all_token_ids = []
        max_len = 0

        for output in outputs:
            # Get prompt tokens + generated tokens
            prompt_tokens = output.prompt_token_ids
            generated_tokens = output.outputs[0].token_ids
            full_sequence = prompt_tokens + generated_tokens
            all_token_ids.append(full_sequence)
            max_len = max(max_len, len(full_sequence))

        # Pad sequences to the same length (right padding for compatibility)
        padded_sequences = []
        for seq in all_token_ids:
            if len(seq) < max_len:
                # Pad on the right
                padded = seq + [pad_token_id] * (max_len - len(seq))
            else:
                padded = seq
            padded_sequences.append(padded)

        # Convert to tensor
        result = torch.tensor(padded_sequences, dtype=torch.long, device=device)

        return result


def create_vllm_generator(
    weights_path: str,
    tokenizer_path: str,
    *,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int | None = None,
    dtype: str = "auto",
) -> VLLMGenerator:
    """Create a vLLM generator for Qwen model.

    Parameters
    ----------
    weights_path
        Path to model weights directory or HuggingFace model name.
    tokenizer_path
        Path to tokenizer file.
    gpu_memory_utilization
        Fraction of GPU memory to use.
    max_model_len
        Maximum sequence length.
    dtype
        Data type for model weights.

    Returns
    -------
    VLLMGenerator
        Initialized vLLM generator.
    """
    return VLLMGenerator(
        model_name_or_path=weights_path,
        tokenizer_path=tokenizer_path,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        dtype=dtype,
    )
