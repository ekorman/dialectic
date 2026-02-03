from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Literal

import torch
from jaxtyping import Int
from torch import Tensor

if TYPE_CHECKING:
    from tokenizers import Tokenizer
    from vllm import LLM

from dialectic.llm.templates import (
    Message,
    get_llama_input_text_from_messages,
    get_qwen_input_text_from_messages,
)


class VLLMGenerator:
    """Wrapper around vLLM's LLM for generation, providing an interface
    compatible with the existing dialectic generation functions.

    Parameters
    ----------
    model : str
        HuggingFace model name or path to load with vLLM.
    **kwargs
        Additional keyword arguments passed to ``vllm.LLM``.
    """

    def __init__(self, model: str, **kwargs) -> None:
        from vllm import LLM

        self.llm: LLM = LLM(model=model, **kwargs)

    def generate_from_tokens(
        self,
        token_ids: Int[Tensor, "B L"],
        eos_token_id: int,
        pad_token_id: int,
        sampling_strategy: Literal["greedy", "sample"] = "sample",
        max_tokens_generated: int = sys.maxsize,
        temperature: float = 1.0,
    ) -> Int[Tensor, "B L_out"]:
        """Generate token continuations using vLLM.

        Parameters
        ----------
        token_ids
            Input prompt token ids of shape ``(B, L)``. Left-padded rows are
            automatically stripped before being sent to vLLM.
        eos_token_id
            Token id that signals end-of-sequence.
        pad_token_id
            Token id used for padding.
        sampling_strategy
            ``"greedy"`` for argmax decoding, ``"sample"`` for multinomial.
        max_tokens_generated
            Maximum number of new tokens to generate.
        temperature
            Sampling temperature (ignored when ``sampling_strategy="greedy"``).

        Returns
        -------
        Int[Tensor, "B L_out"]
            Full sequences (prompt + generated tokens) right-padded to the same
            length, matching the return convention of
            :func:`dialectic.llm.generate.generate_from_tokens`.
        """
        from vllm import SamplingParams

        if sampling_strategy == "greedy":
            temperature = 0.0

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens_generated,
            stop_token_ids=[eos_token_id],
        )

        # Strip left-padding per row so vLLM only sees real tokens.
        prompt_token_ids_list: list[list[int]] = []
        for row in token_ids:
            ids = row.tolist()
            # Find the first non-pad token.
            start = 0
            for start, tid in enumerate(ids):
                if tid != pad_token_id:
                    break
            prompt_token_ids_list.append(ids[start:])

        outputs = self.llm.generate(
            prompt_token_ids=prompt_token_ids_list,
            sampling_params=sampling_params,
        )

        # Reconstruct full sequences (original prompt + generated) and
        # right-pad to a uniform length, mirroring generate_from_tokens.
        all_seqs: list[list[int]] = []
        for row, output in zip(token_ids, outputs):
            prompt = row.tolist()
            generated = list(output.outputs[0].token_ids)
            all_seqs.append(prompt + generated)

        max_len = max(len(s) for s in all_seqs)
        for seq in all_seqs:
            seq.extend([pad_token_id] * (max_len - len(seq)))

        return torch.tensor(all_seqs, dtype=token_ids.dtype, device=token_ids.device)

    def generate_from_text(
        self,
        tokenizer: Tokenizer,
        text_batch: list[str],
        eos_token: str = "<|im_end|>",
        pad_token: str = "<|endoftext|>",
        sampling_strategy: Literal["greedy", "sample"] = "sample",
        max_tokens_generated: int = sys.maxsize,
        temperature: float = 1.0,
    ) -> list[str]:
        """Generate text completions using vLLM.

        Parameters
        ----------
        tokenizer
            A ``tokenizers.Tokenizer`` instance (used only for encoding /
            decoding, not by vLLM itself).
        text_batch
            List of prompt strings.
        eos_token
            End-of-sequence token string.
        pad_token
            Padding token string.
        sampling_strategy
            ``"greedy"`` or ``"sample"``.
        max_tokens_generated
            Maximum number of new tokens.
        temperature
            Sampling temperature.

        Returns
        -------
        list[str]
            Decoded strings (prompt + generation) for each item in the batch.
        """
        pad_token_id = tokenizer.token_to_id(pad_token)
        tokenizer.enable_padding(
            pad_id=pad_token_id, pad_token=pad_token, direction="left"
        )
        tokens = tokenizer.encode_batch(text_batch)
        token_ids = torch.tensor([t.ids for t in tokens])

        eos_token_id = tokenizer.token_to_id(eos_token)

        result_ids = self.generate_from_tokens(
            token_ids=token_ids,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            sampling_strategy=sampling_strategy,
            max_tokens_generated=max_tokens_generated,
            temperature=temperature,
        )

        return [tokenizer.decode(row.tolist()) for row in result_ids]

    def qwen_generate_from_chat(
        self,
        tokenizer: Tokenizer,
        batch_messages: list[list[Message]],
        eos_token: str = "<|im_end|>",
        pad_token: str = "<|endoftext|>",
        sampling_strategy: Literal["greedy", "sample"] = "sample",
        max_tokens_generated: int = 1000,
        temperature: float = 1.0,
        enable_thinking: bool = True,
    ) -> list[str]:
        return self.generate_from_text(
            tokenizer=tokenizer,
            sampling_strategy=sampling_strategy,
            text_batch=[
                get_qwen_input_text_from_messages(
                    message,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
                for message in batch_messages
            ],
            eos_token=eos_token,
            pad_token=pad_token,
            max_tokens_generated=max_tokens_generated,
            temperature=temperature,
        )

    def llama_generate_from_chat(
        self,
        tokenizer: Tokenizer,
        batch_messages: list[list[Message]],
        eos_token: str = "<|eot_id|>",
        pad_token: str = "<|eot_id|>",
        sampling_strategy: Literal["greedy", "sample"] = "sample",
        max_tokens_generated: int = 1000,
        temperature: float = 1.0,
    ) -> list[str]:
        text_batch = [
            get_llama_input_text_from_messages(message, add_generation_prompt=True)
            for message in batch_messages
        ]
        return self.generate_from_text(
            tokenizer=tokenizer,
            sampling_strategy=sampling_strategy,
            text_batch=text_batch,
            eos_token=eos_token,
            pad_token=pad_token,
            max_tokens_generated=max_tokens_generated,
            temperature=temperature,
        )
