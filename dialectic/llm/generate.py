import sys
from abc import abstractmethod
from typing import Literal

import torch
from attr import dataclass
from jaxtyping import Float, Int
from tokenizers import Tokenizer
from torch import Tensor

from dialectic.llm.base import BaseTransformer
from dialectic.llm.components import KVCache
from dialectic.llm.templates import (
    Message,
    get_llama_input_text_from_messages,
    get_qwen_input_text_from_messages,
)


@dataclass
class PreFill:
    condition: Int[torch.Tensor, " N"]  # (space necessary to avoid F821)
    filling: Int[torch.Tensor, " M"]

    def to(self, device: torch.device):
        self.condition = self.condition.to(device)
        self.filling = self.filling.to(device)


def check_and_apply_prefill(
    token_ids: Int[torch.Tensor, "B L"],
    prefill: PreFill,
    pad_token_id: int,
    attention_mask: torch.Tensor | None,
):
    """Checks a batch of token ids and if any match the prefill condition, prefills it and then pads
    the ones not meeting the condition
    """
    if attention_mask is not None:
        if token_ids.shape != attention_mask.shape:
            raise RuntimeError(
                "`token_ids` and `attention_mask` should have the same shape."
            )
    if len(token_ids) < len(prefill.condition):
        return token_ids, attention_mask

    # check if there are any elements in the batch meeting the condition
    cond_met = (
        token_ids[:, -len(prefill.condition) :] == prefill.condition.unsqueeze(0)
    ).all(1)
    if not cond_met.any():
        return token_ids, attention_mask

    new_tensors = torch.where(
        cond_met.unsqueeze(-1),
        prefill.filling.unsqueeze(0),
        torch.full_like(prefill.filling, pad_token_id).unsqueeze(0),
    )

    if attention_mask is not None:
        new_attention_mask = torch.where(
            cond_met.unsqueeze(-1),
            torch.ones_like(new_tensors, dtype=torch.bool),
            torch.zeros_like(new_tensors, dtype=torch.bool),
        )

        attention_mask = torch.cat(
            [attention_mask, new_attention_mask],
            1,
        )

    return torch.cat([token_ids, new_tensors], -1), attention_mask


class BaseTokenGenerator:
    @abstractmethod
    def init_state(
        self,
        initial_input: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ): ...

    @abstractmethod
    def get_next_inputs(
        self, logits: Float[torch.Tensor, "B 1 V"]
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]: ...

    @property
    @abstractmethod
    def finished(self) -> bool: ...

    @abstractmethod
    def get_all_tensors(self) -> torch.Tensor: ...

    def generate(
        self,
        net: BaseTransformer,
        token_ids: Int[Tensor, "B L"],
        max_tokens_generated: int = sys.maxsize,
        use_kv_cache: bool = True,
        attention_mask: torch.Tensor | None = None,  # should be left-padded
        use_bf16: bool = False,
        prefill: PreFill | None = None,
    ):
        if use_kv_cache:
            kv_caches = [
                KVCache(
                    max_seq_len=max_tokens_generated + token_ids.shape[1],
                    num_heads=net.attn_num_kv_heads,
                    head_dim=net.attn_head_d,
                    device=next(net.parameters()).device,
                )
                for _ in range(len(net.layers))
            ]
        else:
            kv_caches = None

        device = token_ids.device
        if prefill:
            prefill = prefill.to(device)
        self.init_state(token_ids, attention_mask)

        input_tokens = token_ids
        tokens_generated = 0
        while tokens_generated < max_tokens_generated:
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
            ):
                logits: Float[torch.Tensor, "B 1 V"] = net(
                    input_tokens, kv_caches=kv_caches, attention_mask=attention_mask
                )

            all_inputs, attention_mask = self.get_next_inputs(logits)

            if all_inputs is None:
                break

            if use_kv_cache:
                input_tokens = all_inputs[:, -1:]
            else:
                input_tokens = all_inputs

            tokens_generated += 1

        return self.get_all_tensors()


class HardGenerator(BaseTokenGenerator):
    # if we want pre-filling conditions then we need to track everything...
    def __init__(
        self,
        sampling_strategy: Literal["greedy", "sample"] | None = "sample",
        temperature: float = 1.0,
        eos_token_id: int = 151645,
        pad_token_id: int = 151643,
        prefill: PreFill | None = None,
    ):
        self.sampling_strategy = sampling_strategy
        self.temperature = temperature
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.prefill = prefill

    def init_state(
        self,
        initial_input: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ):
        self.all_inputs = initial_input
        self.attention_mask = attention_mask
        self._finished = torch.zeros(
            initial_input.shape[0], dtype=torch.bool, device=initial_input.device
        )

    def get_next_inputs(
        self, logits: Float[torch.Tensor, "B 1 V"]
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.sampling_strategy == "greedy":
            next_token = logits.argmax(-1)
        else:
            scaled_logits = logits / self.temperature
            probs = torch.softmax(scaled_logits.squeeze(1), dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        next_token = torch.where(
            self._finished.unsqueeze(-1),
            torch.full_like(next_token, self.pad_token_id),
            next_token,
        )

        self.all_inputs = torch.cat([self.all_inputs, next_token], 1)

        self._finished = self._finished | (self.all_inputs[:, -1] == self.eos_token_id)
        if self.finished:
            return None, None

        if self.attention_mask is not None:
            new_mask = ~self._finished.unsqueeze(-1)
            self.attention_mask = torch.cat([self.attention_mask, new_mask], 1)

        if self.prefill:
            self.all_inputs, self.attention_mask = check_and_apply_prefill(
                token_ids=self.all_inputs,
                prefill=self.prefill,
                pad_token_id=self.pad_token_id,
                attention_mask=self.attention_mask,
            )

            self._finished = self._finished | (
                self.all_inputs[:, -1] == self.eos_token_id
            )

            if self.finished:
                return None, None

        return self.all_inputs, self.attention_mask

    def get_all_tensors(self) -> torch.Tensor:
        return self.all_inputs

    @property
    def finished(self) -> bool:
        return bool(self._finished.all())


class SoftGenerator(BaseTokenGenerator):
    # if we want pre-filling conditions then we need to track everything...
    def __init__(
        self,
        vocab_size: int,
        temperature: float = 1.0,
        eos_token_id: int = 151645,
        pad_token_id: int = 151643,
    ):
        self.vocab_size = vocab_size
        self.temperature = temperature
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

    def init_state(
        self,
        initial_input: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ):
        self.shadow_seq = initial_input
        self.all_inputs = torch.nn.functional.one_hot(
            initial_input, self.vocab_size
        ).float()
        self.attention_mask = attention_mask
        self._finished = torch.zeros(
            initial_input.shape[0], dtype=torch.bool, device=initial_input.device
        )

    def get_next_inputs(
        self, logits: Float[torch.Tensor, "B 1 V"]
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        scaled_logits = logits / self.temperature
        probs = torch.softmax(scaled_logits, dim=-1)
        next_token = probs  # need .detach()?

        next_token = torch.where(
            self._finished.unsqueeze(-1).unsqueeze(-1),
            torch.nn.functional.one_hot(
                torch.tensor(self.pad_token_id), logits.shape[-1]
            ),
            next_token,
        )
        hard_token_id = next_token.argmax(-1)
        self._finished = self._finished | (
            hard_token_id.squeeze(-1) == self.eos_token_id
        )

        self.all_inputs = torch.cat([self.all_inputs, next_token], 1)
        self.shadow_seq = torch.cat([self.shadow_seq, hard_token_id], 1)

        if self.finished:
            return None, None

        if self.attention_mask is not None:
            new_mask = ~self._finished.unsqueeze(-1)
            self.attention_mask = torch.cat([self.attention_mask, new_mask], 1)
        return self.all_inputs, self.attention_mask

    def get_all_tensors(self) -> torch.Tensor:
        return self.all_inputs

    @property
    def finished(self) -> bool:
        return bool(self._finished.all())


@torch.inference_mode()
def generate_from_tokens(
    net: BaseTransformer,
    token_ids: Int[Tensor, "B L"],
    eos_token_id: int = 151645,
    pad_token_id: int = 151643,
    sampling_strategy: Literal["greedy", "sample"] | None = "sample",
    max_tokens_generated: int = sys.maxsize,
    use_kv_cache: bool = True,
    attention_mask: torch.Tensor | None = None,  # should be left-padded
    temperature: float = 1.0,
    use_bf16: bool = False,
    soft_tokens: bool = False,
) -> Int[Tensor, "B L"]:
    if not soft_tokens and sampling_strategy not in ["greedy", "sample"]:
        raise ValueError("`sampling_strategy` must be one of 'greedy' or 'sample'.")

    if pad_token_id is None:
        pad_token_id = eos_token_id

    if soft_tokens:
        return SoftGenerator(
            vocab_size=net.vocab_size,
            temperature=temperature,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        ).generate(
            net, token_ids, max_tokens_generated, use_kv_cache, attention_mask, use_bf16
        )
    else:
        return HardGenerator(
            sampling_strategy=sampling_strategy,
            temperature=temperature,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        ).generate(
            net, token_ids, max_tokens_generated, use_kv_cache, attention_mask, use_bf16
        )


def generate_from_text(
    net: BaseTransformer,
    tokenizer: Tokenizer,
    text_batch: list[str],
    eos_token: str = "<|im_end|>",
    pad_token: str = "<|endoftext|>",
    sampling_strategy: Literal["greedy", "sample"] = "sample",
    max_tokens_generated: int = sys.maxsize,
    device: str | torch.device | None = None,
    use_kv_cache: bool = True,
    temperature: float = 1.0,
) -> list[str]:
    if device is None:
        device = next(net.parameters()).device

    pad_token_id = tokenizer.token_to_id(pad_token)
    tokenizer.enable_padding(pad_id=pad_token_id, pad_token=pad_token, direction="left")
    tokens = tokenizer.encode_batch(text_batch)
    token_ids = torch.tensor([t.ids for t in tokens]).to(device)
    attention_mask = (
        torch.tensor([t.attention_mask for t in tokens], dtype=torch.bool)
    ).to(device)

    eos_token_id = tokenizer.token_to_id(eos_token)

    token_ids = generate_from_tokens(
        net=net,
        token_ids=token_ids,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        sampling_strategy=sampling_strategy,
        max_tokens_generated=max_tokens_generated,
        use_kv_cache=use_kv_cache,
        attention_mask=attention_mask,
        temperature=temperature,
    )

    return [tokenizer.decode(batch.tolist()) for batch in token_ids]


def qwen_generate_from_chat(
    net: BaseTransformer,
    tokenizer: Tokenizer,
    batch_messages: list[list[Message]],
    eos_token: str = "<|im_end|>",
    pad_token: str = "<|endoftext|>",
    sampling_strategy: Literal["greedy", "sample"] = "sample",
    max_tokens_generated: int = 1000,
    device: str | torch.device | None = None,
    temperature: float = 1.0,
    enable_thinking: bool = True,
):
    return generate_from_text(
        net=net,
        tokenizer=tokenizer,
        sampling_strategy=sampling_strategy,
        text_batch=[
            get_qwen_input_text_from_messages(
                message, add_generation_prompt=True, enable_thinking=enable_thinking
            )
            for message in batch_messages
        ],
        eos_token=eos_token,
        pad_token=pad_token,
        max_tokens_generated=max_tokens_generated,
        device=device,
        temperature=temperature,
    )


def llama_generate_from_chat(
    net: BaseTransformer,
    tokenizer: Tokenizer,
    batch_messages: list[list[Message]],
    eos_token: str = "<|eot_id|>",
    pad_token: str = "<|eot_id|>",  # "<|finetune_right_pad_id|>",
    sampling_strategy: Literal["greedy", "sample"] = "sample",
    max_tokens_generated: int = 1000,
    device: str | torch.device | None = None,
    temperature: float = 1.0,
):
    text_batch = [
        get_llama_input_text_from_messages(message, add_generation_prompt=True)
        for message in batch_messages
    ]

    return generate_from_text(
        net=net,
        tokenizer=tokenizer,
        sampling_strategy=sampling_strategy,
        text_batch=text_batch,
        eos_token=eos_token,
        pad_token=pad_token,
        max_tokens_generated=max_tokens_generated,
        device=device,
        temperature=temperature,
    )
