import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Bool, Float, Int
from tokenizers import Tokenizer
from torch import Tensor

from dialectic.distributed import unwrap_model
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
        return self


@dataclass
class HardTokenGeneratorOutput:
    tokens: Int[torch.Tensor, "B L"]
    attention_mask: Bool[torch.Tensor, "B L"] | None


@dataclass
class SoftTokenGeneratorOutput:
    embeddings: Float[torch.Tensor, "B L D"]
    shadow_ids: Int[torch.Tensor, "B L"]
    attention_mask: Bool[torch.Tensor, "B L"] | None
    hard_tokens_mask: Bool[torch.Tensor, "B L"]


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
    if token_ids.shape[1] < len(prefill.condition):
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

        attention_mask = torch.cat([attention_mask, new_attention_mask], 1)

    return torch.cat([token_ids, new_tensors], -1), attention_mask


class _BaseTokenGenerator(ABC):
    def init_state(
        self,
        initial_input: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ):
        self.all_tokens = initial_input
        self.device = initial_input.device
        self.batch_size = initial_input.shape[0]
        self.attention_mask = attention_mask
        self._finished = torch.zeros(
            initial_input.shape[0], dtype=torch.bool, device=self.device
        )
        self.n_generated = 0
        return self

    @abstractmethod
    def get_next_inputs(self, logits: Float[torch.Tensor, "B 1 V"]) -> bool: ...

    @property
    @abstractmethod
    def finished(self) -> bool: ...

    @property
    def _extra_tokens_per_step_bound(self) -> int:
        return 0

    @property
    def _extra_kv_reserve(self) -> int:
        return 0

    def net_forward(
        self,
        net: BaseTransformer,
        input_tokens: torch.Tensor,
        kv_caches: list[KVCache] | None,
        attention_mask: Bool[torch.Tensor, "B L"],
    ):
        return net(
            input_tokens,
            kv_caches=kv_caches,
            attention_mask=attention_mask,
        )

    def _generate(
        self,
        net: BaseTransformer,
        token_ids: Int[Tensor, "B L"],
        max_tokens_generated: int = sys.maxsize,
        use_kv_cache: bool = True,
        attention_mask: torch.Tensor | None = None,  # should be left-padded
        use_bf16: bool = False,
    ):
        raw_net = unwrap_model(net)
        if use_kv_cache:
            kv_caches = [
                KVCache(
                    max_seq_len=max_tokens_generated
                    * (1 + self._extra_tokens_per_step_bound)
                    + self._extra_kv_reserve
                    + token_ids.shape[1],
                    num_heads=raw_net.attn_num_kv_heads,
                    head_dim=raw_net.attn_head_d,
                    device=next(raw_net.parameters()).device,
                )
                for _ in range(len(raw_net.layers))
            ]
        else:
            kv_caches = None

        device = token_ids.device
        self.init_state(token_ids, attention_mask)
        self.max_tokens_generated = max_tokens_generated

        input_tokens = token_ids

        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
        ):
            while self.n_generated < max_tokens_generated:
                logits: Float[torch.Tensor, "B 1 V"] = self.net_forward(
                    net=net,
                    input_tokens=input_tokens,
                    kv_caches=kv_caches,
                    attention_mask=self.attention_mask,
                )

                prev_len = self.all_tokens.shape[1]
                is_done = self.get_next_inputs(logits)

                if is_done:
                    break

                if use_kv_cache:
                    # When get_next_inputs adds >1 token (e.g. prefill), process
                    # the intermediate ones through the model to keep the KV cache
                    # in sync. Note: in batched generation, non-triggering elements
                    # get pad tokens here which shifts their RoPE positions. This is
                    # negligible for small fill lengths since the relative distances
                    # between the element's own real tokens are preserved.
                    n_new = self.all_tokens.shape[1] - prev_len
                    for i in range(n_new - 1):
                        mask = (
                            self.attention_mask[:, : prev_len + i + 1]
                            if self.attention_mask is not None
                            else None
                        )
                        self.net_forward(
                            net=net,
                            input_tokens=self.all_tokens[
                                :, prev_len + i : prev_len + i + 1
                            ],
                            kv_caches=kv_caches,
                            attention_mask=mask,
                        )
                    input_tokens = self.all_tokens[:, -1:]
                else:
                    input_tokens = self.all_tokens

                self.n_generated += 1  # counts generation steps, not tokens (prefill may add multiple per step)

        # If we hit the generation limit before finishing, allow generators to
        # append forced tokens (e.g., to trigger a hard-token switch).
        if self.n_generated >= max_tokens_generated and not self.finished:
            on_max_tokens_reached = getattr(self, "on_max_tokens_reached", None)
            if callable(on_max_tokens_reached):
                on_max_tokens_reached()


class _HardGenerator(_BaseTokenGenerator):
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
        super().init_state(initial_input=initial_input, attention_mask=attention_mask)
        if self.prefill is not None:
            self.prefill = self.prefill.to(self.device)

    def get_next_inputs(self, logits: Float[torch.Tensor, "B 1 V"]) -> bool:
        if self.sampling_strategy == "greedy":
            next_token = logits.argmax(-1)
        else:
            scaled_logits = logits / self.temperature
            probs = torch.softmax(scaled_logits.squeeze(1).float(), dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        next_token = torch.where(
            self._finished.unsqueeze(-1),
            torch.full_like(next_token, self.pad_token_id),
            next_token,
        )

        self.all_tokens = torch.cat([self.all_tokens, next_token], 1)

        self._finished = self._finished | (self.all_tokens[:, -1] == self.eos_token_id)
        if self.finished:
            return True

        if self.attention_mask is not None:
            new_mask = ~self._finished.unsqueeze(-1)
            self.attention_mask = torch.cat([self.attention_mask, new_mask], 1)

        if self.prefill:
            self.all_tokens, self.attention_mask = check_and_apply_prefill(
                token_ids=self.all_tokens,
                prefill=self.prefill,
                pad_token_id=self.pad_token_id,
                attention_mask=self.attention_mask,
            )

            self._finished = self._finished | (
                self.all_tokens[:, -1] == self.eos_token_id
            )

            if self.finished:
                return True

        return False

    def generate(
        self,
        net: BaseTransformer,
        token_ids: Int[Tensor, "B L"],
        max_tokens_generated: int = sys.maxsize,
        use_kv_cache: bool = True,
        attention_mask: torch.Tensor | None = None,  # should be left-padded
        use_bf16: bool = False,
    ) -> HardTokenGeneratorOutput:
        super()._generate(
            net=net,
            token_ids=token_ids,
            max_tokens_generated=max_tokens_generated,
            use_kv_cache=use_kv_cache,
            attention_mask=attention_mask,
            use_bf16=use_bf16,
        )
        return HardTokenGeneratorOutput(
            tokens=self.all_tokens, attention_mask=self.attention_mask
        )

    @property
    def _extra_tokens_per_step_bound(self) -> int:
        if self.prefill:
            return len(self.prefill.filling)
        return 0

    @property
    def finished(self) -> bool:
        return bool(self._finished.all())


@torch.inference_mode()
def generate_hard_tokens(
    net: BaseTransformer,
    token_ids: Int[Tensor, "B L"],
    sampling_strategy: Literal["greedy", "sample"] = "sample",
    eos_token_id: int = 151645,
    pad_token_id: int = 151643,
    max_tokens_generated: int = sys.maxsize,
    use_kv_cache: bool = True,
    attention_mask: torch.Tensor | None = None,
    temperature: float = 1.0,
    use_bf16: bool = False,
    prefill: PreFill | None = None,
) -> HardTokenGeneratorOutput:
    if sampling_strategy not in ["greedy", "sample"]:
        raise ValueError("`sampling_strategy` must be one of 'greedy' or 'sample'.")

    if pad_token_id is None:
        pad_token_id = eos_token_id

    return _HardGenerator(
        sampling_strategy=sampling_strategy,
        temperature=temperature,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        prefill=prefill,
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

    token_ids = generate_hard_tokens(
        net=net,
        token_ids=token_ids,
        sampling_strategy=sampling_strategy,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        max_tokens_generated=max_tokens_generated,
        use_kv_cache=use_kv_cache,
        attention_mask=attention_mask,
        temperature=temperature,
    ).tokens

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
