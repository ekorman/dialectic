import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Bool, Float, Int
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
class InternalReasoningGeneratorOutput:
    hard_token_ids: Int[torch.Tensor, "B C"]
    hard_log_probs: Float[torch.Tensor, "B C"]
    n_cycles: Int[torch.Tensor, " B"]


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
        if use_kv_cache:
            kv_caches = [
                KVCache(
                    max_seq_len=max_tokens_generated
                    * (1 + self._extra_tokens_per_step_bound)
                    + self._extra_kv_reserve
                    + token_ids.shape[1],
                    num_heads=net.attn_num_kv_heads,
                    head_dim=net.attn_head_d,
                    device=next(net.parameters()).device,
                )
                for _ in range(len(net.layers))
            ]
        else:
            kv_caches = None

        device = token_ids.device
        self.init_state(token_ids, attention_mask)
        self.max_tokens_generated = max_tokens_generated

        input_tokens = token_ids

        while self.n_generated < max_tokens_generated:
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
            ):
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
                    with torch.autocast(
                        device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
                    ):
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


class _SoftGenerator(_BaseTokenGenerator):
    def __init__(
        self,
        vocab_size: int,
        temperature: float = 1.0,
        eos_token_id: int = 151645,
        pad_token_id: int = 151643,
        switch_to_hard_tokens_condition: Int[torch.Tensor, " M"] | None = None,
        max_tokens_prefill: Int[torch.Tensor, " N"] | None = None,
        max_tokens_prefill_steps_before_end: int = 0,
        prefill: PreFill | None = None,
        soft_token_noise_std: float | None = None,
        min_soft_steps: int = 0,
    ):
        """soft token generator. The optional parameter `switch_to_hard_tokens_condition`
        determines when to switch from soft token generation to hard token generation: once
        the shadow token sequence ends with `switch_to_hard_tokens_condition` we start sampling
        hard tokens. note to keep the shape the same as the soft tokens, we will one-hot encode them.
        this includes the prefilling
        """
        self.vocab_size = vocab_size
        self.temperature = temperature
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.switch_to_hard_tokens_condition = switch_to_hard_tokens_condition
        self.max_tokens_prefill = max_tokens_prefill
        self.max_tokens_prefill_steps_before_end = max(
            0, max_tokens_prefill_steps_before_end
        )
        self.prefill = prefill
        self.soft_token_noise_std = soft_token_noise_std
        self.min_soft_steps = max(0, min_soft_steps)

    def init_state(
        self,
        initial_input: Int[torch.Tensor, "B L"],
        attention_mask: Bool[torch.Tensor, "B L"] | None,
    ):
        super().init_state(initial_input=initial_input, attention_mask=attention_mask)
        if self.prefill is not None:
            self.prefill = self.prefill.to(self.device)
        self.shadow_seq = initial_input
        self.all_tokens = torch.nn.functional.one_hot(
            self.all_tokens, self.vocab_size
        ).float()

        # mask is True where we use hard tokens
        self.hard_tokens_mask = torch.ones_like(initial_input, dtype=torch.bool)
        self._switched_to_hard = torch.zeros(
            initial_input.shape[0], dtype=torch.bool, device=self.device
        )

        self.all_noise: list[Tensor] = []

        if self.switch_to_hard_tokens_condition is not None:
            self.switch_to_hard_tokens_condition = (
                self.switch_to_hard_tokens_condition.to(self.device)
            )
        if self.max_tokens_prefill is not None:
            self.max_tokens_prefill = self.max_tokens_prefill.to(self.device)
        self._max_tokens_prefilled = False

    def on_max_tokens_reached(self) -> None:
        prefill_ids = (
            self.max_tokens_prefill
            if self.max_tokens_prefill is not None
            else self.switch_to_hard_tokens_condition
        )
        if prefill_ids is None:
            return
        if self._switched_to_hard.all():
            return

        need_prefill = (~self._switched_to_hard) & (~self._finished)
        if not need_prefill.any():
            return

        fill_ids = prefill_ids
        fill_len = fill_ids.numel()
        pad_ids = torch.full_like(fill_ids, self.pad_token_id)

        new_shadow = torch.where(
            need_prefill.unsqueeze(-1),
            fill_ids.unsqueeze(0),
            pad_ids.unsqueeze(0),
        )

        self.shadow_seq = torch.cat([self.shadow_seq, new_shadow], 1)
        new_soft = torch.nn.functional.one_hot(new_shadow, self.vocab_size).float()
        self.all_tokens = torch.cat([self.all_tokens, new_soft], 1)

        new_hard_mask = need_prefill.unsqueeze(-1).expand(-1, fill_len)
        self.hard_tokens_mask = torch.cat([self.hard_tokens_mask, new_hard_mask], 1)

        if self.attention_mask is not None:
            new_attn = new_hard_mask
            self.attention_mask = torch.cat([self.attention_mask, new_attn], 1)
            # TODO: think this is dead code that should never be reached, commenting out
            # for now to test live
            # if self.attention_mask.shape[1] < self.all_tokens.shape[1]:
            #     pad_len = self.all_tokens.shape[1] - self.attention_mask.shape[1]
            #     pad = torch.zeros(
            #         self.attention_mask.shape[0],
            #         pad_len,
            #         dtype=self.attention_mask.dtype,
            #         device=self.attention_mask.device,
            #     )
            #     self.attention_mask = torch.cat([self.attention_mask, pad], 1)

        self._switched_to_hard = self._switched_to_hard | need_prefill
        self._max_tokens_prefilled = True

    def net_forward(
        self,
        net: BaseTransformer,
        input_tokens: torch.Tensor,
        kv_caches: list[KVCache] | None,
        attention_mask: Bool[torch.Tensor, "B L"],
    ):
        if self.soft_token_noise_std is not None and input_tokens.ndim > 2:
            noise = torch.normal(
                0.0,
                self.soft_token_noise_std,
                size=(input_tokens.shape[0], input_tokens.shape[1], net.d),
                device=input_tokens.device,
            )
            self.all_noise.append(noise)
        else:
            noise = None
            if self.soft_token_noise_std is not None:
                self.all_noise.append(
                    torch.zeros(
                        input_tokens.shape[0],
                        input_tokens.shape[1],
                        net.d,
                        device=input_tokens.device,
                    )
                )
        return net(
            input_tokens,
            kv_caches=kv_caches,
            attention_mask=attention_mask,
            soft_token_noise=noise,
        )

    def get_next_inputs(self, logits: Float[torch.Tensor, "B 1 V"]) -> bool:
        if (
            self.max_tokens_prefill_steps_before_end > 0
            and not self._max_tokens_prefilled
            and self.switch_to_hard_tokens_condition is not None
        ):
            prefill_at = max(
                0, self.max_tokens_generated - self.max_tokens_prefill_steps_before_end
            )
            if self.n_generated >= prefill_at:
                # TODO: is there an issue that not replaying prefill through KV cache?
                # probably negligible if any
                self.on_max_tokens_reached()
        scaled_logits = logits / self.temperature
        probs = torch.softmax(scaled_logits.float(), dim=-1)
        next_token = probs

        next_token = torch.where(
            self._finished.unsqueeze(-1).unsqueeze(-1),
            torch.nn.functional.one_hot(
                torch.tensor(self.pad_token_id, device=logits.device),
                logits.shape[-1],
            ),
            next_token,
        )

        hard_token_id = next_token.argmax(-1)

        if self.switch_to_hard_tokens_condition is not None:
            if self.n_generated < self.min_soft_steps:
                cond_met = torch.zeros(
                    self.shadow_seq.shape[0],
                    dtype=torch.bool,
                    device=self.shadow_seq.device,
                )
            else:
                cond_met = (
                    self.shadow_seq[:, -len(self.switch_to_hard_tokens_condition) :]
                    == self.switch_to_hard_tokens_condition
                ).all(1)
            self._switched_to_hard = self._switched_to_hard | cond_met

            self.hard_tokens_mask = torch.cat(
                [self.hard_tokens_mask, self._switched_to_hard.unsqueeze(1)], 1
            )

            if self._switched_to_hard.any():
                next_token = torch.where(
                    self._switched_to_hard.unsqueeze(-1).unsqueeze(-1),
                    torch.nn.functional.one_hot(
                        hard_token_id, num_classes=next_token.shape[-1]
                    ),
                    next_token,
                )
        else:
            self.hard_tokens_mask = torch.cat(
                [
                    self.hard_tokens_mask,
                    torch.zeros(
                        (self.batch_size, 1),
                        dtype=torch.bool,
                        device=self.hard_tokens_mask.device,
                    ),
                ],
                1,
            )

        self._finished = self._finished | (
            hard_token_id.squeeze(-1) == self.eos_token_id
        )

        self.all_tokens = torch.cat([self.all_tokens, next_token], 1)
        self.shadow_seq = torch.cat([self.shadow_seq, hard_token_id], 1)

        if self.finished:
            return True

        if self.attention_mask is not None:
            new_mask = ~self._finished.unsqueeze(-1)
            self.attention_mask = torch.cat([self.attention_mask, new_mask], 1)

        # TODO: think this is dead code that should never be reached, commenting out
        # for now to test live
        # if self.attention_mask is not None and (
        #     self.attention_mask.shape[1] < self.all_tokens.shape[1]
        # ):
        #     pad_len = self.all_tokens.shape[1] - self.attention_mask.shape[1]
        #     pad = torch.zeros(
        #         self.attention_mask.shape[0],
        #         pad_len,
        #         dtype=self.attention_mask.dtype,
        #         device=self.attention_mask.device,
        #     )
        #     self.attention_mask = torch.cat([self.attention_mask, pad], 1)

        if self.prefill:
            self.shadow_seq, self.attention_mask = check_and_apply_prefill(
                token_ids=self.shadow_seq,
                prefill=self.prefill,
                pad_token_id=self.pad_token_id,
                attention_mask=self.attention_mask,
            )

            n_added = self.shadow_seq.shape[1] - self.all_tokens.shape[1]
            if n_added > 0:
                new_hard_tokens = self.shadow_seq[:, -n_added:]
                new_hard_tokens = torch.nn.functional.one_hot(
                    new_hard_tokens, self.vocab_size
                )
                self.all_tokens = torch.cat([self.all_tokens, new_hard_tokens], 1)
                self.hard_tokens_mask = torch.cat(
                    [
                        self.hard_tokens_mask,
                        torch.ones(
                            (self.batch_size, n_added),
                            dtype=torch.bool,
                            device=self.hard_tokens_mask.device,
                        ),
                    ],
                    1,
                )

            self._finished = self._finished | (
                self.shadow_seq[:, -1] == self.eos_token_id
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
    ) -> SoftTokenGeneratorOutput:
        if not use_kv_cache:
            raise ValueError("soft token generation requires use_kv_cache=True")
        super()._generate(
            net=net,
            token_ids=token_ids,
            max_tokens_generated=max_tokens_generated,
            use_kv_cache=use_kv_cache,
            attention_mask=attention_mask,
            use_bf16=use_bf16,
        )

        W = net.embed_tokens.weight
        embeddings = self.all_tokens.float() @ W.float()

        if self.all_noise:
            noise = torch.cat(self.all_noise, dim=1)
            n_pad = self.all_tokens.shape[1] - noise.shape[1]
            if n_pad > 0:
                noise = torch.cat(
                    [
                        noise,
                        torch.zeros(
                            noise.shape[0], n_pad, noise.shape[2], device=noise.device
                        ),
                    ],
                    dim=1,
                )
            embeddings = embeddings + noise.float()

        shadow_ids = self.all_tokens.argmax(-1)

        return SoftTokenGeneratorOutput(
            embeddings=embeddings,
            shadow_ids=shadow_ids,
            attention_mask=self.attention_mask,
            hard_tokens_mask=self.hard_tokens_mask,
        )

    @property
    def finished(self) -> bool:
        return bool(self._finished.all())

    @property
    def _extra_tokens_per_step_bound(self) -> int:
        if self.prefill:
            return len(self.prefill.filling)
        return 0

    @property
    def _extra_kv_reserve(self) -> int:
        prefill_ids = (
            self.max_tokens_prefill
            if self.max_tokens_prefill is not None
            else self.switch_to_hard_tokens_condition
        )
        if prefill_ids is not None:
            return prefill_ids.numel()
        return 0


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


@torch.inference_mode()
def generate_soft_tokens(
    net: BaseTransformer,
    token_ids: Int[Tensor, "B L"],
    eos_token_id: int = 151645,
    pad_token_id: int = 151643,
    max_tokens_generated: int = sys.maxsize,
    use_kv_cache: bool = True,
    attention_mask: torch.Tensor | None = None,
    temperature: float = 1.0,
    use_bf16: bool = False,
    switch_to_hard_tokens_condition: Int[torch.Tensor, " M"] | None = None,
    max_tokens_prefill: Int[torch.Tensor, " N"] | None = None,
    max_tokens_prefill_steps_before_end: int = 0,
    prefill: PreFill | None = None,
    soft_token_noise_std: float | None = None,
    min_soft_steps: int = 0,
) -> SoftTokenGeneratorOutput:
    if pad_token_id is None:
        pad_token_id = eos_token_id

    return _SoftGenerator(
        vocab_size=net.vocab_size,
        temperature=temperature,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        switch_to_hard_tokens_condition=switch_to_hard_tokens_condition,
        max_tokens_prefill=max_tokens_prefill,
        max_tokens_prefill_steps_before_end=max_tokens_prefill_steps_before_end,
        prefill=prefill,
        soft_token_noise_std=soft_token_noise_std,
        min_soft_steps=min_soft_steps,
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


@torch.inference_mode()
def generate_internal_reasoning_tokens(
    net: BaseTransformer,
    token_ids: Int[Tensor, "B L"],
    soft_block_size: int = 4,
    max_cycles: int = 30,
    valid_hard_token_ids: list[int] | None = None,
    done_token_id: int = 151645,
    pad_token_id: int = 151643,
    temperature: float = 1.0,
    attention_mask: Bool[Tensor, "B L"] | None = None,
    use_bf16: bool = False,
    think_token_id: int | None = None,
) -> InternalReasoningGeneratorOutput:
    """Generate with fixed interleaving: soft_block_size hidden-state passes then 1 hard token per cycle.

    Parameters
    ----------
    net
        Transformer model.
    token_ids
        Prompt token IDs [B, L].
    soft_block_size
        Number of soft (hidden-state passthrough) forward passes per cycle.
    max_cycles
        Maximum number of cycles (each cycle produces one hard token).
    valid_hard_token_ids
        Token IDs that can be sampled at hard positions. If None, full vocabulary is used.
    done_token_id
        Token ID that signals generation is complete.
    pad_token_id
        Token ID for padding finished sequences.
    temperature
        Sampling temperature for hard tokens.
    attention_mask
        Left-padded attention mask for prompt [B, L].
    use_bf16
        Whether to use bf16 autocast.
    """
    device = token_ids.device
    B = token_ids.shape[0]
    L_prompt = token_ids.shape[1]

    max_seq_len = L_prompt + max_cycles * (soft_block_size + 1)
    kv_caches = [
        KVCache(
            max_seq_len=max_seq_len,
            num_heads=net.attn_num_kv_heads,
            head_dim=net.attn_head_d,
            device=device,
        )
        for _ in range(len(net.layers))
    ]

    hard_token_ids = torch.full(
        (B, max_cycles), pad_token_id, dtype=torch.long, device=device
    )
    hard_log_probs = torch.zeros(B, max_cycles, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    n_cycles = torch.full((B,), max_cycles, dtype=torch.long, device=device)

    valid_mask: Tensor | None = None
    if valid_hard_token_ids is not None:
        valid_mask = torch.full((net.vocab_size,), float("-inf"), device=device)
        valid_mask[valid_hard_token_ids] = 0.0

    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
    ):
        h: Float[Tensor, "B L D"] = net(
            token_ids,
            kv_caches=kv_caches,
            attention_mask=attention_mask,
            return_hidden_states=True,
        )
    h = h[:, -1:]  # [B, 1, D]

    if think_token_id is not None:
        think_embed = net.embed_tokens(
            torch.full((B, 1), think_token_id, dtype=torch.long, device=device)
        )

    for cycle in range(max_cycles):
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
        ):
            if think_token_id is not None:
                for _ in range(soft_block_size):
                    h = net(think_embed, kv_caches=kv_caches, return_hidden_states=True)
            else:
                for _ in range(soft_block_size):
                    h = net(h, kv_caches=kv_caches, return_hidden_states=True)
                    h = net.apply_soft_projection(h)

            h = net(h, kv_caches=kv_caches, return_hidden_states=True)
            logits = net.lm_head(h)  # [B, 1, V]

        logits_squeezed = logits.squeeze(1).float()  # [B, V]
        if valid_mask is not None:
            logits_squeezed = logits_squeezed + valid_mask.unsqueeze(0)

        log_probs_all = torch.log_softmax(logits_squeezed / temperature, dim=-1)
        probs = torch.softmax(logits_squeezed / temperature, dim=-1)
        token = torch.multinomial(probs, num_samples=1)  # [B, 1]
        log_prob = log_probs_all.gather(1, token).squeeze(1)  # [B]

        token = token.squeeze(1)  # [B]
        token = torch.where(finished, torch.full_like(token, pad_token_id), token)
        log_prob = torch.where(finished, torch.zeros_like(log_prob), log_prob)

        just_finished = ~finished & (token == done_token_id)
        n_cycles[just_finished] = cycle + 1
        finished = finished | (token == done_token_id)

        hard_token_ids[:, cycle] = token
        hard_log_probs[:, cycle] = log_prob

        if finished.all():
            break

        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
        ):
            h = net.embed_tokens(token.unsqueeze(1))  # [B, 1, D]

    return InternalReasoningGeneratorOutput(
        hard_token_ids=hard_token_ids,
        hard_log_probs=hard_log_probs,
        n_cycles=n_cycles,
    )
