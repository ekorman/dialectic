import sys
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
class HardTokenGeneratorOutput:
    tokens: Int[torch.Tensor, "B L"]
    attention_mask: Bool[torch.Tensor, "B L"] | None


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
) -> HardTokenGeneratorOutput:
    if sampling_strategy not in ["greedy", "sample"]:
        raise ValueError("`sampling_strategy` must be one of 'greedy' or 'sample'.")

    if pad_token_id is None:
        pad_token_id = eos_token_id

    raw_net = unwrap_model(net)
    device = token_ids.device
    all_tokens = token_ids
    finished = torch.zeros(token_ids.shape[0], dtype=torch.bool, device=device)
    n_generated = 0

    if use_kv_cache:
        kv_caches = [
            KVCache(
                max_seq_len=max_tokens_generated + token_ids.shape[1],
                num_heads=raw_net.attn_num_kv_heads,
                head_dim=raw_net.attn_head_d,
                device=next(raw_net.parameters()).device,
            )
            for _ in range(len(raw_net.layers))
        ]
    else:
        kv_caches = None

    def _update_state(
        *, logits: Float[torch.Tensor, "B 1 V"], finished, all_tokens, attention_mask
    ):
        if sampling_strategy == "greedy":
            next_token = logits.argmax(-1)
        else:
            scaled_logits = logits / temperature
            probs = torch.softmax(scaled_logits.squeeze(1).float(), dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        next_token = torch.where(
            finished.unsqueeze(-1),
            torch.full_like(next_token, pad_token_id),
            next_token,
        )

        all_tokens = torch.cat([all_tokens, next_token], 1)

        finished = finished | (all_tokens[:, -1] == eos_token_id)
        if bool(finished.all()):
            return finished, all_tokens, attention_mask

        if attention_mask is not None:
            new_mask = ~finished.unsqueeze(-1)
            attention_mask = torch.cat([attention_mask, new_mask], 1)

        return finished, all_tokens, attention_mask

    input_tokens = token_ids
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
    ):
        while n_generated < max_tokens_generated:
            logits: Float[torch.Tensor, "B 1 V"] = net(
                input_tokens,
                kv_caches=kv_caches,
                attention_mask=attention_mask,
            )

            finished, all_tokens, attention_mask = _update_state(
                logits=logits,
                finished=finished,
                all_tokens=all_tokens,
                attention_mask=attention_mask,
            )

            if bool(finished.all()):
                break

            input_tokens = all_tokens[:, -1:] if use_kv_cache else all_tokens
            n_generated += 1

    return HardTokenGeneratorOutput(tokens=all_tokens, attention_mask=attention_mask)


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
