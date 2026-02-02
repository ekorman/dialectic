import sys
from typing import Callable, Literal

import torch
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
    prefill_callback: Callable[[Int[Tensor, "B L"]], Int[Tensor, "B L"]] = lambda x: x,
) -> Int[Tensor, "B L"]:
    if not soft_tokens and sampling_strategy not in ["greedy", "sample"]:
        raise ValueError("`sampling_strategy` must be one of 'greedy' or 'sample'.")

    if pad_token_id is None:
        pad_token_id = eos_token_id

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

    batch_size = token_ids.shape[0]
    device = token_ids.device
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    if soft_tokens:
        shadow_seq = token_ids
        all_tokens = torch.nn.functional.one_hot(token_ids, net.vocab_size).float()
        input_tokens = torch.nn.functional.one_hot(token_ids, net.vocab_size).float()
    else:
        all_tokens = token_ids
        input_tokens = token_ids

    tokens_generated = 0
    while tokens_generated < max_tokens_generated:
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
        ):
            logits: Float[torch.Tensor, "B 1 V"] = net(
                input_tokens, kv_caches=kv_caches, attention_mask=attention_mask
            )

        if (not soft_tokens) and (sampling_strategy == "greedy"):
            next_token = logits.argmax(-1)
            hard_token_id = next_token
        elif not soft_tokens:
            scaled_logits = logits / temperature

            probs = torch.softmax(scaled_logits.squeeze(1), dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            hard_token_id = next_token
        else:
            scaled_logits = logits / temperature
            probs = torch.softmax(scaled_logits, dim=-1)

            hard_token_id = probs.argmax(-1)
            shadow_seq = torch.cat([shadow_seq, hard_token_id], 1)

            next_token = probs  # need .detach()?

        finished = finished | (hard_token_id.squeeze(-1) == eos_token_id)

        if not soft_tokens:
            next_token = torch.where(
                finished.unsqueeze(-1),
                torch.full_like(next_token, pad_token_id),
                next_token,
            )
        else:
            next_token = torch.where(
                finished.unsqueeze(-1).unsqueeze(-1),
                torch.nn.functional.one_hot(
                    torch.tensor(pad_token_id), logits.shape[-1]
                ),
                next_token,
            )

        if finished.all():
            break

        all_tokens = torch.cat([all_tokens, next_token], 1)
        tokens_generated += 1

        if use_kv_cache:
            input_tokens = next_token
        else:
            input_tokens = all_tokens

        if attention_mask is not None:
            new_mask = finished.unsqueeze(-1)
            attention_mask = torch.cat([attention_mask, new_mask], 1)

    return all_tokens


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
    # attention mask is True where we want to mask (i.e. ignore)
    attention_mask = (
        torch.tensor([t.attention_mask for t in tokens], dtype=torch.bool) == 0
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
