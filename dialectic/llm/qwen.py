"""Type conventions:

- 'B': batch size
- 'L': sequence length
- 'D': vector dimension
- 'NH': number of (query) attention heads
- 'NKVH': number of key/value attention heads
- 'DHead': dimension of each head
- 'VC': vocab size
"""

import sys
from typing import Literal

import torch
import torch.nn as nn
from jaxtyping import Int
from tokenizers import Tokenizer
from torch import Tensor

from dialectic.llm.components import DecoderLayer, KVCache, RMSNorm
from dialectic.llm.templates import Message, get_qwen_input_text_from_messages


def create_qwen_decoder_layer(
    d: int,
    *,
    attn_head_d: int,
    attn_num_heads: int,
    attn_num_kv_heads: int,
    mlp_hidden_d: int,
    rope_base_value: float | None = None,
):
    return DecoderLayer(
        d=d,
        attn_head_d=attn_head_d,
        attn_num_heads=attn_num_heads,
        attn_num_kv_heads=attn_num_kv_heads,
        mlp_hidden_d=mlp_hidden_d,
        rope_base_value=rope_base_value,
        causal=True,
        apply_qk_rms_norm=True,
    )


class Qwen(nn.Module):
    def __init__(
        self,
        d: int,
        vocab_size: int,
        n_decoder_layers: int,
        attn_head_d: int,
        attn_num_heads: int,
        attn_num_kv_heads: int,
        mlp_hidden_d: int,
        rope_base_value: float | None = None,
    ):
        super().__init__()
        self.d = d
        self.attn_num_heads = attn_num_heads
        self.attn_num_kv_heads = attn_num_kv_heads
        self.attn_head_d = attn_head_d
        self.vocab_size = vocab_size
        self.embed_tokens = nn.Embedding(vocab_size, d)
        self.layers = nn.ModuleList(
            [
                create_qwen_decoder_layer(
                    d=d,
                    attn_head_d=attn_head_d,
                    attn_num_heads=attn_num_heads,
                    attn_num_kv_heads=attn_num_kv_heads,
                    mlp_hidden_d=mlp_hidden_d,
                    rope_base_value=rope_base_value,
                )
                for _ in range(n_decoder_layers)
            ]
        )
        self.norm = RMSNorm(d)
        self.lm_head = nn.Linear(d, vocab_size, bias=False)

    def forward(
        self,
        x,
        kv_caches: list[KVCache] | None = None,
        attention_mask: torch.Tensor | None = None,
        return_all_logits: bool = False,
        return_hidden_states: bool = False,
    ):
        x = self.embed_tokens(x)

        for layer, kv_cache in zip(self.layers, kv_caches or [None] * len(self.layers)):
            x = layer(x, kv_cache=kv_cache, attention_mask=attention_mask)

        x = self.norm(x)

        if return_hidden_states:
            return x

        # just get last element of output sequence
        # important: if attention_mask is not None then we assume left padding!
        if not return_all_logits:
            x = x[:, -1:]
        return self.lm_head(x)


def load_qwen_06b() -> Qwen:
    return Qwen(
        d=1024,
        vocab_size=151936,
        n_decoder_layers=28,
        attn_head_d=128,
        attn_num_heads=16,
        attn_num_kv_heads=8,
        mlp_hidden_d=3072,
        rope_base_value=1000000,
    )


@torch.inference_mode()
def generate_from_tokens(
    net: Qwen,
    token_ids: Int[Tensor, "B L"],
    eos_token_id: int = 151645,
    pad_token_id: int = 151643,
    sampling_strategy: Literal["greedy", "sample"] = "sample",
    max_tokens_generated: int = sys.maxsize,
    use_kv_cache: bool = True,
    attention_mask: torch.Tensor | None = None,  # should be left-padded
    temperature: float = 1.0,
    use_bf16: bool = False,
) -> Int[Tensor, "B L"]:
    assert sampling_strategy in ["greedy", "sample"]

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

    all_token_ids = token_ids
    input_ids = token_ids

    tokens_generated = 0
    while tokens_generated < max_tokens_generated:
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
        ):
            logits: torch.Tensor = net(
                input_ids, kv_caches=kv_caches, attention_mask=attention_mask
            )

        if sampling_strategy == "greedy":
            next_token_id = logits.argmax(-1)
        else:
            scaled_logits = logits / temperature
            probs = torch.softmax(scaled_logits.squeeze(1), dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)

        finished = finished | (next_token_id.squeeze(-1) == eos_token_id)

        next_token_id = torch.where(
            finished.unsqueeze(-1),
            torch.full_like(next_token_id, pad_token_id),
            next_token_id,
        )

        if finished.all():
            break

        all_token_ids = torch.cat([all_token_ids, next_token_id], 1)
        tokens_generated += 1

        if use_kv_cache:
            input_ids = next_token_id
        else:
            input_ids = all_token_ids

        if attention_mask is not None:
            new_mask = finished.unsqueeze(-1)
            attention_mask = torch.cat([attention_mask, new_mask], 1)

    return all_token_ids


def generate_from_text(
    net: Qwen,
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


def generate_from_chat(
    net: Qwen,
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
