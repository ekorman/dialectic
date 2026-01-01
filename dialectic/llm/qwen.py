"""Type conventions:

- 'B': batch size
- 'L': sequence length
- 'D': vector dimension
- 'NH': number of (query) attention heads
- 'NKVH': number of key/value attention heads
- 'DHead': dimension of each head
"""

from typing import Literal

import torch
import torch.nn as nn
from jaxtyping import Bool, Float
from tokenizers import Tokenizer
from torch import Tensor

from dialectic.llm.tokenizer import Message, get_input_text_from_messages


class KVCache:
    def __init__(
        self,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        device: str | torch.device,
    ):
        self._max_seq_len = max_seq_len
        self._num_heads = num_heads
        self._head_dim = head_dim
        self._device = device
        self._keys = None
        self._values = None
        self._seq_len = 0

    def update_and_get_keys(self, k: torch.Tensor):
        if self._keys is None:
            batch_size = k.shape[0]
            self._keys = torch.zeros(
                batch_size,
                self._num_heads,
                self._max_seq_len,
                self._head_dim,
                device=self._device,
                dtype=k.dtype,
            )
        seq_len = k.shape[2]
        self._keys[:, :, self._seq_len : self._seq_len + seq_len] = k
        return self._keys[:, :, : self._seq_len + seq_len]

    def update_and_get_values(self, v: torch.Tensor):
        if self._values is None:
            batch_size = v.shape[0]
            self._values = torch.zeros(
                batch_size,
                self._num_heads,
                self._max_seq_len,
                self._head_dim,
                device=self._device,
                dtype=v.dtype,
            )
        seq_len = v.shape[2]
        self._values[:, :, self._seq_len : self._seq_len + seq_len] = v
        self._seq_len += seq_len
        return self._values[:, :, : self._seq_len]

    def get_position_offset(self) -> int:
        return self._seq_len


def attention(
    q: Float[Tensor, "B NH L DHead"],
    k: Float[Tensor, "B NKVH L DHead"],
    v: Float[Tensor, "B NKVH L DHead"],
    causal: bool = False,
    attention_mask: Bool[Tensor, "B L"] | None = None,
) -> Float[Tensor, "B NH L DHead"]:
    num_heads, seq_length, head_d = q.shape[-3:]

    num_kv_heads = k.shape[1]

    if num_kv_heads != num_heads:
        k = k.repeat_interleave(num_heads // num_kv_heads, 1)
        v = v.repeat_interleave(num_heads // num_kv_heads, 1)

    dot_prods: torch.Tensor = torch.matmul(q, k.transpose(3, 2)) / (head_d**0.5)

    if attention_mask is not None:
        # attention_mask shape: (B, k_len), True = masked/padding
        # expand to (B, 1, 1, k_len) for broadcasting with (B, NH, q_len, k_len)
        key_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        dot_prods.masked_fill_(key_mask, -torch.inf)

    if causal:
        dot_prods.masked_fill_(
            torch.ones(seq_length, seq_length, device=dot_prods.device)
            .triu(diagonal=1)
            .bool(),
            -torch.inf,
        )
    soft_max_dot_prods = (dot_prods).softmax(-1)
    # NaNs can occur when entire rows are -inf (all keys masked for a query)
    soft_max_dot_prods = torch.nan_to_num(soft_max_dot_prods, nan=0.0)

    return torch.matmul(soft_max_dot_prods, v)


# for our RoPE implementation we follow huggingface where they split the vector into first and second half
# instead of interleaving. this contrasts with the paper where adjacent components in the vector
# dimension are paired
#
# c.f.: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py
# https://github.com/huggingface/transformers/issues/25199
# https://github.com/rasbt/LLMs-from-scratch/pull/747
# https://github.com/rasbt/LLMs-from-scratch/issues/751


def apply_rope(
    x: Float[Tensor, "B NH L D"],
    sin: Float[Tensor, "B L D"],
    cos: Float[Tensor, "B L D"],
) -> Float[Tensor, "B NH L D"]:
    rot = torch.cat((-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1)
    sin = sin.to(x.device)
    cos = cos.to(x.device)
    return x * cos.expand(x.shape[0], x.shape[2], x.shape[3]).unsqueeze(
        1
    ) + rot * sin.expand(x.shape[0], x.shape[2], x.shape[3]).unsqueeze(1)


def create_rope_sine_cosine_tensors(
    dim: int, base_value: float, context_length: int
) -> tuple[Float[Tensor, "1 L D"], Float[Tensor, "dim length"]]:
    sin = torch.zeros([context_length, dim], dtype=torch.float32)
    cos = torch.zeros([context_length, dim], dtype=torch.float32)

    thetas = base_value ** (-2 * (torch.arange(dim // 2)) / dim)
    thetas = thetas.repeat(2)

    freqs = torch.outer(torch.arange(context_length), thetas)

    sin = freqs.sin().view(1, context_length, dim)
    cos = freqs.cos().view(1, context_length, dim)

    return sin, cos


class RMSNorm(nn.Module):
    """RMSNorm, following HuggingFace's implementation"""

    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps  # ty: ignore[unresolved-attribute]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(-1, keepdim=True)
        return self.weight * x * torch.rsqrt(var + self.eps)


class MHSA(nn.Module):
    def __init__(
        self,
        d: int,
        head_d: int,  # dimension for each individual head
        num_heads: int,
        num_kv_heads: int | None = None,
        bias: bool = False,
        causal: bool = False,
        rope_base_value: float | None = None,
        apply_rms_norm: bool = False,
    ):
        super().__init__()

        self.num_heads = num_heads  # ty: ignore[unresolved-attribute]
        self.head_d = head_d  # ty: ignore[unresolved-attribute]
        if num_kv_heads is None:
            num_kv_heads = num_heads
        self.num_kv_heads = num_kv_heads  # ty: ignore[unresolved-attribute]
        self.q_proj = nn.Linear(d, num_heads * head_d, bias=bias)
        self.k_proj = nn.Linear(d, num_kv_heads * head_d, bias=bias)
        self.v_proj = nn.Linear(d, num_kv_heads * head_d, bias=bias)
        self.o_proj = nn.Linear(num_heads * head_d, d, bias=bias)

        self.causal = causal  # ty: ignore[unresolved-attribute]
        self.use_rope = rope_base_value is not None  # ty: ignore[unresolved-attribute]
        self.rope_base_value = rope_base_value  # ty: ignore[unresolved-attribute]
        self.apply_rms_norm = apply_rms_norm  # ty: ignore[unresolved-attribute]

        if apply_rms_norm:
            self.q_norm = RMSNorm(self.head_d)
            self.k_norm = RMSNorm(self.head_d)

    def forward(
        self,
        x: Float[Tensor, "B L D"],
        kv_cache: KVCache | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> Float[Tensor, "B L D"]:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        batch_size = x.shape[0]

        # view tensors as [batch, num_heads, seq_length, head_d] to break into heads
        q = q.view(batch_size, -1, self.num_heads, self.head_d).transpose(2, 1)
        k = k.view(batch_size, -1, self.num_kv_heads, self.head_d).transpose(2, 1)
        v = v.view(batch_size, -1, self.num_kv_heads, self.head_d).transpose(2, 1)

        if self.apply_rms_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.use_rope:
            position_offset = 0 if kv_cache is None else kv_cache.get_position_offset()

            sin, cos = create_rope_sine_cosine_tensors(
                self.head_d,
                base_value=self.rope_base_value,
                context_length=position_offset + x.shape[1],
            )

            sin = sin[:, position_offset : position_offset + x.shape[1]]
            cos = cos[:, position_offset : position_offset + x.shape[1]]

            q = apply_rope(q, sin=sin, cos=cos)
            k = apply_rope(k, sin=sin, cos=cos)

        if kv_cache is not None:
            k = kv_cache.update_and_get_keys(k)
            v = kv_cache.update_and_get_values(v)

        ret = attention(q, k, v, causal=self.causal, attention_mask=attention_mask)

        # move sequence length back to second position and join the heads
        ret = ret.transpose(1, 2).contiguous()
        ret = ret.view(batch_size, x.shape[1], -1)
        ret = self.o_proj(ret)

        return ret


class GatedMLP(nn.Module):
    def __init__(self, d: int, hidden_d: int):
        super().__init__()
        self.gate_proj = nn.Linear(d, hidden_d, bias=False)
        self.up_proj = nn.Linear(d, hidden_d, bias=False)
        self.down_proj = nn.Linear(hidden_d, d, bias=False)

    def forward(self, x: Float[Tensor, "... d"]) -> Float[Tensor, "... d"]:
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class QwenDecoderLayer(nn.Module):
    """
    RMSNorm -> Residual attention -> RMSNorm -> Residual MLP
    """

    def __init__(
        self,
        d: int,
        attn_head_d: int,
        attn_num_heads: int,
        attn_num_kv_heads: int,
        mlp_hidden_d: int,
        rope_base_value: float | None = None,
    ):
        super().__init__()
        self.input_layernorm = RMSNorm(d)
        self.self_attn = MHSA(
            d=d,
            head_d=attn_head_d,
            num_heads=attn_num_heads,
            num_kv_heads=attn_num_kv_heads,
            causal=True,
            apply_rms_norm=True,
            rope_base_value=rope_base_value,
        )
        self.post_attention_layernorm = RMSNorm(d)
        self.mlp = GatedMLP(d=d, hidden_d=mlp_hidden_d)

    def forward(
        self,
        x,
        kv_cache: KVCache | None = None,
        attention_mask: torch.Tensor | None = None,
    ):
        x = x + self.self_attn(
            self.input_layernorm(x), kv_cache=kv_cache, attention_mask=attention_mask
        )
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


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
        rope_base_value: float = None,
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
                QwenDecoderLayer(
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
    ):
        x = self.embed_tokens(x)

        for layer, kv_cache in zip(self.layers, kv_caches or [None] * len(self.layers)):
            x = layer(x, kv_cache=kv_cache, attention_mask=attention_mask)

        x = self.norm(x)

        # just get last element of output sequence
        # important: if attention_mask is not None then we assume left padding!
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
    token_ids: torch.LongTensor,
    eos_token_id: int,
    sampling_strategy: Literal["greedy", "sample"] = "sample",
    max_tokens_generated: int = float("inf"),
    use_kv_cache: bool = True,
    attention_mask: torch.Tensor | None = None,
) -> torch.LongTensor:
    assert sampling_strategy in ["greedy", "sample"]

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

    all_token_ids = token_ids  # Keep track of full sequence
    input_ids = token_ids

    tokens_generated = 0
    while tokens_generated < max_tokens_generated:
        logits: torch.Tensor = net(
            input_ids, kv_caches=kv_caches, attention_mask=attention_mask
        )

        if sampling_strategy == "greedy":
            next_token_id = logits.argmax(-1)
        else:
            next_token_id = torch.distributions.Categorical(logits=logits).sample()

        if (next_token_id == eos_token_id).all():
            break

        all_token_ids = torch.cat([all_token_ids, next_token_id], 1)
        tokens_generated += 1

        if use_kv_cache:
            input_ids = next_token_id  # Only new token for next iteration
        else:
            input_ids = all_token_ids
        if attention_mask is not None:
            attention_mask = torch.cat(
                [attention_mask, torch.zeros(input_ids.shape[0], 1, dtype=torch.bool)],
                1,
            )

    return all_token_ids


def generate_from_text(
    net: Qwen,
    tokenizer: Tokenizer,
    text_batch: list[str],
    eos_token: str,
    pad_token: str,
    sampling_strategy: Literal["greedy", "sample"] = "sample",
    max_tokens_generated: int = float("inf"),
    device: str | torch.device | None = None,
    use_kv_cache: bool = True,
) -> list[str]:
    if device is None:
        device = next(net.parameters()).device

    pad_token_id = tokenizer.token_to_id(pad_token)
    tokenizer.enable_padding(pad_id=pad_token_id, pad_token=pad_token, direction="left")
    tokens = tokenizer.encode_batch(text_batch)
    token_ids = torch.tensor([t.ids for t in tokens]).to(device)
    attention_mask = (
        torch.tensor([t.attention_mask for t in tokens], dtype=torch.bool) == 0
    ).to(device)

    eos_token_id = tokenizer.token_to_id(eos_token)

    token_ids = generate_from_tokens(
        net=net,
        token_ids=token_ids,
        eos_token_id=eos_token_id,
        sampling_strategy=sampling_strategy,
        max_tokens_generated=max_tokens_generated,
        use_kv_cache=use_kv_cache,
        attention_mask=attention_mask,
    )

    return [tokenizer.decode(batch.tolist()) for batch in token_ids]


def generate_from_chat(
    net: Qwen,
    tokenizer: Tokenizer,
    messages: list[Message],
    eos_token: str = "<|endoftext|>",
    pad_token: str = "<|endoftext|>",
    sampling_strategy: Literal["greedy", "sample"] = "sample",
    max_tokens_generated: int = 1000,
    device: str | torch.device | None = None,
):
    return generate_from_text(
        net=net,
        tokenizer=tokenizer,
        sampling_strategy=sampling_strategy,
        text_batch=[get_input_text_from_messages(messages, add_generation_prompt=True)],
        eos_token=eos_token,
        pad_token=pad_token,
        max_tokens_generated=max_tokens_generated,
        device=device,
    )
