from dataclasses import dataclass

import numpy as np
from jaxtyping import Float
from torch import Tensor
import torch
import torch.nn as nn


T = Float[Tensor, "batch seq_length d"]

TMH = Float[Tensor, "batch num_heads seq_length head_d"]


def attention(q: TMH, k: TMH, v: TMH, causal: bool = False) -> TMH:
    # [batch, num_heads, seq_length, head_d]
    num_heads, seq_length, head_d = q.shape[-3:]

    num_kv_heads = k.shape[1]

    if num_kv_heads != num_heads:
        k = k.repeat_interleave(num_heads // num_kv_heads, 1)
        v = v.repeat_interleave(num_heads // num_kv_heads, 1)

    dot_prods = torch.matmul(q, k.transpose(3, 2)) / (head_d**0.5)
    if causal:
        dot_prods.masked_fill_(
            torch.ones(seq_length, seq_length).triu(diagonal=1).bool(), -torch.inf
        )
    soft_max_dot_prods = (dot_prods).softmax(-1)

    return torch.matmul(soft_max_dot_prods, v)


# for our RoPE implementation we follow closely the paper, where adjacent components in the vector
# dimension are paired. this contrasts with huggingface where they split the vector into first and second half
# instead of interleaving
#
# c.f.: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py
# https://github.com/huggingface/transformers/issues/25199
# https://github.com/rasbt/LLMs-from-scratch/pull/747
# https://github.com/rasbt/LLMs-from-scratch/issues/751


def apply_rope(
    x: Float[Tensor, "batch num_heads seq_length d"],
    sin: Float[Tensor, "batch seq_length d"],
    cos: Float[Tensor, "batch seq_length d"],
) -> Float[Tensor, "batch seq_length d"]:
    d = x.shape[-1]
    # rot = torch.cat((-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1)
    rot = torch.stack(
        [-x[:, :, :, torch.arange(1, d, 2)], x[:, :, :, torch.arange(0, d, 2)]], -1
    ).reshape(*x.shape)

    return x * cos.unsqueeze(1) + rot * sin.unsqueeze(1)


@dataclass
class RopeBufferParams:
    context_length: int
    base_value: float = 10000


def create_rope_sine_cosine_tensors(
    dim: int, batch_size: int, rope_params: RopeBufferParams
) -> tuple[Float[Tensor, "dim length"], Float[Tensor, "dim length"]]:
    sin = torch.zeros([rope_params.context_length, dim], dtype=torch.float32)
    cos = torch.zeros([rope_params.context_length, dim], dtype=torch.float32)

    thetas = rope_params.base_value ** (-2 * (torch.arange(dim // 2)) / dim)
    thetas = thetas.repeat_interleave(2)

    freqs = torch.outer(torch.arange(rope_params.context_length), thetas)

    sin = (
        freqs.sin()
        .view(1, rope_params.context_length, dim)
        .expand(batch_size, rope_params.context_length, dim)
    )
    cos = (
        freqs.cos()
        .view(1, rope_params.context_length, dim)
        .expand(batch_size, rope_params.context_length, dim)
    )

    return sin, cos


class MHSA(nn.Module):
    def __init__(
        self,
        d: int,
        num_heads: int,
        bias: bool = False,
        causal: bool = False,
        rope_params: RopeBufferParams | None = None,
    ):
        super().__init__()
        if d % num_heads != 0:
            raise ValueError(
                f"d should be divisible by num_heads but got d={d}, num_heads={num_heads}"
            )
        self.num_heads = num_heads
        self.head_d = d // num_heads
        self.Q = nn.Linear(d, d, bias=bias)
        self.K = nn.Linear(d, d, bias=bias)
        self.V = nn.Linear(d, d, bias=bias)
        self.out_proj = nn.Linear(d, d, bias=bias)

        self.causal = causal
        self.use_rope = rope_params is not None
        if self.use_rope:
            sin, cos = create_rope_sine_cosine_tensors(rope_params.context_length)
            self.register_buffer("rope_sin", sin)
            self.register_buffer("rope_cos", cos)

    def forward(self, x: T) -> T:
        batch_size, seq_length = x.shape[:2]

        q: T = self.Q(x)
        k: T = self.K(x)
        v: T = self.V(x)

        # view tensors as [batch, num_heads, seq_length, head_d] to break into heads
        q = q.view(batch_size, seq_length, self.num_heads, self.head_d).transpose(2, 1)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_d).transpose(2, 1)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_d).transpose(2, 1)

        ret = attention(q, k, v, causal=self.causal)

        # move sequence length back to second position and join the heads
        ret = ret.transpose(1, 2).contiguous()
        ret = ret.view(batch_size, seq_length, -1)

        return self.out_proj(ret)


class Qwen(nn.Module):
    def __init__(self, vocab_size: int, token_embedding_dim: int):
        super().__init__()
        self.embedder = nn.Embedding(vocab_size, token_embedding_dim)
