from dataclasses import dataclass

import numpy as np
from jaxtyping import Float
import torch
import torch.nn as nn


T = Float[torch.Tensor, "batch seq_length d"]

TMH = Float[torch.Tensor, "batch num_heads seq_length head_d"]


def attention(q: TMH, k: TMH, v: TMH, causal: bool = False) -> TMH:
    # [batch, num_heads, seq_length, head_d]
    seq_length, head_d = q.shape[-2:]
    dot_prods = torch.matmul(q, k.transpose(3, 2)) / (head_d**0.5)
    if causal:
        dot_prods.masked_fill_(
            torch.ones(seq_length, seq_length).triu(diagonal=1).bool(), -torch.inf
        )
    soft_max_dot_prods = (dot_prods).softmax(-1)

    return torch.matmul(soft_max_dot_prods, v)


@dataclass
class RopeBufferParams:
    context_length: int
    base_value: float = 10000


def create_rope_sine_cosine_tensors(
    dim: int, rope_params: RopeBufferParams
) -> tuple[Float[torch.Tensor, "dim length"], Float[torch.Tensor, "dim length"]]:
    sin = torch.zeros([rope_params.context_length, dim], dtype=float)
    cos = torch.zeros([rope_params.context_length, dim], dtype=float)

    for m in range(rope_params.context_length):
        for i in range(dim):
            theta = rope_params.base ** (-2 * (i - 1) / dim)
            sin[m, i] = np.sin(m * theta)
            cos[m, i] = np.cos(m * theta)

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
