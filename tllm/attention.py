from jaxtyping import Float
import torch
import torch.nn as nn

T = Float[torch.Tensor, "batch seq-length d"]


class MHSA(nn.Module):
    def __init__(self, d: int, num_heads: int, bias: bool = False):
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

    def forward(self, x: T) -> T:
        batch_size, seq_length = x.shape[:2]

        q: T = self.Q(x)
        k: T = self.K(x)
        v: T = self.V(x)

        q: Float[torch.Tensor, "batch num_heads seq-length head_d"] = q.view(
            batch_size, seq_length, self.num_heads, self.head_d
        ).transpose(2, 1)

        k: Float[torch.Tensor, "batch num_heads seq-length head_d"] = k.view(
            batch_size, seq_length, self.num_heads, self.head_d
        ).transpose(2, 1)

        v: Float[torch.Tensor, "batch num_heads seq-length head_d"] = v.view(
            batch_size, seq_length, self.num_heads, self.head_d
        ).transpose(2, 1)

        soft_max_dot_prods: Float[
            torch.Tensor, "batch num_heads seq-length seq-length"
        ] = (torch.matmul(q, k.transpose(3, 2)) / (self.head_d**0.5)).softmax(-1)

        ret = torch.matmul(soft_max_dot_prods, v)
        ret = ret.transpose(1, 2).contiguous()
        ret = ret.view(batch_size, seq_length, -1)

        return self.out_proj(ret)


class Qwen(nn.Module):
    def __init__(self, vocab_size: int, token_embedding_dim: int):
        super().__init__()
        self.embedder = nn.Embedding(vocab_size, token_embedding_dim)
