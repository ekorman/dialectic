import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

T = Float[Tensor, "batch seq_length d"]

TMH = Float[Tensor, "batch num_heads seq_length head_d"]


def attention(
    q: TMH, k: TMH, v: TMH, causal: bool = False, scaling: float | None = None
) -> TMH:
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


# for our RoPE implementation we follow huggingface where they split the vector into first and second half
# instead of interleaving. this contrasts with the paper where adjacent components in the vector
# dimension are paired
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
    rot = torch.cat((-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1)

    return x * cos.expand(x.shape[0], x.shape[2], x.shape[3]).unsqueeze(
        1
    ) + rot * sin.expand(x.shape[0], x.shape[2], x.shape[3]).unsqueeze(1)


def create_rope_sine_cosine_tensors(
    dim: int, base_value: float, context_length: int
) -> tuple[Float[Tensor, "dim length"], Float[Tensor, "dim length"]]:
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
        self.eps = eps

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
        rope_base_value: float = None,
        apply_rms_norm: bool = False,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_d = head_d
        # kv_head_d = sum of dimensions of k, v heads
        if num_kv_heads is None:
            num_kv_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.q_proj = nn.Linear(d, num_heads * head_d, bias=bias)
        self.k_proj = nn.Linear(d, num_kv_heads * head_d, bias=bias)
        self.v_proj = nn.Linear(d, num_kv_heads * head_d, bias=bias)
        self.o_proj = nn.Linear(num_heads * head_d, d, bias=bias)

        self.causal = causal
        self.use_rope = rope_base_value is not None
        self.rope_base_value = rope_base_value
        self.apply_rms_norm = apply_rms_norm

        # if self.use_rope:
        #     sin, cos = create_rope_sine_cosine_tensors(head_d, rope_params=rope_params)
        # self.register_buffer("rope_sin", sin)
        # self.register_buffer("rope_cos", cos)

        if apply_rms_norm:
            self.q_norm = RMSNorm(self.head_d)
            self.k_norm = RMSNorm(self.head_d)

    def forward(self, x: T) -> T:
        batch_size, seq_length = x.shape[:2]

        q: T = self.q_proj(x)
        k: T = self.k_proj(x)
        v: T = self.v_proj(x)

        # view tensors as [batch, num_heads, seq_length, head_d] to break into heads
        q = q.view(batch_size, seq_length, self.num_heads, self.head_d).transpose(2, 1)
        k = k.view(batch_size, seq_length, self.num_kv_heads, self.head_d).transpose(
            2, 1
        )
        v = v.view(batch_size, seq_length, self.num_kv_heads, self.head_d).transpose(
            2, 1
        )

        if self.apply_rms_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.use_rope:
            sin, cos = create_rope_sine_cosine_tensors(
                self.head_d, base_value=self.rope_base_value, context_length=x.shape[1]
            )
            q = apply_rope(q, sin=sin, cos=cos)
            k = apply_rope(k, sin=sin, cos=cos)
        ret = attention(q, k, v, causal=self.causal)

        # move sequence length back to second position and join the heads
        ret = ret.transpose(1, 2).contiguous()
        ret = ret.view(batch_size, seq_length, -1)

        return self.o_proj(ret)


class GatedMLP(nn.Module):
    def __init__(self, d: int, hidden_d: int):
        super().__init__()
        self.gate_proj = nn.Linear(d, hidden_d, bias=False)
        self.up_proj = nn.Linear(d, hidden_d, bias=False)
        self.down_proj = nn.Linear(hidden_d, d, bias=False)

    def forward(self, x: T) -> T:
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
        rope_base_value: float = None,
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

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
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
        self.lm_head = nn.Linear(d, vocab_size)

    def forward(self, x):
        x = self.embed_tokens(x)

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        # just get last element of output sequence
        x = x[:, -1:]
        return self.lm_head(x)
