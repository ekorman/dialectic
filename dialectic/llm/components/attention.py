import torch
import torch.nn as nn
from jaxtyping import Bool, Float
from torch import Tensor

from dialectic.llm.components.kv_cache import KVCache
from dialectic.llm.components.rms_norm import RMSNorm
from dialectic.llm.components.rope import (
    RopeScaling,
    apply_rope,
    create_rope_sine_cosine_tensors,
)


def attention(
    q: Float[Tensor, "B NH L DHead"],
    k: Float[Tensor, "B NKVH L DHead"],
    v: Float[Tensor, "B NKVH L DHead"],
    causal: bool = False,
    attention_mask: Bool[Tensor, "B L"] | None = None,
) -> Float[Tensor, "B NH L DHead"]:
    sdpa_mask = None
    if attention_mask is not None:
        # reshape to (B, 1, 1, L) for broadcasting
        sdpa_mask = attention_mask.view(
            attention_mask.shape[0], 1, 1, attention_mask.shape[1]
        )

    use_sdpa_causal = False

    seq_length = q.shape[-2]

    if causal:
        if seq_length > 1:
            # Training / Prefill case
            if sdpa_mask is not None:
                causal_mask = torch.ones(
                    seq_length, seq_length, device=q.device, dtype=torch.bool
                ).tril()
                sdpa_mask = sdpa_mask & causal_mask
            else:
                use_sdpa_causal = True
    return nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=sdpa_mask,  # SDPA handles causal masking automatically via is_causal=True
        dropout_p=0.0,
        is_causal=use_sdpa_causal,
        enable_gqa=True,
    )


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
        rms_norm_eps: float | None = None,
        max_position_embeddings: int = 8192,
        rope_scaling: RopeScaling | None = None,
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
        self.apply_rms_norm = apply_rms_norm  # ty: ignore[unresolved-attribute]

        if apply_rms_norm:
            self.q_norm = RMSNorm(self.head_d, rms_norm_eps)
            self.k_norm = RMSNorm(self.head_d, rms_norm_eps)

        # Precompute and cache RoPE sin/cos tensors
        if self.use_rope and rope_base_value is not None:
            sin, cos = create_rope_sine_cosine_tensors(
                head_d,
                base_value=rope_base_value,
                context_length=max_position_embeddings,
                rope_scaling=rope_scaling,
            )
            self.register_buffer("rope_sin", sin, persistent=False)
            self.register_buffer("rope_cos", cos, persistent=False)

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
            end_pos = position_offset + x.shape[1]

            # Slice from precomputed cached tensors
            sin = self.rope_sin[:, position_offset:end_pos]
            cos = self.rope_cos[:, position_offset:end_pos]

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
