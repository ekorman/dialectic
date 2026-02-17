import torch
import torch.nn as nn

from dialectic.llm.components.attention import MHSA
from dialectic.llm.components.gated_mlp import GatedMLP
from dialectic.llm.components.kv_cache import KVCache
from dialectic.llm.components.rms_norm import RMSNorm
from dialectic.llm.components.rope import RopeScaling


class DecoderLayer(nn.Module):
    """
    RMSNorm -> Residual attention -> RMSNorm -> Residual MLP
    """

    def __init__(
        self,
        d: int,
        *,
        attn_head_d: int,
        attn_num_heads: int,
        attn_num_kv_heads: int,
        mlp_hidden_d: int,
        causal: bool,
        apply_qk_rms_norm: bool,
        rms_norm_eps: float,
        rope_max_position_embeddings: int,
        rope_base_value: float | None = None,
        rope_scaling: RopeScaling | None = None,
        upcast_attention: bool = False,
    ):
        super().__init__()
        self.input_layernorm = RMSNorm(d, rms_norm_eps)
        self.self_attn = MHSA(
            d=d,
            head_d=attn_head_d,
            num_heads=attn_num_heads,
            num_kv_heads=attn_num_kv_heads,
            causal=causal,
            apply_rms_norm=apply_qk_rms_norm,
            rope_base_value=rope_base_value,
            rms_norm_eps=rms_norm_eps,
            max_position_embeddings=rope_max_position_embeddings,
            rope_scaling=rope_scaling,
            upcast_attention=upcast_attention,
        )
        self.post_attention_layernorm = RMSNorm(d, rms_norm_eps)
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
