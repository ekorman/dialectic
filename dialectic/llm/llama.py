"""Type conventions:

- 'B': batch size
- 'L': sequence length
- 'D': vector dimension
- 'NH': number of (query) attention heads
- 'NKVH': number of key/value attention heads
- 'DHead': dimension of each head
- 'VC': vocab size
"""

import torch
import torch.nn as nn

from dialectic.llm.components import DecoderLayer, KVCache, RMSNorm
from dialectic.llm.components.rope import RopeScaling


def create_llama_decoder_layer(
    d: int,
    *,
    attn_head_d: int,
    attn_num_heads: int,
    attn_num_kv_heads: int,
    mlp_hidden_d: int,
    rms_norm_eps: float,
    rope_base_value: float = 1e-5,
    rope_max_position_embeddings: int = 8192,
    rope_scaling: RopeScaling = RopeScaling(
        factor=32, high_freq_factor=4, low_freq_factor=1
    ),
):
    return DecoderLayer(
        d=d,
        attn_head_d=attn_head_d,
        attn_num_heads=attn_num_heads,
        attn_num_kv_heads=attn_num_kv_heads,
        mlp_hidden_d=mlp_hidden_d,
        rope_base_value=rope_base_value,
        causal=True,
        apply_qk_rms_norm=False,
        rms_norm_eps=rms_norm_eps,
        rope_max_position_embeddings=rope_max_position_embeddings,
        rope_scaling=rope_scaling,
    )


class Llama(nn.Module):
    def __init__(
        self,
        d: int,
        vocab_size: int,
        n_decoder_layers: int,
        attn_head_d: int,
        attn_num_heads: int,
        attn_num_kv_heads: int,
        mlp_hidden_d: int,
        rms_norm_eps: float = 1e-5,
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
                create_llama_decoder_layer(
                    d=d,
                    attn_head_d=attn_head_d,
                    attn_num_heads=attn_num_heads,
                    attn_num_kv_heads=attn_num_kv_heads,
                    mlp_hidden_d=mlp_hidden_d,
                    rope_base_value=rope_base_value,
                    rms_norm_eps=rms_norm_eps,
                )
                for _ in range(n_decoder_layers)
            ]
        )
        self.norm = RMSNorm(d, rms_norm_eps)
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


def load_llama_1b() -> Llama:
    return Llama(
        d=2048,
        vocab_size=128256,
        n_decoder_layers=16,
        attn_head_d=64,  # 128 for qwen. 64?? d / num_heads
        attn_num_heads=32,
        attn_num_kv_heads=8,
        mlp_hidden_d=8192,
        rope_base_value=500000,
    )
