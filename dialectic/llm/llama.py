"""Type conventions:

- 'B': batch size
- 'L': sequence length
- 'D': vector dimension
- 'NH': number of (query) attention heads
- 'NKVH': number of key/value attention heads
- 'DHead': dimension of each head
- 'VC': vocab size
"""

from dialectic.llm.base import BaseTransformer
from dialectic.llm.components import DecoderLayer
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


def create_llama(
    d: int,
    vocab_size: int,
    n_decoder_layers: int,
    attn_head_d: int,
    attn_num_heads: int,
    attn_num_kv_heads: int,
    mlp_hidden_d: int,
    rope_base_value: int = 500000,
):
    return BaseTransformer(
        d=d,
        vocab_size=vocab_size,
        n_decoder_layers=n_decoder_layers,
        attn_head_d=attn_head_d,
        attn_num_heads=attn_num_heads,
        attn_num_kv_heads=attn_num_kv_heads,
        mlp_hidden_d=mlp_hidden_d,
        rms_norm_eps=1e-5,
        rope_base_value=rope_base_value,
        decoder_layer_factory=create_llama_decoder_layer,
    )


def load_llama_1b() -> BaseTransformer:
    return create_llama(
        d=2048,
        vocab_size=128256,
        n_decoder_layers=16,
        attn_head_d=64,
        attn_num_heads=32,
        attn_num_kv_heads=8,
        mlp_hidden_d=8192,
        rope_base_value=500000,
    )
