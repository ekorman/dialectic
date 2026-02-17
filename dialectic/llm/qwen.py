"""Type conventions:

- 'B': batch size
- 'L': sequence length
- 'D': vector dimension
- 'NH': number of (query) attention heads
- 'NKVH': number of key/value attention heads
- 'DHead': dimension of each head
- 'VC': vocab size
"""

from safetensors.torch import load_file

from dialectic.artifacts import Artifact, get_artifact, map_hf_key_to_dialectic
from dialectic.llm.base import BaseTransformer
from dialectic.llm.components import DecoderLayer


def create_qwen_decoder_layer(
    d: int,
    *,
    attn_head_d: int,
    attn_num_heads: int,
    attn_num_kv_heads: int,
    mlp_hidden_d: int,
    rope_base_value: float | None = None,
    rms_norm_eps: float = 1e-6,
    rope_max_position_embeddings: int = 32768,
    upcast_attention: bool = False,
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
        rms_norm_eps=rms_norm_eps,
        rope_max_position_embeddings=rope_max_position_embeddings,
    )


def create_qwen(
    d: int,
    vocab_size: int,
    n_decoder_layers: int,
    attn_head_d: int,
    attn_num_heads: int,
    attn_num_kv_heads: int,
    mlp_hidden_d: int,
    tie_weights: bool,
    rope_base_value: int = 1000000,
):
    return BaseTransformer(
        d=d,
        vocab_size=vocab_size,
        n_decoder_layers=n_decoder_layers,
        attn_head_d=attn_head_d,
        attn_num_heads=attn_num_heads,
        attn_num_kv_heads=attn_num_kv_heads,
        mlp_hidden_d=mlp_hidden_d,
        rms_norm_eps=1e-6,
        rope_base_value=rope_base_value,
        decoder_layer_factory=create_qwen_decoder_layer,
        tie_weights=tie_weights,
    )


QWEN3_06B_WEIGHTS = Artifact(
    url="https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/model.safetensors?download=true",
    filename="qwen3-0.6b/model.safetensors",
)


def load_qwen_06b(pretrained_weights: bool = False) -> BaseTransformer:
    net = create_qwen(
        d=1024,
        vocab_size=151936,
        n_decoder_layers=28,
        attn_head_d=128,
        attn_num_heads=16,
        attn_num_kv_heads=8,
        mlp_hidden_d=3072,
        rope_base_value=1000000,
        tie_weights=True,
    )
    if pretrained_weights:
        sd = load_file(get_artifact(artifact=QWEN3_06B_WEIGHTS))
        sd = {map_hf_key_to_dialectic(k): v for k, v in sd.items()}

        if not (sd["lm_head.weight"] == sd["embed_tokens.weight"]).all():
            raise ValueError(
                "Expected `lm_head.weight` and `embed_tokens.weight` to be identical."
            )
        net.load_state_dict(sd)

    return net
