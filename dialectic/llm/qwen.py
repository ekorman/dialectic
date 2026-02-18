"""Type conventions:

- 'B': batch size
- 'L': sequence length
- 'D': vector dimension
- 'NH': number of (query) attention heads
- 'NKVH': number of key/value attention heads
- 'DHead': dimension of each head
- 'VC': vocab size
"""

from dialectic.artifacts import Artifact, load_state_dict_from_artifact
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
    urls=[
        "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/model.safetensors?download=true"
    ],
    filenames=["qwen3-0.6b/model.safetensors"],
)

QWEN3_17B_WEIGHTS = Artifact(
    urls=[
        f"https://huggingface.co/Qwen/Qwen3-1.7B/resolve/main/model-0000{i}-of-00002.safetensors?download=true"
        for i in [1, 2]
    ],
    filenames=[f"qwen3-1.7b/model-0000{i}-of-00002.safetensors" for i in [1, 2]],
)


def load_qwen3_06b(pretrained_weights: bool = False) -> BaseTransformer:
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
        sd = load_state_dict_from_artifact(
            QWEN3_06B_WEIGHTS, convert_keys=True, tied_weights=True
        )
        net.load_state_dict(sd)

    return net


def load_qwen3_17b(pretrained_weights: bool = False) -> BaseTransformer:
    net = create_qwen(
        d=2048,
        vocab_size=151936,
        n_decoder_layers=28,
        attn_head_d=128,
        attn_num_heads=16,
        attn_num_kv_heads=8,
        mlp_hidden_d=6144,
        rope_base_value=1000000,
        tie_weights=True,
    )
    if pretrained_weights:
        sd = load_state_dict_from_artifact(
            QWEN3_17B_WEIGHTS, convert_keys=True, tied_weights=True
        )
        net.load_state_dict(sd)

    return net
