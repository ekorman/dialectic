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
        apply_qk_rms_norm=False,
        rms_norm_eps=rms_norm_eps,
        rope_max_position_embeddings=rope_max_position_embeddings,
        rope_scaling=rope_scaling,
        upcast_attention=upcast_attention,
    )


def create_llama(
    d: int,
    vocab_size: int,
    n_decoder_layers: int,
    attn_head_d: int,
    attn_num_heads: int,
    attn_num_kv_heads: int,
    mlp_hidden_d: int,
    tie_weights: bool,
    rope_base_value: int = 500000,
    upcast_attention: bool = False,
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
        tie_weights=tie_weights,
        decoder_layer_factory=create_llama_decoder_layer,
        upcast_attention=upcast_attention,
    )


LLAMA_32_1B_INSTRUCT_WEIGHTS = Artifact(
    urls=[
        "https://public-storage.pols.ai/model-weights/llama-3.2/llama-3.2-1b-instruct/model.safetensors"
    ],
    filenames=["llama-3.2/llama-3.2-1b-instruct/model.safetensors"],
)

LLAMA_32_3B_INSTRUCT_WEIGHTS = Artifact(
    urls=[
        f"https://public-storage.pols.ai/model-weights/llama-3.2/llama-3.2-3b-instruct/model-0000{i}-of-00002.safetensors"
        for i in [1, 2]
    ],
    filenames=[
        f"llama-3.2/llama-3.2-3b-instruct/model-0000{i}-of-00002.safetensors"
        for i in [1, 2]
    ],
)

LLAMA_32_TOKENIZER = Artifact(
    urls=[
        "https://public-storage.pols.ai/model-weights/llama-3.2/llama3.2-tokenizer.json"
    ],
    filenames=["llama-3.2/llama3.2-tokenizer.json"],
)


def load_llama_32_1b_instruct(pretrained_weights: bool = False) -> BaseTransformer:
    net = create_llama(
        d=2048,
        vocab_size=128256,
        n_decoder_layers=16,
        attn_head_d=64,
        attn_num_heads=32,
        attn_num_kv_heads=8,
        mlp_hidden_d=8192,
        rope_base_value=500000,
        tie_weights=True,
        upcast_attention=True,
    )

    if pretrained_weights:
        sd = load_state_dict_from_artifact(
            LLAMA_32_1B_INSTRUCT_WEIGHTS, convert_keys=True, tied_weights=True
        )

        net.load_state_dict(sd)

    return net


def load_llama_32_3b_instruct(pretrained_weights: bool = False) -> BaseTransformer:
    net = create_llama(
        d=3072,
        vocab_size=128256,
        n_decoder_layers=28,
        attn_head_d=128,
        attn_num_heads=24,
        attn_num_kv_heads=8,
        mlp_hidden_d=8192,
        rope_base_value=500000,
        tie_weights=True,
    )

    if pretrained_weights:
        sd = load_state_dict_from_artifact(
            LLAMA_32_3B_INSTRUCT_WEIGHTS, convert_keys=True, tied_weights=True
        )

        net.load_state_dict(sd)

    return net
