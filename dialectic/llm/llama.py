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
    tie_weights: bool,
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
        tie_weights=tie_weights,
        decoder_layer_factory=create_llama_decoder_layer,
    )


LLAMA_32_1B_INSTRUCT_WEIGHTS = Artifact(
    url="https://public-storage.pols.ai/model-weights/llama-3.2-1b-instruct/model.safetensors",
    filename="llama-3.2-1b-instruct/model.safetensors",
)

LLAMA_32_1B_TOKENIZER = Artifact(
    url="https://public-storage.pols.ai/model-weights/llama-3.2-1b-instruct/llama3.2-tokenizer.json",
    filename="llama-3.2-1b-instruct/llama3.2-tokenizer.json",
)


def load_llama_1b(pretrained_weights: bool = False) -> BaseTransformer:
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
    )

    if pretrained_weights:
        sd = load_file(get_artifact(artifact=LLAMA_32_1B_INSTRUCT_WEIGHTS))
        sd = {map_hf_key_to_dialectic(k): v for k, v in sd.items()}

        if "lm_head.weight" not in sd:
            sd["lm_head.weight"] = sd["embed_tokens.weight"]
        elif not (sd["lm_head.weight"] == sd["embed_tokens.weight"]).all():
            raise ValueError(
                "Expected `lm_head.weight` and `embed_tokens.weight` to be identical."
            )

        net.load_state_dict(sd)

    return net
