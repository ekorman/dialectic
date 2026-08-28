"""Export a dialectic Qwen3 ``BaseTransformer`` to a HuggingFace-format directory.

The dialectic Qwen3 module tree mirrors HuggingFace Qwen3 exactly modulo a
``model.`` prefix, so exporting weights is a simple key rename. ``config.json``
is synthesized from attributes already on the net, and the tokenizer is
written by asking the user-supplied ``tokenizers.Tokenizer`` to save itself.
Nothing is fetched from the network.
"""

import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer

from dialectic.llm.base import BaseTransformer


def _dialectic_to_hf_key(k: str) -> str:
    return f"model.{k}"


def _qwen3_config_from_net(
    net: BaseTransformer,
    *,
    eos_token_id: int,
    pad_token_id: int,
) -> dict:
    """Build a HF Qwen3 ``config.json`` dict from a dialectic ``BaseTransformer``.

    All values are read off the net (or its submodules) so there is a single
    source of truth. ``tie_word_embeddings`` is detected by storage-identity
    of the ``embed_tokens`` / ``lm_head`` weight tensors.
    """
    first_layer = net.layers[0]
    attn = first_layer.self_attn
    mlp = first_layer.mlp
    norm = first_layer.input_layernorm

    tie_word_embeddings = (
        net.lm_head.weight.data_ptr() == net.embed_tokens.weight.data_ptr()
    )

    return {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": net.d,
        "num_hidden_layers": len(net.layers),
        "num_attention_heads": net.attn_num_heads,
        "num_key_value_heads": net.attn_num_kv_heads,
        "head_dim": net.attn_head_d,
        "intermediate_size": mlp.gate_proj.out_features,
        "vocab_size": net.vocab_size,
        "max_position_embeddings": attn.max_position_embeddings,
        "rope_theta": float(attn.rope_base_value),
        "rope_scaling": None,
        "rms_norm_eps": norm.eps,
        "hidden_act": "silu",
        "tie_word_embeddings": tie_word_embeddings,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "initializer_range": 0.02,
        "use_cache": True,
        "use_sliding_window": False,
        "sliding_window": None,
        "max_window_layers": len(net.layers),
        "torch_dtype": "bfloat16",
        "bos_token_id": None,
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
    }


def export_qwen3_to_hf_dir(
    net: BaseTransformer,
    tokenizer: Tokenizer,
    out_dir: str | Path,
    *,
    eos_token_id: int,
    pad_token_id: int,
) -> Path:
    """Serialize ``net`` into ``out_dir`` in HuggingFace Qwen3 format.

    Parameters
    ----------
    net
        Loaded (possibly finetuned) dialectic ``BaseTransformer`` whose
        architecture matches HuggingFace Qwen3.
    tokenizer
        The already-loaded ``tokenizers.Tokenizer`` (e.g. from
        ``model_info.load_tokenizer()``). Written to ``tokenizer.json`` via
        ``Tokenizer.save``. The tokenizer object itself is the single source
        of truth — no HF repo is consulted.
    out_dir
        Target directory. Created if missing. Overwritten in place.
    eos_token_id, pad_token_id
        Model-specific special-token ids. Written into ``config.json`` so
        downstream loaders have the same view as the dialectic registry.

    Returns
    -------
    Path
        ``out_dir`` as a ``Path``.

    Notes
    -----
    ``lm_head.weight`` is dropped from the saved state dict because Qwen3
    uses tied embeddings. The consumer re-ties from
    ``model.embed_tokens.weight`` via ``tie_word_embeddings`` in config.json.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    sd: dict[str, torch.Tensor] = {
        _dialectic_to_hf_key(k): v.detach().to("cpu").contiguous()
        for k, v in net.state_dict().items()
        if k != "lm_head.weight"
    }
    save_file(sd, str(out_path / "model.safetensors"))

    config = _qwen3_config_from_net(
        net, eos_token_id=eos_token_id, pad_token_id=pad_token_id
    )
    (out_path / "config.json").write_text(json.dumps(config, indent=2))

    # Round-trip through a fresh Tokenizer so any runtime state the caller
    # set on the original (e.g. `enable_padding` from `generate_from_text`)
    # does not leak into the serialized file. vLLM reloads via HF and would
    # otherwise inherit the mutation, producing subtly different tokenization
    # than the pure-PyTorch path it's supposed to match.
    clean_tokenizer = Tokenizer.from_str(tokenizer.to_str())
    clean_tokenizer.no_padding()
    clean_tokenizer.no_truncation()
    clean_tokenizer.save(str(out_path / "tokenizer.json"))

    return out_path
