"""Load a finetuned dialectic Qwen3 checkpoint into a vLLM engine.

Combines :func:`export_qwen3_to_hf_dir` with a ``vllm.LLM`` constructor so
callers can go from an in-memory ``BaseTransformer`` to a ready-to-serve
vLLM engine in a single call. No network access: the HF-format directory
is synthesized locally from the net plus the passed-in tokenizer.
"""

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tokenizers import Tokenizer

from dialectic.llm.base import BaseTransformer
from dialectic.llm.qwen_hf_export import export_qwen3_to_hf_dir

if TYPE_CHECKING:
    from vllm import LLM


def load_dialectic_qwen_as_vllm(
    net: BaseTransformer,
    tokenizer: Tokenizer,
    *,
    eos_token_id: int,
    pad_token_id: int,
    max_model_len: int,
    gpu_memory_utilization: float = 0.90,
    dtype: str = "bfloat16",
    seed: int | None = None,
    export_dir: str | Path | None = None,
    **llm_kwargs: Any,
) -> "LLM":
    """Export ``net`` to HF format and hand it to ``vllm.LLM``.

    Parameters
    ----------
    net
        Loaded dialectic Qwen3 ``BaseTransformer``. Read once for its state
        dict; the caller is free to delete it after this function returns.
    tokenizer
        The already-loaded ``tokenizers.Tokenizer`` (typically from
        ``model_info.load_tokenizer()``). Saved to the export directory so
        vLLM's internal tokenizer path finds it.
    eos_token_id, pad_token_id
        Special-token ids for the generated ``config.json``.
    max_model_len
        Maximum prompt + completion length vLLM should plan KV cache for.
    gpu_memory_utilization
        Fraction of GPU memory vLLM may consume.
    dtype
        vLLM compute dtype. Defaults to ``"bfloat16"`` to match training.
    seed
        Seed for vLLM's sampler RNG. ``None`` leaves vLLM's default.
    export_dir
        Destination for the HF-format checkpoint. If ``None``, a
        ``TemporaryDirectory`` is created and its lifetime is tied to the
        returned ``LLM`` via an attribute so cleanup happens automatically
        when the engine is garbage-collected.
    **llm_kwargs
        Forwarded verbatim to ``vllm.LLM``.

    Returns
    -------
    vllm.LLM
        An engine with the exported weights already loaded.
    """
    from vllm import LLM

    tmp_ctx: tempfile.TemporaryDirectory | None
    if export_dir is None:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="dialectic_vllm_export_")
        target = Path(tmp_ctx.name)
    else:
        tmp_ctx = None
        target = Path(export_dir)

    export_qwen3_to_hf_dir(
        net,
        tokenizer=tokenizer,
        out_dir=target,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )

    llm_kwargs.setdefault("dtype", dtype)
    llm_kwargs.setdefault("max_model_len", max_model_len)
    llm_kwargs.setdefault("gpu_memory_utilization", gpu_memory_utilization)
    if seed is not None:
        llm_kwargs.setdefault("seed", seed)

    llm = LLM(model=str(target), **llm_kwargs)

    if tmp_ctx is not None:
        llm._dialectic_export_tmpdir = tmp_ctx  # keep temp dir alive
    return llm
