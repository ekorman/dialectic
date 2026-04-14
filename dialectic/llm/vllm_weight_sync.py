"""Build a vLLM engine for GRPO training and keep its weights in sync with ``net``.

The training loop calls :func:`build_vllm_for_training` once before the loop
starts, and :func:`sync_weights_to_vllm` after every optimizer step. Both
helpers unwrap ``DistributedDataParallel`` internally so callers can pass
either the bare ``BaseTransformer`` or a DDP-wrapped one.

Weight updates go through vLLM's public ``collective_rpc`` API rather than
reaching into engine internals. vLLM has churned its ``LLMEngine`` layout
between v0 and v1 (the v1 architecture removed the top-level
``model_executor`` attribute, breaking the older path). ``collective_rpc``
is the documented entrypoint that survives both versions: it dispatches a
callable to every worker and the callable accesses ``model_runner.model``
locally, where it remains stable.

``enforce_eager=True`` on engine construction disables CUDA graphs so
captured graphs don't hold pointers to stale weights between updates. The
prefix cache is flushed after each sync so KV tensors cached from the old
policy are not reused.
"""

import os

# Run vLLM's v1 engine core in-process instead of in a subprocess. By default
# vLLM v1 isolates the engine core in its own process for inference-server
# use cases, which means anything passed to `collective_rpc` must round-trip
# through vLLM's IPC encoder. That encoder is built for sampling params and
# prompt strings, not model weights — it serializes torch.Tensors into a
# list representation that the worker side never reconstitutes, so
# `model.load_weights(...)` sees lists where it expects tensors. In-process
# mode bypasses serialization entirely: the trainer and engine share the
# same Python process and CUDA context, and our worker function is invoked
# directly with the actual tensor objects we passed in. This is the right
# topology for colocated trainer + sampler anyway — there's no benefit to a
# subprocess boundary when the same process owns both sides.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

# Belt and braces: even in in-process mode, vLLM occasionally exercises the
# IPC encoder for ancillary calls. Allowing the pickle fallback for the few
# spots where it's reached is safe in our setup (no untrusted RPC clients).
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

from typing import TYPE_CHECKING  # noqa: E402

from tokenizers import Tokenizer  # noqa: E402

from dialectic.distributed import unwrap_model  # noqa: E402
from dialectic.llm.base import BaseTransformer  # noqa: E402
from dialectic.llm.qwen_hf_export import _dialectic_to_hf_key  # noqa: E402
from dialectic.llm.vllm_loader import load_dialectic_qwen_as_vllm  # noqa: E402
from dialectic.log import log  # noqa: E402

if TYPE_CHECKING:
    from vllm import LLM


def build_vllm_for_training(
    net: BaseTransformer,
    tokenizer: Tokenizer,
    *,
    eos_token_id: int,
    pad_token_id: int,
    max_model_len: int,
    gpu_memory_utilization: float = 0.3,
    dtype: str = "bfloat16",
    seed: int | None = None,
) -> "LLM":
    """Construct a vLLM engine suitable for in-place weight reload during GRPO.

    Parameters
    ----------
    net
        Training model. May be wrapped in ``DistributedDataParallel`` —
        unwrapped internally before export.
    tokenizer
        ``tokenizers.Tokenizer`` for the model. Serialized into the exported
        HF directory so vLLM's internal tokenizer loads identically.
    eos_token_id, pad_token_id
        Written into the synthesized ``config.json`` so vLLM and HF downstream
        tools see consistent special-token ids.
    max_model_len
        Maximum prompt + completion length vLLM should plan KV cache for.
    gpu_memory_utilization
        Fraction of GPU memory vLLM may consume. Default ``0.3`` leaves ~70%
        of the card for the training model, its optimizer state, activations,
        and (optionally) a reference model — sized for 0.6B–1.7B training on a
        single 80+ GB GPU. Bump or drop as needed.
    dtype
        Compute dtype for vLLM. Defaults to bfloat16 to match typical training.
    seed
        Seed for vLLM's sampler RNG.

    Returns
    -------
    vllm.LLM
        Engine constructed with ``enforce_eager=True`` and
        ``tensor_parallel_size=1``. Weight updates must go through
        :func:`sync_weights_to_vllm`.
    """
    bare = unwrap_model(net)
    return load_dialectic_qwen_as_vllm(
        bare,
        tokenizer=tokenizer,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
        seed=seed,
        # Disable CUDA graph capture so graphs don't lock in stale weight
        # pointers between sync calls.
        enforce_eager=True,
        # Force single-process, single-GPU execution. Under torchrun DDP each
        # rank constructs its own engine pinned to its own LOCAL_RANK GPU;
        # cross-rank tensor parallelism would conflict with DDP's process group.
        tensor_parallel_size=1,
    )


def _vllm_worker_load_weights(worker_self, weights: list) -> None:
    """Worker-side: push ``weights`` into the inner model.

    Defined at module level (not as a closure) so vLLM's ``collective_rpc``
    can pickle it for multiprocess executors. Called by
    :func:`sync_weights_to_vllm` via ``llm.collective_rpc(self, args=(...))``.
    """
    inner_model = worker_self.model_runner.model
    inner_model.load_weights(iter(weights))


def sync_weights_to_vllm(llm: "LLM", net: BaseTransformer) -> None:
    """Push current ``net`` weights into the live vLLM engine in place.

    Uses ``LLM.collective_rpc`` to dispatch the weight load to every vLLM
    worker, then invalidates the prefix cache so KV cached from the old
    policy is not reused.

    Parameters
    ----------
    llm
        Engine created by :func:`build_vllm_for_training`.
    net
        Current training model. DDP wrapper is unwrapped before reading
        ``state_dict``.

    Notes
    -----
    ``collective_rpc`` is the public API recommended for cross-version weight
    updates. The previous direct-attribute path
    ``llm.llm_engine.model_executor.driver_worker.model_runner.model`` was
    removed when vLLM v1 restructured ``LLMEngine``. ``model_runner.model``
    on the worker is still the right place to land — what changed is how
    you reach it.

    Prefix cache invalidation tries the public ``LLM.reset_prefix_cache``
    first and falls back to the legacy ``llm_engine.reset_prefix_cache`` for
    older versions. If neither exists, a warning is logged and stale KV may
    persist briefly until the cache turns over naturally.
    """
    bare = unwrap_model(net)
    weights = [
        (_dialectic_to_hf_key(k), v.detach())
        for k, v in bare.state_dict().items()
        # `lm_head.weight` is tied to `embed_tokens.weight`; vLLM re-ties via
        # the `tie_word_embeddings` flag the offline exporter writes into
        # config.json, so don't push it twice.
        if k != "lm_head.weight"
    ]

    llm.collective_rpc(_vllm_worker_load_weights, args=(weights,))

    for try_reset in (
        lambda: llm.reset_prefix_cache(),
        lambda: llm.llm_engine.reset_prefix_cache(),
    ):
        try:
            try_reset()
            return
        except AttributeError:
            continue
    log.warning(
        "could not invalidate vLLM prefix cache after weight sync; "
        "stale KV from the previous policy may be reused on the next generate call"
    )
