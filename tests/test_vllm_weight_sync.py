"""Smoke test: ``sync_weights_to_vllm`` actually pushes new weights.

Gated on ``TEST_LLM_AGAINST_HF`` since it needs a real Qwen3-0.6B checkpoint
and a live vLLM engine. The test is deliberately a behavioral check rather
than a byte-level weight comparison: build an engine, sample greedily,
perturb ``net.embed_tokens.weight`` in place, call ``sync_weights_to_vllm``,
sample again with the same seed, and assert the sampled tokens changed.

This is the minimum evidence that the sync function (a) can reach vLLM's
inner model, (b) successfully writes to it, and (c) invalidates the prefix
cache so the new weights actually take effect on the next generate call.
"""

import os

import pytest
import torch
from tokenizers import Tokenizer

from dialectic.llm.qwen import load_qwen3_06b


@pytest.mark.skipif(
    os.getenv("TEST_LLM_AGAINST_HF") is None,
    reason="skipping `test_sync_weights_to_vllm_changes_output` since env variable `TEST_LLM_AGAINST_HF` not set",
)
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="vLLM requires CUDA",
)
def test_sync_weights_to_vllm_changes_output():
    pytest.importorskip("vllm")
    from vllm import SamplingParams

    from dialectic.llm.vllm_weight_sync import (
        build_vllm_for_training,
        sync_weights_to_vllm,
    )

    net = (
        load_qwen3_06b(pretrained_weights=True).to("cuda", dtype=torch.bfloat16).eval()
    )
    net.requires_grad_(False)
    tokenizer = Tokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    llm = build_vllm_for_training(
        net,
        tokenizer=tokenizer,
        eos_token_id=151645,
        pad_token_id=151643,
        max_model_len=256,
        gpu_memory_utilization=0.5,
        dtype="bfloat16",
        seed=0,
    )

    prompt = "The capital of France is"
    sp = SamplingParams(n=1, temperature=0.0, max_tokens=16, seed=0)

    out_before = llm.generate([prompt], sp, use_tqdm=False)
    tokens_before = tuple(out_before[0].outputs[0].token_ids)

    # Large in-place perturbation of the embedding table is a cheap way to
    # guarantee that the sampled tokens will differ — the model's input
    # representations are now garbage for every token id, so greedy decoding
    # will take a completely different path.
    with torch.no_grad():
        net.embed_tokens.weight.add_(torch.randn_like(net.embed_tokens.weight) * 0.5)

    sync_weights_to_vllm(llm, net)

    out_after = llm.generate([prompt], sp, use_tqdm=False)
    tokens_after = tuple(out_after[0].outputs[0].token_ids)

    assert tokens_before != tokens_after, (
        "sync_weights_to_vllm did not change generated tokens after a large "
        "perturbation of embed_tokens.weight — the sync path is not reaching "
        "vLLM's inner model or the prefix cache is not being invalidated.\n"
        f"before: {tokens_before}\n"
        f"after:  {tokens_after}"
    )
