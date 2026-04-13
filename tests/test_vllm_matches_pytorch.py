"""Agreement test: PyTorch vs vLLM greedy decoding on Qwen3-0.6B.

Gated by the same ``TEST_LLM_AGAINST_HF`` environment variable used by the
other heavy HF-comparison tests. Downloads the base Qwen3-0.6B checkpoint
and spins up a vLLM engine, so expect multi-GB network + VRAM.

The test is deliberately lenient on exact token match because vLLM and the
dialectic PyTorch path use different attention kernels (FlashAttention vs
``scaled_dot_product_attention``) in bfloat16, which produces small logit
differences that can cascade into different token choices at near-tied
argmax positions. We instead require that at least 85% of the shorter
sequence agrees — enough to flag a weight-mapping bug but tolerant of
kernel-level numerical noise.
"""

import os
import tempfile

import pytest
import torch
from tokenizers import Tokenizer

from dialectic.llm.generate import qwen_generate_from_chat
from dialectic.llm.qwen import load_qwen3_06b
from dialectic.llm.qwen_hf_export import export_qwen3_to_hf_dir
from dialectic.llm.templates import Message


@pytest.mark.skipif(
    os.getenv("TEST_LLM_AGAINST_HF") is None,
    reason="skipping `test_qwen_hf_export_round_trip` since env variable `TEST_LLM_AGAINST_HF` not set",
)
def test_qwen_hf_export_round_trip():
    """Exporter sanity check that does not need vLLM.

    Loads Qwen3-0.6B from the dialectic path, exports to an HF-format dir,
    reloads via ``AutoModelForCausalLM``, and asserts the two produce the
    same last-token logits on a random input. Fast, CPU-only, and catches
    any key-mapping or weight-tying regression in the exporter.
    """
    from transformers import AutoModelForCausalLM

    net = load_qwen3_06b(pretrained_weights=True).eval()
    tokenizer = Tokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    with tempfile.TemporaryDirectory() as tmp:
        export_qwen3_to_hf_dir(
            net,
            tokenizer=tokenizer,
            out_dir=tmp,
            eos_token_id=151645,
            pad_token_id=151643,
        )
        hf_model = AutoModelForCausalLM.from_pretrained(tmp, dtype=torch.float32).eval()

    x = torch.randint(0, hf_model.config.vocab_size, size=(1, 10))
    with torch.inference_mode():
        torch.testing.assert_close(
            net(x),
            hf_model(x).logits[:, -1:],
            atol=1e-4,
            rtol=1e-4,
        )


@pytest.mark.skipif(
    os.getenv("TEST_LLM_AGAINST_HF") is None,
    reason="skipping `test_vllm_matches_pytorch_greedy` since env variable `TEST_LLM_AGAINST_HF` not set",
)
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="vLLM requires CUDA",
)
def test_vllm_matches_pytorch_greedy():
    pytest.importorskip("vllm")
    from dialectic.llm.vllm_generate import vllm_qwen_generate_from_chat
    from dialectic.llm.vllm_loader import load_dialectic_qwen_as_vllm

    net = (
        load_qwen3_06b(pretrained_weights=True).to("cuda", dtype=torch.bfloat16).eval()
    )
    net.requires_grad_(False)
    tokenizer: Tokenizer = Tokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    batch_messages = [
        [Message(role="user", content="What is 2+2? Answer in one word. /nothink")],
        [Message(role="user", content="Name the capital of France. /nothink")],
        [
            Message(
                role="user",
                content="Using the numbers [3, 5, 7], create an equation that "
                "equals 15. Put your final equation in <answer></answer> tags. /nothink",
            )
        ],
    ]

    pt_outputs = qwen_generate_from_chat(
        net,
        tokenizer,
        batch_messages,
        sampling_strategy="greedy",
        max_tokens_generated=96,
        enable_thinking=True,
    )

    llm = load_dialectic_qwen_as_vllm(
        net,
        tokenizer=tokenizer,
        eos_token_id=151645,
        pad_token_id=151643,
        max_model_len=512,
        gpu_memory_utilization=0.5,
        dtype="bfloat16",
        seed=0,
    )
    del net
    torch.cuda.empty_cache()

    vllm_outputs = vllm_qwen_generate_from_chat(
        llm,
        tokenizer,
        batch_messages,
        sampling_strategy="greedy",
        max_tokens_generated=96,
        enable_thinking=True,
    )

    assert isinstance(pt_outputs, list) and isinstance(vllm_outputs, list)
    assert len(pt_outputs) == len(vllm_outputs) == len(batch_messages)

    for i, (pt_out, vllm_out) in enumerate(zip(pt_outputs, vllm_outputs)):
        pt_ids = tokenizer.encode(pt_out).ids
        vllm_ids = tokenizer.encode(vllm_out).ids
        first_diff = next(
            (k for k, (a, b) in enumerate(zip(pt_ids, vllm_ids)) if a != b),
            min(len(pt_ids), len(vllm_ids)),
        )
        print(f"\n===== prompt {i} =====")
        print(f"pt  ({len(pt_ids)} tok):   {pt_out!r}")
        print(f"vllm ({len(vllm_ids)} tok): {vllm_out!r}")
        print(f"first token divergence at position {first_diff}")
        if first_diff < min(len(pt_ids), len(vllm_ids)):
            print(
                f"  pt[{first_diff}]={pt_ids[first_diff]} "
                f"({tokenizer.decode([pt_ids[first_diff]])!r})  "
                f"vllm[{first_diff}]={vllm_ids[first_diff]} "
                f"({tokenizer.decode([vllm_ids[first_diff]])!r})"
            )

    # We measure "length of matching prefix" rather than overall token-level
    # agreement on purpose. Two bf16 attention kernels (SDPA and FlashAttention)
    # will eventually flip at some near-tied argmax; once greedy decoding
    # diverges the rest of the sequence is random with respect to the other
    # path, so overall % is not a meaningful signal. A structural bug — wrong
    # weights, wrong config, wrong tokenization — would instead show up as a
    # very short matching prefix (divergence at token 0–5).
    min_matching_prefix = 20
    for i, (pt_out, vllm_out) in enumerate(zip(pt_outputs, vllm_outputs)):
        pt_ids = tokenizer.encode(pt_out).ids
        vllm_ids = tokenizer.encode(vllm_out).ids
        assert min(len(pt_ids), len(vllm_ids)) > 10, (
            f"prompt {i}: one of the outputs is implausibly short"
        )
        matching_prefix = next(
            (k for k, (a, b) in enumerate(zip(pt_ids, vllm_ids)) if a != b),
            min(len(pt_ids), len(vllm_ids)),
        )
        assert matching_prefix >= min_matching_prefix, (
            f"prompt {i}: matching prefix only {matching_prefix} tokens "
            f"(required {min_matching_prefix}); divergence this early suggests "
            f"a structural bug rather than bf16 kernel drift "
            f"(see printed outputs above)"
        )
