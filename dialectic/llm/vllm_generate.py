"""vLLM-backed counterpart to :func:`dialectic.llm.generate.qwen_generate_from_chat`.

Mirrors that function's signature as closely as possible so the two backends
can be plugged into the same test harness.
"""

from typing import TYPE_CHECKING, Literal

from tokenizers import Tokenizer

from dialectic.llm.templates import Message, get_qwen_input_text_from_messages

if TYPE_CHECKING:
    from vllm import LLM


def vllm_qwen_generate_from_chat(
    llm: "LLM",
    tokenizer: Tokenizer,
    batch_messages: list[list[Message]],
    eos_token: str = "<|im_end|>",
    pad_token: str = "<|endoftext|>",
    sampling_strategy: Literal["greedy", "sample"] = "sample",
    max_tokens_generated: int = 1000,
    temperature: float = 1.0,
    enable_thinking: bool = True,
    n: int = 1,
    seed: int | None = None,
) -> list[str] | list[list[str]]:
    """vLLM analogue of :func:`qwen_generate_from_chat`.

    The signature matches the PyTorch version with two differences: the first
    argument is a ``vllm.LLM`` engine instead of a ``BaseTransformer``, and a
    ``device`` kwarg is absent (vLLM manages placement internally).

    Parameters
    ----------
    llm
        Engine built via :func:`dialectic.llm.vllm_loader.load_dialectic_qwen_as_vllm`
        (or constructed directly from any HF-format Qwen3 directory).
    tokenizer
        ``tokenizers.Tokenizer`` used for detokenizing outputs. Passed in —
        instead of using vLLM's internal tokenizer — so the returned strings
        match ``qwen_generate_from_chat`` byte-for-byte and a test can compare
        them directly.
    batch_messages
        Same shape as the PyTorch version: one conversation per element.
    eos_token, pad_token
        Special-token literals. ``pad_token`` is accepted for signature parity
        with the PyTorch version; vLLM manages padding internally.
    sampling_strategy
        ``"greedy"`` maps to ``temperature=0``; ``"sample"`` uses the passed
        ``temperature``.
    max_tokens_generated
        Per-sample decode budget.
    temperature
        Sampling temperature when ``sampling_strategy == "sample"``.
    enable_thinking
        Passed through to the Qwen chat templater.
    n
        Number of samples per prompt. When ``1`` (the default, matching the
        PyTorch version), the return type is ``list[str]``. When ``> 1``,
        ``list[list[str]]`` is returned with the per-prompt sample lists.
    seed
        vLLM sampler seed. ``None`` leaves the engine default in place.

    Returns
    -------
    list[str] | list[list[str]]
        Detokenized ``prompt + completion`` strings (special tokens skipped),
        shape determined by ``n`` as described above.
    """
    from vllm import SamplingParams

    del pad_token  # accepted for signature parity; vLLM manages padding internally

    eos_token_id = tokenizer.token_to_id(eos_token)
    if eos_token_id is None:
        raise ValueError(f"eos_token {eos_token!r} not in tokenizer vocab")

    prompts = [
        get_qwen_input_text_from_messages(
            message, add_generation_prompt=True, enable_thinking=enable_thinking
        )
        for message in batch_messages
    ]

    sampling_params = SamplingParams(
        n=n,
        temperature=0.0 if sampling_strategy == "greedy" else temperature,
        max_tokens=max_tokens_generated,
        stop_token_ids=[eos_token_id],
        seed=seed,
    )

    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

    results: list[list[str]] = []
    for out in outputs:
        prompt_ids = list(out.prompt_token_ids)
        per_prompt: list[str] = []
        for sample in out.outputs:
            gen_ids = [t for t in sample.token_ids if t != eos_token_id]
            per_prompt.append(tokenizer.decode(prompt_ids + gen_ids))
        results.append(per_prompt)

    if n == 1:
        return [samples[0] for samples in results]
    return results
