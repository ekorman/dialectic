from dataclasses import dataclass

import torch
from tokenizers import Tokenizer

from dialectic.llm.base import BaseTransformer
from dialectic.llm.generate import generate_hard_tokens
from dialectic.log import log
from dialectic.rl.extractors import extract_from_answer_tags
from dialectic.rl.inverse_cot_data import PreTokenizedPrompt


def expressions_match(pred: str, gt: str) -> bool:
    """Check if two arithmetic expressions evaluate to the same value."""
    try:
        gt_val = eval(gt, {"__builtins__": {}}, {})
        pred_val = eval(pred, {"__builtins__": {}}, {})
        return abs(pred_val - gt_val) < 1e-6
    except Exception:
        return False


def gsm8k_match(pred: str | None, gt: str) -> bool:
    """Check if a GSM8K-extracted numeric string matches the gold answer.

    Strips thousand-separator commas before float-casting so ``"1,000"`` and
    ``"1000"`` compare equal. Returns ``False`` for unparseable strings or
    ``None`` predictions rather than raising.
    """
    if pred is None:
        return False
    try:
        predicted = float(pred.replace(",", "").strip())
        target = float(gt)
    except (ValueError, TypeError):
        return False
    return abs(predicted - target) < 1e-6


HARD_PROMPT_THRESHOLD = 0.25


@dataclass
class FCRResult:
    fcr: float
    fcr_total: int
    fcr_correct: int
    fcr_all_incorrect: float
    fcr_all_incorrect_total: int
    fcr_all_incorrect_correct: int
    p_baseline: float
    fcr_hard: float
    fcr_hard_total: int


@dataclass
class EvalFcrResult:
    """Multi-sample FCR evaluation result (vLLM-driven, launcher-side).

    All ``*_pass_rate_at_n`` fields are means of per-sample ``is_correct``
    over the ``n_samples`` q-CoTs per prompt; all ``*_pass_at_n`` fields are
    means over prompts of ``any(is_correct[p, :])``. The ``_on_*`` suffixes
    filter the freshly-sampled p baseline to the same bucket the matching
    ``fcr_*`` metric uses — these matter when eval-time ``n_samples`` is
    larger than the artifact's group size, because the all-incorrect / hard
    buckets are fixed by the artifact and a larger N may surface fresh
    successes from p alone that didn't appear in the rollouts.
    """

    n_prompts: int
    n_samples: int

    fcr_at_1: float
    fcr_pass_rate_at_n: float
    fcr_pass_at_n: float

    fcr_all_incorrect_at_1: float
    fcr_all_incorrect_pass_rate_at_n: float
    fcr_all_incorrect_pass_at_n: float
    fcr_all_incorrect_total: int

    fcr_hard_at_1: float
    fcr_hard_pass_rate_at_n: float
    fcr_hard_pass_at_n: float
    fcr_hard_total: int

    p_baseline_artifact: float
    p_pass_rate_at_n: float
    p_pass_at_n: float

    p_pass_rate_at_n_on_all_incorrect: float
    p_pass_at_n_on_all_incorrect: float
    p_pass_rate_at_n_on_hard: float
    p_pass_at_n_on_hard: float

    fcr_pass_rate_lift: float
    fcr_pass_at_n_lift: float
    fcr_pass_rate_lift_on_all_incorrect: float
    fcr_pass_at_n_lift_on_all_incorrect: float
    fcr_pass_rate_lift_on_hard: float
    fcr_pass_at_n_lift_on_hard: float


@dataclass
class BaselineResult:
    accuracy: float
    total: int
    correct: int
    hard_accuracy: float
    hard_total: int


@torch.no_grad()
def compute_baseline(
    *,
    p: BaseTransformer,
    prompts: list[PreTokenizedPrompt],
    tokenizer: Tokenizer,
    eos_token_id: int,
    pad_token_id: int,
    max_tokens_generated: int,
    batch_size: int,
    use_bf16: bool = False,
    temperature: float = 1.0,
    n_samples: int = 1,
) -> BaselineResult:
    """Evaluate p generating from scratch (no q involvement).

    For each prompt, p generates ``n_samples`` completions. A prompt
    counts as correct if **any** of the N samples produces the right
    answer (pass@N).
    """
    device = next(p.parameters()).device
    baseline_prompts = [pr for pr in prompts if pr.equation is not None]

    total = 0
    correct = 0
    hard_total = 0
    hard_correct = 0

    n_batches = (len(baseline_prompts) + batch_size - 1) // batch_size
    for batch_idx, batch_start in enumerate(
        range(0, len(baseline_prompts), batch_size)
    ):
        batch = baseline_prompts[batch_start : batch_start + batch_size]
        B = len(batch)
        log.info(
            f"  val batch {batch_idx + 1}/{n_batches} ({batch_start}/{len(baseline_prompts)} prompts)"
        )

        max_prompt_len = max(len(pr.prompt_ids) for pr in batch)
        token_ids = torch.full((B, max_prompt_len), pad_token_id, device=device)
        attention_mask = torch.zeros(B, max_prompt_len, dtype=torch.bool, device=device)
        for i, pr in enumerate(batch):
            offset = max_prompt_len - len(pr.prompt_ids)
            token_ids[i, offset:] = torch.tensor(pr.prompt_ids, device=device)
            attention_mask[i, offset:] = True

        per_prompt_correct = [False] * B
        for _sample in range(n_samples):
            completions = generate_hard_tokens(
                net=p,
                token_ids=token_ids,
                sampling_strategy="sample" if temperature > 0 else "greedy",
                temperature=max(temperature, 1e-6),
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                max_tokens_generated=max_tokens_generated,
                use_kv_cache=True,
                attention_mask=attention_mask,
                use_bf16=use_bf16,
            ).tokens
            prompt_len = token_ids.shape[1]
            completion_strs = tokenizer.decode_batch(
                completions[:, prompt_len:].tolist()
            )

            for i in range(B):
                if per_prompt_correct[i]:
                    continue
                pr = batch[i]
                extracted = extract_from_answer_tags(completion_strs[i])
                is_match = (
                    extracted is not None
                    and pr.equation is not None
                    and expressions_match(extracted, pr.equation)
                )
                if is_match:
                    per_prompt_correct[i] = True

        for i in range(B):
            pr = batch[i]
            p_rate = (
                sum(c.is_correct for c in pr.completions) / len(pr.completions)
                if pr.completions
                else 0.0
            )

            total += 1
            if per_prompt_correct[i]:
                correct += 1

            if p_rate <= HARD_PROMPT_THRESHOLD:
                hard_total += 1
                if per_prompt_correct[i]:
                    hard_correct += 1

    return BaselineResult(
        accuracy=correct / max(total, 1),
        total=total,
        correct=correct,
        hard_accuracy=hard_correct / max(hard_total, 1),
        hard_total=hard_total,
    )


@torch.no_grad()
def compute_fcr(
    *,
    p: BaseTransformer,
    q: BaseTransformer,
    prompts: list[PreTokenizedPrompt],
    tokenizer: Tokenizer,
    eos_token_id: int,
    pad_token_id: int,
    max_tokens_generated: int,
    batch_size: int,
    use_bf16: bool = False,
    q_temperature: float = 1.0,
) -> FCRResult:
    """Compute Forward Consistency Rate.

    For each prompt with a ground truth equation:
    1. Feed (prompt, gt answer) to q -> generate CoT (sampled at
       ``q_temperature`` — matches the paper spec's "Sample Ĉ ~ q_phi"
       and the inference-time use case for offline data synthesis)
    2. Feed prompt + CoT to p -> generate answer (greedy — we want
       p_theta's most likely response given the CoT, not a noisy sample)
    3. Check if p's answer matches ground truth
    """
    device = next(p.parameters()).device
    fcr_prompts = [pr for pr in prompts if pr.equation is not None]

    fcr_total = 0
    fcr_correct = 0
    fcr_all_incorrect_total = 0
    fcr_all_incorrect_correct = 0
    fcr_hard_total = 0
    fcr_hard_correct = 0
    p_correct_rates: list[float] = []

    for batch_start in range(0, len(fcr_prompts), batch_size):
        batch = fcr_prompts[batch_start : batch_start + batch_size]
        B = len(batch)

        # build q's prefix: prompt_ids + answer_ids
        q_prefix_id_lists = [
            pr.prompt_ids + tokenizer.encode(f" <answer> {pr.equation} </answer>").ids
            for pr in batch
        ]
        q_max_prefix_len = max(len(ids) for ids in q_prefix_id_lists)
        q_prefix_ids = torch.full((B, q_max_prefix_len), pad_token_id, device=device)
        q_prefix_mask = torch.zeros(
            B, q_max_prefix_len, dtype=torch.bool, device=device
        )
        for i, ids in enumerate(q_prefix_id_lists):
            offset = q_max_prefix_len - len(ids)
            q_prefix_ids[i, offset:] = torch.tensor(ids, device=device)
            q_prefix_mask[i, offset:] = True

        # q generates CoT (sampled, matching the inference-time regime)
        q_completions = generate_hard_tokens(
            net=q,
            token_ids=q_prefix_ids,
            sampling_strategy="sample",
            temperature=q_temperature,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            max_tokens_generated=max_tokens_generated,
            use_kv_cache=True,
            use_bf16=use_bf16,
            attention_mask=q_prefix_mask,
        ).tokens
        q_cot_strs = [
            tokenizer.decode(q_completions[i, q_max_prefix_len:].tolist())
            for i in range(B)
        ]

        # build p's input: prompt + <think>CoT</think>
        prompt_strs = [tokenizer.decode(pr.prompt_ids) for pr in batch]
        primed_strs = [
            ps + "<think>\n" + cot.strip() + "\n</think>\n\n"
            for ps, cot in zip(prompt_strs, q_cot_strs)
        ]
        tokenizer.enable_padding(direction="left")
        primed_tokens = tokenizer.encode_batch(primed_strs)
        primed_mask = torch.tensor(
            [t.attention_mask for t in primed_tokens],
            dtype=torch.bool,
            device=device,
        )
        primed_ids = torch.tensor([t.ids for t in primed_tokens], device=device)

        # p generates answer
        p_completions = generate_hard_tokens(
            net=p,
            token_ids=primed_ids,
            sampling_strategy="greedy",
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            max_tokens_generated=max_tokens_generated,
            use_kv_cache=True,
            attention_mask=primed_mask,
            use_bf16=use_bf16,
        ).tokens
        primed_len = primed_ids.shape[1]
        p_strs = tokenizer.decode_batch(p_completions[:, primed_len:].tolist())

        # check correctness
        for i in range(B):
            pr = batch[i]
            extracted = extract_from_answer_tags(p_strs[i])
            is_match = (
                extracted is not None
                and pr.equation is not None
                and expressions_match(extracted, pr.equation)
            )

            p_rate = sum(c.is_correct for c in pr.completions) / len(pr.completions)
            p_correct_rates.append(p_rate)

            fcr_total += 1
            if is_match:
                fcr_correct += 1

            if all(not c.is_correct for c in pr.completions):
                fcr_all_incorrect_total += 1
                if is_match:
                    fcr_all_incorrect_correct += 1

            if p_rate <= HARD_PROMPT_THRESHOLD:
                fcr_hard_total += 1
                if is_match:
                    fcr_hard_correct += 1

    return FCRResult(
        fcr=fcr_correct / max(fcr_total, 1),
        fcr_total=fcr_total,
        fcr_correct=fcr_correct,
        fcr_all_incorrect=fcr_all_incorrect_correct / max(fcr_all_incorrect_total, 1),
        fcr_all_incorrect_total=fcr_all_incorrect_total,
        fcr_all_incorrect_correct=fcr_all_incorrect_correct,
        p_baseline=sum(p_correct_rates) / max(len(p_correct_rates), 1),
        fcr_hard=fcr_hard_correct / max(fcr_hard_total, 1),
        fcr_hard_total=fcr_hard_total,
    )
