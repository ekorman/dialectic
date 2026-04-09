from dataclasses import dataclass

import torch
from tokenizers import Tokenizer

from dialectic.llm.base import BaseTransformer
from dialectic.llm.generate import generate_hard_tokens
from dialectic.llm.inverse_cot import InverseCotModel
from dialectic.rl.extractors import extract_from_answer_tags
from dialectic.rl.inverse_cot_data import PreTokenizedPrompt


def expressions_match(pred: str, gt: str) -> bool:
    """Check if two arithmetic expressions evaluate to the same value."""
    try:
        gt_val = eval(gt, {"__builtins__": {}}, {})
    except Exception:
        return False
    try:
        pred_val = eval(pred, {"__builtins__": {}}, {})
    except Exception:
        return False
    return abs(pred_val - gt_val) < 1e-6


@dataclass
class FCRResult:
    fcr: float
    fcr_total: int
    fcr_correct: int
    fcr_all_incorrect: float
    fcr_all_incorrect_total: int
    fcr_all_incorrect_correct: int


@torch.no_grad()
def compute_fcr(
    *,
    p: BaseTransformer,
    q: InverseCotModel,
    prompts: list[PreTokenizedPrompt],
    tokenizer: Tokenizer,
    eos_token_id: int,
    pad_token_id: int,
    max_tokens_generated: int,
    batch_size: int,
    use_bf16: bool = False,
) -> FCRResult:
    """Compute Forward Consistency Rate.

    For each prompt with a ground truth equation:
    1. Feed (prompt, gt answer) to q -> generate CoT
    2. Feed prompt + CoT to p -> generate answer
    3. Check if p's answer matches ground truth
    """
    device = next(p.parameters()).device
    fcr_prompts = [pr for pr in prompts if pr.equation is not None]

    fcr_total = 0
    fcr_correct = 0
    fcr_all_incorrect_total = 0
    fcr_all_incorrect_correct = 0

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

        # q generates CoT
        q_completions = generate_hard_tokens(
            net=q,
            token_ids=q_prefix_ids,
            sampling_strategy="greedy",
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

            fcr_total += 1
            if is_match:
                fcr_correct += 1

            if all(not c.is_correct for c in pr.completions):
                fcr_all_incorrect_total += 1
                if is_match:
                    fcr_all_incorrect_correct += 1

    return FCRResult(
        fcr=fcr_correct / max(fcr_total, 1),
        fcr_total=fcr_total,
        fcr_correct=fcr_correct,
        fcr_all_incorrect=fcr_all_incorrect_correct / max(fcr_all_incorrect_total, 1),
        fcr_all_incorrect_total=fcr_all_incorrect_total,
        fcr_all_incorrect_correct=fcr_all_incorrect_correct,
    )
