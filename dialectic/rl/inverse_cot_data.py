import json
import random
from dataclasses import dataclass

import extty
import torch
from tokenizers import Tokenizer

from dialectic.log import log


@dataclass
class PreTokenizedCompletion:
    answer_ids: list[int]
    cot_ids: list[int]
    is_correct: bool


@dataclass
class PreTokenizedPrompt:
    prompt_ids: list[int]
    completions: list[PreTokenizedCompletion]


def load_rollout_artifacts(
    artifact_names: list[str], tokenizer: Tokenizer
) -> dict[str, list[PreTokenizedPrompt]]:
    """Load rollout data from multiple artifacts, grouped by split.

    Returns dict mapping split name to list of PreTokenizedPrompt.
    Entries without a split field go under "train".
    """
    by_split: dict[str, list[PreTokenizedPrompt]] = {}
    for name in artifact_names:
        data = extty.load_artifact(name)
        if not isinstance(data, bytes):
            raise ValueError(f"Expected bytes from artifact {name}, got {type(data)}")
        count = 0
        for line in data.decode().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            prompt_ids = tokenizer.encode(entry["prompt_str"]).ids
            completions = []
            for comp in entry["completions"]:
                answer_ids = tokenizer.encode(" " + comp["answer"]).ids
                cot_ids = tokenizer.encode(comp["cot"]).ids
                completions.append(
                    PreTokenizedCompletion(
                        answer_ids=answer_ids,
                        cot_ids=cot_ids,
                        is_correct=comp["is_correct"],
                    )
                )
            split = entry.get("split", "train")
            by_split.setdefault(split, []).append(
                PreTokenizedPrompt(prompt_ids=prompt_ids, completions=completions)
            )
            count += 1
        log.info(f"Loaded {count} prompts from artifact '{name}'")
    for split, prompts in sorted(by_split.items()):
        log.info(f"  {split}: {len(prompts)} prompts")
    return by_split


def subsample_completions(prompt: PreTokenizedPrompt, k: int) -> PreTokenizedPrompt:
    """Subsample k completions, guaranteeing at least 1 positive and 1 negative."""
    positives = [c for c in prompt.completions if c.is_correct]
    negatives = [c for c in prompt.completions if not c.is_correct]

    if not positives or not negatives:
        return prompt

    selected: list[PreTokenizedCompletion] = []
    selected.append(random.choice(positives))
    selected.append(random.choice(negatives))

    remaining = [c for c in prompt.completions if c not in selected]
    n_extra = min(k - 2, len(remaining))
    if n_extra > 0:
        selected.extend(random.sample(remaining, n_extra))

    random.shuffle(selected)
    return PreTokenizedPrompt(prompt_ids=prompt.prompt_ids, completions=selected)


def build_contrastive_batch(
    prompts: list[PreTokenizedPrompt],
    eos_token_id: int,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """Build a batch from pre-tokenized prompts for contrastive training.

    Returns (input_ids, prefix_lengths, loss_mask, is_correct, group_sizes).
    input_ids is [N, L] where N = sum of group sizes across prompts.
    """
    all_ids: list[list[int]] = []
    all_prefix_lengths: list[int] = []
    all_is_correct: list[bool] = []
    group_sizes: list[int] = []

    for prompt in prompts:
        group_sizes.append(len(prompt.completions))
        for comp in prompt.completions:
            seq = prompt.prompt_ids + comp.answer_ids + comp.cot_ids + [eos_token_id]
            all_ids.append(seq)
            all_prefix_lengths.append(len(prompt.prompt_ids) + len(comp.answer_ids))
            all_is_correct.append(comp.is_correct)

    max_len = max(len(ids) for ids in all_ids)
    N = len(all_ids)
    input_ids = torch.full((N, max_len), pad_token_id, device=device)
    for i, ids in enumerate(all_ids):
        input_ids[i, : len(ids)] = torch.tensor(ids, device=device)

    prefix_lengths = torch.tensor(all_prefix_lengths, dtype=torch.long, device=device)
    actual_lengths = torch.tensor(
        [len(ids) for ids in all_ids], dtype=torch.long, device=device
    )
    positions = torch.arange(max_len, device=device).unsqueeze(0)
    loss_mask = (positions >= prefix_lengths.unsqueeze(1)) & (
        positions < actual_lengths.unsqueeze(1)
    )
    is_correct = torch.tensor(all_is_correct, dtype=torch.bool, device=device)

    return input_ids, prefix_lengths, loss_mask, is_correct, group_sizes


def build_shuffled_batch(
    batch_prompts: list[PreTokenizedPrompt],
    eos_token_id: int,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a batch with answer tokens rolled by 1 for shuffled NLL diagnostic.

    Returns (input_ids, prefix_lengths, loss_mask).
    """
    all_prompt_ids: list[list[int]] = []
    all_answer_ids: list[list[int]] = []
    all_cot_ids: list[list[int]] = []
    for prompt_data in batch_prompts:
        for comp in prompt_data.completions:
            all_prompt_ids.append(prompt_data.prompt_ids)
            all_answer_ids.append(comp.answer_ids)
            all_cot_ids.append(comp.cot_ids)

    shuffled_answer = all_answer_ids[1:] + all_answer_ids[:1]
    shuffled_seqs = [
        pi + ai + ci + [eos_token_id]
        for pi, ai, ci in zip(all_prompt_ids, shuffled_answer, all_cot_ids)
    ]
    N = len(shuffled_seqs)
    s_max_len = max(len(s) for s in shuffled_seqs)
    input_ids = torch.full((N, s_max_len), pad_token_id, device=device)
    for i, ids in enumerate(shuffled_seqs):
        input_ids[i, : len(ids)] = torch.tensor(ids, device=device)

    prefix_lengths = torch.tensor(
        [len(pi) + len(ai) for pi, ai in zip(all_prompt_ids, shuffled_answer)],
        dtype=torch.long,
        device=device,
    )
    cot_lens = torch.tensor(
        [len(ci) + 1 for ci in all_cot_ids], dtype=torch.long, device=device
    )
    positions = torch.arange(s_max_len, device=device).unsqueeze(0)
    loss_mask = (positions >= prefix_lengths.unsqueeze(1)) & (
        positions < (prefix_lengths + cot_lens).unsqueeze(1)
    )

    return input_ids, prefix_lengths, loss_mask
