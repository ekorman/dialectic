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
    equation: str | None = None


def load_rollout_artifacts(
    artifact_names: list[str],
    tokenizer: Tokenizer,
    max_cot_tokens: int | None = None,
    min_completions_per_prompt: int = 2,
    filter_train_split: bool = True,
) -> dict[str, list[PreTokenizedPrompt]]:
    """Load rollout data from multiple artifacts, grouped by split.

    Returns dict mapping split name to list of PreTokenizedPrompt.
    Entries without a split field go under "train".

    Tokenization is done via a single ``encode_batch`` call per shard
    rather than per-string ``encode`` calls. For a typical shard (~200
    prompts × 32 completions = ~13k strings per shard) this is ~10-30×
    faster because the tokenizers library parallelizes batch encoding
    across threads on the Rust side.

    Parameters
    ----------
    artifact_names
        Names of extty rollout artifacts to load.
    tokenizer
        Fast tokenizer for the target model. Will be reset (no padding,
        no truncation) before use to protect against caller-side mutation.
    max_cot_tokens
        If set, drop completions whose tokenized CoT exceeds this length.
        Training cost in the InfoNCE contrastive loss scales with the
        max sequence length per batch squared (prefix-LM attention forces
        SDPA's math backend, which materializes ``[N, H, L, L]`` scores),
        so a single outlier CoT can blow the memory budget. Dropping long
        tails at load time bounds ``L`` for every training batch.
    min_completions_per_prompt
        If a prompt has fewer than this many completions after the
        ``max_cot_tokens`` filter, drop the whole prompt. Defaults to 2,
        which is the floor required for the contrastive training loop
        (one correct and one incorrect). If the caller uses a fixed
        ``train_group_size``, pass that value here so post-filter prompts
        retain enough rollouts to satisfy ``subsample_completions``'s
        uniform-K requirement.
    """
    # Defensive: upstream code paths (notably `generate_from_text`) mutate
    # the caller's tokenizer to enable left-padding. `encode_batch` would
    # then pad every string to the batch max, bloating the returned ids and
    # corrupting prefix-length calculations downstream. Reset before use.
    tokenizer.no_padding()
    tokenizer.no_truncation()

    by_split: dict[str, list[PreTokenizedPrompt]] = {}
    total_completions_dropped = 0
    total_prompts_dropped = 0
    for name in artifact_names:
        data = extty.load_artifact(name, cache=True)
        if not isinstance(data, bytes):
            raise ValueError(f"Expected bytes from artifact {name}, got {type(data)}")

        entries: list[dict] = [
            json.loads(line) for line in data.decode().splitlines() if line.strip()
        ]
        if not entries:
            log.info(f"Loaded 0 prompts from artifact '{name}'")
            continue

        # Build a single flat list of all strings to tokenize for this
        # shard. Layout: [prompt_0, prompt_1, ..., prompt_{N-1},
        # ans_0_0, cot_0_0, ans_0_1, cot_0_1, ..., ans_{N-1}_{K-1}, cot_{N-1}_{K-1}]
        texts: list[str] = [entry["prompt_str"] for entry in entries]
        completion_offsets: list[
            int
        ] = []  # one per entry: where its first ans_id lives
        for entry in entries:
            completion_offsets.append(len(texts))
            for comp in entry["completions"]:
                texts.append(" " + comp["answer"])
                texts.append(comp["cot"])

        encoded = tokenizer.encode_batch(texts)

        shard_kept = 0
        shard_comp_dropped = 0
        shard_prompt_dropped = 0
        for i, entry in enumerate(entries):
            prompt_ids = encoded[i].ids
            completions: list[PreTokenizedCompletion] = []
            cursor = completion_offsets[i]
            for comp in entry["completions"]:
                answer_ids = encoded[cursor].ids
                cot_ids = encoded[cursor + 1].ids
                cursor += 2
                if max_cot_tokens is not None and len(cot_ids) > max_cot_tokens:
                    shard_comp_dropped += 1
                    continue
                completions.append(
                    PreTokenizedCompletion(
                        answer_ids=list(answer_ids),
                        cot_ids=list(cot_ids),
                        is_correct=comp["is_correct"],
                    )
                )

            equation = entry.get("equation")
            if equation and "=" in equation:
                equation = equation.split("=")[0].strip()
            split = entry.get("split", "train")

            if split == "train" and filter_train_split:
                has_correct = any(c.is_correct for c in completions)
                has_incorrect = any(not c.is_correct for c in completions)
                if (
                    len(completions) < min_completions_per_prompt
                    or not has_correct
                    or not has_incorrect
                ):
                    shard_prompt_dropped += 1
                    continue
            elif not completions:
                shard_prompt_dropped += 1
                continue
            by_split.setdefault(split, []).append(
                PreTokenizedPrompt(
                    prompt_ids=list(prompt_ids),
                    completions=completions,
                    equation=equation,
                )
            )
            shard_kept += 1

        total_completions_dropped += shard_comp_dropped
        total_prompts_dropped += shard_prompt_dropped
        if shard_comp_dropped or shard_prompt_dropped:
            log.info(
                f"Loaded {shard_kept} prompts from artifact '{name}' "
                f"(dropped {shard_comp_dropped} completions over max_cot_tokens={max_cot_tokens}, "
                f"{shard_prompt_dropped} prompts with too few / unmixed completions)"
            )
        else:
            log.info(f"Loaded {shard_kept} prompts from artifact '{name}'")

    if total_completions_dropped or total_prompts_dropped:
        log.info(
            f"Length filter: dropped {total_completions_dropped} completions and "
            f"{total_prompts_dropped} prompts across all shards "
            f"(max_cot_tokens={max_cot_tokens}, min_completions_per_prompt={min_completions_per_prompt})"
        )
    for split, prompts in sorted(by_split.items()):
        n_all_correct = sum(
            1 for p in prompts if all(c.is_correct for c in p.completions)
        )
        n_all_incorrect = sum(
            1 for p in prompts if all(not c.is_correct for c in p.completions)
        )
        n_mixed = len(prompts) - n_all_correct - n_all_incorrect
        log.info(
            f"  {split}: {len(prompts)} prompts "
            f"({n_mixed} mixed, {n_all_correct} all-correct, {n_all_incorrect} all-incorrect)"
        )
    return by_split


def subsample_completions(
    prompt: PreTokenizedPrompt,
    k: int,
    rng: random.Random | None = None,
) -> PreTokenizedPrompt:
    """Subsample k completions, guaranteeing at least 1 positive and 1 negative.

    Parameters
    ----------
    prompt
        The prompt whose completions to subsample.
    k
        Number of completions to keep.
    rng
        Optional ``random.Random`` instance used for all randomness inside
        this call. ``None`` (the default) uses the global ``random`` module
        state — preserving the nondeterministic behavior the training loop
        relies on for batch diversity. Pass an explicit instance from the
        val loop to make val reproducible across calls within a single run.

    Notes
    -----
    Selection is done on **indices** rather than on completion *objects* so
    duplicate completions (two rollouts that happened to produce byte-
    identical ``(answer_ids, cot_ids, is_correct)`` tuples) don't break the
    "remaining" computation. Under dataclass value-equality, ``c not in
    selected`` would treat all duplicates as already-selected once one was
    picked, shrinking the remaining pool and producing fewer than ``k``
    output completions — which then trips ``build_infonce_batch``'s
    uniform-K assertion downstream. Works correctly with or without
    duplicates.
    """
    r = rng if rng is not None else random
    pos_indices = [i for i, c in enumerate(prompt.completions) if c.is_correct]
    neg_indices = [i for i, c in enumerate(prompt.completions) if not c.is_correct]

    if not pos_indices or not neg_indices:
        return prompt

    selected_indices: set[int] = set()
    selected_indices.add(r.choice(pos_indices))
    selected_indices.add(r.choice(neg_indices))

    remaining_indices = [
        i for i in range(len(prompt.completions)) if i not in selected_indices
    ]
    n_extra = min(k - 2, len(remaining_indices))
    if n_extra > 0:
        selected_indices.update(r.sample(remaining_indices, n_extra))

    selected = [prompt.completions[i] for i in selected_indices]
    r.shuffle(selected)
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

    actual_lens = [len(ids) for ids in all_ids]
    max_len = max(actual_lens)
    # Build the padded [N, max_len] tensor in one CPU allocation + a single
    # host→device copy, rather than N per-row `torch.tensor(...).to(device)`
    # calls. Each of those calls was a separate small H2D transfer.
    padded = [ids + [pad_token_id] * (max_len - len(ids)) for ids in all_ids]
    input_ids = torch.tensor(padded, dtype=torch.long, device=device)

    prefix_lengths = torch.tensor(all_prefix_lengths, dtype=torch.long, device=device)
    actual_lengths = torch.tensor(actual_lens, dtype=torch.long, device=device)
    positions = torch.arange(max_len, device=device).unsqueeze(0)
    loss_mask = (positions >= prefix_lengths.unsqueeze(1)) & (
        positions < actual_lengths.unsqueeze(1)
    )
    is_correct = torch.tensor(all_is_correct, dtype=torch.bool, device=device)

    return input_ids, prefix_lengths, loss_mask, is_correct, group_sizes


def build_infonce_batch(
    prompts: list[PreTokenizedPrompt],
    eos_token_id: int,
    pad_token_id: int,
    device: torch.device,
    n_negatives: int,
    rng: random.Random | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
]:
    """Build a batch laid out for InfoNCE-style contrastive training.

    Only **correct** completions serve as anchors. For each correct anchor,
    the positive pair is ``(P, A_correct, C_i)`` at slot 0, followed by
    ``n_negatives`` contrastive pairs ``(P, A_wrong_j, C_i)`` at slots
    1..M, where the ``A_wrong_j`` are drawn from answer buckets whose
    rollouts are **incorrect**.

    Layout of the returned flat-batch tensor, with ``A = n_anchors``
    (total correct completions across all prompts), ``M = n_negatives``,
    slots = ``1 + M``::

        flat_idx = a * slots + s

    Reshaping via ``.view(A, slots)`` gives the anchor × slot layout.

    Parameters
    ----------
    prompts
        Per-prompt rollout groups.
    eos_token_id, pad_token_id
        Special-token ids for sequence termination and right-padding.
    device
        Target device for the returned tensors.
    n_negatives
        Number of negative (incorrect) answer-conditionings per anchor.
        If a prompt has fewer than ``n_negatives`` distinct incorrect
        answer keys, the missing slots are filled with the positive
        sequence and marked ``valid=False``.
    rng
        Optional ``random.Random`` instance for reproducible negative
        sampling. Defaults to a fresh instance.

    Returns
    -------
    input_ids : torch.Tensor
        Packed ``[A*(1+M), L]`` right-padded token tensor.
    prefix_lengths : torch.Tensor
        Per-sequence prefix length ``|P| + |A_j|``.
    loss_mask : torch.Tensor
        ``[A*(1+M), L]`` bool tensor, True over the CoT + EOS region.
    valid_mask : torch.Tensor
        ``[A*(1+M)]`` bool tensor marking real (non-padding) slots.
    n_anchors : int
        Total number of correct-completion anchors across all prompts.
    """
    if rng is None:
        rng = random.Random()

    if not prompts:
        raise ValueError("build_infonce_batch requires at least one prompt")

    slots_per_anchor = 1 + n_negatives

    all_ids: list[list[int]] = []
    all_prefix_lens: list[int] = []
    all_valid: list[bool] = []
    n_anchors = 0

    for prompt in prompts:
        answer_buckets: dict[tuple[int, ...], list[int]] = {}
        for idx, comp in enumerate(prompt.completions):
            key = tuple(comp.answer_ids)
            answer_buckets.setdefault(key, []).append(idx)

        incorrect_keys = [
            k
            for k in answer_buckets
            if not prompt.completions[answer_buckets[k][0]].is_correct
        ]

        for anchor in prompt.completions:
            if not anchor.is_correct:
                continue
            n_anchors += 1

            pos_seq = (
                prompt.prompt_ids + anchor.answer_ids + anchor.cot_ids + [eos_token_id]
            )
            pos_prefix_len = len(prompt.prompt_ids) + len(anchor.answer_ids)

            all_ids.append(pos_seq)
            all_prefix_lens.append(pos_prefix_len)
            all_valid.append(True)

            if len(incorrect_keys) > n_negatives:
                picked = rng.sample(incorrect_keys, n_negatives)
            else:
                picked = list(incorrect_keys)
                rng.shuffle(picked)
            n_real_negatives = len(picked)

            for slot in range(n_negatives):
                if slot < n_real_negatives:
                    neg_bucket = answer_buckets[picked[slot]]
                    rep = prompt.completions[rng.choice(neg_bucket)]
                    neg_answer_ids = rep.answer_ids
                    neg_seq = (
                        prompt.prompt_ids
                        + neg_answer_ids
                        + anchor.cot_ids
                        + [eos_token_id]
                    )
                    neg_prefix_len = len(prompt.prompt_ids) + len(neg_answer_ids)
                    all_ids.append(neg_seq)
                    all_prefix_lens.append(neg_prefix_len)
                    all_valid.append(True)
                else:
                    all_ids.append(pos_seq)
                    all_prefix_lens.append(pos_prefix_len)
                    all_valid.append(False)

    n_sequences = len(all_ids)
    expected = n_anchors * slots_per_anchor
    if n_sequences != expected:
        raise RuntimeError(
            f"build_infonce_batch produced {n_sequences} sequences, expected {expected}"
        )

    actual_lens = [len(ids) for ids in all_ids]
    max_len = max(actual_lens)
    padded = [ids + [pad_token_id] * (max_len - len(ids)) for ids in all_ids]
    input_ids = torch.tensor(padded, dtype=torch.long, device=device)

    prefix_lengths = torch.tensor(all_prefix_lens, dtype=torch.long, device=device)
    actual_lengths = torch.tensor(actual_lens, dtype=torch.long, device=device)
    positions = torch.arange(max_len, device=device).unsqueeze(0)
    loss_mask = (positions >= prefix_lengths.unsqueeze(1)) & (
        positions < actual_lengths.unsqueeze(1)
    )
    valid_mask = torch.tensor(all_valid, dtype=torch.bool, device=device)

    return input_ids, prefix_lengths, loss_mask, valid_mask, n_anchors


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
    seq_lens = [len(s) for s in shuffled_seqs]
    s_max_len = max(seq_lens)
    # Single CPU allocation + one host→device copy instead of N per-row
    # `torch.tensor(...).to(device)` calls.
    padded = [s + [pad_token_id] * (s_max_len - len(s)) for s in shuffled_seqs]
    input_ids = torch.tensor(padded, dtype=torch.long, device=device)

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
