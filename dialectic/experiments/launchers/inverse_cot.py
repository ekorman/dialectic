import re
import time
from dataclasses import dataclass
from typing import Any, Callable

import extty
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from dialectic.experiments.arg_parser import Experiment, run_experiments_parser
from dialectic.experiments.envs import (
    get_state_to_str,
    load_countdown_dataset_artifacts,
)
from dialectic.experiments.models import load_model_and_opt
from dialectic.experiments.params import (
    InverseCotParams,
    TrainParams,
)
from dialectic.experiments.prompts import PromptCollection
from dialectic.llm.base import BaseTransformer
from dialectic.llm.generate import generate_hard_tokens
from dialectic.llm.inverse_cot import InverseCotModel, create_prefix_lm_mask
from dialectic.llm.registry import MODEL_REGISTRY
from dialectic.log import log
from dialectic.rl.dataset_env import DatasetEnv
from dialectic.rl.env import Countdown, Env
from dialectic.rl.extractors import extract_from_answer_tags
from dialectic.rl.reward import _evaluate_and_verify_countdown
from dialectic.rl.rollout import get_batch
from dialectic.training import StepFunctionReturn


def _parse_cot_and_answer(text: str) -> tuple[str, str] | None:
    """Split model output into (CoT, answer).

    Tries <think>...</think> format first (CoT is inside think tags,
    answer is everything after). Falls back to splitting at <answer> tags.
    Returns None if neither format is found.
    """
    think_match = re.search(r"<think>(.*?)</think>(.*)", text, re.DOTALL)
    if think_match is not None:
        cot = think_match.group(1).strip()
        answer = think_match.group(2).strip()
        if cot and answer:
            return cot, answer

    answer_match = re.search(r"<answer>.*?</answer>", text, re.DOTALL)
    if answer_match is not None:
        cot = text[: answer_match.start()]
        answer = answer_match.group(0)
        if cot.strip():
            return cot, answer

    return None


@dataclass
class TrainingBatch:
    input_ids: torch.Tensor
    prefix_lengths: torch.Tensor
    loss_mask: torch.Tensor
    n_skipped: int
    prompt_token_ids: list[list[int]]
    answer_token_ids: list[list[int]]
    cot_token_ids: list[list[int]]
    is_correct: list[bool]


def _generate_training_data(
    *,
    p: BaseTransformer,
    env: Env,
    state_to_str: Callable[[Countdown], str],
    tokenizer: Tokenizer,
    eos_token_id: int,
    pad_token_id: int,
    batch_size: int,
    temperature: float,
    max_tokens_generated: int,
    use_bf16: bool,
    correct_only: bool = False,
) -> TrainingBatch | None:
    """Generate data from p and construct q's training batch."""
    device = next(p.parameters()).device

    env_responses = get_batch(env, batch_size)
    prompts = [state_to_str(resp.data) for resp in env_responses]

    tokenizer.enable_padding(direction="left")
    tokens = tokenizer.encode_batch(prompts)
    attention_mask = torch.tensor(
        [t.attention_mask for t in tokens], dtype=torch.bool, device=device
    )
    token_ids = torch.tensor([t.ids for t in tokens], device=device)

    was_training = p.training
    p.eval()
    completions = generate_hard_tokens(
        net=p,
        token_ids=token_ids,
        sampling_strategy="sample" if temperature > 0 else "greedy",
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        max_tokens_generated=max_tokens_generated,
        use_kv_cache=True,
        attention_mask=attention_mask,
        temperature=temperature if temperature > 0 else 1.0,
        use_bf16=use_bf16,
    ).tokens
    if was_training:
        p.train()

    prompt_len = token_ids.shape[1]
    completion_strs = tokenizer.decode_batch(completions[:, prompt_len:].tolist())

    valid_prompt_ids: list[list[int]] = []
    valid_answer_strs: list[str] = []
    valid_cot_strs: list[str] = []
    valid_is_correct: list[bool] = []

    n_no_answer_tag = 0
    n_empty_cot = 0
    n_incorrect = 0
    for i, comp_str in enumerate(completion_strs):
        parsed = _parse_cot_and_answer(comp_str)
        if parsed is None:
            n_no_answer_tag += 1
            continue
        cot, answer = parsed
        if not cot.strip():
            n_empty_cot += 1
            continue
        extracted = extract_from_answer_tags(comp_str)
        sample_correct = extracted is not None and _evaluate_and_verify_countdown(
            extracted,
            env_responses[i].data.numbers,
            env_responses[i].data.target,
        )
        if correct_only and not sample_correct:
            n_incorrect += 1
            continue
        # keep original prompt token IDs (no lossy decode→re-encode roundtrip)
        valid_prompt_ids.append(token_ids[i][attention_mask[i]].tolist())
        valid_answer_strs.append(answer)
        valid_cot_strs.append(cot)
        valid_is_correct.append(sample_correct)

    n_skipped = n_no_answer_tag + n_empty_cot + n_incorrect
    if n_skipped > 0:
        log.info(
            f"Skipped {n_no_answer_tag}/{batch_size} (no parse), "
            f"{n_empty_cot}/{batch_size} (empty CoT), "
            f"{n_incorrect}/{batch_size} (incorrect). "
            f"Sample failed completion: {completion_strs[0][:200]!r}"
        )
    if not valid_prompt_ids:
        return None

    # tokenize answer and cot (prompt IDs are kept from original tokenization)
    answer_encs = [tokenizer.encode(" " + s) for s in valid_answer_strs]
    cot_encs = [tokenizer.encode(s) for s in valid_cot_strs]

    prompt_ids = valid_prompt_ids
    answer_ids = [enc.ids for enc in answer_encs]
    cot_ids = [enc.ids for enc in cot_encs]

    prefix_lengths = torch.tensor(
        [len(p) + len(a) for p, a in zip(prompt_ids, answer_ids)],
        dtype=torch.long,
        device=device,
    )

    all_ids = [
        p + a + c + [eos_token_id] for p, a, c in zip(prompt_ids, answer_ids, cot_ids)
    ]
    max_len = max(len(ids) for ids in all_ids)
    input_ids = torch.full((len(all_ids), max_len), pad_token_id, device=device)
    for i, ids in enumerate(all_ids):
        input_ids[i, : len(ids)] = torch.tensor(ids, device=device)

    positions = torch.arange(max_len, device=device).unsqueeze(0)
    actual_lengths = torch.tensor(
        [len(ids) for ids in all_ids], dtype=torch.long, device=device
    ).unsqueeze(1)
    loss_mask = (positions >= prefix_lengths.unsqueeze(1)) & (
        positions < actual_lengths
    )

    return TrainingBatch(
        input_ids=input_ids,
        prefix_lengths=prefix_lengths,
        loss_mask=loss_mask,
        n_skipped=n_skipped,
        prompt_token_ids=prompt_ids,
        answer_token_ids=answer_ids,
        cot_token_ids=cot_ids,
        is_correct=valid_is_correct,
    )


def _compute_nll_loss(
    q: InverseCotModel,
    input_ids: torch.Tensor,
    prefix_lengths: torch.Tensor,
    loss_mask: torch.Tensor,
    normalize_by_sequence_length: bool,
) -> tuple[torch.Tensor, float, list[float]]:
    """Forward through q and compute NLL loss on CoT tokens.

    Returns (loss, mean_nll, per_sample_nlls).
    """
    seq_len = input_ids.shape[1]
    device = input_ids.device

    attention_mask = create_prefix_lm_mask(prefix_lengths, seq_len, device)
    logits = q(input_ids, attention_mask=attention_mask, return_all_logits=True)

    # shift: predict next token from current position
    shift_logits = logits[:, :-1]
    shift_targets = input_ids[:, 1:]
    shift_mask = loss_mask[:, 1:]

    B, L, V = shift_logits.shape
    per_token_loss = F.cross_entropy(
        shift_logits.reshape(B * L, V),
        shift_targets.reshape(B * L),
        reduction="none",
    ).reshape(B, L)

    masked_loss = per_token_loss * shift_mask
    seq_lengths = shift_mask.sum(dim=1).clamp(min=1)
    per_seq_loss = masked_loss.sum(dim=1) / seq_lengths

    if normalize_by_sequence_length:
        loss = per_seq_loss.mean()
    else:
        loss = masked_loss.sum() / shift_mask.sum().clamp(min=1)

    nll = loss.item()
    per_sample_nlls = per_seq_loss.detach().tolist()
    return loss, nll, per_sample_nlls


@dataclass
class PreTokenizedCompletion:
    answer_ids: list[int]
    cot_ids: list[int]
    is_correct: bool


@dataclass
class PreTokenizedPrompt:
    prompt_ids: list[int]
    completions: list[PreTokenizedCompletion]


def _load_rollout_artifacts(
    artifact_names: list[str], tokenizer: Tokenizer
) -> dict[str, list[PreTokenizedPrompt]]:
    """Load rollout data from multiple artifacts, grouped by split.

    Returns dict mapping split name to list of PreTokenizedPrompt.
    Entries without a split field go under "train".
    """
    import json

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


def _subsample_completions(prompt: PreTokenizedPrompt, k: int) -> PreTokenizedPrompt:
    """Subsample k completions, guaranteeing at least 1 positive and 1 negative."""
    import random

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


def _build_contrastive_batch(
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


def _compute_contrastive_loss(
    q: InverseCotModel,
    input_ids: torch.Tensor,
    prefix_lengths: torch.Tensor,
    loss_mask: torch.Tensor,
    is_correct: torch.Tensor,
    group_sizes: list[int],
    contrastive_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute NLL + contrastive loss over grouped completions.

    Returns (total_loss, metrics_dict).
    """
    seq_len = input_ids.shape[1]
    device = input_ids.device

    attention_mask = create_prefix_lm_mask(prefix_lengths, seq_len, device)
    logits = q(input_ids, attention_mask=attention_mask, return_all_logits=True)

    shift_logits = logits[:, :-1]
    shift_targets = input_ids[:, 1:]
    shift_mask = loss_mask[:, 1:]

    N, L, V = shift_logits.shape
    per_token_loss = F.cross_entropy(
        shift_logits.reshape(N * L, V),
        shift_targets.reshape(N * L),
        reduction="none",
    ).reshape(N, L)

    # per-sequence average log prob (negative of per-token avg loss)
    seq_lengths = shift_mask.sum(dim=1).clamp(min=1)
    per_seq_avg_nll = (per_token_loss * shift_mask).sum(dim=1) / seq_lengths
    per_seq_avg_logprob = -per_seq_avg_nll

    # split into groups and compute per-prompt losses
    nll_losses: list[torch.Tensor] = []
    contrastive_losses: list[torch.Tensor] = []
    offset = 0
    for gs in group_sizes:
        group_nll = per_seq_avg_nll[offset : offset + gs]
        group_logprob = per_seq_avg_logprob[offset : offset + gs]
        group_correct = is_correct[offset : offset + gs]

        # NLL: mean over positive examples
        if group_correct.any():
            nll_losses.append(group_nll[group_correct].mean())

        # contrastive: -log(sum_pos exp(s) / sum_all exp(s))
        if contrastive_weight > 0 and group_correct.any() and not group_correct.all():
            log_numerator = torch.logsumexp(group_logprob[group_correct], dim=0)
            log_denominator = torch.logsumexp(group_logprob, dim=0)
            contrastive_losses.append(-(log_numerator - log_denominator))

        offset += gs

    nll_loss = (
        torch.stack(nll_losses).mean()
        if nll_losses
        else torch.tensor(0.0, device=device)
    )
    contrastive_loss = (
        torch.stack(contrastive_losses).mean()
        if contrastive_losses
        else torch.tensor(0.0, device=device)
    )
    total_loss = nll_loss + contrastive_weight * contrastive_loss

    metrics = {
        "train/nll": nll_loss.item(),
        "train/contrastive_loss": contrastive_loss.item(),
        "train/loss": total_loss.item(),
    }
    return total_loss, metrics


@extty.experiment(project="inverse-cot-countdown")
def train_inverse_cot_countdown(
    *,
    train_params: TrainParams,
    inverse_cot_params: InverseCotParams,
    prompt_collection: PromptCollection,
    dataset_artifacts: list[str],
):
    torch.manual_seed(train_params.seed)

    model_info = MODEL_REGISTRY[train_params.model_name]
    tokenizer = model_info.load_tokenizer()

    # Load p from checkpoint
    p, _ = load_model_and_opt(train_params=train_params)
    p.requires_grad_(False)
    p.eval()

    # Build q from p (p is already on device and optionally bf16 from load_model_and_opt)
    q = InverseCotModel(p)
    if inverse_cot_params.freeze_lm_head:
        q.lm_head.requires_grad_(False)
    device = next(p.parameters()).device
    dtype = next(p.parameters()).dtype
    q = q.to(device=device, dtype=dtype)

    trainable_params = [param for param in q.parameters() if param.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=train_params.lr)

    total_params = sum(param.numel() for param in q.parameters())
    trainable_count = sum(param.numel() for param in trainable_params)
    log.info(f"q total params: {total_params:,}, trainable: {trainable_count:,}")

    state_to_str = get_state_to_str(
        format_messages=model_info.format_messages,
        system_prompt=prompt_collection.system_prompt,
        assistant_prefill=prompt_collection.assistant_prefill,
    )

    # load rollout data from artifacts, split by train/val
    by_split = _load_rollout_artifacts(dataset_artifacts, tokenizer)
    rollout_data = by_split.get("train", [])
    val_rollout_data = by_split.get("val", [])
    if not rollout_data:
        raise ValueError("No training data found (split='train')")

    # build val envs from dataset for online val
    all_countdown_problems = load_countdown_dataset_artifacts(
        dataset_artifacts, prompt_template=prompt_collection.env_prompt
    )
    val_countdown = [
        resp for resp, extra in all_countdown_problems if extra.get("split") == "val"
    ]
    val_envs: list = (
        [DatasetEnv(val_countdown, seed=2026, label="countdown_val")]
        if val_countdown
        else []
    )

    def _train_step() -> StepFunctionReturn:
        q.train()
        opt.zero_grad()

        total_metrics: dict[str, float] = {}
        total_episodes = 0

        for _ in range(train_params.accumulation_steps):
            indices = torch.randint(
                len(rollout_data), (train_params.batch_size,)
            ).tolist()
            batch_prompts = [rollout_data[i] for i in indices]

            if inverse_cot_params.train_group_size is not None:
                batch_prompts = [
                    _subsample_completions(p, inverse_cot_params.train_group_size)
                    for p in batch_prompts
                ]

            input_ids, prefix_lengths, loss_mask, is_correct, group_sizes = (
                _build_contrastive_batch(
                    batch_prompts,
                    model_info.eos_token_id,
                    model_info.pad_token_id,
                    device,
                )
            )

            loss, step_metrics = _compute_contrastive_loss(
                q,
                input_ids,
                prefix_lengths,
                loss_mask,
                is_correct,
                group_sizes,
                inverse_cot_params.contrastive_weight,
            )
            (loss / train_params.accumulation_steps).backward()

            for k, v in step_metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + v
            total_episodes += train_params.batch_size

        torch.nn.utils.clip_grad_norm_(trainable_params, train_params.max_grad_norm)
        opt.step()

        metrics = {
            k: v / train_params.accumulation_steps for k, v in total_metrics.items()
        }
        metrics["train/episodes"] = total_episodes
        return StepFunctionReturn(n_episodes_processed=total_episodes, metrics=metrics)

    @torch.no_grad()
    def _val_fn_offline() -> tuple[float, float, float, list[extty.Example]]:
        """Validation using the val rollout file (same code path as training)."""
        assert val_rollout_data is not None
        q.eval()

        all_nll: list[float] = []
        all_nll_correct: list[float] = []
        all_nll_incorrect: list[float] = []
        examples: list[extty.Example] = []

        max_val = (
            train_params.val_episodes
            if train_params.val_episodes is not None
            else len(val_rollout_data)
        )
        n_episodes = 0
        while n_episodes < max_val:
            current_batch = min(train_params.val_batch_size, max_val - n_episodes)
            indices = torch.randint(len(val_rollout_data), (current_batch,)).tolist()
            batch_prompts = [val_rollout_data[i] for i in indices]

            if inverse_cot_params.train_group_size is not None:
                batch_prompts = [
                    _subsample_completions(pr, inverse_cot_params.train_group_size)
                    for pr in batch_prompts
                ]

            input_ids, prefix_lengths, loss_mask, is_correct, group_sizes = (
                _build_contrastive_batch(
                    batch_prompts,
                    model_info.eos_token_id,
                    model_info.pad_token_id,
                    device,
                )
            )

            _, step_metrics = _compute_contrastive_loss(
                q,
                input_ids,
                prefix_lengths,
                loss_mask,
                is_correct,
                group_sizes,
                inverse_cot_params.contrastive_weight,
            )
            all_nll.append(step_metrics["train/nll"])

            _, _, per_sample_nlls = _compute_nll_loss(
                q,
                input_ids,
                prefix_lengths,
                loss_mask,
                inverse_cot_params.normalize_by_sequence_length,
            )
            for sample_nll, correct in zip(per_sample_nlls, is_correct.tolist()):
                if correct:
                    all_nll_correct.append(sample_nll)
                else:
                    all_nll_incorrect.append(sample_nll)

            # qualitative examples (cap at 20)
            if len(examples) < 20:
                offset = 0
                for prompt_data, gs in zip(batch_prompts, group_sizes):
                    if len(examples) >= 20:
                        break
                    prefix_len = prefix_lengths[offset].item()
                    prefix_ids = input_ids[offset, :prefix_len].unsqueeze(0)

                    q_completion = generate_hard_tokens(
                        net=q,
                        token_ids=prefix_ids,
                        sampling_strategy="greedy",
                        eos_token_id=model_info.eos_token_id,
                        pad_token_id=model_info.pad_token_id,
                        max_tokens_generated=train_params.max_tokens_generated,
                        use_kv_cache=True,
                        use_bf16=train_params.use_bf16,
                    ).tokens

                    q_cot = tokenizer.decode(q_completion[0, prefix_len:].tolist())
                    target_cot = tokenizer.decode(prompt_data.completions[0].cot_ids)
                    prompt_str = tokenizer.decode(
                        input_ids[offset, :prefix_len].tolist()
                    )

                    examples.append(
                        extty.Example(
                            prompt=prompt_str,
                            responses=[
                                f"[p's CoT] {target_cot}",
                                f"[q's CoT] {q_cot}",
                            ],
                            rewards=[
                                {"nll": step_metrics["train/nll"]},
                                {"nll": step_metrics["train/nll"]},
                            ],
                        )
                    )
                    offset += gs

            n_episodes += current_batch

        nll_mean = sum(all_nll) / len(all_nll) if all_nll else 0.0
        nll_correct_mean = (
            sum(all_nll_correct) / len(all_nll_correct) if all_nll_correct else 0.0
        )
        nll_incorrect_mean = (
            sum(all_nll_incorrect) / len(all_nll_incorrect)
            if all_nll_incorrect
            else 0.0
        )
        return nll_mean, nll_correct_mean, nll_incorrect_mean, examples

    @torch.no_grad()
    def _val_fn(
        val_env: Env,
    ) -> tuple[float, float, float, float, list[extty.Example]]:
        q.eval()

        all_nll: list[float] = []
        all_nll_correct: list[float] = []
        all_nll_incorrect: list[float] = []
        all_nll_shuffled: list[float] = []
        examples: list[extty.Example] = []

        max_val = (
            train_params.val_episodes
            if train_params.val_episodes is not None
            else len(val_countdown)
        )
        n_episodes = 0
        while n_episodes < max_val:
            current_batch = min(train_params.val_batch_size, max_val - n_episodes)

            data = _generate_training_data(
                p=p,
                env=val_env,
                state_to_str=state_to_str,
                tokenizer=tokenizer,
                eos_token_id=model_info.eos_token_id,
                pad_token_id=model_info.pad_token_id,
                batch_size=current_batch,
                temperature=train_params.temperature,
                max_tokens_generated=train_params.max_tokens_generated,
                use_bf16=train_params.use_bf16,
                correct_only=False,
            )
            if data is None:
                n_episodes += current_batch
                continue

            _, nll, per_sample_nlls = _compute_nll_loss(
                q,
                data.input_ids,
                data.prefix_lengths,
                data.loss_mask,
                inverse_cot_params.normalize_by_sequence_length,
            )
            all_nll.append(nll)
            for sample_nll, correct in zip(per_sample_nlls, data.is_correct):
                if correct:
                    all_nll_correct.append(sample_nll)
                else:
                    all_nll_incorrect.append(sample_nll)

            # shuffled answer diagnostic: keep same prompt and CoT,
            # roll only the answer tokens by 1 within batch
            B = len(data.prompt_token_ids)
            if B > 1:
                shuffled_answer_ids = (
                    data.answer_token_ids[1:] + data.answer_token_ids[:1]
                )
                shuffled_all_ids = [
                    p + a + c + [model_info.eos_token_id]
                    for p, a, c in zip(
                        data.prompt_token_ids, shuffled_answer_ids, data.cot_token_ids
                    )
                ]
                s_max_len = max(len(ids) for ids in shuffled_all_ids)
                s_input_ids = torch.full(
                    (B, s_max_len), model_info.pad_token_id, device=device
                )
                for i, ids in enumerate(shuffled_all_ids):
                    s_input_ids[i, : len(ids)] = torch.tensor(ids, device=device)
                s_prefix_lengths = torch.tensor(
                    [
                        len(p) + len(a)
                        for p, a in zip(data.prompt_token_ids, shuffled_answer_ids)
                    ],
                    dtype=torch.long,
                    device=device,
                )
                positions_s = torch.arange(s_max_len, device=device).unsqueeze(0)
                cot_lengths = data.loss_mask.sum(dim=1)
                s_loss_mask = (positions_s >= s_prefix_lengths.unsqueeze(1)) & (
                    positions_s < (s_prefix_lengths + cot_lengths).unsqueeze(1)
                )
                _, nll_shuffled, _ = _compute_nll_loss(
                    q,
                    s_input_ids,
                    s_prefix_lengths,
                    s_loss_mask,
                    inverse_cot_params.normalize_by_sequence_length,
                )
                all_nll_shuffled.append(nll_shuffled)

            # qualitative: generate CoT from q for a few examples (cap at 20)
            if len(examples) < 20:
                for i in range(min(B, 20 - len(examples))):
                    prefix_len = data.prefix_lengths[i].item()
                    prefix_ids = data.input_ids[i, :prefix_len].unsqueeze(0)

                    q_completion = generate_hard_tokens(
                        net=q,
                        token_ids=prefix_ids,
                        sampling_strategy="greedy",
                        eos_token_id=model_info.eos_token_id,
                        pad_token_id=model_info.pad_token_id,
                        max_tokens_generated=train_params.max_tokens_generated,
                        use_kv_cache=True,
                        use_bf16=train_params.use_bf16,
                    ).tokens

                    q_cot = tokenizer.decode(q_completion[0, prefix_len:].tolist())

                    target_cot = tokenizer.decode(data.cot_token_ids[i])

                    prompt_str = tokenizer.decode(
                        data.input_ids[i, :prefix_len].tolist()
                    )
                    examples.append(
                        extty.Example(
                            prompt=prompt_str,
                            responses=[
                                f"[p's CoT] {target_cot}",
                                f"[q's CoT] {q_cot}",
                            ],
                            rewards=[{"nll": nll}, {"nll": nll}],
                        )
                    )

            n_episodes += current_batch

        nll_mean = sum(all_nll) / len(all_nll) if all_nll else 0.0
        nll_correct_mean = (
            sum(all_nll_correct) / len(all_nll_correct) if all_nll_correct else 0.0
        )
        nll_incorrect_mean = (
            sum(all_nll_incorrect) / len(all_nll_incorrect)
            if all_nll_incorrect
            else 0.0
        )
        nll_shuffled_mean = (
            sum(all_nll_shuffled) / len(all_nll_shuffled) if all_nll_shuffled else 0.0
        )
        return (
            nll_mean,
            nll_correct_mean,
            nll_incorrect_mean,
            nll_shuffled_mean,
            examples,
        )

    @torch.no_grad()
    def _run_validation(step: int):
        was_training = q.training
        q.eval()
        metrics: dict[str, Any] = {}

        # offline val (same code path as training)
        if val_rollout_data is not None:
            nll_mean, nll_correct, nll_incorrect, examples = _val_fn_offline()
            metrics["val/offline/nll_mean"] = nll_mean
            metrics["val/offline/nll_correct"] = nll_correct
            metrics["val/offline/nll_incorrect"] = nll_incorrect
            if examples:
                metrics["val/offline/example"] = extty.BatchExample(
                    prompts=[e.prompt for e in examples],
                    responses=[e.responses for e in examples],
                    rewards=[e.rewards for e in examples],
                )

        # online val (fresh generation from p)
        all_nll: list[float] = []
        all_nll_correct: list[float] = []
        all_nll_incorrect: list[float] = []
        all_nll_shuffled: list[float] = []

        for val_env in val_envs:
            val_env.reseed()
            label = str(val_env)
            nll_mean, nll_correct, nll_incorrect, nll_shuffled_mean, examples = _val_fn(
                val_env
            )
            metrics[f"val/{label}/nll_mean"] = nll_mean
            metrics[f"val/{label}/nll_correct"] = nll_correct
            metrics[f"val/{label}/nll_incorrect"] = nll_incorrect
            metrics[f"val/{label}/nll_shuffled_mean"] = nll_shuffled_mean
            if examples:
                metrics[f"val/{label}/example"] = extty.BatchExample(
                    prompts=[e.prompt for e in examples],
                    responses=[e.responses for e in examples],
                    rewards=[e.rewards for e in examples],
                )
            all_nll.append(nll_mean)
            all_nll_correct.append(nll_correct)
            all_nll_incorrect.append(nll_incorrect)
            all_nll_shuffled.append(nll_shuffled_mean)

        if all_nll:
            metrics["val/nll_mean"] = sum(all_nll) / len(all_nll)
            metrics["val/nll_correct"] = sum(all_nll_correct) / len(all_nll_correct)
            metrics["val/nll_incorrect"] = sum(all_nll_incorrect) / len(
                all_nll_incorrect
            )
        if all_nll_shuffled:
            metrics["val/nll_shuffled_mean"] = sum(all_nll_shuffled) / len(
                all_nll_shuffled
            )

        if was_training:
            q.train()
        if extty.has_active_run():
            extty.log(metrics, step=step)

    n_episodes = 0
    step = 0
    while n_episodes < train_params.max_episodes:
        start_time = time.perf_counter()
        step_ret = _train_step()
        step_time = time.perf_counter() - start_time
        step += 1
        n_episodes += step_ret.n_episodes_processed

        if extty.has_active_run():
            metrics = step_ret.metrics
            metrics["step_time"] = step_time
            extty.log(metrics, step=step)

            if step % train_params.save_ckpt_freq == 0:
                extty.save_checkpoint(
                    step=step,
                    state_dict=q.state_dict(),
                    optimizer_state_dict=opt.state_dict(),
                )
            if train_params.val_freq > 0 and step % train_params.val_freq == 0:
                _run_validation(step)

    if step % train_params.save_ckpt_freq != 0 and extty.has_active_run():
        extty.save_checkpoint(
            step=step,
            state_dict=q.state_dict(),
            optimizer_state_dict=opt.state_dict(),
        )


if __name__ == "__main__":
    run_experiments_parser(
        [
            Experiment(
                env_name="countdown",
                fn=train_inverse_cot_countdown,
                include_prompt_collection_id=True,
                include_dataset_glob=True,
            ),
        ]
    )
