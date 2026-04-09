import random
import re
from dataclasses import dataclass

import extty
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from dialectic.experiments.arg_parser import Experiment, run_experiments_parser
from dialectic.experiments.models import load_model_and_opt
from dialectic.experiments.params import (
    InverseCotParams,
    TrainParams,
)
from dialectic.experiments.prompts import PromptCollection
from dialectic.llm.generate import generate_hard_tokens
from dialectic.llm.inverse_cot import InverseCotModel, create_prefix_lm_mask
from dialectic.llm.registry import MODEL_REGISTRY
from dialectic.log import log
from dialectic.rl.dataset_env import DatasetEnv
from dialectic.rl.env import Env
from dialectic.rl.evaluate import EvaluationResult
from dialectic.training import StepFunctionReturn, train_loop


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

    # load rollout data from artifacts, split by train/val
    by_split = _load_rollout_artifacts(dataset_artifacts, tokenizer)
    rollout_data = by_split.get("train", [])
    val_rollout_data = by_split.get("val", [])
    if not rollout_data:
        raise ValueError("No training data found (split='train')")

    # wrap val data as a dummy env for train_loop compatibility
    val_envs: list[Env] = (
        [DatasetEnv(val_rollout_data, seed=2026, label="inverse_cot_val")]
        if val_rollout_data
        else []
    )

    def _train_step(_step_idx: int) -> StepFunctionReturn:
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

    max_val = (
        train_params.val_episodes
        if train_params.val_episodes is not None
        else len(val_rollout_data)
    )

    def _val_fn(_val_env: Env) -> tuple[EvaluationResult, list[extty.Example]]:
        q.eval()

        all_nll: list[float] = []
        all_nll_correct: list[float] = []
        all_nll_incorrect: list[float] = []
        all_nll_shuffled: list[float] = []
        examples: list[extty.Example] = []

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

            # shuffled answer diagnostic: roll answer tokens by 1
            N = input_ids.shape[0]
            if N > 1:
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
                    pi + ai + ci + [model_info.eos_token_id]
                    for pi, ai, ci in zip(all_prompt_ids, shuffled_answer, all_cot_ids)
                ]
                s_max_len = max(len(s) for s in shuffled_seqs)
                s_input_ids = torch.full(
                    (N, s_max_len), model_info.pad_token_id, device=device
                )
                for i, ids in enumerate(shuffled_seqs):
                    s_input_ids[i, : len(ids)] = torch.tensor(ids, device=device)
                s_prefix_lengths = torch.tensor(
                    [
                        len(pi) + len(ai)
                        for pi, ai in zip(all_prompt_ids, shuffled_answer)
                    ],
                    dtype=torch.long,
                    device=device,
                )
                cot_lens = torch.tensor(
                    [len(ci) + 1 for ci in all_cot_ids],
                    dtype=torch.long,
                    device=device,
                )
                positions_s = torch.arange(s_max_len, device=device).unsqueeze(0)
                s_loss_mask = (positions_s >= s_prefix_lengths.unsqueeze(1)) & (
                    positions_s < (s_prefix_lengths + cot_lens).unsqueeze(1)
                )
                _, nll_shuffled, _ = _compute_nll_loss(
                    q,
                    s_input_ids,
                    s_prefix_lengths,
                    s_loss_mask,
                    inverse_cot_params.normalize_by_sequence_length,
                )
                all_nll_shuffled.append(nll_shuffled)

            # qualitative examples (cap at 20)
            if len(examples) < 20:
                ex_offset = 0
                for prompt_data, gs in zip(batch_prompts, group_sizes):
                    if len(examples) >= 20:
                        break
                    prefix_len = prefix_lengths[ex_offset].item()
                    prefix_ids = input_ids[ex_offset, :prefix_len].unsqueeze(0)

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
                        input_ids[ex_offset, :prefix_len].tolist()
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
                    ex_offset += gs

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

        return EvaluationResult(
            n_episodes=n_episodes,
            reward_mean=nll_mean,
            reward_std=0.0,
            component_means={
                "nll_correct": nll_correct_mean,
                "nll_incorrect": nll_incorrect_mean,
                "nll_shuffled": nll_shuffled_mean,
            },
        ), examples

    train_loop(
        max_episodes=train_params.max_episodes,
        save_ckpt_freq=train_params.save_ckpt_freq,
        val_freq=train_params.val_freq,
        net=q,
        opt=opt,
        train_step=_train_step,
        val_fn=_val_fn,
        val_envs=val_envs,
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
