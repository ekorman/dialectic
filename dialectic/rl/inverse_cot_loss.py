from typing import cast

import torch
import torch.nn.functional as F

from dialectic.distributed import unwrap_model
from dialectic.llm.inverse_cot import InverseCotModel, create_prefix_lm_mask


def compute_nll_loss(
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


def compute_contrastive_loss(
    q: InverseCotModel,
    input_ids: torch.Tensor,
    prefix_lengths: torch.Tensor,
    loss_mask: torch.Tensor,
    is_correct: torch.Tensor,
    group_sizes: list[int],
    contrastive_weight: float,
    contrastive_margin: float = 1.0,
    logprob_chunk_size: int = 64,
) -> tuple[torch.Tensor, dict[str, float]]:
    """NLL (correct-only) + margin contrastive loss over grouped completions.

    Expects the batch layout produced by
    :func:`dialectic.rl.inverse_cot_data.build_contrastive_batch`: each
    prompt contributes all of its completions (correct and incorrect) as
    natural ``(P, A_i, C_i)`` sequences.

    - **NLL term**: mean per-sequence NLL over correct completions only,
      averaged across prompts.
    - **Contrastive term**: per prompt, ``max(0, margin - (mean_correct_logprob
      - mean_incorrect_logprob))``. Pushes the model to assign at least
      ``margin`` nats/token higher average logprob to correct rollouts
      than incorrect ones. Unlike the logsumexp-ratio formulation, this
      keeps providing gradient even after the model can discriminate,
      as long as the gap is below the margin.

    Parameters
    ----------
    q
        The backward model ``q_phi``.
    input_ids, prefix_lengths, loss_mask, is_correct
        Outputs from ``build_contrastive_batch``.
    group_sizes
        Number of completions per prompt (from ``build_contrastive_batch``).
    contrastive_weight
        ``λ`` in the combined loss.
    contrastive_margin
        Target gap in per-token average logprob between correct and
        incorrect rollouts.

    Returns
    -------
    tuple[torch.Tensor, dict[str, float]]
        ``(total_loss, metrics)`` where metrics reports ``train/nll``,
        ``train/contrastive_loss``, ``train/contrastive_gap``, and
        ``train/loss``.
    """
    seq_len = input_ids.shape[1]
    device = input_ids.device

    attention_mask = create_prefix_lm_mask(prefix_lengths, seq_len, device)

    hidden_states = q(
        input_ids, attention_mask=attention_mask, return_hidden_states=True
    )
    raw_q = cast(InverseCotModel, unwrap_model(q))

    shift_hidden = hidden_states[:, :-1]
    shift_targets = input_ids[:, 1:]
    shift_mask = loss_mask[:, 1:]

    N, L_minus_1, _ = shift_hidden.shape
    seq_lengths = shift_mask.sum(dim=1).clamp(min=1)
    per_seq_total_nll = torch.zeros(N, device=device, dtype=torch.float32)

    chunk = logprob_chunk_size if logprob_chunk_size > 0 else L_minus_1
    for start in range(0, L_minus_1, chunk):
        end = min(start + chunk, L_minus_1)
        chunk_hidden = shift_hidden[:, start:end]
        chunk_targets = shift_targets[:, start:end]
        chunk_mask = shift_mask[:, start:end]

        chunk_logits = raw_q.lm_head(chunk_hidden)
        N_c, L_c, V = chunk_logits.shape
        chunk_token_nll = F.cross_entropy(
            chunk_logits.reshape(N_c * L_c, V).float(),
            chunk_targets.reshape(N_c * L_c),
            reduction="none",
        ).reshape(N_c, L_c)
        masked = chunk_token_nll * chunk_mask
        per_seq_total_nll = per_seq_total_nll + masked.sum(dim=1)
        del chunk_logits

    per_seq_avg_nll = per_seq_total_nll / seq_lengths
    per_seq_avg_logprob = -per_seq_avg_nll

    nll_losses: list[torch.Tensor] = []
    contrastive_losses: list[torch.Tensor] = []
    contrastive_gaps: list[torch.Tensor] = []
    offset = 0
    for gs in group_sizes:
        group_nll = per_seq_avg_nll[offset : offset + gs]
        group_logprob = per_seq_avg_logprob[offset : offset + gs]
        group_correct = is_correct[offset : offset + gs]

        if group_correct.any():
            nll_losses.append(group_nll[group_correct].mean())

        if contrastive_weight > 0 and group_correct.any() and not group_correct.all():
            correct_mean = group_logprob[group_correct].mean()
            incorrect_mean = group_logprob[~group_correct].mean()
            gap = correct_mean - incorrect_mean
            contrastive_gaps.append(gap.detach())
            contrastive_losses.append(torch.clamp(contrastive_margin - gap, min=0.0))

        offset += gs

    nll_loss = (
        torch.stack(nll_losses).mean()
        if nll_losses
        else torch.tensor(0.0, device=device, requires_grad=True)
    )
    contrastive_loss = (
        torch.stack(contrastive_losses).mean()
        if contrastive_losses
        else torch.tensor(0.0, device=device, requires_grad=True)
    )
    contrastive_gap = (
        sum(g.item() for g in contrastive_gaps) / len(contrastive_gaps)
        if contrastive_gaps
        else 0.0
    )
    total_loss = nll_loss + contrastive_weight * contrastive_loss

    metrics = {
        "train/nll": nll_loss.item(),
        "train/contrastive_loss": contrastive_loss.item(),
        "train/contrastive_gap": contrastive_gap,
        "train/loss": total_loss.item(),
    }
    return total_loss, metrics
