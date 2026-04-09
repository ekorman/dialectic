import torch
import torch.nn.functional as F

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

    seq_lengths = shift_mask.sum(dim=1).clamp(min=1)
    per_seq_avg_nll = (per_token_loss * shift_mask).sum(dim=1) / seq_lengths
    per_seq_avg_logprob = -per_seq_avg_nll

    nll_losses: list[torch.Tensor] = []
    contrastive_losses: list[torch.Tensor] = []
    offset = 0
    for gs in group_sizes:
        group_nll = per_seq_avg_nll[offset : offset + gs]
        group_logprob = per_seq_avg_logprob[offset : offset + gs]
        group_correct = is_correct[offset : offset + gs]

        if group_correct.any():
            nll_losses.append(group_nll[group_correct].mean())

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
