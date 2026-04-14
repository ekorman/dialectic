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
    valid_mask: torch.Tensor,
    group_size: int,
    n_negatives: int,
    contrastive_weight: float,
    contrastive_temperature: float,
    logprob_chunk_size: int = 64,
) -> tuple[torch.Tensor, dict[str, float]]:
    """NLL + InfoNCE contrastive loss over ``(C_i, A_j)`` pairs.

    Expects the flat batch layout produced by
    :func:`dialectic.rl.inverse_cot_data.build_infonce_batch`: a tensor of
    shape ``[B * K * (1+M), L]`` where ``B`` is the number of prompts,
    ``K = group_size`` rollouts per prompt, and ``M = n_negatives``
    contrastive negatives per anchor. Slot 0 of each anchor is the
    positive pair ``(P, A_i, C_i)``; slots 1..M are negative pairs
    ``(P, A_j, C_i)`` where ``A_j`` comes from a rollout with a different
    answer token sequence.

    The two loss terms from paper_spec.md § "Training q_phi":

    - **NLL term**: mean per-token negative log-likelihood over the
      positive (slot-0) sequences, averaged across the K anchors of each
      prompt and across prompts in the batch.
    - **InfoNCE term**: for each anchor, softmax over ``(1+M)`` candidate
      scores ``s(C_i, A_j) / τ``, minimizing the negative log-likelihood
      of the positive slot. Invalid slots (padding for prompts with fewer
      than M distinct other answers) are masked to ``-inf`` in the
      denominator. Anchors whose entire negative set is invalid (the
      degenerate "all rollouts share one answer" case) contribute zero to
      the contrastive loss, exactly as the spec describes.

    Parameters
    ----------
    q
        The backward model ``q_phi``.
    input_ids, prefix_lengths, loss_mask, valid_mask
        Outputs from ``build_infonce_batch``.
    group_size
        ``K`` — rollouts per prompt. Used to reshape the flat batch.
    n_negatives
        ``M`` — contrastive negatives per anchor. Used to reshape the flat
        batch; also determines the softmax width (``1 + M``).
    contrastive_weight
        ``λ`` in the combined loss. Setting to ``0`` recovers NLL-only
        training (ablation A2 in the paper).
    contrastive_temperature
        ``τ`` in the InfoNCE softmax.

    Returns
    -------
    tuple[torch.Tensor, dict[str, float]]
        ``(total_loss, metrics)`` where metrics reports ``train/nll``,
        ``train/contrastive_loss``, ``train/loss``, and
        ``train/contrastive_frac_valid_anchors`` (the fraction of anchors
        across the batch that had at least one valid negative — useful to
        watch for runs where diversity collapses).
    """
    seq_len = input_ids.shape[1]
    device = input_ids.device

    attention_mask = create_prefix_lm_mask(prefix_lengths, seq_len, device)

    # Get hidden states, not logits. Materializing the full
    # [N, L, V=151936] logits tensor dies on memory at the Option B batch
    # sizes (N = batch_size * group_size * (1 + n_negatives)). Instead,
    # run the transformer once to get [N, L, D] hidden states, then
    # chunk the lm_head projection + cross-entropy over the L dimension
    # so the peak logits footprint is bounded by chunk_size.
    hidden_states = q(
        input_ids, attention_mask=attention_mask, return_hidden_states=True
    )

    # Standard causal-LM shift: the loss at position t predicts token t+1.
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

        chunk_logits = q.lm_head(chunk_hidden)  # [N, chunk, V]
        N_c, L_c, V = chunk_logits.shape
        # Cast to fp32 for numerically stable softmax over the full Qwen
        # vocab (~152k). Matches `_compute_log_probs_chunked` in the GRPO
        # path — doubles the per-chunk logits footprint from bf16 to fp32,
        # but chunk_size is small (default 64) so this is still well under
        # the unchunked bf16 tensor we replaced.
        chunk_token_nll = F.cross_entropy(
            chunk_logits.reshape(N_c * L_c, V).float(),
            chunk_targets.reshape(N_c * L_c),
            reduction="none",
        ).reshape(N_c, L_c)
        masked = chunk_token_nll * chunk_mask
        per_seq_total_nll = per_seq_total_nll + masked.sum(dim=1)
        del chunk_logits  # free this chunk's vocab projection before the next

    per_seq_avg_nll = per_seq_total_nll / seq_lengths
    per_seq_avg_logprob = -per_seq_avg_nll

    slots = 1 + n_negatives
    if N % (group_size * slots) != 0:
        raise RuntimeError(
            f"compute_contrastive_loss: flat batch size {N} not divisible by "
            f"group_size * (1 + n_negatives) = {group_size * slots}. "
            f"Did build_infonce_batch run with the same group_size and n_negatives?"
        )
    batch_size = N // (group_size * slots)

    per_seq_nll_bks = per_seq_avg_nll.view(batch_size, group_size, slots)
    per_seq_logprob_bks = per_seq_avg_logprob.view(batch_size, group_size, slots)
    valid_bks = valid_mask.view(batch_size, group_size, slots)

    # NLL: mean over positives only (slot 0). Three normalizations — token
    # (inside per_seq_avg_nll), per-prompt rollouts (mean over K), batch
    # (mean over B) — fall out of a flat .mean() since K is uniform.
    positive_nll = per_seq_nll_bks[:, :, 0]
    nll_loss = positive_nll.mean()

    # InfoNCE: softmax per anchor over all (1+M) candidates.
    scores = per_seq_logprob_bks / contrastive_temperature
    scores_masked = scores.masked_fill(~valid_bks, float("-inf"))
    log_partition = torch.logsumexp(scores_masked, dim=-1)
    log_positive = scores[:, :, 0]
    per_anchor_nce = -(log_positive - log_partition)

    # Anchors with no valid negatives contribute zero gradient (degenerate
    # all-same-answer prompts, per spec). Zero-out those anchors' NCE and
    # renormalize against the number of anchors that had at least one real
    # negative so a prompt with only invalid slots doesn't drag the average
    # toward zero.
    has_valid_neg = valid_bks[:, :, 1:].any(dim=-1)
    n_valid_anchors = has_valid_neg.sum().clamp(min=1)
    contrastive_loss = (per_anchor_nce * has_valid_neg.float()).sum() / n_valid_anchors

    # Fraction of allocated negative slots (B × K × M) that were filled with
    # real distinct-other-answer rollouts rather than invalid padding. This
    # tells you how much of the nominal `n_negatives` budget the contrastive
    # softmax is actually using — if it's near 1, M=4 is doing full work;
    # if it's ~0.25, most slots are masked to -inf and you're paying forward
    # cost for denominator entries the model isn't actually seeing. Drives
    # the decision to drop M. The old "frac of anchors with ≥1 negative"
    # metric was tautologically 1.0 for any dataset that passed an n_pos/
    # n_neg filter at generation time, so it's been replaced.
    negatives_filled_frac = valid_bks[:, :, 1:].float().mean()

    total_loss = nll_loss + contrastive_weight * contrastive_loss

    metrics = {
        "train/nll": nll_loss.item(),
        "train/contrastive_loss": contrastive_loss.item(),
        "train/loss": total_loss.item(),
        "train/contrastive_negatives_filled_frac": negatives_filled_frac.item(),
    }
    return total_loss, metrics
