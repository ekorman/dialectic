import torch
from jaxtyping import Bool, Int
from torch import Tensor


def create_prefix_lm_mask(
    prefix_lengths: Int[Tensor, " B"],
    seq_len: int,
    device: torch.device,
) -> Bool[Tensor, "B 1 L L"]:
    """Create a prefix-LM attention mask.

    Within the prefix: bidirectional (all-to-all).
    CoT tokens: attend to all prefix tokens + causally to prior CoT tokens.
    Prefix tokens: cannot see CoT tokens.
    """
    positions = torch.arange(seq_len, device=device)

    prefix_mask = positions.unsqueeze(0) < prefix_lengths.unsqueeze(1)

    is_prefix_query = prefix_mask.unsqueeze(2)
    is_prefix_key = prefix_mask.unsqueeze(1)

    # prefix-to-prefix: True (bidirectional)
    pp = is_prefix_query & is_prefix_key
    # cot-to-prefix: True
    cp = ~is_prefix_query & is_prefix_key
    # cot-to-cot: causal (j <= i)
    causal = positions.unsqueeze(0) <= positions.unsqueeze(1)
    cc = ~is_prefix_query & ~is_prefix_key & causal.unsqueeze(0)

    mask = pp | cp | cc
    return mask.unsqueeze(1)
