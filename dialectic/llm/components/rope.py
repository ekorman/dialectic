import torch
from jaxtyping import Float
from torch import Tensor

# for our RoPE implementation we follow huggingface where they split the vector into first and second half
# instead of interleaving. this contrasts with the paper where adjacent components in the vector
# dimension are paired
#
# c.f.: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py
# https://github.com/huggingface/transformers/issues/25199
# https://github.com/rasbt/LLMs-from-scratch/pull/747
# https://github.com/rasbt/LLMs-from-scratch/issues/751


def apply_rope(
    x: Float[Tensor, "B NH L D"],
    sin: Float[Tensor, "B L D"],
    cos: Float[Tensor, "B L D"],
) -> Float[Tensor, "B NH L D"]:
    rot = torch.cat((-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1)
    # sin/cos should already be on correct device (cached as buffers)
    return x * cos.expand(x.shape[0], x.shape[2], x.shape[3]).unsqueeze(
        1
    ) + rot * sin.expand(x.shape[0], x.shape[2], x.shape[3]).unsqueeze(1)


def create_rope_sine_cosine_tensors(
    dim: int,
    base_value: float,
    context_length: int,
    device: str | torch.device | None = None,
) -> tuple[Float[Tensor, "1 L D"], Float[Tensor, "1 L D"]]:
    thetas = base_value ** (-2 * (torch.arange(dim // 2, device=device)) / dim)
    thetas = thetas.repeat(2)

    freqs = torch.outer(torch.arange(context_length, device=device), thetas)

    sin = freqs.sin().view(1, context_length, dim)
    cos = freqs.cos().view(1, context_length, dim)

    return sin, cos
