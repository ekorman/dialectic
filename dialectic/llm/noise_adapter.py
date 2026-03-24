import torch
import torch.nn as nn
from jaxtyping import Float


class NoiseAdapter(nn.Module):
    """Transforms raw noise vectors into adapted representations via cross-attention.

    The adapter cross-attends to context embeddings (prompt + hard tokens so far)
    to produce noise representations that the frozen transformer can usefully
    attend to.

    Parameters
    ----------
    d_model
        Model hidden dimension.
    n_heads
        Number of cross-attention heads.
    d_ff
        Feed-forward hidden dimension.
    """

    def __init__(self, d_model: int, n_heads: int = 4, d_ff: int = 256):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        noise: Float[torch.Tensor, "B k D"],
        context: Float[torch.Tensor, "B L D"],
    ) -> Float[torch.Tensor, "B k D"]:
        h, _ = self.cross_attn(noise, context, context)
        h = self.norm1(noise + h)
        h = self.norm2(h + self.ffn(h))
        return h
