import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class GatedMLP(nn.Module):
    def __init__(self, d: int, hidden_d: int):
        super().__init__()
        self.gate_proj = nn.Linear(d, hidden_d, bias=False)
        self.up_proj = nn.Linear(d, hidden_d, bias=False)
        self.down_proj = nn.Linear(hidden_d, d, bias=False)

    def forward(self, x: Float[Tensor, "... d"]) -> Float[Tensor, "... d"]:
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))
