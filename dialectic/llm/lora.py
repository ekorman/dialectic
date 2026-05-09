import math

import torch
import torch.nn as nn

DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base = base
        self.scaling = alpha / rank
        device = base.weight.device
        dtype = base.weight.dtype
        self.lora_A = nn.Linear(
            base.in_features, rank, bias=False, device=device, dtype=dtype
        )
        self.lora_B = nn.Linear(
            rank, base.out_features, bias=False, device=device, dtype=dtype
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # rank-stabilized init: scale kaiming by 1/sqrt(rank) so that
        # ||grad_B|| doesn't grow with rank during the B=0 warmup phase
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        self.lora_A.weight.data /= math.sqrt(rank)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.base(x) + self.scaling * self.lora_B(self.lora_A(self.dropout(x)))


def apply_lora(
    net: nn.Module,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES,
) -> nn.Module:
    for _, module in net.named_modules():
        for attr_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and attr_name in target_modules:
                setattr(
                    module,
                    attr_name,
                    LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout),
                )
    return net


def freeze_base_params(net: nn.Module) -> None:
    for name, param in net.named_parameters():
        param.requires_grad_("lora_" in name)


def get_lora_params(net: nn.Module) -> list[nn.Parameter]:
    return [p for p in net.parameters() if p.requires_grad]


def get_lora_state_dict(net: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v for k, v in net.state_dict().items() if "lora_" in k}


def merge_lora(net: nn.Module) -> nn.Module:
    """Merge LoRA weights into base and replace LoRALinear with plain Linear."""
    for _, module in net.named_modules():
        for attr_name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                merged = child.base
                merged.weight.data += (
                    child.scaling * child.lora_B.weight @ child.lora_A.weight
                )
                setattr(module, attr_name, merged)
    return net
