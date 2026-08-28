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


def resolve_lora_targets(target_modules: str) -> tuple[str, ...]:
    if target_modules == "all":
        return DEFAULT_TARGET_MODULES
    if target_modules == "attn":
        return ("q_proj", "k_proj", "v_proj", "o_proj")
    if target_modules == "mlp":
        return ("gate_proj", "up_proj", "down_proj")
    raise ValueError(
        f"Unknown lora_target_modules: {target_modules!r} (expected 'all', 'attn', or 'mlp')"
    )


def freeze_base_params(net: nn.Module) -> None:
    for name, param in net.named_parameters():
        param.requires_grad_("lora_" in name)


def get_lora_params(net: nn.Module) -> list[nn.Parameter]:
    return [p for p in net.parameters() if p.requires_grad]


def get_lora_state_dict(net: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v for k, v in net.state_dict().items() if "lora_" in k}


def merged_state_dict(net: nn.Module) -> dict[str, torch.Tensor]:
    """Return a plain-keyed state dict with LoRA deltas merged into the base.

    Unlike :func:`merge_lora`, ``net`` is not mutated — safe to call inside a
    training loop (e.g. to push merged weights into a vLLM engine between
    optimizer steps). Adapter keys (``lora_A``/``lora_B``) are dropped and
    every ``<prefix>.base.<param>`` key is rewritten to ``<prefix>.<param>``,
    so the result matches the state dict of the un-wrapped model.

    Parameters
    ----------
    net
        Module tree, with or without ``LoRALinear`` layers. Without them the
        state dict passes through unchanged.

    Returns
    -------
    dict[str, torch.Tensor]
        State dict keyed as if ``apply_lora`` had never run. Merged weight
        tensors are freshly allocated; all other tensors are the live ones
        from ``net.state_dict()``.
    """
    lora_modules = {
        name: module
        for name, module in net.named_modules()
        if isinstance(module, LoRALinear)
    }
    sd: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for key, value in net.state_dict().items():
            prefix, sep, rest = key.rpartition(".base.")
            if sep and prefix in lora_modules:
                if rest == "weight":
                    module = lora_modules[prefix]
                    sd[f"{prefix}.weight"] = value + module.scaling * (
                        module.lora_B.weight @ module.lora_A.weight
                    )
                else:
                    sd[f"{prefix}.{rest}"] = value
            elif ".lora_A." in key or ".lora_B." in key:
                continue
            else:
                sd[key] = value
    return sd


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
