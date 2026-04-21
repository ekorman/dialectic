import extty
import torch

from dialectic.llm.base import BaseTransformer
from dialectic.llm.registry import MODEL_REGISTRY
from dialectic.llm.utils import get_default_device
from dialectic.log import log


def load_model_and_opt(
    *,
    model_name: str,
    start_ckpt_run: str | None = None,
    start_ckpt_step: int | None = None,
    device: str | torch.device | None = None,
    use_bf16: bool,
    compile_model: bool,
    load_opt: bool = True,
    lr: float | None = None,
    **kwargs,
) -> tuple[BaseTransformer, torch.optim.Optimizer | None]:
    model_info = MODEL_REGISTRY[model_name]
    net = model_info.load_net(**kwargs, pretrained_weights=start_ckpt_run is None)
    ckpt = None
    if start_ckpt_run is not None:
        if start_ckpt_step is None:
            raise ValueError("`ckpt_step` cannot be none if `ckpt_run` is not None")
        log.info(
            f"Loading checkpoint from run {start_ckpt_run}, step {start_ckpt_step}"
        )
        project, run_name = start_ckpt_run.split("/")
        ckpt = extty.load_checkpoint_from(
            project=project,
            run_name=run_name,
            step=start_ckpt_step,
            load_optimizer=load_opt,
        )
        net.load_state_dict(ckpt["model_state_dict"])

    if use_bf16:
        net = net.to(dtype=torch.bfloat16)
    if compile_model:
        net.compile()
    if device is None:
        device = get_default_device()
    net = net.to(device)
    log.info(f"loaded net on device {device} (dtype={next(net.parameters()).dtype})")

    if load_opt:
        if lr is None:
            raise ValueError("lr must be set if `load_opt` is True")
        opt = torch.optim.AdamW(net.parameters(), lr=lr)

        if ckpt is not None and "optimizer_state_dict" in ckpt:
            opt.load_state_dict(ckpt["optimizer_state_dict"])
            log.info("Loaded optimizer state from checkpoint")
    else:
        opt = None

    return net, opt
