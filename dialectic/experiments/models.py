import extty
import torch

from dialectic.experiments.params import TrainParams
from dialectic.llm.base import BaseTransformer
from dialectic.llm.registry import MODEL_REGISTRY
from dialectic.llm.utils import get_default_device
from dialectic.log import log


def load_model_and_opt(
    *, train_params: TrainParams, **kwargs
) -> tuple[BaseTransformer, torch.optim.Optimizer]:
    model_info = MODEL_REGISTRY[train_params.model_name]
    net = model_info.load_net(
        **kwargs, pretrained_weights=train_params.start_ckpt_run is None
    )
    ckpt = None
    if train_params.start_ckpt_run is not None:
        if train_params.start_ckpt_step is None:
            raise ValueError("`ckpt_step` cannot be none if `ckpt_run` is not None")
        log.info(
            f"Loading checkpoint from run {train_params.start_ckpt_run}, step {train_params.start_ckpt_step}"
        )
        project, run_name = train_params.start_ckpt_run.split("/")
        ckpt = extty.load_checkpoint_from(
            project=project, run_name=run_name, step=train_params.start_ckpt_step
        )
        net.load_state_dict(ckpt["model_state_dict"])

    if train_params.use_bf16:
        net = net.to(dtype=torch.bfloat16)
    if train_params.compile_model:
        net.compile()
    device = get_default_device()
    net = net.to(device)
    log.info(f"loaded net on device {device} (dtype={next(net.parameters()).dtype})")
    opt = torch.optim.AdamW(net.parameters(), lr=train_params.lr)

    if (
        train_params.load_ckpt_opt
        and ckpt is not None
        and "optimizer_state_dict" in ckpt
    ):
        opt.load_state_dict(ckpt["optimizer_state_dict"])
        log.info("Loaded optimizer state from checkpoint")

    return net, opt
