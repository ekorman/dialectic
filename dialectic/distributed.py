import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from dialectic.llm.utils import get_default_device

_distributed_active = False


def init_distributed() -> bool:
    """Initialize distributed training if RANK env var is set (e.g. via torchrun).

    Idempotent — safe to call multiple times. Returns whether distributed mode
    is active.
    """
    global _distributed_active
    if _distributed_active:
        return True
    if "RANK" not in os.environ:
        return False

    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    _distributed_active = True
    return True


def is_distributed() -> bool:
    return _distributed_active


def get_rank() -> int:
    return dist.get_rank() if _distributed_active else 0


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0")) if _distributed_active else 0


def get_world_size() -> int:
    return dist.get_world_size() if _distributed_active else 1


def is_main_process() -> bool:
    return get_rank() == 0


def get_device() -> torch.device:
    if _distributed_active:
        return torch.device("cuda", get_local_rank())
    return torch.device(get_default_device())


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def wrap_ddp(
    model: nn.Module,
    device_id: int,
    *,
    find_unused_parameters: bool = False,
) -> DDP:
    return DDP(
        model,
        device_ids=[device_id],
        find_unused_parameters=find_unused_parameters,
    )


def all_gather_rewards(rewards: torch.Tensor) -> torch.Tensor:
    """All-gather reward tensor across ranks along dim=1.

    Parameters
    ----------
    rewards
        Shape ``[G, local_B]``. All ranks must have the same shape.

    Returns
    -------
    torch.Tensor
        Shape ``[G, local_B * world_size]`` with rewards from all ranks
        concatenated along the batch dimension.
    """
    if not _distributed_active:
        return rewards
    gathered = [torch.zeros_like(rewards) for _ in range(get_world_size())]
    dist.all_gather(gathered, rewards)
    return torch.cat(gathered, dim=1)


def barrier() -> None:
    if _distributed_active:
        dist.barrier()


def cleanup() -> None:
    if _distributed_active:
        dist.destroy_process_group()
