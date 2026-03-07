import torch

from dialectic.experiments.params import TrainParams
from dialectic.llm.base import BaseTransformer
from dialectic.llm.registry import MODEL_REGISTRY


def load_model_and_opt(
    *, train_params: TrainParams, **kwargs
) -> tuple[BaseTransformer, torch.optim.Optimizer]:
    model_info = MODEL_REGISTRY[train_params.model_name]
    net = model_info.load_net(**kwargs)
    if train_params.compile_model:
        net.compile()
    opt = torch.optim.AdamW(net.parameters(), lr=train_params.lr)
    return net, opt
