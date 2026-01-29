import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

from dialectic.llm.components import GatedMLP


def test_gated_mlp():
    b, l, d, hidden_d = 4, 6, 20, 32
    x = torch.rand(b, l, d)

    conf = Qwen3Config()
    conf.hidden_size = d
    conf.intermediate_size = hidden_d

    mlp1 = GatedMLP(d, hidden_d)
    mlp2 = Qwen3MLP(conf)

    mlp1.load_state_dict(mlp2.state_dict())

    torch.testing.assert_close(mlp1(x), mlp2(x))
