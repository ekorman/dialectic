import pytest
import torch
from tokenizers import Tokenizer

from dialectic.llm.qwen import create_qwen
from dialectic.rl.env import CountdownEnv


@pytest.fixture
def MockGenerateModel():
    class _MockGenerateModel(torch.nn.Module):
        def __init__(self, token_schedule: list[torch.Tensor], vocab_size: int):
            super().__init__()
            self.token_schedule = token_schedule
            self.vocab_size = vocab_size
            self.step = 0
            self.attn_num_kv_heads = 1
            self.attn_head_d = 1
            self.layers = torch.nn.ModuleList([torch.nn.Identity()])
            self.dummy_param = torch.nn.Parameter(torch.zeros(1))
            self.embed_tokens = torch.nn.Embedding(vocab_size, 2)

        def forward(
            self, input_ids, kv_caches=None, attention_mask=None, soft_token_noise=None
        ):
            tokens = self.token_schedule[self.step].to(input_ids.device)
            self.step += 1
            batch_size = input_ids.shape[0]
            assert tokens.shape[0] == batch_size
            logits = torch.full(
                (batch_size, 1, self.vocab_size),
                fill_value=-1e4,
                device=input_ids.device,
            )
            logits[torch.arange(batch_size), 0, tokens] = 0.0
            return logits

    return _MockGenerateModel


TINY_QWEN_KWARGS = dict(
    d=32,
    vocab_size=151936,
    n_decoder_layers=2,
    attn_head_d=16,
    attn_num_heads=4,
    attn_num_kv_heads=2,
    mlp_hidden_d=64,
    tie_weights=True,
)


@pytest.fixture
def tiny_model():
    torch.manual_seed(1000)
    return create_qwen(**TINY_QWEN_KWARGS)


@pytest.fixture
def tiny_model_with_soft_projection():
    torch.manual_seed(1000)
    return create_qwen(**TINY_QWEN_KWARGS, soft_projection=True)


@pytest.fixture
def env():
    return CountdownEnv()


@pytest.fixture
def tokenizer():
    return Tokenizer.from_pretrained("Qwen/Qwen3-0.6B")
