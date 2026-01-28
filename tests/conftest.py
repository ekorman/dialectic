import pytest
from tokenizers import Tokenizer

from dialectic.llm.qwen import Qwen
from dialectic.rl.env import CountdownEnv


@pytest.fixture
def tiny_model():
    return Qwen(
        d=32,
        vocab_size=151936,
        n_decoder_layers=2,
        attn_head_d=16,
        attn_num_heads=4,
        attn_num_kv_heads=2,
        mlp_hidden_d=64,
    )


@pytest.fixture
def env():
    return CountdownEnv()


@pytest.fixture
def tokenizer():
    return Tokenizer.from_pretrained("Qwen/Qwen3-0.6B")
