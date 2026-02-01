import torch

from dialectic.llm.base import BaseTransformer
from dialectic.llm.generate import generate_from_tokens


@torch.inference_mode()
def test_soft_tokens_forward_pass(tiny_model: BaseTransformer):
    """Tests that the forward pass gives the same thing if we pass an integer tensor
    of token ids or the corresponding one-hot float tensor
    """
    b, l = 4, 6
    token_ids = torch.randint(0, tiny_model.vocab_size, (b, l))
    one_hot_tokens = torch.nn.functional.one_hot(
        token_ids, tiny_model.vocab_size
    ).float()
    assert one_hot_tokens.shape == torch.Size((b, l, tiny_model.vocab_size))

    out1 = tiny_model(token_ids)
    out2 = tiny_model(one_hot_tokens)

    torch.testing.assert_close(out1, out2)


@torch.inference_mode()
def test_soft_tokens_generation(tiny_model: BaseTransformer):
    b, l = 4, 6

    input_token_ids = torch.randint(0, tiny_model.vocab_size, (b, l))
    output = generate_from_tokens(
        net=tiny_model,
        token_ids=input_token_ids,
        eos_token_id=-1,
        pad_token_id=2,
        max_tokens_generated=24,
        use_kv_cache=False,
        sampling_strategy=None,
        soft_tokens=True,
    )

    assert output.shape == torch.Size((b, l + 24, tiny_model.vocab_size))
