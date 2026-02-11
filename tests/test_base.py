import torch

from dialectic.llm.base import BaseTransformer


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

    soft_tokens = one_hot_tokens @ tiny_model.embed_tokens.weight

    out1 = tiny_model(token_ids)
    out2 = tiny_model(soft_tokens)

    torch.testing.assert_close(out1, out2)
