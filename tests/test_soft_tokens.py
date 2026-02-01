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


def test_generate_from_tokens_stopping_condition_partial_batch_soft(MockGenerateModel):
    pad_token_id = 9
    eos_token_id = 2
    vocab_size = 12
    token_ids = torch.tensor([[4, 5], [6, 7], [8, 9]])
    token_schedule = [
        torch.tensor([3, 4, eos_token_id]),
        torch.tensor([5, 6, 7]),
        torch.tensor([6, 7, 8]),
        torch.tensor([7, 8, 9]),
    ]

    model = MockGenerateModel(
        token_schedule=token_schedule, vocab_size=vocab_size
    ).eval()

    max_tokens_generated = 4
    output = generate_from_tokens(
        net=model,
        token_ids=token_ids,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        sampling_strategy="greedy",
        max_tokens_generated=max_tokens_generated,
        use_kv_cache=True,
        soft_tokens=True,
    )

    assert output.shape == torch.Size((3, 2 + max_tokens_generated, vocab_size))
    generated_tokens = output[:, 2:]

    pad_token_one_hot = torch.zeros(vocab_size)
    pad_token_one_hot[pad_token_id] = 1

    # last element of batch should terminated immediately and just have pad token distribution
    assert (generated_tokens[2] == pad_token_one_hot).all()

    # others should always have no pad token component
    assert (generated_tokens[:2, :, pad_token_id] == 0).all()


def test_generate_from_tokens_stopping_condition_full_batch_soft(MockGenerateModel):
    pad_token_id = 8
    eos_token_id = 2
    vocab_size = 12
    token_ids = torch.tensor([[4, 5], [6, 7], [8, 9]])
    token_schedule = [
        torch.tensor([eos_token_id, 5, 6]),
        torch.tensor([7, eos_token_id, 8]),
        torch.tensor([9, 10, eos_token_id]),
    ]

    model = MockGenerateModel(
        token_schedule=token_schedule, vocab_size=vocab_size
    ).eval()

    max_tokens_generated = 5
    output = generate_from_tokens(
        net=model,
        token_ids=token_ids,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        sampling_strategy="greedy",
        max_tokens_generated=max_tokens_generated,
        use_kv_cache=True,
        soft_tokens=True,
    )

    assert output.shape[1] < token_ids.shape[1] + max_tokens_generated
    assert output.shape == torch.Size((3, 4, vocab_size))
    generated_tokens = output[:, 2:]

    def _create_one_hot(token_id):
        ret = torch.zeros(vocab_size)
        ret[token_id] = 1
        return ret

    pad_token_one_hot = _create_one_hot(pad_token_id)

    assert (
        generated_tokens[0] == torch.stack([pad_token_one_hot, pad_token_one_hot])
    ).all()
    assert (
        generated_tokens[1] == torch.stack([_create_one_hot(5), pad_token_one_hot])
    ).all()
    assert (
        generated_tokens[2] == torch.stack([_create_one_hot(6), _create_one_hot(8)])
    ).all()
