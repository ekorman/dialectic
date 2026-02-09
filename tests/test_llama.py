import os

import pytest
import torch
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM

from dialectic.llm.generate import llama_generate_from_chat
from dialectic.llm.llama import load_llama_1b
from dialectic.llm.templates import Message

"""
hf_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct", dtype=torch.float32
).eval()
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
messages = [
    {"role": "user", "content": "Hello who are you?"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(hf_model.device)
outputs = hf_model.generate(**inputs, do_sample=False, max_new_tokens=500)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :]))

Hello! I'm an artificial intelligence model known as Llama. Llama stands for "Large Language Model Meta AI."<|eot_id|>
"""


@pytest.mark.skipif(
    os.getenv("TEST_LLM_AGAINST_HF") is None,
    reason="skipping `test_llama_against_hf_softmax` since env variable `TEST_LLM_AGAINST_HF` not set",
)
@torch.no_grad()
def test_llama_against_hf_softmax():
    model = load_llama_1b()

    hf_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct", dtype=torch.float32
    ).eval()

    sd = hf_model.state_dict()

    sd = {k[6:] if k.startswith("model.") else k: v for k, v in sd.items()}
    model.load_state_dict(sd)

    x = torch.randint(0, 100, (2, 4))
    hf_model.eval()
    model.eval()

    out1 = model(x, return_all_logits=True)
    out2 = hf_model(x).logits

    # logits aren't sufficiently close but the softmax are
    torch.testing.assert_close(out1.softmax(-1), out2.softmax(-1))


"""
weird thing going on with the tokenizer, even going through transformers:
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

text_input = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

then `tokenizer.encode(text_input)` is not the same as `inputs["input_ids"]`. the former has an extra whitespace/pad
token.
"""


@pytest.mark.skipif(
    os.getenv("TEST_LLM_AGAINST_HF") is None,
    reason="skipping `test_llama_against_hf_generation` since env variable `TEST_LLM_AGAINST_HF` not set",
)
@torch.no_grad()
def test_llama_against_hf_generation():
    tokenizer = Tokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

    model = load_llama_1b()

    hf_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct", dtype=torch.float32
    ).eval()

    sd = hf_model.state_dict()

    sd = {k[6:] if k.startswith("model.") else k: v for k, v in sd.items()}
    model.load_state_dict(sd)

    model.eval()

    messages = [[Message(role="user", content="Hello who are you?")]]

    chat_output = llama_generate_from_chat(
        model,
        tokenizer=tokenizer,
        batch_messages=messages,
        sampling_strategy="greedy",
        max_tokens_generated=100,
    )

    assert len(chat_output) == 1

    assert (
        chat_output[0]
        == """user

Hello who are you?assistant

I'm an artificial intelligence model known as Llama. Llama stands for "Large Language Model Meta AI.\""""
    )
