from tokenizers import Encoding, Tokenizer
from transformers import AutoTokenizer

from dialectic.tokenizer import Message, get_input_text_from_messages


def test_tokenizer():
    auto_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    tokenizer: Tokenizer = Tokenizer.from_file("qwen-tokenizer/tokenizer.json")

    prompt = "Give me a short introduction to large language model."
    messages = [{"role": "user", "content": prompt}]
    text = auto_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,  # Switches between thinking and non-thinking modes. Default is True.
    )

    assert (
        text
        == get_input_text_from_messages(
            messages=[Message(**m) for m in messages], add_generation_prompt=True
        )
        == "<|im_start|>user\nGive me a short introduction to large language model.<|im_end|>\n<|im_start|>assistant\n"
    )

    x: Encoding = tokenizer.encode_batch([text])  # or could do .encode(text)
    y = auto_tokenizer([text], return_tensors="pt")

    assert y.tokens() == x[0].tokens
    assert y.input_ids.tolist()[0] == x[0].ids
