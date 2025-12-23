import timeit

import torch

from dialectic.qwen import generate_from_tokens, load_qwen_06b

n_iters = 10


net = load_qwen_06b()

n_input_tokens = 10
n_output_tokens = 20

x = torch.randint(0, net.vocab_size, size=(1, n_input_tokens))


def gen_with_cache():
    generate_from_tokens(
        net, x, -1, max_tokens_generated=n_output_tokens, use_kv_cache=True
    )


def gen_without_cache():
    generate_from_tokens(
        net, x, -1, max_tokens_generated=n_output_tokens, use_kv_cache=False
    )


print(
    f"Running benchmark with n_input_tokens: {n_input_tokens}, n_output_tokens: {n_output_tokens}"
)
print(f"With kv-cache: {timeit.timeit(gen_with_cache, number=n_iters)}")
print(f"Without kv-cache: {timeit.timeit(gen_without_cache, number=n_iters)}")
