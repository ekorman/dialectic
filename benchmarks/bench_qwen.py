"""Benchmark script for BaseTransformer model implementation.

Usage:
    uv run python benchmarks/bench_qwen.py
    uv run python benchmarks/bench_qwen.py --devices cpu mps
    uv run python benchmarks/bench_qwen.py --batch-sizes 1 4 8
    uv run python benchmarks/bench_qwen.py --no-compile
"""

import argparse
import time
from dataclasses import dataclass

import torch

from dialectic.llm.base import BaseTransformer
from dialectic.llm.generate import generate_hard_tokens
from dialectic.llm.qwen import create_qwen


@dataclass
class BenchmarkConfig:
    name: str
    d: int
    head_d: int
    num_heads: int
    num_kv_heads: int
    mlp_hidden_d: int
    vocab_size: int
    n_decoder_layers: int
    rope_base_value: float = 10000


CONFIGS = {
    "small": BenchmarkConfig(
        name="small",
        d=20,
        head_d=16,
        num_heads=8,
        num_kv_heads=2,
        mlp_hidden_d=32,
        vocab_size=500,
        n_decoder_layers=3,
    ),
    "medium": BenchmarkConfig(
        name="medium",
        d=512,
        head_d=64,
        num_heads=8,
        num_kv_heads=4,
        mlp_hidden_d=1408,
        vocab_size=32000,
        n_decoder_layers=8,
    ),
    "large": BenchmarkConfig(
        name="large",
        d=1024,
        head_d=128,
        num_heads=16,
        num_kv_heads=8,
        mlp_hidden_d=3072,
        vocab_size=151936,
        n_decoder_layers=28,
    ),
}


def get_available_devices() -> list[str]:
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


def create_model(
    config: BenchmarkConfig, device: str, compiled: bool = False
) -> BaseTransformer:
    model = create_qwen(
        d=config.d,
        vocab_size=config.vocab_size,
        n_decoder_layers=config.n_decoder_layers,
        attn_head_d=config.head_d,
        attn_num_heads=config.num_heads,
        attn_num_kv_heads=config.num_kv_heads,
        mlp_hidden_d=config.mlp_hidden_d,
        rope_base_value=config.rope_base_value,
        tie_weights=True,
    )
    model = model.to(device).eval()
    if compiled:
        model = torch.compile(model)
    return model


def sync_device(device: str):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def benchmark_forward(
    model: BaseTransformer,
    device: str,
    batch_size: int,
    seq_length: int,
    n_warmup: int,
    n_iterations: int = 10,
) -> dict:
    x = torch.randint(0, model.vocab_size, size=(batch_size, seq_length), device=device)

    warmup_times = []
    for _ in range(n_warmup):
        sync_device(device)
        start = time.perf_counter()
        with torch.inference_mode():
            _ = model(x)
        sync_device(device)
        end = time.perf_counter()
        warmup_times.append(end - start)

    times = []
    for _ in range(n_iterations):
        sync_device(device)
        start = time.perf_counter()
        with torch.inference_mode():
            _ = model(x)
        sync_device(device)
        end = time.perf_counter()
        times.append(end - start)

    return {
        "warmup_first_ms": warmup_times[0] * 1000,
        "warmup_mean_ms": sum(warmup_times) / len(warmup_times) * 1000,
        "mean_ms": sum(times) / len(times) * 1000,
        "min_ms": min(times) * 1000,
        "max_ms": max(times) * 1000,
    }


def benchmark_generation(
    model: BaseTransformer,
    device: str,
    batch_size: int,
    prompt_length: int,
    max_new_tokens: int,
    use_kv_cache: bool,
    n_warmup: int,
    n_iterations: int = 5,
) -> dict:
    x = torch.randint(
        0, model.vocab_size, size=(batch_size, prompt_length), device=device
    )

    total_tokens = max_new_tokens * batch_size

    warmup_times = []
    for _ in range(n_warmup):
        sync_device(device)
        start = time.perf_counter()
        _ = generate_hard_tokens(
            net=model,
            token_ids=x,
            eos_token_id=-1,
            max_tokens_generated=max_new_tokens,
            use_kv_cache=use_kv_cache,
            sampling_strategy="greedy",
        )
        sync_device(device)
        end = time.perf_counter()
        warmup_times.append(end - start)

    times = []
    for _ in range(n_iterations):
        sync_device(device)
        start = time.perf_counter()
        _ = generate_hard_tokens(
            net=model,
            token_ids=x,
            eos_token_id=-1,
            max_tokens_generated=max_new_tokens,
            use_kv_cache=use_kv_cache,
            sampling_strategy="greedy",
        )
        sync_device(device)
        end = time.perf_counter()
        times.append(end - start)

    mean_time = sum(times) / len(times)

    return {
        "warmup_first_ms": warmup_times[0] * 1000,
        "warmup_mean_ms": sum(warmup_times) / len(warmup_times) * 1000,
        "mean_ms": mean_time * 1000,
        "min_ms": min(times) * 1000,
        "max_ms": max(times) * 1000,
        "tokens_per_sec": total_tokens / mean_time,
        "warmup_tokens_per_sec": total_tokens / (sum(warmup_times) / len(warmup_times)),
    }


def _run_forward_benchmarks(
    config: BenchmarkConfig,
    devices: list[str],
    batch_sizes: list[int],
    seq_lengths: list[int],
    compiled: bool,
    n_warmup: int,
):
    label = (
        "Forward Pass Benchmark (compiled)" if compiled else "Forward Pass Benchmark"
    )
    print("-" * 100)
    print(label)
    print("-" * 100)
    print(
        f"{'Device':<8} {'Batch':<6} {'SeqLen':<8} {'Warmup 1st':<12} {'Warmup Mean':<12} {'Mean (ms)':<12} {'Min (ms)':<12}"
    )
    print("-" * 100)

    for device in devices:
        model = create_model(config, device, compiled=compiled)
        for batch_size in batch_sizes:
            for seq_length in seq_lengths:
                result = benchmark_forward(
                    model, device, batch_size, seq_length, n_warmup
                )
                print(
                    f"{device:<8} {batch_size:<6} {seq_length:<8} "
                    f"{result['warmup_first_ms']:<12.3f} {result['warmup_mean_ms']:<12.3f} "
                    f"{result['mean_ms']:<12.3f} {result['min_ms']:<12.3f}"
                )
        del model
        sync_device(device)


def _run_generation_benchmarks(
    config: BenchmarkConfig,
    devices: list[str],
    batch_sizes: list[int],
    seq_lengths: list[int],
    generation_tokens: int,
    use_kv_cache: bool,
    compiled: bool,
    n_warmup: int,
):
    cache_label = "with KV cache" if use_kv_cache else "without KV cache"
    compiled_label = " (compiled)" if compiled else ""
    label = f"Generation Benchmark ({cache_label}){compiled_label}"
    print()
    print("-" * 100)
    print(label)
    print("-" * 100)
    print(
        f"{'Device':<8} {'Batch':<6} {'Prompt':<8} {'NewToks':<8} {'Warmup 1st':<12} {'Mean (ms)':<12} {'Warmup Tok/s':<14} {'Tok/s':<12}"
    )
    print("-" * 100)

    for device in devices:
        model = create_model(config, device, compiled=compiled)
        for batch_size in batch_sizes:
            for seq_length in seq_lengths:
                result = benchmark_generation(
                    model,
                    device,
                    batch_size,
                    seq_length,
                    generation_tokens,
                    n_warmup=n_warmup,
                    use_kv_cache=use_kv_cache,
                )
                print(
                    f"{device:<8} {batch_size:<6} {seq_length:<8} {generation_tokens:<8} "
                    f"{result['warmup_first_ms']:<12.3f} {result['mean_ms']:<12.3f} "
                    f"{result['warmup_tokens_per_sec']:<14.1f} {result['tokens_per_sec']:<12.1f}"
                )
        del model
        sync_device(device)


def run_benchmarks(
    devices: list[str],
    batch_sizes: list[int],
    seq_lengths: list[int],
    generation_tokens: int,
    configs: list[BenchmarkConfig],
    n_warmup: int,
    include_compiled: bool = True,
):
    print("=" * 100)
    print("BaseTransformer Benchmark")
    print("=" * 100)
    print(f"Configs: {[c.name for c in configs]}")
    print(f"Devices: {devices}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Sequence lengths: {seq_lengths}")
    print(f"Generation tokens: {generation_tokens}")
    print(f"Include compiled: {include_compiled}")
    print()

    for config in configs:
        print()
        print("=" * 100)
        print(
            f"Config: {config.name} (d={config.d}, layers={config.n_decoder_layers}, "
            f"heads={config.num_heads}, kv_heads={config.num_kv_heads}, mlp_hidden={config.mlp_hidden_d})"
        )
        print("=" * 100)

        _run_forward_benchmarks(
            config, devices, batch_sizes, seq_lengths, compiled=False, n_warmup=n_warmup
        )

        if include_compiled:
            _run_forward_benchmarks(
                config,
                devices,
                batch_sizes,
                seq_lengths,
                compiled=True,
                n_warmup=n_warmup,
            )

        _run_generation_benchmarks(
            config,
            devices,
            batch_sizes,
            seq_lengths,
            generation_tokens,
            use_kv_cache=True,
            compiled=False,
            n_warmup=n_warmup,
        )

        if include_compiled:
            _run_generation_benchmarks(
                config,
                devices,
                batch_sizes,
                seq_lengths,
                generation_tokens,
                use_kv_cache=True,
                compiled=True,
                n_warmup=n_warmup,
            )

        _run_generation_benchmarks(
            config,
            devices,
            batch_sizes,
            seq_lengths,
            generation_tokens,
            use_kv_cache=False,
            compiled=False,
            n_warmup=n_warmup,
        )

        if include_compiled:
            _run_generation_benchmarks(
                config,
                devices,
                batch_sizes,
                seq_lengths,
                generation_tokens,
                use_kv_cache=False,
                compiled=True,
                n_warmup=n_warmup,
            )


def main():
    parser = argparse.ArgumentParser(description="Benchmark BaseTransformer model")
    parser.add_argument(
        "--devices",
        nargs="+",
        default=None,
        help="Devices to benchmark (default: all available)",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=list(CONFIGS.keys()),
        default=["small"],
        help=f"Model configs to test (choices: {list(CONFIGS.keys())})",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 4],
        help="Batch sizes to test",
    )
    parser.add_argument(
        "--seq-lengths",
        nargs="+",
        type=int,
        default=[4, 16],
        help="Sequence lengths to test",
    )
    parser.add_argument(
        "--generation-tokens",
        type=int,
        default=24,
        help="Number of tokens to generate",
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=10,
        help="Number of times to warmup",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        default=False,
        help="Skip compiled model benchmarks",
    )
    args = parser.parse_args()

    devices = args.devices or get_available_devices()
    available = get_available_devices()
    for d in devices:
        if d not in available:
            print(f"Warning: device '{d}' not available, skipping")
    devices = [d for d in devices if d in available]

    if not devices:
        print("No devices available for benchmarking")
        return

    configs = [CONFIGS[name] for name in args.configs]
    run_benchmarks(
        devices=devices,
        batch_sizes=args.batch_sizes,
        seq_lengths=args.seq_lengths,
        generation_tokens=args.generation_tokens,
        configs=configs,
        include_compiled=not args.no_compile,
        n_warmup=args.n_warmup,
    )


if __name__ == "__main__":
    main()
