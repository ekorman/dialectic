# dialectic

Code for *[Inverse Reasoning Models for Efficient Post-Training and Verification from Answer-Only Data](https://openreview.net/forum?id=DtSUbLSHLn&nesting=2&sort=date-desc)*. This method trains an inverse model q(CoT | prompt, answer) on the rollouts of a GRPO-trained forward model p. q is then used two ways: to synthesize CoTs that are distilled back into p with SFT, and as a best-of-N verifier. All experiments use Qwen3-0.6B (thinking mode) on Countdown and GSM8K.

## Setup

This repository makes use of [`extty`](https://github.com/ekorman/extty) for experiment tracking. If you wish to use S3-syncing or use the TUI, please see that repo for setup. To install `dialectic` with all required dependencies simply run

```bash
uv sync --group vllm --group eval-q
```

Each stage is a launcher in `dialectic/experiments/launchers/`. Stages can pass their outputs (generated datasets, rollouts, model checkpoints, etc) to each other through `extty` artifacts.

On first use, the Qwen3-0.6B weights are downloaded from Hugging Face. For GSM8K experiments, the official `train.jsonl` and `test.jsonl` from [openai/grade-school-math](https://github.com/openai/grade-school-math) should be put in `data/gsm8k/`.

## Reproducing the paper

[`docs/paper_experiments.md`](docs/paper_experiments.md) has the full command sequence for every experiment in the paper, on both Countdown and GSM8K. It also lists the extty run and artifact names the paper used.
