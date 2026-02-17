# Overview of experiments

## Baseline tests

Some hard token hyperparameter experiments are summarized in table below.


| Parameter | Value(s) |
| - |-|
| `model`* | "qwen3-0.6b",  "llama-3.2-1b-instruct"|
| `max-episodes` | 10000 |
| `batch_size` | 2 |
| `group_size` | 8 |
| `max-tokens` | 300 |
| `lr` | 1e-5 |
| `beta`* | 0, 0.04, 0.1|
| `eps`* | 0.1, 0.2 |
| `n-ops` | `3,2` |
| `n-total` | `4,3`|
| `n-larges` | `1,1`|
| `mu` | 1 |
| `accumulation-steps` | 16 |
| `max-grad-norm` | 1.0 |
| `update-ref-net-batch-cadence` | 100 |
| `use-bf16`* | both `--use-bf16` and `--no-bf16` |
| `no-qwen-thinking` | `--no-qwen-thinking` |
| `seed`* | 20, 80, 140 |
| `compile-model` | `--compile-model`|
| `temperature`* | 0.3, 0.7, 1.0 |
| `answer-tags-weight` | 0.1 |
| `think-tags-weight`* | 0.05 (only when using prompt that says to use it), 0 |
| `normalize-advantages`* | both `--normalize-advantages` and `--no-normalize-advantages` |
| `prompt-collections-id`* | 0, 1 |


### First sweep
| Parameter | Value(s) |
| - |-|
| `model`* | "qwen3-0.6b",  "llama-3.2-1b-instruct"|
| `max-episodes` | 10000 |
| `batch_size` | 2 |
| `group_size` | 8 |
| `max-tokens` | 300 |
| `lr` | 1e-5 |
| `beta` | 0.04|
| `eps`* | 0.1, 0.2 |
| `n-ops` | `3,2` |
| `n-total` | `4,3`|
| `n-larges` | `1,1`|
| `mu` | 1 |
| `accumulation-steps` | 16 |
| `max-grad-norm` | 1.0 |
| `update-ref-net-batch-cadence` | 100 |
| `use-bf16` | `use-bf16` |
| `no-qwen-thinking` | `--no-qwen-thinking` |
| `seed` | 20 |
| `compile-model` | `--compile-model`|
| `temperature`* | 0.3, 0.7, 1.0 |
| `answer-tags-weight` | 0.1 |
| `think-tags-weight` | 0 |
| `normalize-advantages` | `--normalize-advantages` |
| `prompt-collections-id` | 1 |


```shell
uv run train_scripts/grpo_countdown.py --seed 20 --prompt-collections-id 2 --no-qwen-thinking --max-episodes 10000 --max-tokens 300 --n-total 4,3 --n-larges 1,1 --n-ops 3,2 --temperature TEMPERATURE --eps EPS --model MODEL --think-tags-weight 0.0
```

### Second sweep
For each model take top temperature and `eps` combination and sweep `beta` in {0, 0.1}



- TODO: Implement RLOO and compare
