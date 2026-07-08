from dataclasses import dataclass

import extty
import matplotlib.pyplot as plt
import numpy as np

MIX_RATIOS = [0, 0.25, 0.5, 0.75, 1]


@dataclass
class Metric:
    seed: int
    step: int
    value: float
    mix_ratio: float


def get_mix_ratio_and_seed(eval_run: extty.RunData) -> tuple[float, int]:
    # env = eval_run.project.split("-")[-1]

    run_name = eval_run.config["eval_params"]["ckpt_run"]

    proj, run_name = run_name.split("/")
    run = extty.get_run(proj, run_name)

    return run.config["sft_params"]["mix_ratio"], run.config["sft_params"]["seed"]


def get_mix_ratio(eval_run: extty.RunData) -> tuple[float, int]:
    run_name = eval_run.config["eval_params"]["ckpt_run"]

    proj, run_name = run_name.split("/")
    run = extty.get_run(proj, run_name)

    return run.config["sft_params"]["mix_ratio"]


def get_seed(eval_run: extty.RunData) -> tuple[float, int]:
    run_name = eval_run.config["eval_params"]["ckpt_run"]

    proj, run_name = run_name.split("/")
    run = extty.get_run(proj, run_name)

    return run.config["sft_params"]["seed"]


def get_metrics_from_run(eval_run: extty.RunData, metric_name: str) -> list[Metric]:
    mix_ratio, seed = get_mix_ratio_and_seed(eval_run)

    return [
        Metric(seed=seed, mix_ratio=mix_ratio, step=m.step, value=m.value)
        for m in eval_run.metric(metric_name)
    ]


def get_plot(metric_name: str, mix_ratio: float, ax: plt.Axes):
    runs = [r for r in sfts if get_mix_ratio(r) == mix_ratio]
    seed_to_metrics = {}

    assert len(set([get_seed(r) for r in runs])) == len(runs)

    for r in runs:
        seed_to_metrics[get_seed(r)] = [
            (m.step, m.value)
            for m in sorted(r.metric(metric_name), key=lambda m: m.step)
        ]

    assert len(seed_to_metrics) == 2

    max_step = min([metrics[-1][0] for metrics in seed_to_metrics.values()])

    for seed, metrics in seed_to_metrics.items():
        seed_to_metrics[seed] = [m for m in metrics if m[0] <= max_step]

    xs = np.array([[m[0] for m in metrics] for metrics in seed_to_metrics.values()])
    ys = np.array([[m[1] for m in metrics] for metrics in seed_to_metrics.values()])

    means = ys.mean(0)
    stds = ys.std(0)

    (line,) = ax.plot(xs[0], means, lw=2, label=f"mix_ratio = {mix_ratio}")

    ax.fill_between(
        xs[0], means - stds, means + stds, alpha=0.2, color=line.get_color()
    )


project = "eval-grpo-gsm8k"

metric_names = ["all_incorrect_pass_at_1", "all_incorrect_pass_at_32", "pass_at_32"]

runs = extty.get_runs(project=project)

baseline = None
sfts = []
for run in runs:
    if len(run.metric(metric_names[0])) == 1:
        assert baseline is None
        baseline = run
    else:
        sfts.append(run)

assert baseline
assert len(sfts) == 10


for metric_name in metric_names:
    fig, ax = plt.subplots(figsize=(8, 5))

    for mix_ratio in MIX_RATIOS:
        get_plot(metric_name, mix_ratio=mix_ratio, ax=ax)

    ax.set_title(metric_name)
    ax.legend()
    fig.tight_layout()
    plt.show()
