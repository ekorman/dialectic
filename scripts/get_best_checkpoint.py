import argparse

import extty

from dialectic.log import log


def main(project: str, run_name: str, metric: str):
    run = extty.get_run(project=project, name=run_name)

    ckpt_metrics = []
    for ckpt in run.checkpoints:
        m = run.metric(metric, ckpt.step)
        if m is not None:
            ckpt_metrics.append((ckpt.step, m.value))

    all_vals = [x[1] for x in ckpt_metrics]

    best = max(ckpt_metrics, key=lambda x: x[1])

    log.info(f"min metric value: {min(all_vals)}")
    log.info(f"max metric value: {max(all_vals)}")
    log.info(f"average metric value: {sum(all_vals) / len(all_vals)}")
    log.info(
        f"best checkpoint for metric: {metric}: {best[0]} with metric value {best[1]}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=str)
    parser.add_argument("metric", type=str)

    args = parser.parse_args()

    project, run_name = args.run.split("/")
    main(project, run_name, args.metric)
