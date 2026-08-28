import dataclasses
import json
import os
import random
import tempfile
from collections import Counter
from datetime import datetime

import extty
from tqdm import tqdm

from dialectic.experiments.arg_parser import Experiment, run_experiments_parser
from dialectic.experiments.params import CountdownParams, DatasetGenParams
from dialectic.log import log
from dialectic.rl.env import CountdownEnv, build_countdown_equation


@extty.experiment(project="generate-countdown-dataset")
def generate_countdown_dataset(
    *,
    dataset_gen_params: DatasetGenParams,
    countdown_params: CountdownParams,
):
    p = dataset_gen_params
    if abs(p.train_pct + p.val_pct + p.test_pct - 1.0) > 1e-6:
        raise ValueError(
            f"train_pct + val_pct + test_pct must equal 1.0, "
            f"got {p.train_pct} + {p.val_pct} + {p.test_pct} = {p.train_pct + p.val_pct + p.test_pct}"
        )

    env = CountdownEnv(
        seed=p.seed,
        n_larges=countdown_params.n_larges,
        n_total=countdown_params.n_total,
        n_ops=countdown_params.n_ops,
    )

    seen: set[tuple[frozenset, int]] = set()
    n_excluded_keys = 0
    if p.exclude_artifacts is not None:
        for artifact_name in p.exclude_artifacts.split(","):
            artifact_name = artifact_name.strip()
            data = extty.load_artifact(artifact_name, cache=True)
            if not isinstance(data, bytes):
                raise ValueError(f"Expected bytes from artifact {artifact_name!r}")
            for line in data.decode().splitlines():
                if not line.strip():
                    continue
                ex = json.loads(line)
                seen.add((frozenset(Counter(ex["numbers"]).items()), ex["target"]))
        n_excluded_keys = len(seen)
        log.info(
            f"Excluding {n_excluded_keys} problems from "
            f"{p.exclude_artifacts!r} (all splits)"
        )
    examples: list[dict] = []

    pbar = tqdm(total=p.n_examples, desc="Generating problems")
    n_generated = 0
    while len(examples) < p.n_examples:
        resp = env.reset()
        n_generated += 1
        numbers = resp.data.numbers
        target = resp.data.target

        key = (frozenset(Counter(numbers).items()), target)
        if key in seen:
            continue
        seen.add(key)

        assert resp.data.solution is not None
        equation = build_countdown_equation(numbers, resp.data.solution, target)
        examples.append(
            {
                "numbers": numbers,
                "target": target,
                "equation": equation,
            }
        )
        pbar.update(1)
        pbar.set_postfix(generated=n_generated, kept=len(examples))
    pbar.close()

    log.info(
        f"Generated {n_generated} total, kept {len(examples)} unique "
        f"({n_generated - len(examples)} rejected as duplicates"
        f"{' or excluded-artifact collisions' if n_excluded_keys else ''})"
    )

    rng = random.Random(p.seed)
    rng.shuffle(examples)

    n_train = round(p.n_examples * p.train_pct)
    n_val = round(p.n_examples * p.val_pct)

    for i, ex in enumerate(examples):
        if i < n_train:
            ex["split"] = "train"
        elif i < n_train + n_val:
            ex["split"] = "val"
        else:
            ex["split"] = "test"

    split_counts = Counter(ex["split"] for ex in examples)
    log.info(f"Splits: {dict(split_counts)}")

    tmpfile = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, prefix="countdown_dataset_"
    )
    try:
        for ex in examples:
            tmpfile.write(json.dumps(ex) + "\n")
        tmpfile.close()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        artifact_name = (
            f"countdown-dataset-n{p.n_examples}-"
            f"ops{'_'.join(map(str, env._n_ops))}-{ts}"
        )
        meta = extty.save_artifact(
            name=artifact_name,
            path=tmpfile.name,
            description=(
                f"Countdown dataset: {p.n_examples} unique problems, "
                f"splits={dict(split_counts)}"
            ),
            metadata={
                "dataset_gen_params": dataclasses.asdict(p),
                "countdown_params": dataclasses.asdict(countdown_params),
                "n_excluded_keys": n_excluded_keys,
            },
        )
        log.info(f"Uploaded artifact: {meta}")
    finally:
        os.unlink(tmpfile.name)


if __name__ == "__main__":
    run_experiments_parser(
        [
            Experiment(
                env_name=None,
                fn=generate_countdown_dataset,
                include_prompt_collection_id=False,
            ),
        ]
    )
