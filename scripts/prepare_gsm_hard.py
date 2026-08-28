"""Convert reasoning-machines/gsm-hard into the gsm8k rollout-generation JSONL schema.

GSM-Hard (Gao et al., PAL) is the gsm8k *test* set with numbers substituted for
much larger values — the reasoning structure (and hence p's CoT distribution)
stays gsm8k-shaped while arithmetic execution gets much harder. Complements
GSM-Symbolic, which shifts reasoning structure instead: together they separate
the two difficulty axes for frozen-verifier evaluation. No contamination
against p/q trained on gsm8k train.

Known dataset noise: number substitution can make problems semantically
nonsensical, most visibly the ~11% of rows with negative targets (counts/money
can't be negative). ``--drop-negative-targets`` filters those; default keeps
the benchmark as published.

Output rows are ``{"question", "answer"}`` with ``answer`` ending in
``#### <value>`` — the shape ``generate_inverse_cot_rollouts._load_gsm8k_problems``
parses. Pass the file as ``--gsm8k_params.train-path`` (rollouts land under
split="train"; evaluate with ``--eval_params.split train``).

Usage::

    uv run python scripts/prepare_gsm_hard.py [--drop-negative-targets]
        [--out data/gsm_hard.jsonl]
"""

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download


def _format_target(target: float) -> str:
    if target == int(target):
        return str(int(target))
    return repr(target)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop-negative-targets", action="store_true")
    ap.add_argument("--out", default="data/gsm_hard.jsonl")
    args = ap.parse_args()

    src = hf_hub_download(
        "reasoning-machines/gsm-hard", "gsmhardv2.jsonl", repo_type="dataset"
    )
    rows: list[dict] = []
    n_negative = 0
    with open(src) as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            target = float(entry["target"])
            if target < 0:
                n_negative += 1
                if args.drop_negative_targets:
                    continue
            rows.append(
                {
                    "question": entry["input"],
                    "answer": f"#### {_format_target(target)}",
                }
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    dropped = " (dropped)" if args.drop_negative_targets else " (kept)"
    print(
        f"gsm-hard: {len(rows)} problems -> {out_path}  "
        f"[{n_negative} negative targets{dropped}]"
    )


if __name__ == "__main__":
    main()
