"""Convert apple/GSM-Symbolic into the gsm8k rollout-generation JSONL schema.

GSM-Symbolic (Mirzadeh et al.) perturbs 100 GSM8K *test* templates — ``main``
matches gsm8k difficulty, ``p1``/``p2`` add one/two extra clauses per problem
(harder reasoning, same style). Since templates derive from the gsm8k test
set, there is no contamination against p/q trained on gsm8k train.

Each template has up to 50 generated instances that are near-duplicates of
each other (same story, different numbers), so rows are subsampled per
template (``--instances-per-template``, seeded) — otherwise per-prompt stats
would be dominated by 50-way-clustered clones.

Output: ``data/gsm_symbolic/<variant>.jsonl`` with ``{"question", "answer"}``
rows where ``answer`` ends in ``#### <value>`` — the exact shape
``generate_inverse_cot_rollouts._load_gsm8k_problems`` parses. Pass the file
as ``--gsm8k_params.train-path`` (rollouts land under split="train"; evaluate
with ``--eval_params.split train``).

Usage::

    uv run python scripts/prepare_gsm_symbolic.py [--variants main p1 p2]
        [--instances-per-template 13] [--seed 0] [--out-dir data/gsm_symbolic]
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download

ANSWER_RE = re.compile(r"####\s*([^\n]+)")


def prepare_variant(
    variant: str, instances_per_template: int, seed: int, out_dir: Path
) -> None:
    src = hf_hub_download(
        "apple/GSM-Symbolic", f"{variant}/test.jsonl", repo_type="dataset"
    )
    by_template: dict[str, list[dict]] = defaultdict(list)
    with open(src) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                by_template[str(row["id"])].append(row)

    rng = random.Random(seed)
    rows: list[dict] = []
    n_bad = 0
    for template_id in sorted(by_template):
        pool = sorted(by_template[template_id], key=lambda r: r["instance"])
        picked = (
            rng.sample(pool, instances_per_template)
            if len(pool) > instances_per_template
            else pool
        )
        for row in picked:
            m = ANSWER_RE.search(row["answer"])
            if m is None:
                n_bad += 1
                continue
            try:
                float(m.group(1).strip().replace(",", ""))
            except ValueError:
                n_bad += 1
                continue
            rows.append({"question": row["question"], "answer": row["answer"]})

    rng.shuffle(rows)
    out_path = out_dir / f"{variant}.jsonl"
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(
        f"{variant}: {len(by_template)} templates -> {len(rows)} problems "
        f"({n_bad} dropped for unparseable answers) -> {out_path}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["main", "p1", "p2"])
    ap.add_argument("--instances-per-template", type=int, default=13)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="data/gsm_symbolic")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for variant in args.variants:
        prepare_variant(variant, args.instances_per_template, args.seed, out_dir)


if __name__ == "__main__":
    main()
