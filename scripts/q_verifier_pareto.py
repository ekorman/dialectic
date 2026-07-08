"""Early-stopping compute/accuracy Pareto: q-threshold vs adaptive self-consistency.

Reads the per-completion scores artifact emitted by ``eval_q_verifier`` with
``--eval_params.dump-scores`` (one JSON line per prompt, each a list of
``{"q", "p", "correct", "val", "cot_len"}`` records) and simulates two
verifier-guided best-of-N schemes that stop *before* drawing all N completions:

- **q-threshold**: draw completions one at a time; accept the first whose q-score
  clears a threshold ``tau``; if none clear it, fall back to the best q-score
  seen (== full best-of-N). Sweeping ``tau`` traces accuracy vs expected draws.
- **adaptive self-consistency**: draw one at a time, keep a running tally of
  answer values, and stop once the plurality's lead (top count − second count)
  reaches a margin ``m``; return a random completion of the plurality value.
  Sweeping ``m`` traces its own accuracy vs expected draws.

Rollouts have no intrinsic generation order, so each prompt is replayed over
``--n-orderings`` random shuffles and averaged — that both de-biases the order
and smooths the curves. Both schemes are scored on the SAME mixed prompts
(those with a correct and an incorrect completion) so the comparison is apples
to apples with the launcher's mixed-only headline.

The point of the comparison: q can be Pareto-better (more accuracy per expected
generation) even where it merely ties self-consistency at full N, because it is
a per-item scorer and can stop early. Adaptive self-consistency is the fair
rival — it also stops early — so beating *it* is the real efficiency claim.

A second, separate advantage is reported as a margin-stratified best-of-N table
(``_margin_breakdown``): on vote ties (plurality margin 0 — e.g. three answers
each appearing twice) self-consistency has no signal and falls to the base
rate, while q keeps ranking. q's lift at margin 0 isolates that tie-break
advantage.

Finally, a reliability table (``_calibration_table``) sanity-checks the
threshold rule: empirical P(correct) by q-score decile, which must be monotone
for early stopping to be trustworthy. It runs on ALL prompts, not just mixed,
since a deployed accept rule sees every completion. This is descriptive
(threshold-free); picking a concrete deployment tau would require fitting on a
separate calibration split and is deliberately out of scope here.

Usage::

    uv run python scripts/q_verifier_pareto.py \
        --artifact q-verifier-scores-<split>-<q_run>-s<step> \
        [--n-orderings 50] [--max-prompts N] [--out /path/to/pareto.png]

The artifact name is printed by the launcher on save. ``--out`` requires
matplotlib, which is not a project dependency — run without it for the text
tables, or ``uv run --with matplotlib`` for the plot.
"""

import argparse
import json
import random
from collections import Counter

import extty


def _load_prompts(artifact: str) -> list[list[dict]]:
    data = extty.load_artifact(artifact, cache=True)
    if not isinstance(data, bytes):
        raise ValueError(f"expected bytes from artifact {artifact!r}")
    return [json.loads(line) for line in data.decode().splitlines() if line.strip()]


def _mixed(prompts: list[list[dict]]) -> list[list[dict]]:
    out = []
    for comps in prompts:
        has_c = any(c["correct"] for c in comps)
        has_i = any(not c["correct"] for c in comps)
        if has_c and has_i and len(comps) >= 2:
            out.append(comps)
    return out


def _q_threshold_point(
    prompts: list[list[dict]], tau: float, orderings: dict[int, list[list[int]]]
) -> tuple[float, float]:
    """Return (mean_draws, accuracy) for q-threshold early-stop at ``tau``."""
    draws_sum = 0
    correct_sum = 0
    n = 0
    for comps in prompts:
        order_idx = orderings[len(comps)]
        for order in order_idx:
            best = None  # (q, correct) best-so-far, for the no-accept fallback
            accepted = None
            for pos, i in enumerate(order, start=1):
                c = comps[i]
                if best is None or c["q"] > best[0]:
                    best = (c["q"], c["correct"])
                if c["q"] >= tau:
                    accepted = (pos, c["correct"])
                    break
            assert best is not None  # set on the first iteration
            if accepted is None:
                accepted = (len(comps), best[1])  # drew all N, keep best-so-far
            draws_sum += accepted[0]
            correct_sum += int(accepted[1])
            n += 1
    return draws_sum / n, correct_sum / n


def _adaptive_sc_point(
    prompts: list[list[dict]],
    margin: int,
    orderings: dict[int, list[list[int]]],
    rng: random.Random,
) -> tuple[float, float]:
    """Return (mean_draws, accuracy) for adaptive self-consistency at ``margin``."""
    draws_sum = 0
    correct_sum = 0
    n = 0
    for comps in prompts:
        order_idx = orderings[len(comps)]
        for order in order_idx:
            counts: Counter = Counter()
            seen: list[dict] = []
            stop_at = len(comps)
            for pos, i in enumerate(order, start=1):
                c = comps[i]
                seen.append(c)
                if c["val"] is not None:
                    counts[c["val"]] += 1
                if counts:
                    top2 = counts.most_common(2)
                    lead = top2[0][1] - (top2[1][1] if len(top2) > 1 else 0)
                    if lead >= margin:
                        stop_at = pos
                        break
            # Return a random completion of the plurality value among those seen.
            if counts:
                winner = counts.most_common(1)[0][0]
                pool = [c for c in seen if c["val"] == winner]
            else:
                pool = seen
            pick = rng.choice(pool)
            draws_sum += stop_at
            correct_sum += int(pick["correct"])
            n += 1
    return draws_sum / n, correct_sum / n


def _margin_breakdown(prompts: list[list[dict]]) -> dict[int, dict]:
    """Best-of-N accuracy stratified by self-consistency's plurality margin.

    For each prompt (at full N) the margin is ``top_vote_count −
    second_vote_count`` over answer values. Margin 0 is a vote tie: two or more
    values are co-modal, so self-consistency has no information to pick between
    them and degrades to a random choice among the tied clusters — exactly the
    "three answers each appear twice" case. q, a continuous per-item scorer,
    still ranks within the tie, so its accuracy should stay roughly flat while
    self-consistency collapses toward the base rate as margin → 0.

    q is deterministic (argmax q). Self-consistency accuracy is computed
    analytically as the expected correctness of a uniform pick from the
    modal-vote pool (the same random tie-break the launcher samples), which is
    lower-variance than sampling. Buckets margins ``>= 5`` together.
    """
    from collections import defaultdict

    buckets: dict[int, dict] = defaultdict(
        lambda: {"n": 0, "q": 0.0, "sc": 0.0, "rand": 0.0}
    )
    for comps in prompts:
        counts = Counter(c["val"] for c in comps if c["val"] is not None)
        if counts:
            top2 = counts.most_common(2)
            max_votes = top2[0][1]
            margin = max_votes - (top2[1][1] if len(top2) > 1 else 0)
            pool = [
                c
                for c in comps
                if c["val"] is not None and counts[c["val"]] == max_votes
            ]
        else:
            margin = 0
            pool = comps  # no parseable answers to vote on
        q_correct = float(max(comps, key=lambda c: c["q"])["correct"])
        sc_acc = sum(c["correct"] for c in pool) / len(pool) if pool else 0.0
        rand_acc = sum(c["correct"] for c in comps) / len(comps)

        b = buckets[min(margin, 5)]
        b["n"] += 1
        b["q"] += q_correct
        b["sc"] += sc_acc
        b["rand"] += rand_acc
    return buckets


def _calibration_table(prompts: list[list[dict]], n_bins: int = 10) -> list[dict]:
    """Reliability table: empirical P(correct) by q-score bin.

    Bins are equal-count (score quantiles), not equal-width — q scores are
    heavy-tailed, so equal-width bins would put nearly everything in one bin.
    The early-stop threshold rule is only trustworthy if this table is
    monotone: higher q-score bins should have higher empirical correctness.
    """
    comps = sorted((c for p in prompts for c in p), key=lambda c: c["q"])
    n = len(comps)
    bins = []
    for b in range(n_bins):
        chunk = comps[b * n // n_bins : (b + 1) * n // n_bins]
        if not chunk:
            continue
        bins.append(
            {
                "lo": chunk[0]["q"],
                "hi": chunk[-1]["q"],
                "n": len(chunk),
                "mean_q": sum(c["q"] for c in chunk) / len(chunk),
                "p_correct": sum(c["correct"] for c in chunk) / len(chunk),
            }
        )
    return bins


def _frontier(curve: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Upper-left Pareto frontier: sort by draws, keep accuracy-improving points."""
    best = -1.0
    out = []
    for d, a in sorted(curve):
        if a > best:
            out.append((d, a))
            best = a
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--artifact", required=True, help="per-completion scores artifact name"
    )
    ap.add_argument("--n-orderings", type=int, default=50)
    ap.add_argument("--max-prompts", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="optional PNG path for the Pareto plot")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    all_prompts = _load_prompts(args.artifact)
    if args.max_prompts is not None and len(all_prompts) > args.max_prompts:
        all_prompts = rng.sample(all_prompts, args.max_prompts)
    prompts = _mixed(all_prompts)
    print(f"prompts: {len(all_prompts)} total, {len(prompts)} mixed")

    # Precompute the random orderings once per distinct completion-count so both
    # schemes replay the SAME shuffles (fair, and reproducible).
    sizes = {len(c) for c in prompts}
    orderings = {
        k: [rng.sample(range(k), k) for _ in range(args.n_orderings)] for k in sizes
    }

    # Quantile-spaced tau grid: q scores are heavy-tailed, so a uniform grid
    # over [min, max] would waste nearly every point on the empty tail and
    # leave the dense region (where the draws/accuracy trade-off actually
    # happens) unresolved — the curve would jump straight to full N.
    q_scores = sorted(c["q"] for comps in prompts for c in comps)
    n_taus = 40
    taus = sorted(
        {q_scores[int(i * (len(q_scores) - 1) / (n_taus - 1))] for i in range(n_taus)}
    ) + [float("inf")]
    q_curve = [_q_threshold_point(prompts, t, orderings) for t in taus]

    max_n = max(sizes)
    margins = list(range(1, min(max_n, 20) + 1))
    sc_curve = [_adaptive_sc_point(prompts, m, orderings, rng) for m in margins]

    print("\nq-threshold  (draws, acc):")
    for d, a in _frontier(q_curve):
        print(f"  {d:5.2f}  {a:.4f}")
    print("\nadaptive self-consistency  (draws, acc):")
    for d, a in _frontier(sc_curve):
        print(f"  {d:5.2f}  {a:.4f}")

    # At matched budgets, how much more accurate is q?
    print("\nq accuracy − adaptive-SC accuracy at matched expected-draws:")
    for budget in (2, 3, 4, 6, 8):
        qa = _interp_acc(q_curve, budget)
        sa = _interp_acc(sc_curve, budget)
        if qa is not None and sa is not None:
            print(f"  ~{budget} draws: q={qa:.4f}  sc={sa:.4f}  lift={qa - sa:+.4f}")

    # Full-N best-of-N accuracy stratified by vote margin. Margin 0 is the
    # vote-tie stratum: self-consistency has no signal there and drops to the
    # base rate, while q keeps ranking. q's lift at margin 0 is the tie-break
    # advantage isolated.
    print("\nBest-of-N accuracy by self-consistency vote margin (full N):")
    print("  margin   n     q       sc      random   q−sc")
    breakdown = _margin_breakdown(prompts)
    total = sum(b["n"] for b in breakdown.values())
    for m in sorted(breakdown):
        b = breakdown[m]
        n = b["n"]
        q_a, sc_a, r_a = b["q"] / n, b["sc"] / n, b["rand"] / n
        label = f"{m}+ " if m == 5 else f"{m}  "
        tie = "  <- vote tie" if m == 0 else ""
        print(
            f"  {label:5s} {n:5d} ({n / total:4.0%})  "
            f"{q_a:.4f}  {sc_a:.4f}  {r_a:.4f}  {q_a - sc_a:+.4f}{tie}"
        )

    # Reliability diagram (descriptive, threshold-free): the early-stop rule is
    # only trustworthy if q-score maps monotonically to empirical correctness.
    # Uses ALL prompts (not just mixed) — a deployed accept rule sees every
    # completion, and mixed-filtering would bias P(correct | bin).
    print(
        f"\nCalibration — empirical P(correct) by q-score decile "
        f"({len(all_prompts)} prompts, monotone = trustworthy):"
    )
    print("  bin   [q range]            n      mean_q   P(correct)")
    for i, b in enumerate(_calibration_table(all_prompts)):
        print(
            f"  {i:3d}   [{b['lo']:7.3f},{b['hi']:7.3f}] {b['n']:6d}  "
            f"{b['mean_q']:7.3f}   {b['p_correct']:.4f}"
        )

    if args.out:
        _plot(q_curve, sc_curve, args.out)
        print(f"\nsaved plot: {args.out}")


def _interp_acc(curve, budget: float):
    """Linear-interpolate accuracy at a target expected-draws budget."""
    pts = sorted(curve)
    if budget < pts[0][0] or budget > pts[-1][0]:
        return None
    for (d0, a0), (d1, a1) in zip(pts, pts[1:]):
        if d0 <= budget <= d1:
            if d1 == d0:
                return a0
            return a0 + (a1 - a0) * (budget - d0) / (d1 - d0)
    return None


def _plot(q_curve, sc_curve, out: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for curve, label, marker in (
        (q_curve, "q-threshold", "o"),
        (sc_curve, "adaptive self-consistency", "s"),
    ):
        pts = sorted(curve)
        ax.plot([d for d, _ in pts], [a for _, a in pts], marker=marker, label=label)
    ax.set_xlabel("expected generations per prompt")
    ax.set_ylabel("selection accuracy (mixed prompts)")
    ax.set_title("Early-stopping compute/accuracy frontier")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)


if __name__ == "__main__":
    main()
