"""Qualitative spot-check of an SFT checkpoint on the held-out TEST split.

Loads an SFT checkpoint (base + trained LoRA, merged), generates fresh CoT+answer
on the rollout artifact's *test* split (never touched in training), grades, prints
aggregate accuracy, and dumps example generations to a markdown file so you can eyeball
whether the reasoning is genuine.

Use ``--compare-step`` to load a SECOND checkpoint of the same run (e.g. a pre-leap
step) and bucket test prompts by how the two differ:
  fixed      — pre wrong, post right  (the leap, made concrete)
  broke      — pre right, post wrong
  both_right / both_wrong
This is the cleanest check that a val leap is real generalization, not a val artifact:
if `fixed` >> `broke` on the *test* split and the CoTs read as real reasoning, it's real.

Run via extty on the GPU box, e.g.:
    extty run uv run python scripts/spot_check_test.py \
        --ckpt-run sft-inverse-cot-countdown/2026-06-17_18-08-10_b19c \
        --ckpt-step 80000 --compare-step 60000 --env countdown
"""

import argparse
import random

import extty
import torch

from dialectic.experiments.launchers._eval_helpers import (
    load_run_config,
    strip_rng_state,
)
from dialectic.llm.generate import generate_hard_tokens
from dialectic.llm.lora import DEFAULT_TARGET_MODULES, apply_lora, merge_lora
from dialectic.llm.registry import MODEL_REGISTRY
from dialectic.log import log
from dialectic.rl.extractors import extract_from_answer_tags
from dialectic.rl.inverse_cot_data import load_rollout_artifact
from dialectic.rl.inverse_cot_eval import (
    HARD_PROMPT_THRESHOLD,
    expressions_match,
    gsm8k_match,
)


def _grade(extracted: str | None, gold: str, env: str) -> bool:
    if extracted is None:
        return False
    return (
        gsm8k_match(extracted, gold)
        if env == "gsm8k"
        else expressions_match(extracted, gold)
    )


def _lora_targets(spec: str) -> tuple[str, ...]:
    if spec == "attn":
        return ("q_proj", "k_proj", "v_proj", "o_proj")
    if spec == "mlp":
        return ("gate_proj", "up_proj", "down_proj")
    return DEFAULT_TARGET_MODULES


def _load_sft_ckpt(ckpt_run, ckpt_step, model_info, sft_params, device, dtype):
    """Base architecture + trained LoRA from the SFT checkpoint, merged for inference.

    The SFT checkpoint's state_dict carries the frozen base (= GRPO weights) AND the
    trained LoRA, so we reconstruct the base shell, attach LoRA at the run's rank, load
    the full state, and merge.
    """
    p = model_info.load_net(pretrained_weights=False)
    rank = sft_params.get("lora_rank")
    if rank is not None:
        apply_lora(
            p,
            rank=rank,
            alpha=sft_params.get("lora_alpha", 16.0),
            target_modules=_lora_targets(sft_params.get("lora_target_modules", "all")),
        )
    proj, name = ckpt_run.split("/")
    ckpt = extty.load_checkpoint_from(
        project=proj, run_name=name, step=ckpt_step, load_optimizer=False
    )
    state = ckpt["model_state_dict"]
    strip_rng_state(state)
    p.load_state_dict(state)
    if rank is not None:
        merge_lora(p)
    return p.to(device=device, dtype=dtype).requires_grad_(False).eval()


@torch.no_grad()
def _generate(
    p,
    prompts,
    tokenizer,
    model_info,
    *,
    n_samples,
    temperature,
    max_tokens,
    batch_size,
    device,
    use_bf16,
    env,
):
    """Return per-prompt list of dicts: {gold, artifact_rate, samples:[(text,extracted,correct)]}."""
    tokenizer.no_padding()
    tokenizer.no_truncation()
    results = [
        {
            "gold": pr.equation,
            "artifact_rate": (
                sum(c.is_correct for c in pr.completions) / len(pr.completions)
                if pr.completions
                else 0.0
            ),
            "prompt_str": tokenizer.decode(pr.prompt_ids),
            "samples": [],
        }
        for pr in prompts
    ]
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        max_len = max(len(pr.prompt_ids) for pr in batch)
        ids = torch.full((len(batch), max_len), model_info.pad_token_id, device=device)
        mask = torch.zeros(len(batch), max_len, dtype=torch.bool, device=device)
        for i, pr in enumerate(batch):
            off = max_len - len(pr.prompt_ids)
            ids[i, off:] = torch.tensor(pr.prompt_ids, device=device)
            mask[i, off:] = True
        for _ in range(n_samples):
            out = generate_hard_tokens(
                net=p,
                token_ids=ids,
                sampling_strategy="sample" if temperature > 0 else "greedy",
                temperature=max(temperature, 1e-6),
                eos_token_id=model_info.eos_token_id,
                pad_token_id=model_info.pad_token_id,
                max_tokens_generated=max_tokens,
                use_kv_cache=True,
                attention_mask=mask,
                use_bf16=use_bf16,
            ).tokens
            gen_lists = out[:, max_len:].tolist()
            texts = tokenizer.decode_batch(gen_lists)
            for i, txt in enumerate(texts):
                ex = extract_from_answer_tags(txt)
                gen_ids = gen_lists[i]
                if model_info.eos_token_id in gen_ids:
                    gen_len = gen_ids.index(model_info.eos_token_id) + 1
                    truncated = False
                else:
                    gen_len = len(gen_ids)
                    truncated = True  # ran to the token budget without EOS
                results[start + i]["samples"].append(
                    (txt, ex, _grade(ex, batch[i].equation, env), gen_len, truncated)
                )
        log.info(f"  generated {min(start + batch_size, len(prompts))}/{len(prompts)}")
    return results


def _aggregate(results):
    def stats(rs):
        if not rs:
            return None
        samples = [s for r in rs for s in r["samples"]]
        n_samp = len(samples)
        n_corr = sum(s[2] for s in samples)
        any_corr = sum(1 for r in rs if any(s[2] for s in r["samples"]))
        return {
            "n_prompts": len(rs),
            "pass_rate": n_corr / max(n_samp, 1),
            "pass_at_n": any_corr / len(rs),
            "mean_gen_len": sum(s[3] for s in samples) / max(n_samp, 1),
            "trunc_rate": sum(s[4] for s in samples) / max(n_samp, 1),
            "no_tag_rate": sum(1 for s in samples if s[1] is None) / max(n_samp, 1),
        }

    hard = [r for r in results if r["artifact_rate"] <= HARD_PROMPT_THRESHOLD]
    return {"all": stats(results), "hard": stats(hard), "n_hard": len(hard)}


def _correct(r):
    return any(s[2] for s in r["samples"])


def _first_correct_or_first(r):
    for s in r["samples"]:
        if s[2]:
            return s
    return r["samples"][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-run", required=True)
    ap.add_argument("--ckpt-step", type=int, required=True)
    ap.add_argument("--env", required=True, choices=["gsm8k", "countdown"])
    ap.add_argument(
        "--compare-step",
        type=int,
        default=None,
        help="second (e.g. pre-leap) step of the same run",
    )
    ap.add_argument(
        "--rollout-artifact",
        default=None,
        help="defaults to the SFT run's p_rollout_artifact",
    )
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-prompts", type=int, default=150)
    ap.add_argument("--examples", type=int, default=12)
    ap.add_argument("--out", default="spot_check_test.md")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_run_config(args.ckpt_run)
    if cfg is None:
        raise ValueError(f"could not load config for {args.ckpt_run}")
    sft = cfg.get("sft_params", {})
    model_name = sft.get("model_name")
    use_bf16 = bool(sft.get("use_bf16"))
    artifact = args.rollout_artifact or sft.get("p_rollout_artifact")
    if model_name is None or artifact is None:
        raise ValueError(
            "could not resolve model_name / p_rollout_artifact from the SFT run config"
        )
    log.info(
        f"model={model_name} use_bf16={use_bf16} lora_rank={sft.get('lora_rank')} artifact={artifact}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    model_info = MODEL_REGISTRY[model_name]
    tokenizer = model_info.load_tokenizer()

    by_split = load_rollout_artifact(artifact, tokenizer, filter_train_split=False)
    prompts = [pr for pr in by_split.get(args.split, []) if pr.equation is not None]
    if not prompts:
        raise ValueError(f"no prompts for split={args.split} in {artifact}")
    rng = random.Random(args.seed)
    if len(prompts) > args.max_prompts:
        prompts = rng.sample(prompts, args.max_prompts)
    log.info(
        f"Spot-checking {len(prompts)} {args.split} prompts (n_samples={args.n_samples}, temp={args.temperature})"
    )

    gen_kw = dict(
        prompts=prompts,
        tokenizer=tokenizer,
        model_info=model_info,
        n_samples=args.n_samples,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        device=device,
        use_bf16=use_bf16,
        env=args.env,
    )

    log.info(f"=== checkpoint step {args.ckpt_step} ===")
    p_post = _load_sft_ckpt(
        args.ckpt_run, args.ckpt_step, model_info, sft, device, dtype
    )
    res_post = _generate(p_post, **gen_kw)
    agg_post = _aggregate(res_post)
    del p_post
    torch.cuda.empty_cache()

    agg_pre = res_pre = None
    if args.compare_step is not None:
        log.info(f"=== compare checkpoint step {args.compare_step} ===")
        p_pre = _load_sft_ckpt(
            args.ckpt_run, args.compare_step, model_info, sft, device, dtype
        )
        res_pre = _generate(p_pre, **gen_kw)
        agg_pre = _aggregate(res_pre)
        del p_pre
        torch.cuda.empty_cache()

    def fmt(agg):
        a, h = agg["all"], agg["hard"]
        return (
            f"all: pass_rate={a['pass_rate']:.3f} pass@{args.n_samples}={a['pass_at_n']:.3f} "
            f"len={a['mean_gen_len']:.0f} trunc={a['trunc_rate']:.2f} notag={a['no_tag_rate']:.2f}  |  "
            f"hard(n={agg['n_hard']}): pass_rate={h['pass_rate']:.3f} pass@{args.n_samples}={h['pass_at_n']:.3f} "
            f"len={h['mean_gen_len']:.0f} trunc={h['trunc_rate']:.2f}"
        )

    log.info(f"\nTEST split aggregate — step {args.ckpt_step}:  {fmt(agg_post)}")
    if agg_pre:
        log.info(f"TEST split aggregate — step {args.compare_step}:  {fmt(agg_pre)}")

    with open(args.out, "w") as fh:
        fh.write(f"# Test-split spot check — `{args.ckpt_run}`\n\n")
        fh.write(
            f"split={args.split}, {len(prompts)} prompts, n_samples={args.n_samples}, temp={args.temperature}\n\n"
        )
        fh.write(f"- step {args.ckpt_step}: {fmt(agg_post)}\n")
        if agg_pre:
            fh.write(f"- step {args.compare_step}: {fmt(agg_pre)}\n")

        if res_pre is None:
            for label, pred in [("CORRECT", True), ("WRONG", False)]:
                pool = [r for r in res_post if _correct(r) == pred]
                fh.write(f"\n## {label} ({len(pool)})\n")
                for r in pool[: args.examples]:
                    s = _first_correct_or_first(r)
                    txt, ex = s[0], s[1]
                    fh.write(
                        f"\n**gold:** `{r['gold']}` · **extracted:** `{ex}` · artifact_rate={r['artifact_rate']:.2f}\n\n"
                    )
                    fh.write(
                        f"**prompt:**\n```\n{r['prompt_str'][-800:].strip()}\n```\n\n"
                    )
                    fh.write(
                        f"**generation:**\n```\n{txt[:2000].strip()}\n```\n\n---\n"
                    )
        else:
            buckets = {"fixed": [], "broke": [], "both_wrong": [], "both_right": []}
            for rp, rq in zip(res_pre, res_post):
                pre_ok, post_ok = _correct(rp), _correct(rq)
                key = (
                    "fixed"
                    if (not pre_ok and post_ok)
                    else "broke"
                    if (pre_ok and not post_ok)
                    else "both_right"
                    if pre_ok
                    else "both_wrong"
                )
                buckets[key].append((rp, rq))
            fh.write(
                f"\n**Bucket counts (test):** fixed={len(buckets['fixed'])} "
                f"broke={len(buckets['broke'])} both_right={len(buckets['both_right'])} "
                f"both_wrong={len(buckets['both_wrong'])}\n"
            )
            log.info(
                f"compare buckets: fixed={len(buckets['fixed'])} broke={len(buckets['broke'])} "
                f"both_right={len(buckets['both_right'])} both_wrong={len(buckets['both_wrong'])}"
            )
            for key in ("fixed", "broke", "both_wrong"):
                fh.write(f"\n## {key} ({len(buckets[key])})\n")
                for rp, rq in buckets[key][: args.examples]:
                    sp, sq = _first_correct_or_first(rp), _first_correct_or_first(rq)
                    tp, ep = sp[0], sp[1]
                    tq, eq = sq[0], sq[1]
                    fh.write(
                        f"\n**gold:** `{rq['gold']}` · artifact_rate={rq['artifact_rate']:.2f}\n\n"
                    )
                    fh.write(
                        f"**prompt:**\n```\n{rq['prompt_str'][-800:].strip()}\n```\n\n"
                    )
                    fh.write(
                        f"**pre (step {args.compare_step}) extracted=`{ep}`:**\n```\n{tp[:1500].strip()}\n```\n\n"
                    )
                    fh.write(
                        f"**post (step {args.ckpt_step}) extracted=`{eq}`:**\n```\n{tq[:1500].strip()}\n```\n\n---\n"
                    )
    log.info(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
