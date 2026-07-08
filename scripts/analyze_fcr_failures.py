"""Categorize FCR=0 q-CoT examples: did q reason wrong, or did p fail to extract?

For a q-CoT artifact, take ``fcr_correct=False`` entries, re-prime p (the forward
GRPO checkpoint) with q's *stored* CoT, greedy-decode the answer exactly as
``generate_q_cot`` does, and bucket each failure:

  (a) bad_reasoning  — q's CoT does not reach the gold answer; p follows it to a
                       wrong answer. FCR is correct to reject these.
  (b) extraction_gap — q's CoT reaches the gold answer, but p's greedy decode
                       commits to a *different* answer. FCR rejects good data here
                       (the greedy-mode-checking strictness: p(A|C,P) may be high
                       even when the argmax continuation isn't A).
  (c) no_answer_tag  — q's CoT reaches the gold answer, but p emits no parseable
                       <answer> tag. A format failure, not a reasoning failure.

Only p is needed — q's CoTs are read from the artifact, so no q forward pass.

Usage:
    uv run python scripts/analyze_fcr_failures.py \
        --artifact q-cot-gsm8k-qwen3-0.6b-thinking-20260610_160830-train \
        --env gsm8k
"""

import argparse
import json
import random
import re

import extty
import torch

from dialectic.experiments.launchers._eval_helpers import (
    load_with_optional_lora,
    resolve_eval_common_params,
    strip_rng_state,
)
from dialectic.experiments.params import EvalCommonParams
from dialectic.llm.generate import generate_hard_tokens
from dialectic.llm.registry import MODEL_REGISTRY
from dialectic.log import log
from dialectic.rl.extractors import extract_from_answer_tags
from dialectic.rl.inverse_cot_eval import expressions_match, gsm8k_match


def _grade(extracted: str | None, gold: str, env: str) -> bool:
    if extracted is None:
        return False
    if env == "gsm8k":
        return gsm8k_match(extracted, gold)
    return expressions_match(extracted, gold)


def _cot_reaches_gold(cot: str, gold: str, env: str) -> bool:
    """Heuristic: does q's CoT *conclude* with the gold answer?

    gsm8k — compare the last number in the CoT to the gold number (robust).
    countdown — approximate: an <answer> tag inside the CoT that matches, or the
    gold expression appearing as a substring (q often writes the equation out).
    """
    if env == "gsm8k":
        nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", cot)
        return bool(nums) and gsm8k_match(nums[-1], gold)
    tag = extract_from_answer_tags(cot)
    if tag is not None and expressions_match(tag, gold):
        return True
    return gold.replace(" ", "") in cot.replace(" ", "")


def _trace_q_ckpt(artifact_name: str) -> tuple[str, int]:
    """artifact -> producing generate-q-cot run -> (q_ckpt_run, q_ckpt_step)."""
    meta = extty.get_artifact(artifact_name)
    if meta.run_project is None or meta.run_name is None:
        raise ValueError(f"artifact {artifact_name!r} has no producing-run linkage")
    run = extty.get_run(meta.run_project, meta.run_name)
    gp = run.config.get("gen_params", {}) if isinstance(run.config, dict) else {}
    q_run, q_step = gp.get("q_ckpt_run"), gp.get("q_ckpt_step")
    if q_run is None or q_step is None:
        raise ValueError(
            f"could not trace q_ckpt from {meta.run_project}/{meta.run_name}"
        )
    return q_run, q_step


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--env", required=True, choices=["gsm8k", "countdown"])
    ap.add_argument(
        "--max-examples", type=int, default=300, help="FCR=0 entries to process"
    )
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        help="match the generate_q_cot run's value for faithful FCR re-derivation",
    )
    ap.add_argument("--examples-per-bucket", type=int, default=8)
    ap.add_argument("--out", default="fcr_failure_examples.md")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    q_run, q_step = _trace_q_ckpt(args.artifact)
    log.info(f"Traced q ckpt {q_run} step {q_step}; resolving p from its lineage")
    cfg = resolve_eval_common_params(
        EvalCommonParams(q_ckpt_run=q_run, q_ckpt_step=q_step)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if cfg.use_bf16 else torch.float32
    model_info = MODEL_REGISTRY[cfg.model_name]
    tokenizer = model_info.load_tokenizer()

    log.info(f"Loading p from {cfg.forward_ckpt_run} step {cfg.forward_ckpt_step}")
    p = model_info.load_net(pretrained_weights=False)
    proj, name = cfg.forward_ckpt_run.split("/")
    ckpt = extty.load_checkpoint_from(
        project=proj, run_name=name, step=cfg.forward_ckpt_step, load_optimizer=False
    )
    state = ckpt["model_state_dict"]
    strip_rng_state(state)
    load_with_optional_lora(
        p,
        state,
        lora_rank=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_target_modules=cfg.lora_target_modules,
    )
    p = p.to(device=device, dtype=dtype)
    p.requires_grad_(False)
    p.eval()

    data = extty.load_artifact(args.artifact, cache=True)
    if not isinstance(data, bytes):
        raise ValueError(f"expected bytes from {args.artifact}")
    entries = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
    failures = [e for e in entries if not e.get("fcr_correct", False)]
    log.info(f"{len(failures)}/{len(entries)} entries have fcr_correct=False")

    rng = random.Random(args.seed)
    if len(failures) > args.max_examples:
        failures = rng.sample(failures, args.max_examples)

    tokenizer.no_padding()
    tokenizer.no_truncation()

    buckets = {"a_bad_reasoning": [], "b_extraction_gap": [], "c_no_answer_tag": []}
    n_redrive_correct = 0  # sanity: FCR=0 entries that now grade correct (drift)

    for start in range(0, len(failures), args.batch_size):
        batch = failures[start : start + args.batch_size]
        prompt_strs = [tokenizer.decode(e["prompt_ids"]) for e in batch]
        primed = [
            ps + "<think>\n" + e["cot"].strip() + "\n</think>\n\n"
            for ps, e in zip(prompt_strs, batch)
        ]
        tokenizer.enable_padding(direction="left")
        encs = tokenizer.encode_batch(primed)
        ids = torch.tensor([e.ids for e in encs], device=device)
        mask = torch.tensor(
            [e.attention_mask for e in encs], dtype=torch.bool, device=device
        )
        tokenizer.no_padding()

        out = generate_hard_tokens(
            net=p,
            token_ids=ids,
            sampling_strategy="greedy",
            eos_token_id=model_info.eos_token_id,
            pad_token_id=model_info.pad_token_id,
            max_tokens_generated=args.max_tokens,
            use_kv_cache=True,
            attention_mask=mask,
            use_bf16=cfg.use_bf16,
        ).tokens
        p_strs = tokenizer.decode_batch(out[:, ids.shape[1] :].tolist())

        for e, prompt_str, p_str in zip(batch, prompt_strs, p_strs):
            gold = e["answer"]
            p_ext = extract_from_answer_tags(p_str)
            has_tag = p_ext is not None and len(p_ext) > 0
            if _grade(p_ext, gold, args.env):
                n_redrive_correct += 1
            reaches = _cot_reaches_gold(e["cot"], gold, args.env)

            if not reaches:
                key = "a_bad_reasoning"
            elif not has_tag:
                key = "c_no_answer_tag"
            else:
                key = "b_extraction_gap"
            buckets[key].append(
                {
                    "prompt": prompt_str,
                    "gold": gold,
                    "cot": e["cot"],
                    "p_output": p_str,
                    "p_extracted": p_ext,
                }
            )
        log.info(
            f"  processed {min(start + args.batch_size, len(failures))}/{len(failures)}"
        )

    total = sum(len(v) for v in buckets.values())
    log.info(f"\nFCR=0 failure breakdown ({total} processed):")
    for key, items in buckets.items():
        log.info(
            f"  {key:18s}: {len(items):5d}  ({100 * len(items) / max(total, 1):.1f}%)"
        )
    log.info(
        f"  [sanity] re-derived as correct (greedy drift vs stored fcr): {n_redrive_correct}"
    )

    with open(args.out, "w") as fh:
        fh.write(f"# FCR=0 failure analysis — `{args.artifact}` ({args.env})\n\n")
        fh.write(f"Processed {total} fcr_correct=False entries.\n\n")
        for key, items in buckets.items():
            fh.write(
                f"- **{key}**: {len(items)} ({100 * len(items) / max(total, 1):.1f}%)\n"
            )
        fh.write(
            f"\n_Sanity: {n_redrive_correct} re-derived as correct (greedy/bf16 drift)._\n"
        )
        for key, items in buckets.items():
            fh.write(
                f"\n## {key} (showing {min(args.examples_per_bucket, len(items))})\n"
            )
            for ex in items[: args.examples_per_bucket]:
                fh.write(
                    f"\n**gold:** `{ex['gold']}` · **p extracted:** `{ex['p_extracted']}`\n\n"
                )
                fh.write(f"**prompt:**\n```\n{ex['prompt'][-1200:].strip()}\n```\n\n")
                fh.write(f"**q CoT:**\n```\n{ex['cot'][:2000].strip()}\n```\n\n")
                fh.write(
                    f"**p output:**\n```\n{ex['p_output'][:800].strip()}\n```\n\n---\n"
                )
    log.info(f"Wrote examples to {args.out}")


if __name__ == "__main__":
    main()
