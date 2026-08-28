"""Generate q-CoTs at scale for offline data synthesis.

For each prompt in a rollout artifact:
1. q (prefix-LM, conditioned on prompt + gold answer) samples one CoT.
2. p (causal) greedy-decodes given prompt + ``<think>q_cot</think>`` — used to
   record whether q's CoT successfully primes p to the right answer
   (``fcr_correct``).

The emitted artifact is the SFT material for bootstrapping p past its
sampling ceiling: every entry pairs (prompt, q's CoT, gold answer) with a
correctness flag from p's downstream greedy decode.

Like the eval launchers, every field except ``--q-ckpt-run`` / ``--q-ckpt-step``
is auto-derived from q's extty config (``model_name``, ``use_bf16``,
``forward_ckpt_run/step``, both LoRA configs, and ``p_rollout_artifact``). p
and q are both run via native PyTorch — q under prefix-LM attention (its
training regime), p under causal attention (its training regime).
"""

import copy
import json
import os
import tempfile
from datetime import datetime

import extty
import torch
from tqdm import tqdm

from dialectic.experiments.arg_parser import Experiment, run_experiments_parser
from dialectic.experiments.launchers._eval_helpers import (
    load_with_optional_lora,
    resolve_eval_common_params,
    strip_rng_state,
)
from dialectic.experiments.params import GenerateQCotParams
from dialectic.llm.generate import generate_hard_tokens
from dialectic.llm.registry import MODEL_REGISTRY
from dialectic.log import log
from dialectic.rl.extractors import extract_from_answer_tags
from dialectic.rl.inverse_cot_data import load_rollout_artifact
from dialectic.rl.inverse_cot_eval import expressions_match, gsm8k_match


def _generate_q_cot(
    *,
    gen_params: GenerateQCotParams,
    grade_fn,
    env_name: str,
) -> None:
    cfg = resolve_eval_common_params(gen_params)

    torch.manual_seed(cfg.seed)
    model_info = MODEL_REGISTRY[cfg.model_name]
    tokenizer = model_info.load_tokenizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if cfg.use_bf16 else torch.float32

    # ---------------- p ----------------
    log.info(f"Loading p from {cfg.forward_ckpt_run} step {cfg.forward_ckpt_step}")
    p = model_info.load_net(pretrained_weights=False)
    p_project, p_run_name = cfg.forward_ckpt_run.split("/")
    p_ckpt = extty.load_checkpoint_from(
        project=p_project,
        run_name=p_run_name,
        step=cfg.forward_ckpt_step,
        load_optimizer=False,
    )
    p_state = p_ckpt["model_state_dict"]
    strip_rng_state(p_state)
    load_with_optional_lora(
        p,
        p_state,
        lora_rank=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_target_modules=cfg.lora_target_modules,
    )
    p = p.to(device=device, dtype=dtype)
    p.requires_grad_(False)
    p.eval()

    # ---------------- q ----------------
    q = copy.deepcopy(p)
    for layer in q.layers:
        layer.self_attn.causal = False
    q = q.to(device=device, dtype=dtype)

    log.info(f"Loading q from {cfg.q_ckpt_run} step {cfg.q_ckpt_step}")
    q_project, q_run_name = cfg.q_ckpt_run.split("/")
    q_ckpt = extty.load_checkpoint_from(
        project=q_project,
        run_name=q_run_name,
        step=cfg.q_ckpt_step,
        load_optimizer=False,
    )
    q_state = q_ckpt["model_state_dict"]
    strip_rng_state(q_state)
    load_with_optional_lora(
        q,
        q_state,
        lora_rank=cfg.q_lora_rank,
        lora_alpha=cfg.q_lora_alpha,
        lora_target_modules=cfg.q_lora_target_modules,
    )
    q.requires_grad_(False)
    q.eval()

    # ---------------- data ----------------
    # filter_train_split=False: the default mixed-correctness filter exists
    # for q TRAINING (the contrastive loss needs ≥1 correct and ≥1 incorrect
    # completion per prompt). For q-CoT synthesis we want every prompt —
    # especially the all-incorrect ones, which are exactly the rescue
    # targets where a q-generated CoT adds the most SFT value.
    by_split = load_rollout_artifact(
        cfg.p_rollout_artifact, tokenizer, filter_train_split=False
    )
    # Only the requested split (default "train") is processed — q-CoT
    # synthesis targets training data; generating for val/test produces
    # artifacts nothing consumes.
    if cfg.split not in by_split:
        raise ValueError(
            f"split {cfg.split!r} not found in rollout artifact "
            f"(available: {sorted(by_split)})"
        )
    by_split = {cfg.split: by_split[cfg.split]}

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_generated = 0
    total_fcr_correct = 0

    for split_name, prompts in sorted(by_split.items()):
        fcr_prompts = [pr for pr in prompts if pr.equation is not None]
        if not fcr_prompts:
            log.info(f"Skipping split {split_name}: no prompts with equations")
            continue
        if cfg.max_prompts is not None:
            fcr_prompts = fcr_prompts[: cfg.max_prompts]

        log.info(f"Generating CoTs for {len(fcr_prompts)} prompts (split={split_name})")

        tmpfile = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, prefix="q_cot_"
        )
        split_generated = 0
        split_fcr_correct = 0

        pbar = tqdm(total=len(fcr_prompts), desc=f"Generating ({split_name})")

        for batch_start in range(0, len(fcr_prompts), gen_params.batch_size):
            batch = fcr_prompts[batch_start : batch_start + gen_params.batch_size]
            B = len(batch)

            # q's prefix: prompt + " <answer> {gold} </answer>"
            q_prefix_id_lists = [
                pr.prompt_ids
                + tokenizer.encode(f" <answer> {pr.equation} </answer>").ids
                for pr in batch
            ]
            q_max_prefix_len = max(len(ids) for ids in q_prefix_id_lists)
            q_prefix_ids = torch.full(
                (B, q_max_prefix_len), model_info.pad_token_id, device=device
            )
            q_prefix_mask = torch.zeros(
                B, q_max_prefix_len, dtype=torch.bool, device=device
            )
            for i, ids in enumerate(q_prefix_id_lists):
                offset = q_max_prefix_len - len(ids)
                q_prefix_ids[i, offset:] = torch.tensor(ids, device=device)
                q_prefix_mask[i, offset:] = True

            with torch.no_grad():
                q_completions = generate_hard_tokens(
                    net=q,
                    token_ids=q_prefix_ids,
                    sampling_strategy="sample"
                    if gen_params.temperature > 0
                    else "greedy",
                    temperature=max(gen_params.temperature, 1e-6),
                    eos_token_id=model_info.eos_token_id,
                    pad_token_id=model_info.pad_token_id,
                    max_tokens_generated=gen_params.max_tokens_generated,
                    use_kv_cache=True,
                    use_bf16=cfg.use_bf16,
                    attention_mask=q_prefix_mask,
                ).tokens

            q_cot_strs = [
                tokenizer.decode(q_completions[i, q_max_prefix_len:].tolist())
                for i in range(B)
            ]

            # Prime p with prompt + <think>q_cot</think>, greedy-decode the answer
            prompt_strs = [tokenizer.decode(pr.prompt_ids) for pr in batch]
            primed_strs = [
                ps + "<think>\n" + cot.strip() + "\n</think>\n\n"
                for ps, cot in zip(prompt_strs, q_cot_strs)
            ]
            tokenizer.enable_padding(direction="left")
            primed_tokens = tokenizer.encode_batch(primed_strs)
            primed_mask = torch.tensor(
                [t.attention_mask for t in primed_tokens],
                dtype=torch.bool,
                device=device,
            )
            primed_ids = torch.tensor([t.ids for t in primed_tokens], device=device)

            with torch.no_grad():
                p_completions = generate_hard_tokens(
                    net=p,
                    token_ids=primed_ids,
                    sampling_strategy="greedy",
                    eos_token_id=model_info.eos_token_id,
                    pad_token_id=model_info.pad_token_id,
                    max_tokens_generated=gen_params.max_tokens_generated,
                    use_kv_cache=True,
                    attention_mask=primed_mask,
                    use_bf16=cfg.use_bf16,
                ).tokens
            primed_len = primed_ids.shape[1]
            p_strs = tokenizer.decode_batch(p_completions[:, primed_len:].tolist())

            for i in range(B):
                pr = batch[i]
                extracted = extract_from_answer_tags(p_strs[i])
                assert pr.equation is not None
                fcr_correct = grade_fn(extracted, pr.equation)

                entry = {
                    "prompt_ids": pr.prompt_ids,
                    "cot": q_cot_strs[i],
                    "answer": pr.equation,
                    "split": split_name,
                    "fcr_correct": fcr_correct,
                }
                tmpfile.write(json.dumps(entry) + "\n")
                split_generated += 1
                if fcr_correct:
                    split_fcr_correct += 1

            pbar.update(B)
            pbar.set_postfix(fcr=f"{split_fcr_correct}/{split_generated}")
            # step = prompts processed, so the x-axis reads as raw progress
            # (a batch-index step axis is easy to misread as a percentage).
            extty.log(
                {
                    "pct_processed": 100.0 * split_generated / len(fcr_prompts),
                    "fcr_rate": split_fcr_correct / split_generated,
                },
                step=split_generated,
            )

        pbar.close()
        tmpfile.close()

        fcr_rate = split_fcr_correct / max(split_generated, 1)
        log.info(
            f"  {split_name}: {split_generated} generated, "
            f"FCR={fcr_rate:.3f} ({split_fcr_correct}/{split_generated})"
        )

        artifact_name = f"q-cot-{env_name}-{cfg.model_name}-{ts}-{split_name}"
        extty.save_artifact(
            name=artifact_name,
            path=tmpfile.name,
            description=(
                f"q-generated CoTs ({split_name}): {split_generated} prompts, "
                f"FCR={fcr_rate:.3f}"
            ),
        )
        os.unlink(tmpfile.name)
        log.info(f"  Saved artifact: {artifact_name}")

        total_generated += split_generated
        total_fcr_correct += split_fcr_correct

    log.info(
        f"Done: {total_generated} total, "
        f"FCR={total_fcr_correct / max(total_generated, 1):.3f}"
    )


def _countdown_grade(extracted: str | None, gold: str) -> bool:
    return extracted is not None and expressions_match(extracted, gold)


@extty.experiment(project="generate-q-cot-countdown")
def generate_q_cot_countdown(*, gen_params: GenerateQCotParams) -> None:
    _generate_q_cot(
        gen_params=gen_params,
        grade_fn=_countdown_grade,
        env_name="countdown",
    )


@extty.experiment(project="generate-q-cot-gsm8k")
def generate_q_cot_gsm8k(*, gen_params: GenerateQCotParams) -> None:
    _generate_q_cot(
        gen_params=gen_params,
        grade_fn=gsm8k_match,
        env_name="gsm8k",
    )


if __name__ == "__main__":
    run_experiments_parser(
        [
            Experiment(
                env_name="countdown",
                fn=generate_q_cot_countdown,
                include_prompt_collection_id=False,
            ),
            Experiment(
                env_name="gsm8k",
                fn=generate_q_cot_gsm8k,
                include_prompt_collection_id=False,
            ),
        ]
    )
