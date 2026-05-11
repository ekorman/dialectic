import copy
import json
import os
import tempfile
from datetime import datetime

import extty
import torch
from tqdm import tqdm

from dialectic.experiments.arg_parser import Experiment, run_experiments_parser
from dialectic.experiments.params import GenerateQCotParams
from dialectic.llm.generate import generate_hard_tokens
from dialectic.llm.registry import MODEL_REGISTRY
from dialectic.log import log
from dialectic.rl.extractors import extract_from_answer_tags
from dialectic.rl.inverse_cot_data import load_rollout_artifacts
from dialectic.rl.inverse_cot_eval import expressions_match


@extty.experiment(project="generate-q-cot")
def generate_q_cot_countdown(
    *,
    gen_params: GenerateQCotParams,
    dataset_artifacts: list[str],
):
    torch.manual_seed(gen_params.seed)
    model_info = MODEL_REGISTRY[gen_params.model_name]
    tokenizer = model_info.load_tokenizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if gen_params.use_bf16 else torch.float32

    # Load p
    log.info(
        f"Loading p from {gen_params.forward_ckpt_run} step {gen_params.forward_ckpt_step}"
    )
    p = model_info.load_net(pretrained_weights=False)
    project, run_name = gen_params.forward_ckpt_run.split("/")
    p_ckpt = extty.load_checkpoint_from(
        project=project,
        run_name=run_name,
        step=gen_params.forward_ckpt_step,
        load_optimizer=False,
    )
    p.load_state_dict(p_ckpt["model_state_dict"])
    p = p.to(device=device, dtype=dtype)
    p.requires_grad_(False)
    p.eval()

    q = copy.deepcopy(p)
    for layer in q.layers:
        layer.self_attn.causal = False

    q = q.to(device=device, dtype=dtype)

    # Load q checkpoint
    log.info(f"Loading q from {gen_params.q_ckpt_run} step {gen_params.q_ckpt_step}")
    q_project, q_run_name = gen_params.q_ckpt_run.split("/")
    q_ckpt = extty.load_checkpoint_from(
        project=q_project,
        run_name=q_run_name,
        step=gen_params.q_ckpt_step,
        load_optimizer=False,
    )
    state_dict = q_ckpt["model_state_dict"]
    state_dict.pop("_rng_torch", None)
    state_dict.pop("_rng_python", None)
    state_dict.pop("_rng_cuda", None)
    q.load_state_dict(state_dict)
    q.eval()

    # Load data
    by_split = load_rollout_artifacts(dataset_artifacts, tokenizer)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_generated = 0
    total_fcr_correct = 0

    for split_name, prompts in sorted(by_split.items()):
        fcr_prompts = [pr for pr in prompts if pr.equation is not None]
        if not fcr_prompts:
            log.info(f"Skipping split {split_name}: no prompts with equations")
            continue

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

            # Build q's prefix: prompt_ids + answer_ids
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

            # q generates CoT
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
                    use_bf16=gen_params.use_bf16,
                    attention_mask=q_prefix_mask,
                ).tokens

            q_cot_strs = [
                tokenizer.decode(q_completions[i, q_max_prefix_len:].tolist())
                for i in range(B)
            ]

            # Feed q's CoT to p, check if p gets the right answer
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
                    use_bf16=gen_params.use_bf16,
                ).tokens
            primed_len = primed_ids.shape[1]
            p_strs = tokenizer.decode_batch(p_completions[:, primed_len:].tolist())

            for i in range(B):
                pr = batch[i]
                extracted = extract_from_answer_tags(p_strs[i])
                fcr_correct = (
                    extracted is not None
                    and pr.equation is not None
                    and expressions_match(extracted, pr.equation)
                )

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

        pbar.close()
        tmpfile.close()

        fcr_rate = split_fcr_correct / max(split_generated, 1)
        log.info(
            f"  {split_name}: {split_generated} generated, "
            f"FCR={fcr_rate:.3f} ({split_fcr_correct}/{split_generated})"
        )

        artifact_name = f"q-cot-{gen_params.model_name}-{ts}-{split_name}"
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


if __name__ == "__main__":
    run_experiments_parser(
        [
            Experiment(
                env_name="countdown",
                fn=generate_q_cot_countdown,
                include_dataset_glob=True,
                include_prompt_collection_id=False,
            ),
        ]
    )
