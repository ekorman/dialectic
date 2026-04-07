import dataclasses
import json
import os
import tempfile
from datetime import datetime

import extty
import torch
from tokenizers import Tokenizer
from tqdm import tqdm

from dialectic.experiments.arg_parser import Experiment, run_experiments_parser
from dialectic.experiments.envs import get_state_to_str
from dialectic.experiments.launchers.inverse_cot import _parse_cot_and_answer
from dialectic.experiments.params import RolloutGenParams
from dialectic.experiments.prompts import PromptCollection
from dialectic.llm.base import BaseTransformer
from dialectic.llm.generate import generate_hard_tokens
from dialectic.llm.registry import MODEL_REGISTRY
from dialectic.llm.utils import get_default_device
from dialectic.rl.env import Countdown
from dialectic.rl.extractors import extract_from_answer_tags
from dialectic.rl.reward import _evaluate_and_verify_countdown
from dialectic.rl.types import EnvResponse


def _load_dataset_problems(
    artifact_name: str,
    prompt_template: str,
) -> list[tuple[EnvResponse[Countdown], dict]]:
    """Load problems from a dataset artifact.

    Returns list of (env_response, extra_fields) tuples, where extra_fields
    contains passthrough fields like 'split' and 'equation'.
    """
    data = extty.load_artifact(artifact_name)
    if not isinstance(data, bytes):
        raise ValueError(f"Expected bytes from artifact, got {type(data)}")

    problems = []
    for line in data.decode().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        numbers = entry["numbers"]
        target = entry["target"]
        prompt = prompt_template.format(numbers=numbers, target=target)
        extra: dict = {}
        if "split" in entry:
            extra["split"] = entry["split"]
        if "equation" in entry:
            extra["equation"] = entry["equation"]
        problems.append(
            (
                EnvResponse(
                    is_done=True,
                    data=Countdown(
                        prompt=prompt, numbers=numbers, target=target, solution=None
                    ),
                ),
                extra,
            )
        )
    print(f"Loaded {len(problems)} problems from artifact '{artifact_name}'")
    return problems


def _generate_rollouts_for_batch(
    *,
    p: BaseTransformer,
    env_responses: list[EnvResponse[Countdown]],
    extra_fields: list[dict],
    state_to_str,
    tokenizer: Tokenizer,
    eos_token_id: int,
    pad_token_id: int,
    group_size: int,
    temperature: float,
    max_tokens_generated: int,
    use_bf16: bool,
    n_pos_min: int,
    n_neg_min: int,
) -> list[dict]:
    device = next(p.parameters()).device
    batch_size = len(env_responses)

    prompts = [state_to_str(resp.data) for resp in env_responses]

    tokenizer.enable_padding(direction="left")
    tokens = tokenizer.encode_batch(prompts)
    attention_mask = torch.tensor(
        [t.attention_mask for t in tokens], dtype=torch.bool, device=device
    )
    token_ids = torch.tensor([t.ids for t in tokens], device=device)

    expanded_token_ids = token_ids.repeat_interleave(group_size, dim=0)
    expanded_attention_mask = attention_mask.repeat_interleave(group_size, dim=0)

    completions = generate_hard_tokens(
        net=p,
        token_ids=expanded_token_ids,
        sampling_strategy="sample" if temperature > 0 else "greedy",
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        max_tokens_generated=max_tokens_generated,
        use_kv_cache=True,
        attention_mask=expanded_attention_mask,
        temperature=temperature if temperature > 0 else 1.0,
        use_bf16=use_bf16,
    ).tokens

    prompt_len = token_ids.shape[1]
    completion_strs = tokenizer.decode_batch(completions[:, prompt_len:].tolist())

    kept = []
    total_correct = 0
    total_incorrect = 0
    total_unparsed = 0
    for b in range(batch_size):
        group_completions = []
        for g in range(group_size):
            idx = b * group_size + g
            comp_str = completion_strs[idx]
            parsed = _parse_cot_and_answer(comp_str)
            if parsed is None:
                total_unparsed += 1
                continue
            cot, answer = parsed
            if not cot.strip():
                total_unparsed += 1
                continue

            extracted = extract_from_answer_tags(comp_str)
            is_correct = extracted is not None and _evaluate_and_verify_countdown(
                extracted,
                env_responses[b].data.numbers,
                env_responses[b].data.target,
            )
            if is_correct:
                total_correct += 1
            else:
                total_incorrect += 1
            group_completions.append(
                {"cot": cot, "answer": answer, "is_correct": is_correct}
            )

        n_pos = sum(1 for c in group_completions if c["is_correct"])
        n_neg = sum(1 for c in group_completions if not c["is_correct"])
        if n_pos >= n_pos_min and n_neg >= n_neg_min:
            entry = {
                "prompt_str": prompts[b],
                "numbers": env_responses[b].data.numbers,
                "target": env_responses[b].data.target,
                "completions": group_completions,
                **extra_fields[b],
            }
            kept.append(entry)

    total = total_correct + total_incorrect + total_unparsed
    print(
        f"Batch: {total_correct}/{total} correct, "
        f"{total_incorrect}/{total} incorrect, "
        f"{total_unparsed}/{total} unparsed, "
        f"{len(kept)}/{batch_size} prompts kept"
    )
    return kept


@extty.experiment(project="generate-inverse-cot-rollouts")
def generate_inverse_cot_rollouts(
    *,
    rollout_gen_params: RolloutGenParams,
    prompt_collection: PromptCollection,
):
    torch.manual_seed(rollout_gen_params.seed)

    model_info = MODEL_REGISTRY[rollout_gen_params.model_name]
    tokenizer = model_info.load_tokenizer()

    p = model_info.load_net(
        pretrained_weights=rollout_gen_params.start_ckpt_run is None
    )
    if rollout_gen_params.start_ckpt_run is not None:
        if rollout_gen_params.start_ckpt_step is None:
            raise ValueError("`start_ckpt_step` required when `start_ckpt_run` is set")
        project, run_name = rollout_gen_params.start_ckpt_run.split("/")
        p.load_state_dict(
            extty.load_checkpoint_from(
                project=project,
                run_name=run_name,
                step=rollout_gen_params.start_ckpt_step,
            )["model_state_dict"]
        )

    device = get_default_device()
    if rollout_gen_params.use_bf16:
        p = p.to(device=device, dtype=torch.bfloat16)
    else:
        p = p.to(device)
    p.requires_grad_(False)
    p.eval()

    state_to_str = get_state_to_str(
        format_messages=model_info.format_messages,
        system_prompt=prompt_collection.system_prompt,
        assistant_prefill=prompt_collection.assistant_prefill,
    )

    all_problems = _load_dataset_problems(
        rollout_gen_params.dataset_artifact,
        prompt_template=prompt_collection.env_prompt,
    )
    n_total = len(all_problems)
    n_shards = rollout_gen_params.n_shards
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    shard_size = (n_total + n_shards - 1) // n_shards
    total_kept = 0

    pbar = tqdm(total=n_total, desc="Generating rollouts")
    for shard_idx in range(n_shards):
        shard_start = shard_idx * shard_size
        shard_end = min(shard_start + shard_size, n_total)
        shard_problems = all_problems[shard_start:shard_end]

        tmpfile = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, prefix="inverse_cot_rollouts_"
        )
        print(
            f"Shard {shard_idx + 1}/{n_shards}: {len(shard_problems)} problems -> {tmpfile.name}"
        )

        shard_kept = 0
        problem_idx = 0
        try:
            while problem_idx < len(shard_problems):
                batch_end = min(
                    problem_idx + rollout_gen_params.batch_size, len(shard_problems)
                )
                batch = shard_problems[problem_idx:batch_end]
                batch_responses = [resp for resp, _ in batch]
                batch_extra = [extra for _, extra in batch]
                problem_idx = batch_end

                kept = _generate_rollouts_for_batch(
                    p=p,
                    env_responses=batch_responses,
                    extra_fields=batch_extra,
                    state_to_str=state_to_str,
                    tokenizer=tokenizer,
                    eos_token_id=model_info.eos_token_id,
                    pad_token_id=model_info.pad_token_id,
                    group_size=rollout_gen_params.group_size,
                    temperature=rollout_gen_params.temperature,
                    max_tokens_generated=rollout_gen_params.max_tokens_generated,
                    use_bf16=rollout_gen_params.use_bf16,
                    n_pos_min=rollout_gen_params.n_pos_min,
                    n_neg_min=rollout_gen_params.n_neg_min,
                )
                for entry in kept:
                    tmpfile.write(json.dumps(entry) + "\n")
                    tmpfile.flush()
                shard_kept += len(kept)
                pbar.update(len(batch))
                pbar.set_postfix(
                    shard=f"{shard_idx + 1}/{n_shards}", kept=total_kept + shard_kept
                )
        finally:
            tmpfile.close()

            shard_suffix = f"-shard{shard_idx + 1}of{n_shards}" if n_shards > 1 else ""
            artifact_name = (
                f"inverse-cot-rollouts-{rollout_gen_params.model_name}-"
                f"g{rollout_gen_params.group_size}-n{shard_kept}{shard_suffix}-{ts}"
            )
            meta = extty.save_artifact(
                name=artifact_name,
                path=tmpfile.name,
                description=(
                    f"Inverse CoT rollouts shard {shard_idx + 1}/{n_shards}: "
                    f"{shard_kept}/{len(shard_problems)} prompts, "
                    f"group_size={rollout_gen_params.group_size}"
                ),
                metadata={
                    "rollout_gen_params": dataclasses.asdict(rollout_gen_params),
                    "prompt_collection": dataclasses.asdict(prompt_collection),
                    "shard_idx": shard_idx,
                    "n_shards": n_shards,
                },
            )
            os.unlink(tmpfile.name)
            print(f"Uploaded shard {shard_idx + 1}/{n_shards}: {meta}")

        total_kept += shard_kept

    pbar.close()
    print(f"Done: {total_kept}/{n_total} prompts kept across {n_shards} shard(s)")


if __name__ == "__main__":
    run_experiments_parser(
        [
            Experiment(
                env_name="countdown",
                fn=generate_inverse_cot_rollouts,
                include_prompt_collection_id=True,
            ),
        ]
    )
