"""vLLM-backed variant of ``generate_inverse_cot_rollouts``.

Shares dataset loading, sharding, filtering and artifact upload with the
pure-PyTorch launcher next door. The only differences are the generation
backend (``vllm.LLM`` instead of ``generate_hard_tokens``) and the one-time
checkpoint export that produces a HuggingFace-format directory for vLLM to
read.

For a small model like Qwen3-0.6B this is typically 10–25× faster than the
PyTorch path thanks to vLLM's continuous batching, PagedAttention, and
automatic prefix-cache reuse across the ``group_size`` samples per prompt.
"""

import dataclasses
import json
import os
import tempfile
from datetime import datetime
from typing import TYPE_CHECKING

import extty
import torch
from tqdm import tqdm

from dialectic.distributed import (
    barrier,
    cleanup,
    get_rank,
    get_world_size,
    init_distributed,
    is_main_process,
)
from dialectic.experiments.arg_parser import Experiment, run_experiments_parser
from dialectic.experiments.envs import get_state_to_str
from dialectic.experiments.launchers._rollout_filter import filter_countdown_rollouts
from dialectic.experiments.params import RolloutGenParams
from dialectic.experiments.prompts import PromptCollection
from dialectic.llm.registry import MODEL_REGISTRY
from dialectic.llm.vllm_loader import load_dialectic_qwen_as_vllm
from dialectic.log import log
from dialectic.rl.env import Countdown
from dialectic.rl.types import EnvResponse

if TYPE_CHECKING:
    from vllm import LLM


def _load_dataset_problems(
    artifact_name: str,
    prompt_template: str,
) -> list[tuple[EnvResponse[Countdown], dict]]:
    data = extty.load_artifact(artifact_name, cache=True)
    if not isinstance(data, bytes):
        raise ValueError(f"Expected bytes from artifact, got {type(data)}")

    problems: list[tuple[EnvResponse[Countdown], dict]] = []
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
    log.info(f"Loaded {len(problems)} problems from artifact '{artifact_name}'")
    return problems


def _vllm_generate_group(
    llm: "LLM",
    prompts: list[str],
    *,
    group_size: int,
    temperature: float,
    max_tokens_generated: int,
    eos_token_id: int,
    seed: int,
) -> list[list[str]]:
    """Run ``llm.generate`` and return completions shaped as ``[problem][sample]``."""
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        n=group_size,
        temperature=temperature if temperature > 0 else 0.0,
        max_tokens=max_tokens_generated,
        stop_token_ids=[eos_token_id],
        seed=seed,
    )

    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

    completions_by_problem: list[list[str]] = []
    for out in outputs:
        completions_by_problem.append([sample.text for sample in out.outputs])
    return completions_by_problem


@extty.experiment(project="generate-inverse-cot-rollouts")
def generate_inverse_cot_rollouts_vllm(
    *,
    rollout_gen_params: RolloutGenParams,
    prompt_collection: PromptCollection,
    dataset_artifacts: list[str],
):
    init_distributed()
    rank = get_rank()
    world_size = get_world_size()
    torch.manual_seed(rollout_gen_params.seed + rank)

    model_info = MODEL_REGISTRY[rollout_gen_params.model_name]
    tokenizer = model_info.load_tokenizer()

    net = model_info.load_net(
        pretrained_weights=rollout_gen_params.start_ckpt_run is None
    )
    if rollout_gen_params.start_ckpt_run is not None:
        if rollout_gen_params.start_ckpt_step is None:
            raise ValueError("`start_ckpt_step` required when `start_ckpt_run` is set")
        project, run_name = rollout_gen_params.start_ckpt_run.split("/")
        net.load_state_dict(
            extty.load_checkpoint_from(
                project=project,
                run_name=run_name,
                step=rollout_gen_params.start_ckpt_step,
            )["model_state_dict"]
        )
    net.eval()
    net.requires_grad_(False)

    state_to_str = get_state_to_str(
        format_messages=model_info.format_messages,
        system_prompt=prompt_collection.system_prompt,
        assistant_prefill=prompt_collection.assistant_prefill,
    )

    all_problems: list[tuple[EnvResponse[Countdown], dict]] = []
    for artifact_name in dataset_artifacts:
        all_problems.extend(
            _load_dataset_problems(
                artifact_name, prompt_template=prompt_collection.env_prompt
            )
        )
    log.info(
        f"Total: {len(all_problems)} problems from {len(dataset_artifacts)} artifact(s)"
    )

    max_model_len = rollout_gen_params.max_tokens_generated + 1024

    llm = load_dialectic_qwen_as_vllm(
        net,
        tokenizer=tokenizer,
        eos_token_id=model_info.eos_token_id,
        pad_token_id=model_info.pad_token_id,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.90,
        dtype="bfloat16" if rollout_gen_params.use_bf16 else "float16",
        seed=rollout_gen_params.seed + rank,
    )
    del net
    torch.cuda.empty_cache()

    n_total = len(all_problems)
    n_shards = rollout_gen_params.n_shards
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if n_shards < world_size:
        raise ValueError(f"n_shards ({n_shards}) must be >= world_size ({world_size})")

    shard_size = (n_total + n_shards - 1) // n_shards
    total_kept = 0

    my_shard_indices = list(range(rank, n_shards, world_size))
    my_n_problems = sum(
        min((idx + 1) * shard_size, n_total) - idx * shard_size
        for idx in my_shard_indices
    )
    pbar = tqdm(
        total=my_n_problems,
        desc=f"Generating rollouts (rank {rank}, vllm)",
        disable=not is_main_process(),
    )
    for shard_idx in my_shard_indices:
        shard_start = shard_idx * shard_size
        shard_end = min(shard_start + shard_size, n_total)
        shard_problems = all_problems[shard_start:shard_end]

        tmpfile = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".jsonl",
            delete=False,
            prefix="inverse_cot_rollouts_vllm_",
        )
        log.info(
            f"Shard {shard_idx + 1}/{n_shards}: "
            f"{len(shard_problems)} problems -> {tmpfile.name}"
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
                batch_prompts = [state_to_str(resp.data) for resp in batch_responses]
                problem_idx = batch_end

                completions_by_problem = _vllm_generate_group(
                    llm,
                    batch_prompts,
                    group_size=rollout_gen_params.group_size,
                    temperature=rollout_gen_params.temperature,
                    max_tokens_generated=rollout_gen_params.max_tokens_generated,
                    eos_token_id=model_info.eos_token_id,
                    seed=rollout_gen_params.seed + rank + shard_start + problem_idx,
                )

                kept = filter_countdown_rollouts(
                    prompts=batch_prompts,
                    env_responses=batch_responses,
                    extra_fields=batch_extra,
                    completions_by_problem=completions_by_problem,
                    n_pos_min=rollout_gen_params.n_pos_min,
                    n_neg_min=rollout_gen_params.n_neg_min,
                )
                for entry in kept:
                    tmpfile.write(json.dumps(entry) + "\n")
                shard_kept += len(kept)
                pbar.update(len(batch))
                pbar.set_postfix(
                    shard=f"{shard_idx + 1}/{n_shards}", kept=total_kept + shard_kept
                )
                if is_main_process() and extty.has_active_run():
                    extty.log(
                        {
                            "processed": shard_start + problem_idx,
                            "kept": total_kept + shard_kept,
                            "shard": shard_idx + 1,
                        },
                        step=shard_start + problem_idx,
                    )
        finally:
            tmpfile.close()

            base_name = (
                f"inverse-cot-rollouts-{rollout_gen_params.model_name}-"
                f"g{rollout_gen_params.group_size}-{ts}"
            )
            if n_shards > 1:
                n_digits = len(str(n_shards))
                artifact_name = f"{base_name}-shard{str(shard_idx + 1).zfill(n_digits)}"
            else:
                artifact_name = base_name
            meta = extty.save_artifact(
                name=artifact_name,
                path=tmpfile.name,
                description=(
                    f"Inverse CoT rollouts shard {shard_idx + 1}/{n_shards}: "
                    f"{shard_kept}/{len(shard_problems)} prompts, "
                    f"group_size={rollout_gen_params.group_size} (vllm)"
                ),
                metadata={
                    "rollout_gen_params": dataclasses.asdict(rollout_gen_params),
                    "prompt_collection": dataclasses.asdict(prompt_collection),
                    "shard_idx": shard_idx,
                    "n_shards": n_shards,
                    "backend": "vllm",
                },
            )
            os.unlink(tmpfile.name)
            log.info(f"Uploaded shard {shard_idx + 1}/{n_shards}: {meta}")

        total_kept += shard_kept

    pbar.close()
    log.info(
        f"Rank {rank} done: {total_kept} prompts kept "
        f"across {len(my_shard_indices)} shard(s)"
    )
    barrier()
    cleanup()


if __name__ == "__main__":
    run_experiments_parser(
        [
            Experiment(
                env_name="countdown",
                fn=generate_inverse_cot_rollouts_vllm,
                include_prompt_collection_id=True,
                include_dataset_glob=True,
            ),
        ]
    )
