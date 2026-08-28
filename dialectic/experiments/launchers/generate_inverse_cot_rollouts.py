"""vLLM-backed launcher for generating inverse-CoT rollout artifacts.

The file is split into four layers so adding a new env costs only a small
loader + an entry in ``__main__``:

1. Env-specific problem loaders (``_load_countdown_problems`` /
   ``_load_gsm8k_problems``).
2. Shared model + vLLM init (``_init_vllm_and_state_to_str``).
3. Shared generate / parse / upload loop
   (``_run_rollout_generation_loop``), which is env-agnostic and dispatches
   grading via the caller-supplied ``parse_fn``.
4. Thin ``@extty.experiment`` wrappers per env.

Single-process by design: vLLM's continuous batching saturates one GPU, and
for a small model like Qwen3-0.6B it's typically 10–25× faster than the
PyTorch equivalent thanks to PagedAttention and automatic prefix-cache reuse
across the ``group_size`` samples per prompt.
"""

import dataclasses
import json
import os
import re
import tempfile
from datetime import datetime
from typing import TYPE_CHECKING, Callable

import extty
import torch
from tqdm import tqdm

from dialectic.experiments.arg_parser import Experiment, run_experiments_parser
from dialectic.experiments.envs import get_state_to_str
from dialectic.experiments.launchers._rollout_parsing import (
    parse_countdown_rollouts,
    parse_gsm8k_rollouts,
)
from dialectic.experiments.params import GSM8kParams, RolloutGenParams
from dialectic.experiments.prompts import PromptCollection
from dialectic.llm.registry import MODEL_REGISTRY, ModelInfo
from dialectic.llm.vllm_loader import load_dialectic_qwen_as_vllm
from dialectic.log import log
from dialectic.rl.env import Countdown, MathState
from dialectic.rl.types import EnvResponse

if TYPE_CHECKING:
    from vllm import LLM


# --- Env-specific problem loaders -------------------------------------------


def _load_countdown_problems(
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


def _load_gsm8k_problems(
    path: str,
    prompt_template: str,
    split: str,
) -> list[tuple[EnvResponse[MathState], dict]]:
    """Load GSM8K problems from a local JSONL, tag with split.

    The parsed gold answer is stashed both in the ``MathState`` payload
    (for grading by ``parse_gsm8k_rollouts``) and in the passthrough
    ``equation`` field (so it lands in the emitted artifact's entry and
    flows into ``PreTokenizedPrompt.equation`` downstream).
    """
    problems: list[tuple[EnvResponse[MathState], dict]] = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            question = entry["question"]
            m = re.search(r"####\s*([^\n]+)", entry["answer"])
            if m is None:
                raise RuntimeError(f"Error extracting answer from {entry['answer']!r}")
            answer_str = m.group(1).strip().replace(",", "")
            problems.append(
                (
                    EnvResponse(
                        is_done=True,
                        data=MathState(
                            prompt=prompt_template.format(question=question),
                            answer=answer_str,
                            problem_type="gsm8k",
                        ),
                    ),
                    {"split": split, "equation": answer_str},
                )
            )
    log.info(f"Loaded {len(problems)} problems from '{path}' (split={split})")
    return problems


# --- Shared init + generation loop ------------------------------------------


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


def _init_vllm_and_state_to_str(
    rollout_gen_params: RolloutGenParams,
    prompt_collection: PromptCollection,
) -> tuple["LLM", ModelInfo, Callable]:
    """Build the vLLM engine + chat-formatted ``state_to_str``."""
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise RuntimeError(
            "rollout generation is single-process (sharding support was "
            "removed); run without torchrun — vLLM saturates one GPU"
        )
    torch.manual_seed(rollout_gen_params.seed)

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
                load_optimizer=False,
            )["model_state_dict"],
            strict=False,
        )
    net.eval()
    net.requires_grad_(False)

    state_to_str = get_state_to_str(
        format_messages=model_info.format_messages,
        system_prompt=prompt_collection.system_prompt,
        assistant_prefill=prompt_collection.assistant_prefill,
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
        seed=rollout_gen_params.seed,
    )
    del net
    torch.cuda.empty_cache()
    return llm, model_info, state_to_str


def _run_rollout_generation_loop(
    *,
    llm: "LLM",
    model_info: ModelInfo,
    state_to_str: Callable,
    rollout_gen_params: RolloutGenParams,
    prompt_collection: PromptCollection,
    all_problems: list[tuple[EnvResponse, dict]],
    parse_fn: Callable[..., list[dict]],
    env_name: str,
) -> None:
    """Env-agnostic generate / parse / upload loop (single process, one
    artifact).

    ``parse_fn`` must accept the same kwargs as ``parse_countdown_rollouts``
    and ``parse_gsm8k_rollouts``; ``_rollout_parsing`` exposes both with that
    contract. Every prompt is emitted (artifacts are complete); consumers
    apply their own filtering at load time.
    """
    n_total = len(all_problems)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log.info(f"Total: {n_total} problems")

    tmpfile = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".jsonl",
        delete=False,
        prefix="inverse_cot_rollouts_vllm_",
    )
    log.info(f"Writing rollouts to {tmpfile.name}")

    n_written = 0
    pbar = tqdm(total=n_total, desc="Generating rollouts (vllm)")
    try:
        problem_idx = 0
        while problem_idx < n_total:
            batch_end = min(problem_idx + rollout_gen_params.batch_size, n_total)
            batch = all_problems[problem_idx:batch_end]
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
                seed=rollout_gen_params.seed + problem_idx,
            )

            entries = parse_fn(
                prompts=batch_prompts,
                env_responses=batch_responses,
                extra_fields=batch_extra,
                completions_by_problem=completions_by_problem,
            )
            for entry in entries:
                tmpfile.write(json.dumps(entry) + "\n")
            n_written += len(entries)
            pbar.update(len(batch))
            if extty.has_active_run():
                extty.log({"processed": problem_idx}, step=problem_idx)
    finally:
        tmpfile.close()

        artifact_name = (
            f"inverse-cot-rollouts-{env_name}-{rollout_gen_params.model_name}-"
            f"g{rollout_gen_params.group_size}-{ts}"
        )
        meta = extty.save_artifact(
            name=artifact_name,
            path=tmpfile.name,
            description=(
                f"Inverse CoT rollouts: {n_written}/{n_total} prompts, "
                f"group_size={rollout_gen_params.group_size} (vllm)"
            ),
            metadata={
                "rollout_gen_params": dataclasses.asdict(rollout_gen_params),
                "prompt_collection": dataclasses.asdict(prompt_collection),
                "backend": "vllm",
            },
        )
        os.unlink(tmpfile.name)
        log.info(f"Uploaded: {meta}")

    pbar.close()
    log.info(f"Done: {n_written} prompts written")


# --- Experiment wrappers ----------------------------------------------------


@extty.experiment(project="generate-inverse-cot-rollouts-countdown")
def generate_inverse_cot_rollouts_vllm_countdown(
    *,
    rollout_gen_params: RolloutGenParams,
    prompt_collection: PromptCollection,
    dataset_artifacts: list[str],
):
    llm, model_info, state_to_str = _init_vllm_and_state_to_str(
        rollout_gen_params, prompt_collection
    )

    all_problems: list[tuple[EnvResponse, dict]] = []
    for artifact_name in dataset_artifacts:
        all_problems.extend(
            _load_countdown_problems(
                artifact_name, prompt_template=prompt_collection.env_prompt
            )
        )

    _run_rollout_generation_loop(
        llm=llm,
        model_info=model_info,
        state_to_str=state_to_str,
        rollout_gen_params=rollout_gen_params,
        prompt_collection=prompt_collection,
        all_problems=all_problems,
        parse_fn=parse_countdown_rollouts,
        env_name="countdown",
    )


@extty.experiment(project="generate-inverse-cot-rollouts-gsm8k")
def generate_inverse_cot_rollouts_vllm_gsm8k(
    *,
    rollout_gen_params: RolloutGenParams,
    prompt_collection: PromptCollection,
    gsm8k_params: GSM8kParams,
):
    llm, model_info, state_to_str = _init_vllm_and_state_to_str(
        rollout_gen_params, prompt_collection
    )

    all_problems: list[tuple[EnvResponse, dict]] = _load_gsm8k_problems(
        gsm8k_params.train_path, prompt_collection.env_prompt, "train"
    )
    if gsm8k_params.val_path:
        all_problems += _load_gsm8k_problems(
            gsm8k_params.val_path, prompt_collection.env_prompt, "val"
        )

    _run_rollout_generation_loop(
        llm=llm,
        model_info=model_info,
        state_to_str=state_to_str,
        rollout_gen_params=rollout_gen_params,
        prompt_collection=prompt_collection,
        all_problems=all_problems,
        parse_fn=parse_gsm8k_rollouts,
        env_name="gsm8k",
    )


if __name__ == "__main__":
    run_experiments_parser(
        [
            Experiment(
                env_name="countdown",
                fn=generate_inverse_cot_rollouts_vllm_countdown,
                include_prompt_collection_id=True,
                include_dataset_glob=True,
            ),
            Experiment(
                env_name="gsm8k",
                fn=generate_inverse_cot_rollouts_vllm_gsm8k,
                include_prompt_collection_id=True,
                include_dataset_glob=False,
            ),
        ]
    )
