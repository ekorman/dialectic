from typing import Callable

import extty
import torch

from dialectic.experiments.arg_parser import Experiment, run_experiments_parser
from dialectic.experiments.params import GrpoEvalParams
from dialectic.llm.lora import DEFAULT_TARGET_MODULES, apply_lora, merge_lora
from dialectic.llm.registry import MODEL_REGISTRY
from dialectic.llm.vllm_loader import load_dialectic_qwen_as_vllm
from dialectic.log import log
from dialectic.rl.extractors import extract_from_answer_tags
from dialectic.rl.inverse_cot_data import load_rollout_artifacts
from dialectic.rl.inverse_cot_eval import (
    HARD_PROMPT_THRESHOLD,
    expressions_match,
    gsm8k_match,
)

try:
    from vllm import SamplingParams, TokensPrompt
except ModuleNotFoundError:
    log.error(
        "vllm is required to run `eval_grpo.py`. please make sure dialectic is installed with the vllm extra."
    )


GradeFn = Callable[[str | None, str], bool]


def _resolve_lora_targets(target_modules: str) -> tuple[str, ...]:
    if target_modules == "all":
        return DEFAULT_TARGET_MODULES
    if target_modules == "attn":
        return ("q_proj", "k_proj", "v_proj", "o_proj")
    if target_modules == "mlp":
        return ("gate_proj", "up_proj", "down_proj")
    raise ValueError(
        f"Unknown lora_target_modules: {target_modules!r} (expected 'all', 'attn', or 'mlp')"
    )


def _eval_grpo(
    *,
    eval_params: GrpoEvalParams,
    dataset_artifacts: list[str],
    grade_fn: GradeFn,
) -> None:
    torch.manual_seed(eval_params.seed)
    model_info = MODEL_REGISTRY[eval_params.model_name]
    tokenizer = model_info.load_tokenizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if eval_params.use_bf16 else torch.float32

    log.info(
        f"Loading p from {eval_params.forward_ckpt_run} step {eval_params.forward_ckpt_step}"
    )
    p = model_info.load_net(pretrained_weights=False)
    p_project, p_run_name = eval_params.forward_ckpt_run.split("/")
    p_ckpt = extty.load_checkpoint_from(
        project=p_project,
        run_name=p_run_name,
        step=eval_params.forward_ckpt_step,
        load_optimizer=False,
    )
    p_state = p_ckpt["model_state_dict"]
    for key in ("_rng_torch", "_rng_python", "_rng_cuda"):
        p_state.pop(key, None)

    if eval_params.lora_rank is not None:
        targets = _resolve_lora_targets(eval_params.lora_target_modules)
        apply_lora(
            p,
            rank=eval_params.lora_rank,
            alpha=eval_params.lora_alpha,
            target_modules=targets,
        )
        p.load_state_dict(p_state)
        merge_lora(p)
        log.info(f"Loaded and merged LoRA weights (rank={eval_params.lora_rank})")
    else:
        p.load_state_dict(p_state)

    p = p.to(device=device, dtype=dtype)
    p.requires_grad_(False)
    p.eval()

    by_split = load_rollout_artifacts(
        dataset_artifacts, tokenizer, filter_train_split=False
    )
    prompts = by_split.get(eval_params.split, [])
    prompts = [pr for pr in prompts if pr.equation is not None]
    if not prompts:
        raise ValueError(
            f"No prompts found for split={eval_params.split} with non-null `equation`"
        )
    if eval_params.max_prompts is not None:
        prompts = prompts[: eval_params.max_prompts]
    log.info(
        f"Evaluating on {len(prompts)} prompts "
        f"(split={eval_params.split}, n_samples={eval_params.n_samples}, "
        f"temperature={eval_params.temperature})"
    )

    llm = load_dialectic_qwen_as_vllm(
        p,
        tokenizer=tokenizer,
        eos_token_id=model_info.eos_token_id,
        pad_token_id=model_info.pad_token_id,
        max_model_len=eval_params.max_tokens_generated + 1024,
        gpu_memory_utilization=eval_params.gpu_memory_utilization,
        dtype="bfloat16" if eval_params.use_bf16 else "float16",
        seed=eval_params.seed,
    )
    del p
    torch.cuda.empty_cache()

    sampling_params = SamplingParams(
        n=eval_params.n_samples,
        temperature=eval_params.temperature if eval_params.temperature > 0 else 0.0,
        max_tokens=eval_params.max_tokens_generated,
        stop_token_ids=[model_info.eos_token_id],
        seed=eval_params.seed,
    )
    vllm_outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=pr.prompt_ids) for pr in prompts],
        sampling_params,
        use_tqdm=True,
    )

    total = 0
    any_correct_total = 0
    hard_total = 0
    hard_any_correct_total = 0
    sample_total = 0
    sample_correct_total = 0
    hard_sample_total = 0
    hard_sample_correct_total = 0

    for pr, out in zip(prompts, vllm_outputs):
        assert pr.equation is not None
        per_sample_correct = [
            grade_fn(extract_from_answer_tags(sample.text), pr.equation)
            for sample in out.outputs
        ]
        any_correct = any(per_sample_correct)
        n_correct = sum(per_sample_correct)

        artifact_rate = (
            sum(c.is_correct for c in pr.completions) / len(pr.completions)
            if pr.completions
            else 0.0
        )
        is_hard = artifact_rate <= HARD_PROMPT_THRESHOLD

        total += 1
        sample_total += len(per_sample_correct)
        sample_correct_total += n_correct
        if any_correct:
            any_correct_total += 1
        if is_hard:
            hard_total += 1
            hard_sample_total += len(per_sample_correct)
            hard_sample_correct_total += n_correct
            if any_correct:
                hard_any_correct_total += 1

    pass_at_n = any_correct_total / max(total, 1)
    pass_rate_at_n = sample_correct_total / max(sample_total, 1)
    hard_pass_at_n = hard_any_correct_total / max(hard_total, 1)
    hard_pass_rate_at_n = hard_sample_correct_total / max(hard_sample_total, 1)

    log.info(
        f"Results ({total} prompts, n_samples={eval_params.n_samples}, "
        f"temp={eval_params.temperature}):"
    )
    log.info(f"  pass_at_n:        {pass_at_n:.4f} ({any_correct_total}/{total})")
    log.info(f"  pass_rate_at_n:   {pass_rate_at_n:.4f}")
    log.info(
        f"  hard_pass_at_n:   {hard_pass_at_n:.4f} ({hard_any_correct_total}/{hard_total} hard)"
    )
    log.info(f"  hard_pass_rate_at_n: {hard_pass_rate_at_n:.4f}")

    if extty.has_active_run():
        extty.log(
            {
                "n_prompts": total,
                "n_samples": eval_params.n_samples,
                "pass_at_n": pass_at_n,
                "pass_rate_at_n": pass_rate_at_n,
                "hard_pass_at_n": hard_pass_at_n,
                "hard_pass_rate_at_n": hard_pass_rate_at_n,
                "hard_total": hard_total,
            },
            step=0,
        )


def _countdown_grade(extracted: str | None, gold: str) -> bool:
    return extracted is not None and expressions_match(extracted, gold)


@extty.experiment(project="eval-grpo-countdown")
def eval_grpo_countdown(
    *,
    eval_params: GrpoEvalParams,
    dataset_artifacts: list[str],
) -> None:
    _eval_grpo(
        eval_params=eval_params,
        dataset_artifacts=dataset_artifacts,
        grade_fn=_countdown_grade,
    )


@extty.experiment(project="eval-grpo-gsm8k")
def eval_grpo_gsm8k(
    *,
    eval_params: GrpoEvalParams,
    dataset_artifacts: list[str],
) -> None:
    _eval_grpo(
        eval_params=eval_params,
        dataset_artifacts=dataset_artifacts,
        grade_fn=gsm8k_match,
    )


if __name__ == "__main__":
    run_experiments_parser(
        [
            Experiment(
                env_name="countdown",
                fn=eval_grpo_countdown,
                include_dataset_glob=True,
                include_prompt_collection_id=False,
            ),
            Experiment(
                env_name="gsm8k",
                fn=eval_grpo_gsm8k,
                include_dataset_glob=True,
                include_prompt_collection_id=False,
            ),
        ]
    )
