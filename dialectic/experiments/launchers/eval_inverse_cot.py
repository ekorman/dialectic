import copy

import extty
import torch

from dialectic.experiments.arg_parser import Experiment, run_experiments_parser
from dialectic.experiments.params import InverseCotEvalParams
from dialectic.llm.inverse_cot import InverseCotModel
from dialectic.llm.registry import MODEL_REGISTRY
from dialectic.log import log
from dialectic.rl.extractors import extract_from_answer_tags
from dialectic.rl.inverse_cot_data import load_rollout_artifacts
from dialectic.rl.inverse_cot_eval import (
    compute_fcr,
    expressions_match,
)


@extty.experiment(project="eval-inverse-cot")
def eval_inverse_cot_countdown(
    *,
    eval_params: InverseCotEvalParams,
    dataset_artifacts: list[str],
):
    torch.manual_seed(eval_params.seed)
    model_info = MODEL_REGISTRY[eval_params.model_name]
    tokenizer = model_info.load_tokenizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load p
    log.info(
        f"Loading p from {eval_params.forward_ckpt_run} step {eval_params.forward_ckpt_step}"
    )
    p = model_info.load_net(pretrained_weights=False)
    project, run_name = eval_params.forward_ckpt_run.split("/")
    p_ckpt = extty.load_checkpoint_from(
        project=project,
        run_name=run_name,
        step=eval_params.forward_ckpt_step,
        load_optimizer=False,
    )
    p_state = p_ckpt["model_state_dict"]
    p_state.pop("_rng_torch", None)
    p_state.pop("_rng_python", None)
    p_state.pop("_rng_cuda", None)
    p.load_state_dict(p_state)
    dtype = torch.bfloat16 if eval_params.use_bf16 else torch.float32
    p = p.to(device=device, dtype=dtype)
    p.requires_grad_(False)
    p.eval()

    # Load data
    by_split = load_rollout_artifacts(
        dataset_artifacts, tokenizer, filter_train_split=False
    )
    prompts = by_split.get(eval_params.split, [])
    if not prompts:
        raise ValueError(f"No data found for split={eval_params.split}")
    log.info(f"Evaluating on {len(prompts)} prompts (split={eval_params.split})")

    if eval_params.baseline_only:
        from dialectic.llm.vllm_loader import load_dialectic_qwen_as_vllm
        from dialectic.rl.inverse_cot_eval import HARD_PROMPT_THRESHOLD, BaselineResult

        llm = load_dialectic_qwen_as_vllm(
            p,
            tokenizer=tokenizer,
            eos_token_id=model_info.eos_token_id,
            pad_token_id=model_info.pad_token_id,
            max_model_len=eval_params.max_tokens_generated + 1024,
            gpu_memory_utilization=0.90,
            dtype="bfloat16" if eval_params.use_bf16 else "float16",
            seed=eval_params.seed,
        )
        del p
        torch.cuda.empty_cache()

        from vllm import SamplingParams, TokensPrompt

        baseline_prompts = [pr for pr in prompts if pr.equation is not None]
        sampling_params = SamplingParams(
            n=eval_params.n_samples,
            temperature=eval_params.temperature if eval_params.temperature > 0 else 0.0,
            max_tokens=eval_params.max_tokens_generated,
            stop_token_ids=[model_info.eos_token_id],
            seed=eval_params.seed,
        )

        vllm_outputs = llm.generate(
            [TokensPrompt(prompt_token_ids=pr.prompt_ids) for pr in baseline_prompts],
            sampling_params,
            use_tqdm=True,
        )

        total = 0
        correct = 0
        hard_total = 0
        hard_correct = 0

        for pr, out in zip(baseline_prompts, vllm_outputs):
            p_rate = (
                sum(c.is_correct for c in pr.completions) / len(pr.completions)
                if pr.completions
                else 0.0
            )
            any_correct = False
            for sample in out.outputs:
                extracted = extract_from_answer_tags(sample.text)
                if (
                    extracted is not None
                    and pr.equation is not None
                    and expressions_match(extracted, pr.equation)
                ):
                    any_correct = True
                    break

            total += 1
            if any_correct:
                correct += 1
            if p_rate <= HARD_PROMPT_THRESHOLD:
                hard_total += 1
                if any_correct:
                    hard_correct += 1

        result = BaselineResult(
            accuracy=correct / max(total, 1),
            total=total,
            correct=correct,
            hard_accuracy=hard_correct / max(hard_total, 1),
            hard_total=hard_total,
        )

        n_label = (
            f"pass@{eval_params.n_samples}" if eval_params.n_samples > 1 else "accuracy"
        )
        log.info(
            f"Baseline results ({result.total} prompts, temp={eval_params.temperature}, n={eval_params.n_samples}):"
        )
        log.info(
            f"  {n_label}:      {result.accuracy:.4f} ({result.correct}/{result.total})"
        )
        log.info(
            f"  hard_{n_label}: {result.hard_accuracy:.4f} ({result.hard_total} hard prompts)"
        )

        if extty.has_active_run():
            extty.log(
                {
                    "baseline_accuracy": result.accuracy,
                    "baseline_hard_accuracy": result.hard_accuracy,
                    "baseline_hard_total": result.hard_total,
                    "n_prompts": result.total,
                    "n_samples": eval_params.n_samples,
                },
                step=0,
            )
        return

    # Build q
    if eval_params.full_finetune or eval_params.finetune_freeze_mlp:
        q = copy.deepcopy(p)
        for layer in q.layers:
            layer.self_attn.causal = False
    else:
        q = InverseCotModel(p, unfreeze_mlp=eval_params.unfreeze_mlp)
    q = q.to(device=device, dtype=dtype)

    # Load q checkpoint
    log.info(f"Loading q from {eval_params.q_ckpt_run} step {eval_params.q_ckpt_step}")
    q_project, q_run_name = eval_params.q_ckpt_run.split("/")
    q_ckpt = extty.load_checkpoint_from(
        project=q_project,
        run_name=q_run_name,
        step=eval_params.q_ckpt_step,
        load_optimizer=False,
    )
    state_dict = q_ckpt["model_state_dict"]
    state_dict.pop("_rng_torch", None)
    state_dict.pop("_rng_python", None)
    state_dict.pop("_rng_cuda", None)
    q.load_state_dict(state_dict)
    q.eval()

    # Compute FCR
    fcr_result = compute_fcr(
        p=p,
        q=q,
        prompts=prompts,
        tokenizer=tokenizer,
        eos_token_id=model_info.eos_token_id,
        pad_token_id=model_info.pad_token_id,
        max_tokens_generated=eval_params.max_tokens_generated,
        batch_size=eval_params.batch_size,
        use_bf16=eval_params.use_bf16,
        q_temperature=eval_params.temperature,
    )

    log.info(f"Results ({len(prompts)} prompts, temp={eval_params.temperature}):")
    log.info(
        f"  FCR:              {fcr_result.fcr:.4f} ({fcr_result.fcr_correct}/{fcr_result.fcr_total})"
    )
    log.info(f"  p_baseline:       {fcr_result.p_baseline:.4f}")
    log.info(f"  FCR - p_baseline: {fcr_result.fcr - fcr_result.p_baseline:+.4f}")
    log.info(
        f"  fcr_hard:         {fcr_result.fcr_hard:.4f} ({fcr_result.fcr_hard_total} hard prompts)"
    )
    log.info(
        f"  fcr_all_incorrect:{fcr_result.fcr_all_incorrect:.4f} ({fcr_result.fcr_all_incorrect_total} all-incorrect prompts)"
    )

    if extty.has_active_run():
        extty.log(
            {
                "fcr": fcr_result.fcr,
                "p_baseline": fcr_result.p_baseline,
                "fcr_lift": fcr_result.fcr - fcr_result.p_baseline,
                "fcr_hard": fcr_result.fcr_hard,
                "fcr_hard_total": fcr_result.fcr_hard_total,
                "fcr_all_incorrect": fcr_result.fcr_all_incorrect,
                "fcr_all_incorrect_total": fcr_result.fcr_all_incorrect_total,
                "n_prompts": fcr_result.fcr_total,
            },
            step=0,
        )


if __name__ == "__main__":
    run_experiments_parser(
        [
            Experiment(
                env_name="countdown",
                fn=eval_inverse_cot_countdown,
                include_dataset_glob=True,
                include_prompt_collection_id=False,
            ),
        ]
    )
