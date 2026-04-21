import copy

import extty
import torch

from dialectic.experiments.arg_parser import Experiment, run_experiments_parser
from dialectic.experiments.params import InverseCotEvalParams
from dialectic.llm.inverse_cot import InverseCotModel
from dialectic.llm.registry import MODEL_REGISTRY
from dialectic.log import log
from dialectic.rl.inverse_cot_data import load_rollout_artifacts
from dialectic.rl.inverse_cot_eval import compute_baseline, compute_fcr


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
    p.load_state_dict(p_ckpt["model_state_dict"])
    dtype = torch.bfloat16 if eval_params.use_bf16 else torch.float32
    p = p.to(device=device, dtype=dtype)
    p.requires_grad_(False)
    p.eval()

    # Load data
    by_split = load_rollout_artifacts(dataset_artifacts, tokenizer)
    prompts = by_split.get(eval_params.split, [])
    if not prompts:
        raise ValueError(f"No data found for split={eval_params.split}")
    log.info(f"Evaluating on {len(prompts)} prompts (split={eval_params.split})")

    if eval_params.baseline_only:
        result = compute_baseline(
            p=p,
            prompts=prompts,
            tokenizer=tokenizer,
            eos_token_id=model_info.eos_token_id,
            pad_token_id=model_info.pad_token_id,
            max_tokens_generated=eval_params.max_tokens_generated,
            batch_size=eval_params.batch_size,
            use_bf16=eval_params.use_bf16,
            temperature=eval_params.temperature,
        )

        log.info(
            f"Baseline results ({result.total} prompts, temp={eval_params.temperature}):"
        )
        log.info(
            f"  accuracy:      {result.accuracy:.4f} ({result.correct}/{result.total})"
        )
        log.info(
            f"  hard_accuracy: {result.hard_accuracy:.4f} ({result.hard_total} hard prompts)"
        )

        if extty.has_active_run():
            extty.log(
                {
                    "baseline_accuracy": result.accuracy,
                    "baseline_hard_accuracy": result.hard_accuracy,
                    "baseline_hard_total": result.hard_total,
                    "n_prompts": result.total,
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
