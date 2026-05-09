"""Evaluate q as a verifier: does q(CoT | answer, P) separate correct from incorrect rollouts?

For each prompt in the rollout data:
1. For each completion (CoT + answer from p):
   - Compute q(CoT | extracted_answer, P) — log prob of the CoT under q conditioned on p's own answer
2. Compare scores for correct vs incorrect completions
3. Report: AUC, mean score gap, and accuracy of "pick highest q-score" as a selection strategy
"""

import copy

import extty
import torch
import torch.nn.functional as F

from dialectic.experiments.arg_parser import Experiment, run_experiments_parser
from dialectic.experiments.params import InverseCotEvalParams
from dialectic.llm.inverse_cot import InverseCotModel, create_prefix_lm_mask
from dialectic.llm.registry import MODEL_REGISTRY
from dialectic.log import log
from dialectic.rl.extractors import extract_from_answer_tags
from dialectic.rl.inverse_cot_data import load_rollout_artifacts


@extty.experiment(project="eval-q-verifier")
def eval_q_verifier_countdown(
    *,
    eval_params: InverseCotEvalParams,
    dataset_artifacts: list[str],
):
    torch.manual_seed(eval_params.seed)
    model_info = MODEL_REGISTRY[eval_params.model_name]
    tokenizer = model_info.load_tokenizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if eval_params.use_bf16 else torch.float32

    # Load p (for architecture reference)
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
    p = p.to(device=device, dtype=dtype)
    p.requires_grad_(False)
    p.eval()

    # Build q
    if eval_params.full_finetune or eval_params.finetune_freeze_mlp:
        q = copy.deepcopy(p)
        for layer in q.layers:
            layer.self_attn.causal = False
    else:
        q = InverseCotModel(p, unfreeze_mlp=eval_params.unfreeze_mlp)
    q = q.to(device=device, dtype=dtype)

    # Load q checkpoint
    q_project, q_run_name = eval_params.q_ckpt_run.split("/")
    q_ckpt = extty.load_checkpoint_from(
        project=q_project,
        run_name=q_run_name,
        step=eval_params.q_ckpt_step,
        load_optimizer=False,
    )
    q_state = q_ckpt["model_state_dict"]
    q_state.pop("_rng_torch", None)
    q_state.pop("_rng_python", None)
    q_state.pop("_rng_cuda", None)
    q.load_state_dict(q_state)
    q.eval()

    # Load data
    by_split = load_rollout_artifacts(
        dataset_artifacts, tokenizer, filter_train_split=False
    )
    prompts = by_split.get(eval_params.split, [])
    if not prompts:
        raise ValueError(f"No data found for split={eval_params.split}")
    if eval_params.max_prompts is not None and len(prompts) > eval_params.max_prompts:
        import random

        rng = random.Random(eval_params.seed)
        prompts = rng.sample(prompts, eval_params.max_prompts)
    log.info(f"Evaluating on {len(prompts)} prompts (split={eval_params.split})")

    tokenizer.no_padding()
    tokenizer.no_truncation()

    correct_scores: list[float] = []
    incorrect_scores: list[float] = []
    n_prompts_evaluated = 0
    best_of_n_correct = 0
    best_of_n_total = 0
    random_correct = 0
    random_total = 0

    for pr_idx, pr in enumerate(prompts):
        if not pr.completions or pr.equation is None:
            continue

        # Score each completion
        scored_completions: list[tuple[float, bool]] = []

        for comp in pr.completions:
            # Extract answer from this completion
            extracted = extract_from_answer_tags(
                comp.answer if hasattr(comp, "answer") else ""
            )
            if extracted is None:
                # Try to get answer from the answer_ids
                answer_str = tokenizer.decode(comp.answer_ids)
                extracted = extract_from_answer_tags(answer_str)
                if extracted is None:
                    extracted = answer_str.strip()

            # Build q's input: prompt + answer + CoT
            prefix_ids = pr.prompt_ids + comp.answer_ids
            cot_ids = comp.cot_ids

            full_ids = prefix_ids + cot_ids + [model_info.eos_token_id]
            prefix_len = len(prefix_ids)

            if len(full_ids) > 2048:
                continue

            input_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
            prefix_lengths = torch.tensor([prefix_len], dtype=torch.long, device=device)

            with torch.no_grad():
                attention_mask = create_prefix_lm_mask(
                    prefix_lengths, len(full_ids), device
                )
                logits = q(
                    input_tensor, attention_mask=attention_mask, return_all_logits=True
                )

                shift_logits = logits[0, prefix_len - 1 : -1]
                shift_targets = input_tensor[0, prefix_len:]

                log_probs = -F.cross_entropy(
                    shift_logits.float(), shift_targets, reduction="none"
                )
                avg_log_prob = log_probs.mean().item()

            is_correct = comp.is_correct
            scored_completions.append((avg_log_prob, is_correct))

            if is_correct:
                correct_scores.append(avg_log_prob)
            else:
                incorrect_scores.append(avg_log_prob)

        if not scored_completions:
            continue

        n_prompts_evaluated += 1

        # Best-of-N: does the highest q-scored completion happen to be correct?
        best_score_comp = max(scored_completions, key=lambda x: x[0])
        best_of_n_total += 1
        if best_score_comp[1]:
            best_of_n_correct += 1

        # Random baseline: pick a random completion
        import random

        random_comp = random.choice(scored_completions)
        random_total += 1
        if random_comp[1]:
            random_correct += 1

        if pr_idx % 50 == 0 and pr_idx > 0:
            log.info(
                f"  [{pr_idx}/{len(prompts)}] "
                f"correct_mean={sum(correct_scores) / max(len(correct_scores), 1):.3f} "
                f"incorrect_mean={sum(incorrect_scores) / max(len(incorrect_scores), 1):.3f} "
                f"best_of_n={best_of_n_correct}/{best_of_n_total}"
            )

    # Results
    correct_mean = sum(correct_scores) / max(len(correct_scores), 1)
    incorrect_mean = sum(incorrect_scores) / max(len(incorrect_scores), 1)
    gap = correct_mean - incorrect_mean
    best_of_n_acc = best_of_n_correct / max(best_of_n_total, 1)
    random_acc = random_correct / max(random_total, 1)

    log.info(f"\nResults ({n_prompts_evaluated} prompts evaluated):")
    log.info(
        f"  q score (correct completions):   mean={correct_mean:.4f} (n={len(correct_scores)})"
    )
    log.info(
        f"  q score (incorrect completions): mean={incorrect_mean:.4f} (n={len(incorrect_scores)})"
    )
    log.info(f"  Score gap (correct - incorrect):  {gap:.4f}")
    log.info(
        f"  Best-of-N accuracy (pick highest q score): {best_of_n_acc:.4f} ({best_of_n_correct}/{best_of_n_total})"
    )
    log.info(
        f"  Random selection accuracy:                  {random_acc:.4f} ({random_correct}/{random_total})"
    )
    log.info(
        f"  Lift over random:                           {best_of_n_acc - random_acc:+.4f}"
    )

    if extty.has_active_run():
        extty.log(
            {
                "correct_score_mean": correct_mean,
                "incorrect_score_mean": incorrect_mean,
                "score_gap": gap,
                "best_of_n_accuracy": best_of_n_acc,
                "random_accuracy": random_acc,
                "lift": best_of_n_acc - random_acc,
                "n_correct_scores": len(correct_scores),
                "n_incorrect_scores": len(incorrect_scores),
                "n_prompts": n_prompts_evaluated,
            },
            step=0,
        )


if __name__ == "__main__":
    run_experiments_parser(
        [
            Experiment(
                env_name="countdown",
                fn=eval_q_verifier_countdown,
                include_dataset_glob=True,
                include_prompt_collection_id=False,
            ),
        ]
    )
