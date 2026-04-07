import extty
import torch
from tqdm import tqdm

from dialectic.experiments.arg_parser import Experiment, run_experiments_parser
from dialectic.experiments.envs import get_state_to_str
from dialectic.experiments.launchers.generate_inverse_cot_rollouts import (
    _load_dataset_problems,
)
from dialectic.experiments.params import InverseCotEvalParams
from dialectic.experiments.prompts import PromptCollection
from dialectic.llm.generate import generate_hard_tokens
from dialectic.llm.inverse_cot import InverseCotModel
from dialectic.llm.registry import MODEL_REGISTRY
from dialectic.llm.utils import get_default_device
from dialectic.rl.extractors import extract_from_answer_tags
from dialectic.rl.reward import _evaluate_and_verify_countdown


@extty.experiment(project="eval-inverse-cot")
def eval_inverse_cot_countdown(
    *,
    eval_params: InverseCotEvalParams,
    prompt_collection: PromptCollection,
):
    torch.manual_seed(eval_params.seed)

    model_info = MODEL_REGISTRY[eval_params.model_name]
    tokenizer = model_info.load_tokenizer()

    # Load p
    p = model_info.load_net(pretrained_weights=eval_params.start_ckpt_run is None)
    if eval_params.start_ckpt_run is not None:
        if eval_params.start_ckpt_step is None:
            raise ValueError("`start_ckpt_step` required when `start_ckpt_run` is set")
        project, run_name = eval_params.start_ckpt_run.split("/")
        p.load_state_dict(
            extty.load_checkpoint_from(
                project=project, run_name=run_name, step=eval_params.start_ckpt_step
            )["model_state_dict"]
        )

    device = get_default_device()
    if eval_params.use_bf16:
        p = p.to(device=device, dtype=torch.bfloat16)
    else:
        p = p.to(device)
    p.requires_grad_(False)
    p.eval()

    # Build q from p, load q's checkpoint
    q = InverseCotModel(p)
    dtype = next(p.parameters()).dtype
    q = q.to(device=device, dtype=dtype)

    q_project, q_run_name = eval_params.q_ckpt_run.split("/")
    q_state_dict = extty.load_checkpoint_from(
        project=q_project, run_name=q_run_name, step=eval_params.q_ckpt_step
    )["model_state_dict"]
    q.load_state_dict(q_state_dict, strict=False)
    q.eval()

    state_to_str = get_state_to_str(
        format_messages=model_info.format_messages,
        system_prompt=prompt_collection.system_prompt,
        assistant_prefill=prompt_collection.assistant_prefill,
    )

    # Load dataset
    all_problems = _load_dataset_problems(
        eval_params.dataset_artifact,
        prompt_template=prompt_collection.env_prompt,
    )
    problems = [
        (resp, extra)
        for resp, extra in all_problems
        if extra.get("split") == eval_params.split
    ]
    print(f"Evaluating on {len(problems)} problems (split={eval_params.split})")

    baseline_correct = 0
    q_primed_correct = 0
    n_evaluated = 0

    baseline_examples: list[extty.Example] = []
    q_primed_examples: list[extty.Example] = []

    pbar = tqdm(total=len(problems), desc="Evaluating")
    problem_idx = 0
    while problem_idx < len(problems):
        batch_end = min(problem_idx + eval_params.batch_size, len(problems))
        batch = problems[problem_idx:batch_end]
        batch_responses = [resp for resp, _ in batch]
        batch_extras = [extra for _, extra in batch]
        B = len(batch)
        problem_idx = batch_end

        prompts = [state_to_str(resp.data) for resp in batch_responses]

        tokenizer.enable_padding(direction="left")
        tokens = tokenizer.encode_batch(prompts)
        attention_mask = torch.tensor(
            [t.attention_mask for t in tokens], dtype=torch.bool, device=device
        )
        token_ids = torch.tensor([t.ids for t in tokens], device=device)

        # --- Baseline: p generates from scratch ---
        baseline_completions = generate_hard_tokens(
            net=p,
            token_ids=token_ids,
            sampling_strategy="sample" if eval_params.temperature > 0 else "greedy",
            eos_token_id=model_info.eos_token_id,
            pad_token_id=model_info.pad_token_id,
            max_tokens_generated=eval_params.max_tokens_generated,
            use_kv_cache=True,
            attention_mask=attention_mask,
            temperature=eval_params.temperature if eval_params.temperature > 0 else 1.0,
        ).tokens
        prompt_len = token_ids.shape[1]
        baseline_strs = tokenizer.decode_batch(
            baseline_completions[:, prompt_len:].tolist()
        )

        # --- q-primed: q generates CoT from (prompt, ground truth answer) ---
        # tokenize prompt and answer separately, concatenate IDs
        # (matches how training constructs q's input)
        prompt_id_lists = [tokenizer.encode(prompt).ids for prompt in prompts]
        # strip "= target" from equation to match p's answer format
        equations = [extra["equation"].split("=")[0].strip() for extra in batch_extras]
        answer_id_lists = [
            tokenizer.encode(f" <answer> {eq} </answer>").ids for eq in equations
        ]
        q_prefix_id_lists = [
            p_ids + a_ids for p_ids, a_ids in zip(prompt_id_lists, answer_id_lists)
        ]
        q_max_prefix_len = max(len(ids) for ids in q_prefix_id_lists)
        # left-pad to align
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

        q_completions = generate_hard_tokens(
            net=q,
            token_ids=q_prefix_ids,
            sampling_strategy="greedy",
            eos_token_id=model_info.eos_token_id,
            pad_token_id=model_info.pad_token_id,
            max_tokens_generated=eval_params.q_max_tokens_generated,
            use_kv_cache=True,
            attention_mask=q_prefix_mask,
        ).tokens

        q_cot_strs = [
            tokenizer.decode(q_completions[i, q_max_prefix_len:].tolist())
            for i in range(B)
        ]

        # Build p's primed input: original prompt + q's CoT wrapped in think tags
        primed_strs = [
            prompt + "<think>\n" + q_cot.strip() + "\n</think>\n\n"
            for prompt, q_cot in zip(prompts, q_cot_strs)
        ]

        tokenizer.enable_padding(direction="left")
        primed_tokens = tokenizer.encode_batch(primed_strs)
        primed_mask = torch.tensor(
            [t.attention_mask for t in primed_tokens], dtype=torch.bool, device=device
        )
        primed_ids = torch.tensor([t.ids for t in primed_tokens], device=device)

        primed_completions = generate_hard_tokens(
            net=p,
            token_ids=primed_ids,
            sampling_strategy="sample" if eval_params.temperature > 0 else "greedy",
            eos_token_id=model_info.eos_token_id,
            pad_token_id=model_info.pad_token_id,
            max_tokens_generated=eval_params.max_tokens_generated,
            use_kv_cache=True,
            attention_mask=primed_mask,
            temperature=eval_params.temperature if eval_params.temperature > 0 else 1.0,
        ).tokens

        primed_prompt_len = primed_ids.shape[1]
        primed_strs_out = tokenizer.decode_batch(
            primed_completions[:, primed_prompt_len:].tolist()
        )

        # --- Evaluate correctness ---
        for b in range(B):
            numbers = batch_responses[b].data.numbers
            target = batch_responses[b].data.target

            baseline_extracted = extract_from_answer_tags(baseline_strs[b])
            baseline_ok = (
                baseline_extracted is not None
                and _evaluate_and_verify_countdown(baseline_extracted, numbers, target)
            )
            if baseline_ok:
                baseline_correct += 1

            primed_extracted = extract_from_answer_tags(primed_strs_out[b])
            primed_ok = primed_extracted is not None and _evaluate_and_verify_countdown(
                primed_extracted, numbers, target
            )
            if primed_ok:
                q_primed_correct += 1

            n_evaluated += 1

            if len(baseline_examples) < 20:
                baseline_examples.append(
                    extty.Example(
                        prompt=prompts[b],
                        responses=[baseline_strs[b]],
                        rewards=[{"correct": float(baseline_ok)}],
                    )
                )
                q_primed_examples.append(
                    extty.Example(
                        prompt=prompts[b],
                        responses=[
                            f"[q's CoT] {q_cot_strs[b]}\n[p's answer] {primed_strs_out[b]}"
                        ],
                        rewards=[{"correct": float(primed_ok)}],
                    )
                )

        pbar.update(B)
        pbar.set_postfix(
            baseline=f"{baseline_correct}/{n_evaluated}",
            q_primed=f"{q_primed_correct}/{n_evaluated}",
        )

        if extty.has_active_run():
            batch_metrics: dict = {
                "baseline/accuracy": baseline_correct / max(n_evaluated, 1),
                "q_primed/accuracy": q_primed_correct / max(n_evaluated, 1),
                "n_problems": n_evaluated,
            }
            if baseline_examples:
                batch_metrics["baseline/example"] = extty.BatchExample(
                    prompts=[e.prompt for e in baseline_examples],
                    responses=[e.responses for e in baseline_examples],
                    rewards=[e.rewards for e in baseline_examples],
                )
            if q_primed_examples:
                batch_metrics["q_primed/example"] = extty.BatchExample(
                    prompts=[e.prompt for e in q_primed_examples],
                    responses=[e.responses for e in q_primed_examples],
                    rewards=[e.rewards for e in q_primed_examples],
                )
            extty.log(batch_metrics, step=n_evaluated)

    pbar.close()

    baseline_acc = baseline_correct / max(n_evaluated, 1)
    q_primed_acc = q_primed_correct / max(n_evaluated, 1)

    print(f"Baseline accuracy: {baseline_acc:.4f} ({baseline_correct}/{n_evaluated})")
    print(f"q-primed accuracy: {q_primed_acc:.4f} ({q_primed_correct}/{n_evaluated})")


if __name__ == "__main__":
    run_experiments_parser(
        [
            Experiment(
                env_name="countdown",
                fn=eval_inverse_cot_countdown,
                include_prompt_collection_id=True,
            ),
        ]
    )
