import extty
import torch

from dialectic.experiments.arg_parser import Experiment, run_experiments_parser
from dialectic.experiments.models import load_model_and_opt
from dialectic.experiments.params import InverseCotParams, TrainParams
from dialectic.experiments.prompts import PromptCollection
from dialectic.llm.generate import generate_hard_tokens
from dialectic.llm.inverse_cot import InverseCotModel
from dialectic.llm.registry import MODEL_REGISTRY
from dialectic.log import log
from dialectic.rl.dataset_env import DatasetEnv
from dialectic.rl.env import Env
from dialectic.rl.evaluate import EvaluationResult
from dialectic.rl.inverse_cot_data import (
    build_contrastive_batch,
    build_infonce_batch,
    build_shuffled_batch,
    load_rollout_artifacts,
    subsample_completions,
)
from dialectic.rl.inverse_cot_eval import compute_fcr
from dialectic.rl.inverse_cot_loss import compute_contrastive_loss, compute_nll_loss
from dialectic.training import StepFunctionReturn, train_loop


@extty.experiment(project="inverse-cot-countdown")
def train_inverse_cot_countdown(
    *,
    train_params: TrainParams,
    inverse_cot_params: InverseCotParams,
    prompt_collection: PromptCollection,
    dataset_artifacts: list[str],
):
    torch.manual_seed(train_params.seed)

    model_info = MODEL_REGISTRY[train_params.model_name]
    tokenizer = model_info.load_tokenizer()

    p, _ = load_model_and_opt(train_params=train_params)
    p.requires_grad_(False)
    p.eval()

    q = InverseCotModel(p)
    q.use_gradient_checkpointing = inverse_cot_params.gradient_checkpointing
    if inverse_cot_params.freeze_lm_head:
        q.lm_head.requires_grad_(False)
    device = next(p.parameters()).device
    dtype = next(p.parameters()).dtype
    q = q.to(device=device, dtype=dtype)

    trainable_params = [param for param in q.parameters() if param.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=train_params.lr)

    total_params = sum(param.numel() for param in q.parameters())
    trainable_count = sum(param.numel() for param in trainable_params)
    log.info(f"q total params: {total_params:,}, trainable: {trainable_count:,}")

    by_split = load_rollout_artifacts(
        dataset_artifacts,
        tokenizer,
        max_cot_tokens=inverse_cot_params.max_cot_tokens,
        # Ensure post-filter prompts retain enough rollouts for
        # `subsample_completions`'s uniform-K assumption. Default to
        # `train_group_size` if set, otherwise 2 (the minimum for the
        # mixed-correctness constraint).
        min_completions_per_prompt=inverse_cot_params.train_group_size or 2,
    )
    rollout_data = by_split.get("train", [])
    val_rollout_data = by_split.get("val", [])
    if not rollout_data:
        raise ValueError("No training data found (split='train')")

    val_envs: list[Env] = (
        [DatasetEnv(val_rollout_data, seed=2026, label="inverse_cot_val")]
        if val_rollout_data
        else []
    )

    def _train_step(_step_idx: int) -> StepFunctionReturn:
        q.train()
        opt.zero_grad()

        total_metrics: dict[str, float] = {}
        total_episodes = 0

        for _ in range(train_params.accumulation_steps):
            indices = torch.randint(
                len(rollout_data), (train_params.batch_size,)
            ).tolist()
            batch_prompts = [rollout_data[i] for i in indices]

            if inverse_cot_params.train_group_size is not None:
                batch_prompts = [
                    subsample_completions(pr, inverse_cot_params.train_group_size)
                    for pr in batch_prompts
                ]

            input_ids, prefix_lengths, loss_mask, valid_mask, _, group_size = (
                build_infonce_batch(
                    batch_prompts,
                    eos_token_id=model_info.eos_token_id,
                    pad_token_id=model_info.pad_token_id,
                    device=device,
                    n_negatives=inverse_cot_params.contrastive_n_negatives,
                )
            )

            loss, step_metrics = compute_contrastive_loss(
                q,
                input_ids,
                prefix_lengths,
                loss_mask,
                valid_mask,
                group_size=group_size,
                n_negatives=inverse_cot_params.contrastive_n_negatives,
                contrastive_weight=inverse_cot_params.contrastive_weight,
                contrastive_temperature=inverse_cot_params.contrastive_temperature,
                logprob_chunk_size=train_params.logprob_chunk_size,
            )
            (loss / train_params.accumulation_steps).backward()

            for k, v in step_metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + v
            total_episodes += train_params.batch_size

        torch.nn.utils.clip_grad_norm_(trainable_params, train_params.max_grad_norm)
        opt.step()

        metrics = {
            k: v / train_params.accumulation_steps for k, v in total_metrics.items()
        }
        metrics["train/episodes"] = total_episodes
        return StepFunctionReturn(n_episodes_processed=total_episodes, metrics=metrics)

    max_val = (
        train_params.val_episodes
        if train_params.val_episodes is not None
        else len(val_rollout_data)
    )

    def _val_fn(_val_env: Env) -> tuple[EvaluationResult, list[extty.Example]]:
        q.eval()

        all_nll: list[float] = []
        all_nll_correct: list[float] = []
        all_nll_incorrect: list[float] = []
        all_nll_shuffled: list[float] = []
        examples: list[extty.Example] = []

        n_episodes = 0
        while n_episodes < max_val:
            current_batch = min(train_params.val_batch_size, max_val - n_episodes)
            indices = torch.randint(len(val_rollout_data), (current_batch,)).tolist()
            batch_prompts = [val_rollout_data[i] for i in indices]

            if inverse_cot_params.train_group_size is not None:
                batch_prompts = [
                    subsample_completions(pr, inverse_cot_params.train_group_size)
                    for pr in batch_prompts
                ]

            # Two separate batches at val time: the old
            # `build_contrastive_batch` for per-sample NLL diagnostics
            # (stratified by correctness) and a fresh `build_infonce_batch`
            # for the training-style NLL + InfoNCE metric. They share
            # prompt data but lay out sequences differently.
            input_ids, prefix_lengths, loss_mask, is_correct, _ = (
                build_contrastive_batch(
                    batch_prompts,
                    model_info.eos_token_id,
                    model_info.pad_token_id,
                    device,
                )
            )

            _, _, per_sample_nlls = compute_nll_loss(
                q,
                input_ids,
                prefix_lengths,
                loss_mask,
                inverse_cot_params.normalize_by_sequence_length,
            )
            for sample_nll, correct in zip(per_sample_nlls, is_correct.tolist()):
                if correct:
                    all_nll_correct.append(sample_nll)
                else:
                    all_nll_incorrect.append(sample_nll)

            (
                nce_input_ids,
                nce_prefix_lengths,
                nce_loss_mask,
                nce_valid_mask,
                _,
                nce_group_size,
            ) = build_infonce_batch(
                batch_prompts,
                eos_token_id=model_info.eos_token_id,
                pad_token_id=model_info.pad_token_id,
                device=device,
                n_negatives=inverse_cot_params.contrastive_n_negatives,
            )

            _, step_metrics = compute_contrastive_loss(
                q,
                nce_input_ids,
                nce_prefix_lengths,
                nce_loss_mask,
                nce_valid_mask,
                group_size=nce_group_size,
                n_negatives=inverse_cot_params.contrastive_n_negatives,
                contrastive_weight=inverse_cot_params.contrastive_weight,
                contrastive_temperature=inverse_cot_params.contrastive_temperature,
                logprob_chunk_size=train_params.logprob_chunk_size,
            )
            all_nll.append(step_metrics["train/nll"])

            # shuffled answer diagnostic
            N = input_ids.shape[0]
            if N > 1:
                s_input_ids, s_prefix_lengths, s_loss_mask = build_shuffled_batch(
                    batch_prompts,
                    model_info.eos_token_id,
                    model_info.pad_token_id,
                    device,
                )
                _, nll_shuffled, _ = compute_nll_loss(
                    q,
                    s_input_ids,
                    s_prefix_lengths,
                    s_loss_mask,
                    inverse_cot_params.normalize_by_sequence_length,
                )
                all_nll_shuffled.append(nll_shuffled)

            # qualitative examples (cap at 20). `build_contrastive_batch`
            # emits exactly `len(prompt.completions)` rows per prompt, so we
            # can stride by the uniform group size to pick one example per
            # prompt.
            if len(examples) < 20:
                ex_offset = 0
                gs = len(batch_prompts[0].completions) if batch_prompts else 0
                for prompt_data in batch_prompts:
                    if len(examples) >= 20:
                        break
                    prefix_len = prefix_lengths[ex_offset].item()
                    prefix_ids = input_ids[ex_offset, :prefix_len].unsqueeze(0)

                    q_completion = generate_hard_tokens(
                        net=q,
                        token_ids=prefix_ids,
                        sampling_strategy="sample",
                        temperature=train_params.temperature,
                        eos_token_id=model_info.eos_token_id,
                        pad_token_id=model_info.pad_token_id,
                        max_tokens_generated=train_params.max_tokens_generated,
                        use_kv_cache=True,
                        use_bf16=train_params.use_bf16,
                    ).tokens

                    q_cot = tokenizer.decode(q_completion[0, prefix_len:].tolist())
                    target_cot = tokenizer.decode(prompt_data.completions[0].cot_ids)
                    prompt_str = tokenizer.decode(
                        input_ids[ex_offset, :prefix_len].tolist()
                    )
                    examples.append(
                        extty.Example(
                            prompt=prompt_str,
                            responses=[
                                f"[p's CoT] {target_cot}",
                                f"[q's CoT] {q_cot}",
                            ],
                            rewards=[
                                {"nll": step_metrics["train/nll"]},
                                {"nll": step_metrics["train/nll"]},
                            ],
                        )
                    )
                    ex_offset += gs

            n_episodes += current_batch

        nll_mean = sum(all_nll) / len(all_nll) if all_nll else 0.0
        nll_correct_mean = (
            sum(all_nll_correct) / len(all_nll_correct) if all_nll_correct else 0.0
        )
        nll_incorrect_mean = (
            sum(all_nll_incorrect) / len(all_nll_incorrect)
            if all_nll_incorrect
            else 0.0
        )
        nll_shuffled_mean = (
            sum(all_nll_shuffled) / len(all_nll_shuffled) if all_nll_shuffled else 0.0
        )

        # Forward Consistency Rate (FCR)
        fcr_result = compute_fcr(
            p=p,
            q=q,
            prompts=val_rollout_data[:max_val],
            tokenizer=tokenizer,
            eos_token_id=model_info.eos_token_id,
            pad_token_id=model_info.pad_token_id,
            max_tokens_generated=train_params.max_tokens_generated,
            batch_size=train_params.val_batch_size,
            use_bf16=train_params.use_bf16,
            q_temperature=train_params.temperature,
        )

        return EvaluationResult(
            n_episodes=n_episodes,
            reward_mean=nll_mean,
            reward_std=0.0,
            component_means={
                "nll_correct": nll_correct_mean,
                "nll_incorrect": nll_incorrect_mean,
                "nll_shuffled": nll_shuffled_mean,
                "fcr": fcr_result.fcr,
                "fcr_all_incorrect": fcr_result.fcr_all_incorrect,
                "fcr_n_all_incorrect": fcr_result.fcr_all_incorrect_total,
            },
        ), examples

    train_loop(
        max_episodes=train_params.max_episodes,
        save_ckpt_freq=train_params.save_ckpt_freq,
        val_freq=train_params.val_freq,
        net=q,
        opt=opt,
        train_step=_train_step,
        val_fn=_val_fn,
        val_envs=val_envs,
    )


if __name__ == "__main__":
    run_experiments_parser(
        [
            Experiment(
                env_name="countdown",
                fn=train_inverse_cot_countdown,
                include_prompt_collection_id=True,
                include_dataset_glob=True,
            ),
        ]
    )
