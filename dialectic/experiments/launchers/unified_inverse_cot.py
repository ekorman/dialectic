import random
from typing import TYPE_CHECKING

import extty
import torch
import torch.nn.functional as F

from dialectic.distributed import (
    barrier,
    cleanup,
    get_device,
    get_local_rank,
    get_rank,
    get_world_size,
    init_distributed,
    is_distributed,
    wrap_ddp,
)
from dialectic.experiments.arg_parser import Experiment, run_experiments_parser
from dialectic.experiments.envs import load_countdown_dataset_artifacts
from dialectic.experiments.models import load_model_and_opt
from dialectic.experiments.params import UnifiedInverseCotParams
from dialectic.experiments.prompts import PromptCollection
from dialectic.llm.registry import MODEL_REGISTRY
from dialectic.llm.vllm_weight_sync import build_vllm_for_training, sync_weights_to_vllm
from dialectic.log import log
from dialectic.rl.dataset_env import DatasetEnv
from dialectic.rl.env import Env
from dialectic.rl.evaluate import EvaluationResult
from dialectic.rl.extractors import extract_from_answer_tags
from dialectic.rl.reward import _evaluate_and_verify_countdown
from dialectic.training import StepFunctionReturn, train_loop

if TYPE_CHECKING:
    from vllm import LLM


def _vllm_generate(
    llm: "LLM",
    prompts: list[str],
    tokenizer,
    *,
    n: int,
    temperature: float,
    max_tokens: int,
    eos_token_id: int,
    seed: int | None = None,
) -> list[list[str]]:
    from vllm import SamplingParams

    tokenizer.no_padding()
    tokenizer.no_truncation()
    encs = tokenizer.encode_batch(prompts)
    prompt_token_ids = [list(e.ids) for e in encs]

    sampling_params = SamplingParams(
        n=n,
        temperature=temperature if temperature > 0 else 0.0,
        max_tokens=max_tokens,
        stop_token_ids=[eos_token_id],
        seed=seed,
    )

    from vllm import TokensPrompt

    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=ids) for ids in prompt_token_ids],
        sampling_params,
        use_tqdm=False,
    )

    return [[sample.text for sample in out.outputs] for out in outputs]


def _build_nll_batch(
    sequences: list[list[int]],
    prefix_lens: list[int],
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(s) for s in sequences)
    padded = [s + [pad_token_id] * (max_len - len(s)) for s in sequences]
    input_ids = torch.tensor(padded, dtype=torch.long, device=device)

    actual_lens = torch.tensor(
        [len(s) for s in sequences], dtype=torch.long, device=device
    )
    prefix = torch.tensor(prefix_lens, dtype=torch.long, device=device)
    positions = torch.arange(max_len, device=device).unsqueeze(0)
    loss_mask = (positions >= prefix.unsqueeze(1)) & (
        positions < actual_lens.unsqueeze(1)
    )

    return input_ids, loss_mask


def _compute_nll(
    net,
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    logits = net(input_ids, return_all_logits=True)
    shift_logits = logits[:, :-1]
    shift_targets = input_ids[:, 1:]
    shift_mask = loss_mask[:, 1:]

    n, l, v = shift_logits.shape
    per_token_loss = F.cross_entropy(
        shift_logits.reshape(n * l, v),
        shift_targets.reshape(n * l),
        reduction="none",
    ).reshape(n, l)

    masked = per_token_loss * shift_mask
    seq_lengths = shift_mask.sum(dim=1).clamp(min=1)
    return (masked.sum(dim=1) / seq_lengths).mean()


@extty.experiment(project="unified-inverse-cot-countdown")
def train_unified_inverse_cot_countdown(
    *,
    params: UnifiedInverseCotParams,
    prompt_collection: PromptCollection,
    dataset_artifacts: list[str],
):
    init_distributed()
    rank = get_rank()
    world_size = get_world_size()
    torch.manual_seed(params.seed + rank)
    random.seed(params.seed + rank)
    device = get_device()

    batch_size = params.batch_size
    if batch_size % world_size != 0:
        raise ValueError(
            f"batch_size ({batch_size}) must be divisible by world_size ({world_size})"
        )
    local_batch_size = batch_size // world_size

    model_info = MODEL_REGISTRY[params.model_name]
    tokenizer = model_info.load_tokenizer()

    # Load model
    if is_distributed() and not rank == 0:
        barrier()
    net, _ = load_model_and_opt(
        model_name=params.model_name,
        start_ckpt_run=params.start_ckpt_run,
        start_ckpt_step=params.start_ckpt_step,
        device=device,
        use_bf16=params.use_bf16,
        compile_model=False,
        load_opt=False,
    )
    if is_distributed() and rank == 0:
        barrier()

    trainable_params = list(net.parameters())
    opt = torch.optim.AdamW(trainable_params, lr=params.lr)

    net_raw = net
    if is_distributed():
        net = wrap_ddp(net, get_local_rank())

    log.info(
        f"params: {sum(p.numel() for p in net_raw.parameters()):,} "
        f"(rank {rank}/{world_size}, local_batch_size={local_batch_size})"
    )

    # Load dataset
    all_problems = load_countdown_dataset_artifacts(
        dataset_artifacts, prompt_template=prompt_collection.env_prompt
    )
    train_problems = [
        (resp, extra) for resp, extra in all_problems if extra.get("split") == "train"
    ]
    val_problems = [resp for resp, extra in all_problems if extra.get("split") == "val"]
    if not train_problems:
        raise ValueError("No training problems found")
    log.info(f"Train: {len(train_problems)}, Val: {len(val_problems)}")

    # Build forward and backward prompt formatters
    forward_state_to_str = _make_state_to_str(
        model_info.format_messages,
        system_prompt=prompt_collection.system_prompt,
        assistant_prefill=prompt_collection.assistant_prefill,
    )
    backward_state_to_str = _make_state_to_str(
        model_info.format_messages,
        system_prompt=params.backward_system_prompt,
        assistant_prefill=prompt_collection.assistant_prefill,
    )

    # Build vLLM engine
    vllm_max_model_len = params.max_tokens_generated + 1024
    llm = build_vllm_for_training(
        net,
        tokenizer=tokenizer,
        eos_token_id=model_info.eos_token_id,
        pad_token_id=model_info.pad_token_id,
        max_model_len=vllm_max_model_len,
        gpu_memory_utilization=params.vllm_gpu_memory_utilization,
        dtype="bfloat16" if params.use_bf16 else "float16",
        seed=params.seed + rank,
    )

    eos_token_id = model_info.eos_token_id
    pad_token_id = model_info.pad_token_id

    def _train_step(_step_idx: int) -> StepFunctionReturn:
        net.train()
        opt.zero_grad()

        total_forward_loss = 0.0
        total_backward_nll = 0.0
        total_contrastive_loss = 0.0
        total_contrastive_gap = 0.0
        n_contrastive_groups = 0
        local_episodes = 0

        for _ in range(params.accumulation_steps):
            # Sample problems
            indices = [
                random.randint(0, len(train_problems) - 1)
                for _ in range(local_batch_size)
            ]
            batch = [train_problems[i] for i in indices]
            batch_responses = [resp for resp, _ in batch]

            # --- Forward generation via vLLM ---
            forward_prompts = [
                forward_state_to_str(resp.data) for resp in batch_responses
            ]
            forward_completions = _vllm_generate(
                llm,
                forward_prompts,
                tokenizer,
                n=params.group_size,
                temperature=params.temperature,
                max_tokens=params.max_tokens_generated,
                eos_token_id=eos_token_id,
            )

            # Parse forward results
            per_prompt_correct_answers: list[list[str]] = []
            per_prompt_incorrect_answers: list[list[str]] = []
            per_prompt_correct_completions: list[list[str]] = []

            for p_idx, completions in enumerate(forward_completions):
                correct_answers = []
                incorrect_answers = []
                correct_completions = []
                numbers = batch_responses[p_idx].data.numbers
                target = batch_responses[p_idx].data.target

                for comp_str in completions:
                    extracted = extract_from_answer_tags(comp_str)
                    is_correct = (
                        extracted is not None
                        and _evaluate_and_verify_countdown(extracted, numbers, target)
                    )
                    if is_correct:
                        correct_answers.append(extracted)
                        correct_completions.append(comp_str)
                    elif extracted is not None:
                        incorrect_answers.append(extracted)

                per_prompt_correct_answers.append(correct_answers)
                per_prompt_incorrect_answers.append(incorrect_answers)
                per_prompt_correct_completions.append(correct_completions)

            # --- Forward NLL: train on correct rollouts ---
            forward_sequences: list[list[int]] = []
            forward_prefix_lens: list[int] = []

            for p_idx in range(local_batch_size):
                if not per_prompt_correct_completions[p_idx]:
                    continue
                prompt_ids = tokenizer.encode(forward_prompts[p_idx]).ids
                for comp_str in per_prompt_correct_completions[p_idx]:
                    response_ids = tokenizer.encode(comp_str).ids
                    forward_sequences.append(
                        list(prompt_ids) + list(response_ids) + [eos_token_id]
                    )
                    forward_prefix_lens.append(len(prompt_ids))

            forward_loss = torch.tensor(0.0, device=device, requires_grad=True)
            if forward_sequences:
                fwd_ids, fwd_mask = _build_nll_batch(
                    forward_sequences, forward_prefix_lens, pad_token_id, device
                )
                forward_loss = _compute_nll(net, fwd_ids, fwd_mask)

            # --- Backward generation via vLLM ---
            backward_prompts_list: list[str] = []
            backward_is_correct: list[bool] = []
            backward_group_boundaries: list[int] = []

            for p_idx in range(local_batch_size):
                correct = list(set(per_prompt_correct_answers[p_idx]))
                incorrect = list(set(per_prompt_incorrect_answers[p_idx]))
                if not correct and not incorrect:
                    continue

                prompt_str = backward_state_to_str(batch_responses[p_idx].data)
                group_count = 0
                for ans in correct:
                    backward_prompts_list.append(
                        prompt_str + f" <answer> {ans} </answer>"
                    )
                    backward_is_correct.append(True)
                    group_count += 1
                for ans in incorrect:
                    backward_prompts_list.append(
                        prompt_str + f" <answer> {ans} </answer>"
                    )
                    backward_is_correct.append(False)
                    group_count += 1

                prev = backward_group_boundaries[-1] if backward_group_boundaries else 0
                backward_group_boundaries.append(prev + group_count)

            backward_nll_loss = torch.tensor(0.0, device=device, requires_grad=True)
            contrastive_loss = torch.tensor(0.0, device=device, requires_grad=True)
            contrastive_gap = 0.0

            if backward_prompts_list:
                backward_completions = _vllm_generate(
                    llm,
                    backward_prompts_list,
                    tokenizer,
                    n=1,
                    temperature=params.temperature,
                    max_tokens=params.max_tokens_generated,
                    eos_token_id=eos_token_id,
                )

                # Build backward training sequences
                backward_sequences: list[list[int]] = []
                backward_prefix_lens: list[int] = []

                for b_idx, (prompt_str, comp_list) in enumerate(
                    zip(backward_prompts_list, backward_completions)
                ):
                    cot_str = comp_list[0]
                    prompt_ids = tokenizer.encode(prompt_str).ids
                    cot_ids = tokenizer.encode(cot_str).ids
                    backward_sequences.append(
                        list(prompt_ids) + list(cot_ids) + [eos_token_id]
                    )
                    backward_prefix_lens.append(len(prompt_ids))

                bwd_ids, bwd_mask = _build_nll_batch(
                    backward_sequences, backward_prefix_lens, pad_token_id, device
                )

                # Compute per-sequence NLL for contrastive
                logits = net(bwd_ids, return_all_logits=True)
                shift_logits = logits[:, :-1]
                shift_targets = bwd_ids[:, 1:]
                shift_mask = bwd_mask[:, 1:]

                n_seq, l_seq, v_size = shift_logits.shape
                per_token_loss = F.cross_entropy(
                    shift_logits.reshape(n_seq * l_seq, v_size),
                    shift_targets.reshape(n_seq * l_seq),
                    reduction="none",
                ).reshape(n_seq, l_seq)
                masked = per_token_loss * shift_mask
                seq_lengths = shift_mask.sum(dim=1).clamp(min=1)
                per_seq_nll = masked.sum(dim=1) / seq_lengths
                per_seq_logprob = -per_seq_nll

                is_correct_t = torch.tensor(
                    backward_is_correct, dtype=torch.bool, device=device
                )

                # NLL on correct backward CoTs
                if is_correct_t.any():
                    backward_nll_loss = per_seq_nll[is_correct_t].mean()

                # Contrastive: margin loss per group
                contrastive_losses: list[torch.Tensor] = []
                contrastive_gaps: list[float] = []
                prev_boundary = 0
                for boundary in backward_group_boundaries:
                    group_logprob = per_seq_logprob[prev_boundary:boundary]
                    group_correct = is_correct_t[prev_boundary:boundary]
                    prev_boundary = boundary

                    if group_correct.any() and not group_correct.all():
                        correct_mean = group_logprob[group_correct].mean()
                        incorrect_mean = group_logprob[~group_correct].mean()
                        gap = correct_mean - incorrect_mean
                        contrastive_gaps.append(gap.item())
                        contrastive_losses.append(
                            torch.clamp(params.contrastive_margin - gap, min=0.0)
                        )

                if contrastive_losses:
                    contrastive_loss = torch.stack(contrastive_losses).mean()
                    contrastive_gap = sum(contrastive_gaps) / len(contrastive_gaps)
                    n_contrastive_groups += len(contrastive_losses)

            # Combined loss — skip backward during warmup
            total_loss = (
                params.forward_weight * forward_loss
                + params.backward_weight
                * (backward_nll_loss + params.contrastive_weight * contrastive_loss)
            )
            (total_loss / params.accumulation_steps).backward()

            total_forward_loss += forward_loss.item()
            total_backward_nll += backward_nll_loss.item()
            total_contrastive_loss += contrastive_loss.item()
            total_contrastive_gap += contrastive_gap
            local_episodes += local_batch_size

        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable_params, params.max_grad_norm
        ).item()
        opt.step()

        # Sync weights to vLLM after optimizer step
        sync_weights_to_vllm(llm, net)

        accum = params.accumulation_steps
        global_episodes = local_episodes * world_size
        n_correct_total = sum(len(c) for c in per_prompt_correct_answers)
        n_incorrect_total = sum(len(c) for c in per_prompt_incorrect_answers)

        # Log a few forward completions as examples
        ex_prompts: list[str] = []
        ex_responses: list[list[str]] = []
        ex_rewards: list[dict] = []
        for p_idx, completions in enumerate(forward_completions):
            if len(ex_prompts) >= 5:
                break
            numbers = batch_responses[p_idx].data.numbers
            target = batch_responses[p_idx].data.target
            for comp_str in completions[:2]:
                extracted = extract_from_answer_tags(comp_str)
                is_correct = extracted is not None and _evaluate_and_verify_countdown(
                    extracted,
                    numbers,
                    target,
                )
                ex_prompts.append(forward_prompts[p_idx])
                ex_responses.append([comp_str])
                ex_rewards.append({"correct": float(is_correct)})

        metrics: dict = {
            "train/forward_loss": total_forward_loss / accum,
            "train/backward_nll": total_backward_nll / accum,
            "train/contrastive_loss": total_contrastive_loss / accum,
            "train/contrastive_gap": total_contrastive_gap / accum,
            "train/loss": (
                total_forward_loss + total_backward_nll + total_contrastive_loss
            )
            / accum,
            "train/grad_norm": grad_norm,
            "train/n_correct": n_correct_total,
            "train/n_incorrect": n_incorrect_total,
            "train/accuracy": n_correct_total
            / max(n_correct_total + n_incorrect_total, 1),
            "train/episodes": global_episodes,
        }
        if ex_prompts:
            metrics["train/example"] = extty.BatchExample(
                prompts=ex_prompts,
                responses=ex_responses,
                rewards=ex_rewards,
            )

        return StepFunctionReturn(
            n_episodes_processed=global_episodes,
            metrics=metrics,
        )

    # Val: generate from scratch (forward mode) and check accuracy
    val_envs: list[Env] = []
    if val_problems:
        val_envs = [DatasetEnv(val_problems, seed=2026, label="unified_val")]

    def _val_fn(_val_env: Env) -> tuple[EvaluationResult, list[extty.Example]]:
        net_raw.eval()
        max_val = params.val_episodes or len(val_problems)
        eval_problems = val_problems[:max_val]

        val_prompts = [forward_state_to_str(resp.data) for resp in eval_problems]
        completions = _vllm_generate(
            llm,
            val_prompts,
            tokenizer,
            n=1,
            temperature=params.temperature,
            max_tokens=params.max_tokens_generated,
            eos_token_id=eos_token_id,
        )

        correct = 0
        total = 0
        examples: list[extty.Example] = []
        for p_idx, comp_list in enumerate(completions):
            resp = eval_problems[p_idx]
            comp_str = comp_list[0]
            extracted = extract_from_answer_tags(comp_str)
            is_match = extracted is not None and _evaluate_and_verify_countdown(
                extracted, resp.data.numbers, resp.data.target
            )
            total += 1
            if is_match:
                correct += 1

            if len(examples) < 20:
                examples.append(
                    extty.Example(
                        prompt=val_prompts[p_idx][-100:],
                        responses=[comp_str[:300]],
                        rewards=[{"correct": float(is_match)}],
                    )
                )

        accuracy = correct / max(total, 1)

        return EvaluationResult(
            n_episodes=total,
            reward_mean=accuracy,
            reward_std=0.0,
            component_means={
                "accuracy": accuracy,
                "n_correct": correct,
                "n_total": total,
            },
        ), examples

    train_loop(
        max_episodes=params.max_episodes,
        save_ckpt_freq=params.save_ckpt_freq,
        val_freq=params.val_freq,
        net=net,
        opt=opt,
        train_step=_train_step,
        val_fn=_val_fn,
        val_envs=val_envs,
    )
    cleanup()


def _make_state_to_str(format_messages, system_prompt, assistant_prefill):
    from dialectic.experiments.envs import Message

    def _fn(data):
        msgs = []
        if system_prompt:
            msgs.append(Message(role="system", content=system_prompt))
        msgs.append(Message(role="user", content=data.prompt))
        ret = format_messages(msgs, True)
        if assistant_prefill:
            ret += assistant_prefill
        return ret

    return _fn


if __name__ == "__main__":
    run_experiments_parser(
        [
            Experiment(
                env_name="countdown",
                fn=train_unified_inverse_cot_countdown,
                include_prompt_collection_id=True,
                include_dataset_glob=True,
            ),
        ]
    )
