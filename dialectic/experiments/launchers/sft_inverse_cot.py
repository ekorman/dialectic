import json
import random

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
from dialectic.experiments.models import load_model_and_opt
from dialectic.experiments.params import SftParams
from dialectic.llm.generate import generate_hard_tokens
from dialectic.llm.registry import MODEL_REGISTRY
from dialectic.log import log
from dialectic.rl.env import Env
from dialectic.rl.evaluate import EvaluationResult
from dialectic.rl.inverse_cot_data import load_rollout_artifacts
from dialectic.rl.inverse_cot_eval import compute_baseline
from dialectic.training import StepFunctionReturn, train_loop


def _load_sft_data(
    artifact_names: list[str],
    tokenizer,
    eos_token_id: int,
    fcr_filter: bool = False,
    use_rollout_data: bool = False,
) -> dict[str, list[dict]]:
    """Load training data for SFT.

    When ``use_rollout_data`` is False, loads q-generated CoT artifacts
    (one CoT per prompt). When True, loads original rollout artifacts
    and uses p's own correct CoTs (one example per correct completion).

    Returns dict mapping split name to list of tokenized examples.
    Each example has: prompt_ids, response_ids, full_ids.
    """
    by_split: dict[str, list[dict]] = {}

    for name in artifact_names:
        data = extty.load_artifact(name, cache=True)
        if not isinstance(data, bytes):
            raise ValueError(f"Expected bytes from artifact {name}, got {type(data)}")

        for line in data.decode().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)

            if use_rollout_data:
                prompt_str = entry["prompt_str"]
                prompt_ids = tokenizer.encode(prompt_str).ids
                split = entry.get("split", "train")
                for comp in entry["completions"]:
                    if not comp["is_correct"]:
                        continue
                    cot = comp["cot"]
                    answer = comp["answer"]
                    response_str = f"<think>\n{cot.strip()}\n</think>\n\n{answer}"
                    response_ids = tokenizer.encode(response_str).ids
                    full_ids = prompt_ids + response_ids + [eos_token_id]
                    by_split.setdefault(split, []).append(
                        {
                            "prompt_ids": prompt_ids,
                            "response_ids": response_ids,
                            "full_ids": full_ids,
                            "prefix_len": len(prompt_ids),
                        }
                    )
            else:
                if fcr_filter and not entry.get("fcr_correct", False):
                    continue

                prompt_ids = entry["prompt_ids"]
                cot = entry["cot"]
                answer = entry["answer"]
                split = entry.get("split", "train")

                response_str = (
                    f"<think>\n{cot.strip()}\n</think>\n\n<answer> {answer} </answer>"
                )
                response_ids = tokenizer.encode(response_str).ids

                full_ids = prompt_ids + response_ids + [eos_token_id]

                by_split.setdefault(split, []).append(
                    {
                        "prompt_ids": prompt_ids,
                        "response_ids": response_ids,
                        "full_ids": full_ids,
                        "prefix_len": len(prompt_ids),
                    }
                )

    for split, examples in sorted(by_split.items()):
        log.info(f"  {split}: {len(examples)} examples")
        if examples:
            ex = examples[0]
            decoded = tokenizer.decode(ex["full_ids"])
            log.info(f"  Sample ({split}): {decoded[:500]!r}")
            log.info(
                f"  prompt_ids len={len(ex['prompt_ids'])} response_ids len={len(ex['response_ids'])} total={len(ex['full_ids'])}"
            )
    return by_split


def _build_sft_batch(
    examples: list[dict],
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a padded batch from SFT examples.

    Returns (input_ids, loss_mask).
    """
    max_len = max(len(ex["full_ids"]) for ex in examples)

    padded = [
        ex["full_ids"] + [pad_token_id] * (max_len - len(ex["full_ids"]))
        for ex in examples
    ]
    input_ids = torch.tensor(padded, dtype=torch.long, device=device)

    prefix_lens = torch.tensor(
        [ex["prefix_len"] for ex in examples], dtype=torch.long, device=device
    )
    actual_lens = torch.tensor(
        [len(ex["full_ids"]) for ex in examples], dtype=torch.long, device=device
    )
    positions = torch.arange(max_len, device=device).unsqueeze(0)
    loss_mask = (positions >= prefix_lens.unsqueeze(1)) & (
        positions < actual_lens.unsqueeze(1)
    )

    return input_ids, loss_mask


@extty.experiment(project="sft-inverse-cot-countdown")
def train_sft_inverse_cot_countdown(
    *,
    sft_params: SftParams,
    dataset_artifacts: list[str],
):
    init_distributed()
    rank = get_rank()
    world_size = get_world_size()
    torch.manual_seed(sft_params.seed + rank)
    random.seed(sft_params.seed + rank)
    device = get_device()

    batch_size = sft_params.batch_size
    if batch_size % world_size != 0:
        raise ValueError(
            f"batch_size ({batch_size}) must be divisible by world_size ({world_size})"
        )
    local_batch_size = batch_size // world_size

    model_info = MODEL_REGISTRY[sft_params.model_name]
    tokenizer = model_info.load_tokenizer()

    # Load model
    if is_distributed() and not rank == 0:
        barrier()
    p, _ = load_model_and_opt(
        model_name=sft_params.model_name,
        start_ckpt_run=sft_params.start_ckpt_run,
        start_ckpt_step=sft_params.start_ckpt_step,
        device=device,
        use_bf16=sft_params.use_bf16,
        compile_model=sft_params.compile_model,
        load_opt=False,
    )
    if is_distributed() and rank == 0:
        barrier()

    trainable_params = list(p.parameters())
    opt = torch.optim.AdamW(trainable_params, lr=sft_params.lr)

    p_raw = p
    if is_distributed():
        p = wrap_ddp(p, get_local_rank())

    total_params = sum(pa.numel() for pa in p_raw.parameters())
    log.info(
        f"p total params: {total_params:,} "
        f"(rank {rank}/{world_size}, local_batch_size={local_batch_size})"
    )

    # Load SFT data
    by_split = _load_sft_data(
        dataset_artifacts,
        tokenizer,
        eos_token_id=model_info.eos_token_id,
        fcr_filter=sft_params.fcr_filter,
        use_rollout_data=sft_params.use_rollout_data,
    )
    train_data = by_split.get("train", [])
    val_data = by_split.get("val", [])
    if not train_data:
        raise ValueError("No training data found")

    # Load original rollout data for val accuracy evaluation
    val_prompts = []
    if sft_params.val_rollout_artifact is not None:
        from dialectic.experiments.arg_parser import resolve_artifact_glob

        val_artifact_names = resolve_artifact_glob(sft_params.val_rollout_artifact)
        rollout_splits = load_rollout_artifacts(val_artifact_names, tokenizer)
        val_prompts = rollout_splits.get("val", [])
        log.info(f"Loaded {len(val_prompts)} val prompts for accuracy eval")

    def _train_step(_step_idx: int) -> StepFunctionReturn:
        p.train()
        opt.zero_grad()

        total_loss_val = 0.0
        local_episodes = 0

        for _ in range(sft_params.accumulation_steps):
            indices = torch.randint(len(train_data), (local_batch_size,)).tolist()
            batch_examples = [train_data[i] for i in indices]

            input_ids, loss_mask = _build_sft_batch(
                batch_examples, model_info.pad_token_id, device
            )

            logits = p(input_ids, return_all_logits=True)
            shift_logits = logits[:, :-1]
            shift_targets = input_ids[:, 1:]
            shift_mask = loss_mask[:, 1:]

            B, L, V = shift_logits.shape
            per_token_loss = F.cross_entropy(
                shift_logits.reshape(B * L, V),
                shift_targets.reshape(B * L),
                reduction="none",
            ).reshape(B, L)

            masked_loss = per_token_loss * shift_mask
            seq_lengths = shift_mask.sum(dim=1).clamp(min=1)
            loss = (masked_loss.sum(dim=1) / seq_lengths).mean()

            (loss / sft_params.accumulation_steps).backward()
            total_loss_val += loss.item()
            local_episodes += local_batch_size

        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable_params, sft_params.max_grad_norm
        ).item()
        opt.step()

        global_episodes = local_episodes * world_size
        return StepFunctionReturn(
            n_episodes_processed=global_episodes,
            metrics={
                "train/loss": total_loss_val / sft_params.accumulation_steps,
                "train/grad_norm": grad_norm,
                "train/episodes": global_episodes,
            },
        )

    def _val_fn(_val_env: Env) -> tuple[EvaluationResult, list[extty.Example]]:
        p_raw.eval()

        max_val = sft_params.val_episodes or max(len(val_data), len(val_prompts))
        component_means: dict[str, float] = {}

        # Val loss on SFT data (if available)
        if val_data:
            val_examples = val_data[:max_val]
            total_val_loss = 0.0
            n_batches = 0
            for batch_start in range(0, len(val_examples), sft_params.val_batch_size):
                batch = val_examples[
                    batch_start : batch_start + sft_params.val_batch_size
                ]
                input_ids, loss_mask = _build_sft_batch(
                    batch, model_info.pad_token_id, device
                )
                with torch.no_grad():
                    logits = p_raw(input_ids, return_all_logits=True)
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
                batch_loss = (masked.sum(dim=1) / seq_lengths).mean().item()
                total_val_loss += batch_loss
                n_batches += 1
            component_means["val_loss"] = total_val_loss / max(n_batches, 1)

        # Accuracy: have p generate from scratch on val prompts
        examples: list[extty.Example] = []
        if val_prompts:
            eval_prompts = val_prompts[:max_val]
            baseline_result = compute_baseline(
                p=p_raw,
                prompts=eval_prompts,
                tokenizer=tokenizer,
                eos_token_id=model_info.eos_token_id,
                pad_token_id=model_info.pad_token_id,
                max_tokens_generated=sft_params.max_tokens_generated,
                batch_size=sft_params.val_batch_size,
                use_bf16=sft_params.use_bf16,
                temperature=sft_params.temperature,
            )
            component_means["accuracy"] = baseline_result.accuracy
            component_means["hard_accuracy"] = baseline_result.hard_accuracy

            # Qualitative examples: generate from p on first few val prompts
            n_examples = min(20, len(eval_prompts))
            example_prompts = eval_prompts[:n_examples]
            max_prompt_len = max(len(pr.prompt_ids) for pr in example_prompts)
            ex_ids = torch.full(
                (n_examples, max_prompt_len), model_info.pad_token_id, device=device
            )
            ex_mask = torch.zeros(
                n_examples, max_prompt_len, dtype=torch.bool, device=device
            )
            for i, pr in enumerate(example_prompts):
                offset = max_prompt_len - len(pr.prompt_ids)
                ex_ids[i, offset:] = torch.tensor(pr.prompt_ids, device=device)
                ex_mask[i, offset:] = True

            with torch.no_grad():
                ex_completions = generate_hard_tokens(
                    net=p_raw,
                    token_ids=ex_ids,
                    sampling_strategy="sample"
                    if sft_params.temperature > 0
                    else "greedy",
                    temperature=max(sft_params.temperature, 1e-6),
                    eos_token_id=model_info.eos_token_id,
                    pad_token_id=model_info.pad_token_id,
                    max_tokens_generated=sft_params.max_tokens_generated,
                    use_kv_cache=True,
                    attention_mask=ex_mask,
                    use_bf16=sft_params.use_bf16,
                ).tokens
            ex_strs = tokenizer.decode_batch(
                ex_completions[:, max_prompt_len:].tolist()
            )
            for i in range(n_examples):
                prompt_str = tokenizer.decode(example_prompts[i].prompt_ids)
                examples.append(
                    extty.Example(
                        prompt=prompt_str,
                        responses=[f"[p's generation] {ex_strs[i]}"],
                        rewards=[{"equation": example_prompts[i].equation or ""}],
                    )
                )

        return EvaluationResult(
            n_episodes=max_val,
            reward_mean=component_means.get(
                "accuracy", component_means.get("val_loss", 0.0)
            ),
            reward_std=0.0,
            component_means=component_means,
        ), examples

    val_envs: list[Env] = []
    if val_data or val_prompts:
        from dialectic.rl.dataset_env import DatasetEnv

        dummy_list = val_prompts if val_prompts else val_data
        val_envs = [DatasetEnv(dummy_list, seed=2026, label="sft_val")]

    train_loop(
        max_episodes=sft_params.max_episodes,
        save_ckpt_freq=sft_params.save_ckpt_freq,
        val_freq=sft_params.val_freq,
        net=p,
        opt=opt,
        train_step=_train_step,
        val_fn=_val_fn,
        val_envs=val_envs,
    )
    cleanup()


if __name__ == "__main__":
    run_experiments_parser(
        [
            Experiment(
                env_name="countdown",
                fn=train_sft_inverse_cot_countdown,
                include_dataset_glob=True,
                include_prompt_collection_id=False,
            ),
        ]
    )
