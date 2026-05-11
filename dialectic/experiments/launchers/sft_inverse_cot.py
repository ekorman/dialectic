import copy
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
    import time as _time

    by_split: dict[str, list[dict]] = {}

    tokenizer.no_padding()
    tokenizer.no_truncation()

    for name in artifact_names:
        t0 = _time.perf_counter()
        data = extty.load_artifact(name, cache=True)
        if not isinstance(data, bytes):
            raise ValueError(f"Expected bytes from artifact {name}, got {type(data)}")
        log.info(f"Downloaded artifact '{name}' in {_time.perf_counter() - t0:.1f}s")

        t1 = _time.perf_counter()
        entries = [
            json.loads(line) for line in data.decode().splitlines() if line.strip()
        ]
        log.info(f"Parsed {len(entries)} entries in {_time.perf_counter() - t1:.1f}s")

        t2 = _time.perf_counter()
        if use_rollout_data:
            # Batch tokenize: collect all strings first, encode in one call
            prompt_strs: list[str] = []
            response_strs: list[str] = []
            entry_indices: list[int] = []
            splits: list[str] = []

            cot_only_strs: list[str] = []
            for i, entry in enumerate(entries):
                split = entry.get("split", "train")
                for comp in entry["completions"]:
                    if not comp["is_correct"]:
                        continue
                    cot_str = f"<think>\n{comp['cot'].strip()}\n</think>\n\n"
                    prompt_strs.append(entry["prompt_str"])
                    response_strs.append(cot_str + comp["answer"])
                    cot_only_strs.append(cot_str)
                    entry_indices.append(i)
                    splits.append(split)

            log.info(
                f"Collected {len(prompt_strs)} correct completions in {_time.perf_counter() - t2:.1f}s"
            )
            t3 = _time.perf_counter()

            prompt_encs = tokenizer.encode_batch(prompt_strs)
            response_encs = tokenizer.encode_batch(response_strs)
            cot_encs = tokenizer.encode_batch(cot_only_strs)
            log.info(
                f"Batch tokenized {len(prompt_strs)} pairs in {_time.perf_counter() - t3:.1f}s"
            )

            for idx in range(len(prompt_strs)):
                prompt_ids = list(prompt_encs[idx].ids)
                response_ids = list(response_encs[idx].ids)
                cot_len = len(cot_encs[idx].ids)
                full_ids = prompt_ids + response_ids + [eos_token_id]
                by_split.setdefault(splits[idx], []).append(
                    {
                        "prompt_ids": prompt_ids,
                        "response_ids": response_ids,
                        "full_ids": full_ids,
                        "prefix_len": len(prompt_ids),
                        "cot_len": cot_len,
                    }
                )
        else:
            for entry in entries:
                if fcr_filter and not entry.get("fcr_correct", False):
                    continue

                prompt_ids = entry["prompt_ids"]
                cot = entry["cot"]
                answer = entry["answer"]
                split = entry.get("split", "train")

                cot_str = f"<think>\n{cot.strip()}\n</think>\n\n"
                response_str = cot_str + f"<answer> {answer} </answer>"
                response_ids = tokenizer.encode(response_str).ids
                cot_len = len(tokenizer.encode(cot_str).ids)

                full_ids = prompt_ids + response_ids + [eos_token_id]

                by_split.setdefault(split, []).append(
                    {
                        "prompt_ids": prompt_ids,
                        "response_ids": response_ids,
                        "full_ids": full_ids,
                        "prefix_len": len(prompt_ids),
                        "cot_len": cot_len,
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
    examples: list[dict], pad_token_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a padded batch from SFT examples.

    Returns (input_ids, loss_mask)
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

    end_lens = torch.tensor(
        [len(ex["full_ids"]) for ex in examples], dtype=torch.long, device=device
    )
    positions = torch.arange(max_len, device=device).unsqueeze(0)
    loss_mask = (positions >= prefix_lens.unsqueeze(1)) & (
        positions < end_lens.unsqueeze(1)
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

    if sft_params.lora_rank is not None:
        from dialectic.llm.lora import (
            DEFAULT_TARGET_MODULES,
            apply_lora,
            freeze_base_params,
        )

        targets = DEFAULT_TARGET_MODULES
        if sft_params.lora_target_modules == "attn":
            targets = ("q_proj", "k_proj", "v_proj", "o_proj")
        elif sft_params.lora_target_modules == "mlp":
            targets = ("gate_proj", "up_proj", "down_proj")

        apply_lora(
            p,
            rank=sft_params.lora_rank,
            alpha=sft_params.lora_alpha,
            dropout=sft_params.lora_dropout,
            target_modules=targets,
        )
        freeze_base_params(p)

    if sft_params.freeze_embeddings:
        p.embed_tokens.requires_grad_(False)
    trainable_params = [pa for pa in p.parameters() if pa.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=sft_params.lr)

    scheduler = None
    if sft_params.warmup_steps > 0:
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        warmup = LinearLR(
            opt, start_factor=1e-8, end_factor=1.0, total_iters=sft_params.warmup_steps
        )
        total_steps = sft_params.max_episodes // (
            sft_params.batch_size * sft_params.accumulation_steps
        )
        cosine = CosineAnnealingLR(
            opt, T_max=max(total_steps - sft_params.warmup_steps, 1)
        )
        scheduler = SequentialLR(
            opt, schedulers=[warmup, cosine], milestones=[sft_params.warmup_steps]
        )
        log.info(
            f"LR schedule: warmup {sft_params.warmup_steps} steps, cosine decay over {total_steps} total steps"
        )

    p_raw = p
    if is_distributed():
        p = wrap_ddp(p, get_local_rank())

    total_params = sum(pa.numel() for pa in p_raw.parameters())
    trainable_count = sum(pa.numel() for pa in trainable_params)
    log.info(
        f"p total params: {total_params:,}, trainable: {trainable_count:,} "
        f"(rank {rank}/{world_size}, local_batch_size={local_batch_size})"
    )

    # Load SFT data (rank 0 downloads first, others wait for cache)
    if is_distributed() and rank != 0:
        barrier()
    by_split = _load_sft_data(
        dataset_artifacts,
        tokenizer,
        eos_token_id=model_info.eos_token_id,
        fcr_filter=sft_params.fcr_filter,
        use_rollout_data=sft_params.use_rollout_data,
    )
    train_data = by_split.get("train", [])
    val_data = by_split.get("val", [])

    mix_train: list[dict] = []
    if sft_params.mix_rollout_artifact is not None:
        from dialectic.experiments.arg_parser import resolve_artifact_glob as _resolve

        mix_names = _resolve(sft_params.mix_rollout_artifact)
        mix_split = _load_sft_data(
            mix_names,
            tokenizer,
            eos_token_id=model_info.eos_token_id,
            use_rollout_data=True,
        )
        mix_train = mix_split.get("train", [])
        log.info(
            f"Mixing {len(mix_train)} rollout examples with {len(train_data)} q-cot examples "
            f"(mix_ratio={sft_params.mix_ratio}: {sft_params.mix_ratio:.0%} q-cot, "
            f"{1 - sft_params.mix_ratio:.0%} rollout per batch)"
        )

    if is_distributed() and rank == 0:
        barrier()

    if not train_data:
        raise ValueError("No training data found")

    # Load q for importance weighting
    q_model = None
    if sft_params.importance_weight:
        if sft_params.q_ckpt_run is None or sft_params.q_ckpt_step is None:
            raise ValueError(
                "--q-ckpt-run and --q-ckpt-step required with --importance-weight"
            )

        q_model = copy.deepcopy(p_raw)
        for layer in q_model.layers:
            layer.self_attn.causal = False

        q_model = q_model.to(device=device, dtype=next(p_raw.parameters()).dtype)
        q_project, q_run_name = sft_params.q_ckpt_run.split("/")
        q_ckpt = extty.load_checkpoint_from(
            project=q_project,
            run_name=q_run_name,
            step=sft_params.q_ckpt_step,
            load_optimizer=False,
        )
        q_state = q_ckpt["model_state_dict"]
        q_state.pop("_rng_torch", None)
        q_state.pop("_rng_python", None)
        q_state.pop("_rng_cuda", None)
        q_model.load_state_dict(q_state)
        q_model.eval()
        q_model.requires_grad_(False)
        log.info(f"Loaded q from {sft_params.q_ckpt_run} step {sft_params.q_ckpt_step}")

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
            if mix_train:
                n_q = max(1, int(local_batch_size * sft_params.mix_ratio))
                n_p = local_batch_size - n_q
                q_indices = torch.randint(len(train_data), (n_q,)).tolist()
                p_indices = torch.randint(len(mix_train), (n_p,)).tolist()
                batch_examples = [train_data[i] for i in q_indices] + [
                    mix_train[i] for i in p_indices
                ]
            else:
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
            p_log_probs = -F.cross_entropy(
                shift_logits.reshape(B * L, V),
                shift_targets.reshape(B * L),
                reduction="none",
            ).reshape(B, L)

            if q_model is not None:
                with torch.no_grad():
                    q_logits = q_model(input_ids, return_all_logits=True)
                q_shift_logits = q_logits[:, :-1]
                q_log_probs = -F.cross_entropy(
                    q_shift_logits.reshape(B * L, V),
                    shift_targets.reshape(B * L),
                    reduction="none",
                ).reshape(B, L)

                # Sequence-level importance weight: p(C|P) / q(C|A*,P)
                seq_lengths = shift_mask.sum(dim=1).clamp(min=1)
                p_seq_logprob = (p_log_probs.detach() * shift_mask).sum(
                    dim=1
                ) / seq_lengths
                q_seq_logprob = (q_log_probs * shift_mask).sum(dim=1) / seq_lengths
                seq_ratio = torch.exp(p_seq_logprob - q_seq_logprob)
                clipped_seq_ratio = torch.clamp(
                    seq_ratio,
                    1.0 - sft_params.importance_eps,
                    1.0 + sft_params.importance_eps,
                )
                # Apply one weight per sequence to all tokens
                per_token_loss = -clipped_seq_ratio.unsqueeze(1) * p_log_probs
            else:
                per_token_loss = -p_log_probs

            masked_loss = per_token_loss * shift_mask
            seq_lengths = shift_mask.sum(dim=1).clamp(min=1)
            nll_loss = (masked_loss.sum(dim=1) / seq_lengths).mean()

            loss = nll_loss

            (loss / sft_params.accumulation_steps).backward()

            total_loss_val += nll_loss.item()
            local_episodes += local_batch_size

        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable_params, sft_params.max_grad_norm
        ).item()
        opt.step()
        if scheduler is not None:
            scheduler.step()

        global_episodes = local_episodes * world_size
        metrics = {
            "train/nll_loss": total_loss_val / sft_params.accumulation_steps,
            "train/grad_norm": grad_norm,
            "train/episodes": global_episodes,
            "train/lr": opt.param_groups[0]["lr"],
        }

        metrics["train/loss"] = total_loss_val / sft_params.accumulation_steps

        return StepFunctionReturn(
            n_episodes_processed=global_episodes,
            metrics=metrics,
        )

    def _val_fn(_val_env: Env) -> tuple[EvaluationResult, list[extty.Example]]:
        p_raw.eval()

        # Merge LoRA weights for faster generation, restore after
        if sft_params.lora_rank is not None:
            from dialectic.llm.lora import LoRALinear

            lora_state: list[tuple[torch.nn.Module, str, LoRALinear]] = []
            for _, module in p_raw.named_modules():
                for attr_name, child in list(module.named_children()):
                    if isinstance(child, LoRALinear):
                        lora_state.append((module, attr_name, child))
                        merged = child.base
                        merged.weight.data += (
                            child.scaling * child.lora_B.weight @ child.lora_A.weight
                        )
                        setattr(module, attr_name, merged)

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

        # Accuracy: have p generate from scratch on val prompts via vLLM
        examples: list[extty.Example] = []
        if val_prompts:
            eval_prompts = val_prompts[:max_val]
            val_rng_state = torch.random.get_rng_state()
            torch.manual_seed(42)
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
                n_samples=sft_params.val_pass_at_n,
            )
            n_label = (
                f"pass@{sft_params.val_pass_at_n}"
                if sft_params.val_pass_at_n > 1
                else "accuracy"
            )
            component_means[n_label] = baseline_result.accuracy
            component_means[f"hard_{n_label}"] = baseline_result.hard_accuracy
            torch.random.set_rng_state(val_rng_state)

        n_label = (
            f"pass@{sft_params.val_pass_at_n}"
            if sft_params.val_pass_at_n > 1
            else "accuracy"
        )
        # Restore LoRA layers after val
        if sft_params.lora_rank is not None:
            for module, attr_name, lora_child in lora_state:
                # Undo the merge
                lora_child.base.weight.data -= (
                    lora_child.scaling
                    * lora_child.lora_B.weight
                    @ lora_child.lora_A.weight
                )
                setattr(module, attr_name, lora_child)

        return EvaluationResult(
            n_episodes=max_val,
            reward_mean=component_means.get(
                n_label,
                component_means.get("accuracy", component_means.get("val_loss", 0.0)),
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
