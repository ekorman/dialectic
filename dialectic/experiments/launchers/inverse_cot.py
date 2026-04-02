import re
from typing import Callable

import extty
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from dialectic.experiments.arg_parser import Experiment, run_experiments_parser
from dialectic.experiments.envs import get_state_to_str
from dialectic.experiments.models import load_model_and_opt
from dialectic.experiments.params import (
    CountdownParams,
    InverseCotParams,
    TrainParams,
)
from dialectic.experiments.prompts import PromptCollection
from dialectic.llm.base import BaseTransformer
from dialectic.llm.generate import generate_hard_tokens
from dialectic.llm.inverse_cot import InverseCotModel, create_prefix_lm_mask
from dialectic.llm.registry import MODEL_REGISTRY
from dialectic.llm.utils import get_default_device
from dialectic.rl.env import Countdown, CountdownEnv
from dialectic.rl.rollout import get_batch
from dialectic.training import StepFunctionReturn, train_loop


def _parse_cot_and_answer(text: str) -> tuple[str, str] | None:
    """Split model output into (CoT, answer).

    Tries <think>...</think> format first (CoT is inside think tags,
    answer is everything after). Falls back to splitting at <answer> tags.
    Returns None if neither format is found.
    """
    think_match = re.search(
        r"<think>(.*?)</think>(.*)", text, re.DOTALL
    )
    if think_match is not None:
        cot = think_match.group(1).strip()
        answer = think_match.group(2).strip()
        if cot and answer:
            return cot, answer

    answer_match = re.search(r"<answer>.*?</answer>", text, re.DOTALL)
    if answer_match is not None:
        cot = text[: answer_match.start()]
        answer = answer_match.group(0)
        if cot.strip():
            return cot, answer

    return None


def _generate_training_data(
    *,
    p: BaseTransformer,
    env: CountdownEnv,
    state_to_str: Callable[[Countdown], str],
    tokenizer: Tokenizer,
    eos_token_id: int,
    pad_token_id: int,
    batch_size: int,
    temperature: float,
    max_tokens_generated: int,
    use_bf16: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] | None:
    """Generate data from p and construct q's training batch.

    Returns (input_ids, prefix_lengths, loss_mask, n_skipped) or None if no
    valid samples. n_skipped is the number of samples that failed parsing.
    """
    device = next(p.parameters()).device

    env_responses = get_batch(env, batch_size)
    prompts = [state_to_str(resp.data) for resp in env_responses]

    tokenizer.enable_padding(direction="left")
    tokens = tokenizer.encode_batch(prompts)
    attention_mask = torch.tensor(
        [t.attention_mask for t in tokens], dtype=torch.bool, device=device
    )
    token_ids = torch.tensor([t.ids for t in tokens], device=device)

    was_training = p.training
    p.eval()
    completions = generate_hard_tokens(
        net=p,
        token_ids=token_ids,
        sampling_strategy="sample" if temperature > 0 else "greedy",
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        max_tokens_generated=max_tokens_generated,
        use_kv_cache=True,
        attention_mask=attention_mask,
        temperature=temperature if temperature > 0 else 1.0,
        use_bf16=use_bf16,
    ).tokens
    if was_training:
        p.train()

    prompt_len = token_ids.shape[1]
    completion_strs = tokenizer.decode_batch(completions[:, prompt_len:].tolist())

    valid_prompt_strs: list[str] = []
    valid_answer_strs: list[str] = []
    valid_cot_strs: list[str] = []

    n_no_answer_tag = 0
    n_empty_cot = 0
    for i, comp_str in enumerate(completion_strs):
        parsed = _parse_cot_and_answer(comp_str)
        if parsed is None:
            n_no_answer_tag += 1
            continue
        cot, answer = parsed
        if not cot.strip():
            n_empty_cot += 1
            continue
        # reconstruct prompt string (remove left-padding artifacts)
        prompt_token_ids = token_ids[i][attention_mask[i]].tolist()
        prompt_str = tokenizer.decode(prompt_token_ids)
        valid_prompt_strs.append(prompt_str)
        valid_answer_strs.append(answer)
        valid_cot_strs.append(cot)

    if n_no_answer_tag > 0 or n_empty_cot > 0:
        print(
            f"Skipped {n_no_answer_tag}/{batch_size} (no parse), "
            f"{n_empty_cot}/{batch_size} (empty CoT). "
            f"Sample failed completion: {completion_strs[0][:200]!r}"
        )
    n_skipped = n_no_answer_tag + n_empty_cot
    if not valid_prompt_strs:
        return None

    eos_str = tokenizer.decode([eos_token_id])

    # tokenize prefix and cot separately, then concatenate
    # this avoids cross-boundary tokenization issues and gives exact prefix lengths
    prefix_strs = [
        prompt + " " + answer
        for prompt, answer in zip(valid_prompt_strs, valid_answer_strs)
    ]
    cot_strs = [cot + eos_str for cot in valid_cot_strs]

    prefix_encodings = [tokenizer.encode(s) for s in prefix_strs]
    cot_encodings = [tokenizer.encode(s) for s in cot_strs]

    prefix_lengths = torch.tensor(
        [len(enc.ids) for enc in prefix_encodings], dtype=torch.long, device=device
    )

    all_ids = [
        p_enc.ids + c_enc.ids
        for p_enc, c_enc in zip(prefix_encodings, cot_encodings)
    ]
    max_len = max(len(ids) for ids in all_ids)
    input_ids = torch.full((len(all_ids), max_len), pad_token_id, device=device)
    for i, ids in enumerate(all_ids):
        input_ids[i, : len(ids)] = torch.tensor(ids, device=device)

    positions = torch.arange(max_len, device=device).unsqueeze(0)
    actual_lengths = torch.tensor(
        [len(ids) for ids in all_ids], dtype=torch.long, device=device
    ).unsqueeze(1)
    loss_mask = (positions >= prefix_lengths.unsqueeze(1)) & (
        positions < actual_lengths
    )

    return input_ids, prefix_lengths, loss_mask, n_skipped


def _compute_nll_loss(
    q: InverseCotModel,
    input_ids: torch.Tensor,
    prefix_lengths: torch.Tensor,
    loss_mask: torch.Tensor,
    normalize_by_sequence_length: bool,
) -> tuple[torch.Tensor, float]:
    """Forward through q and compute NLL loss on CoT tokens."""
    seq_len = input_ids.shape[1]
    device = input_ids.device

    attention_mask = create_prefix_lm_mask(prefix_lengths, seq_len, device)
    logits = q(input_ids, attention_mask=attention_mask, return_all_logits=True)

    # shift: predict next token from current position
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
    if normalize_by_sequence_length:
        seq_lengths = shift_mask.sum(dim=1).clamp(min=1)
        per_seq_loss = masked_loss.sum(dim=1) / seq_lengths
        loss = per_seq_loss.mean()
    else:
        loss = masked_loss.sum() / shift_mask.sum().clamp(min=1)

    nll = loss.item()
    return loss, nll


@extty.experiment(project="inverse-cot-countdown")
def train_inverse_cot_countdown(
    *,
    train_params: TrainParams,
    inverse_cot_params: InverseCotParams,
    countdown_params: CountdownParams,
    prompt_collection: PromptCollection,
):
    torch.manual_seed(train_params.seed)

    model_info = MODEL_REGISTRY[train_params.model_name]
    tokenizer = model_info.load_tokenizer()

    # Load p from checkpoint
    p, _ = load_model_and_opt(train_params=train_params)
    p.requires_grad_(False)
    p.eval()

    # Build q from p
    q = InverseCotModel(p)
    device = get_default_device()
    q = q.to(device)

    trainable_params = [param for param in q.parameters() if param.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=train_params.lr)

    total_params = sum(param.numel() for param in q.parameters())
    trainable_count = sum(param.numel() for param in trainable_params)
    print(f"q total params: {total_params:,}, trainable: {trainable_count:,}")

    state_to_str = get_state_to_str(
        format_messages=model_info.format_messages,
        system_prompt=prompt_collection.system_prompt,
        assistant_prefill=prompt_collection.assistant_prefill,
    )

    n_ops = countdown_params.n_ops
    n_total = countdown_params.n_total
    n_larges = countdown_params.n_larges

    env = CountdownEnv(
        seed=train_params.seed,
        n_larges=n_larges,
        n_total=n_total,
        n_ops=n_ops,
        prompt_template=prompt_collection.env_prompt,
    )

    n_ops_list = [n_ops] if isinstance(n_ops, int) else n_ops
    n_total_list = [n_total] if isinstance(n_total, int) else n_total
    n_larges_list = [n_larges] if isinstance(n_larges, int) else n_larges
    val_envs = [
        CountdownEnv(
            seed=2026 + i,
            n_larges=n_larges_list[i],
            n_total=n_total_list[i],
            n_ops=n_ops_list[i],
            prompt_template=prompt_collection.env_prompt,
        )
        for i in range(len(n_ops_list))
    ]

    def _train_step(_step_idx: int) -> StepFunctionReturn:
        q.train()
        opt.zero_grad()

        total_loss = 0.0
        total_nll = 0.0
        total_episodes = 0
        skipped = 0

        for _ in range(train_params.accumulation_steps):
            data = _generate_training_data(
                p=p,
                env=env,
                state_to_str=state_to_str,
                tokenizer=tokenizer,
                eos_token_id=model_info.eos_token_id,
                pad_token_id=model_info.pad_token_id,
                batch_size=train_params.batch_size,
                temperature=train_params.temperature,
                max_tokens_generated=train_params.max_tokens_generated,
                use_bf16=train_params.use_bf16,
            )
            if data is None:
                skipped += train_params.batch_size
                continue

            input_ids, prefix_lengths, loss_mask, batch_skipped = data
            skipped += batch_skipped
            loss, nll = _compute_nll_loss(
                q,
                input_ids,
                prefix_lengths,
                loss_mask,
                inverse_cot_params.normalize_by_sequence_length,
            )
            (loss / train_params.accumulation_steps).backward()
            total_loss += loss.item()
            total_nll += nll
            total_episodes += input_ids.shape[0]

        if total_episodes > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, train_params.max_grad_norm)
            opt.step()

        metrics = {
            "train/loss": total_loss / max(train_params.accumulation_steps, 1),
            "train/nll": total_nll / max(train_params.accumulation_steps, 1),
            "train/episodes": total_episodes,
            "train/skipped": skipped,
        }
        return StepFunctionReturn(
            n_episodes_processed=total_episodes + skipped, metrics=metrics
        )

    @torch.no_grad()
    def _val_fn(val_env: CountdownEnv) -> tuple[dict, list[extty.Example]]:
        q.eval()

        all_nll: list[float] = []
        examples: list[extty.Example] = []

        n_episodes = 0
        while n_episodes < train_params.val_episodes:
            current_batch = min(
                train_params.val_batch_size,
                train_params.val_episodes - n_episodes,
            )

            data = _generate_training_data(
                p=p,
                env=val_env,
                state_to_str=state_to_str,
                tokenizer=tokenizer,
                eos_token_id=model_info.eos_token_id,
                pad_token_id=model_info.pad_token_id,
                batch_size=current_batch,
                temperature=train_params.temperature,
                max_tokens_generated=train_params.max_tokens_generated,
                use_bf16=train_params.use_bf16,
            )
            if data is None:
                n_episodes += current_batch
                continue

            input_ids, prefix_lengths, loss_mask, batch_skipped = data
            _, nll = _compute_nll_loss(
                q,
                input_ids,
                prefix_lengths,
                loss_mask,
                inverse_cot_params.normalize_by_sequence_length,
            )
            all_nll.append(nll)

            # qualitative: generate CoT from q for a few examples
            if len(examples) < train_params.val_episodes:
                for i in range(min(current_batch, train_params.val_episodes - len(examples))):
                    prefix_len = prefix_lengths[i].item()
                    prefix_ids = input_ids[i, :prefix_len].unsqueeze(0)

                    q_completion = generate_hard_tokens(
                        net=q,
                        token_ids=prefix_ids,
                        sampling_strategy="greedy",
                        eos_token_id=model_info.eos_token_id,
                        pad_token_id=model_info.pad_token_id,
                        max_tokens_generated=train_params.max_tokens_generated,
                        use_kv_cache=True,
                        use_bf16=train_params.use_bf16,
                    ).tokens

                    q_cot = tokenizer.decode(
                        q_completion[0, prefix_len:].tolist()
                    )

                    # decode the target CoT from training data
                    cot_mask = loss_mask[i]
                    cot_ids = input_ids[i][cot_mask].tolist()
                    target_cot = tokenizer.decode(cot_ids)

                    prompt_str = tokenizer.decode(
                        input_ids[i, :prefix_len].tolist()
                    )
                    examples.append(
                        extty.Example(
                            prompt=prompt_str,
                            responses=[
                                f"[p's CoT] {target_cot}",
                                f"[q's CoT] {q_cot}",
                            ],
                            rewards=[{"nll": nll}, {"nll": nll}],
                        )
                    )

            n_episodes += current_batch

        nll_mean = sum(all_nll) / len(all_nll) if all_nll else 0.0
        return {"nll_mean": nll_mean}, examples

    def val_fn_adapter(val_env):
        from dialectic.rl.evaluate import EvaluationResult

        metrics, examples = _val_fn(val_env)
        return EvaluationResult(
            n_episodes=train_params.val_episodes,
            reward_mean=-metrics["nll_mean"],
            reward_std=0.0,
            component_means=metrics,
        ), examples

    train_loop(
        max_episodes=train_params.max_episodes,
        save_ckpt_freq=train_params.save_ckpt_freq,
        val_freq=train_params.val_freq,
        net=q,
        opt=opt,
        train_step=_train_step,
        val_fn=val_fn_adapter,
        val_envs=val_envs,
    )


if __name__ == "__main__":
    run_experiments_parser(
        [
            Experiment(
                env_name="countdown",
                fn=train_inverse_cot_countdown,
                include_prompt_collection_id=True,
            ),
        ]
    )
