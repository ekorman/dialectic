import dataclasses
import json
import os
import tempfile

import extty
import torch
from tokenizers import Tokenizer
from tqdm import tqdm

from dialectic.experiments.arg_parser import Experiment, run_experiments_parser
from dialectic.experiments.envs import get_state_to_str
from dialectic.experiments.launchers.inverse_cot import _parse_cot_and_answer
from dialectic.experiments.params import CountdownParams, RolloutGenParams
from dialectic.experiments.prompts import PromptCollection
from dialectic.llm.base import BaseTransformer
from dialectic.llm.generate import generate_hard_tokens
from dialectic.llm.registry import MODEL_REGISTRY
from dialectic.llm.utils import get_default_device
from dialectic.rl.env import CountdownEnv
from dialectic.rl.extractors import extract_from_answer_tags
from dialectic.rl.reward import _evaluate_and_verify_countdown
from dialectic.rl.rollout import get_batch


def _generate_rollouts_for_batch(
    *,
    p: BaseTransformer,
    env: CountdownEnv,
    state_to_str,
    tokenizer: Tokenizer,
    eos_token_id: int,
    pad_token_id: int,
    batch_size: int,
    group_size: int,
    temperature: float,
    max_tokens_generated: int,
    use_bf16: bool,
    n_pos_min: int,
    n_neg_min: int,
) -> list[dict]:
    device = next(p.parameters()).device

    env_responses = get_batch(env, batch_size)
    prompts = [state_to_str(resp.data) for resp in env_responses]

    tokenizer.enable_padding(direction="left")
    tokens = tokenizer.encode_batch(prompts)
    attention_mask = torch.tensor(
        [t.attention_mask for t in tokens], dtype=torch.bool, device=device
    )
    token_ids = torch.tensor([t.ids for t in tokens], device=device)

    expanded_token_ids = token_ids.repeat_interleave(group_size, dim=0)
    expanded_attention_mask = attention_mask.repeat_interleave(group_size, dim=0)

    completions = generate_hard_tokens(
        net=p,
        token_ids=expanded_token_ids,
        sampling_strategy="sample" if temperature > 0 else "greedy",
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        max_tokens_generated=max_tokens_generated,
        use_kv_cache=True,
        attention_mask=expanded_attention_mask,
        temperature=temperature if temperature > 0 else 1.0,
        use_bf16=use_bf16,
    ).tokens

    prompt_len = token_ids.shape[1]
    completion_strs = tokenizer.decode_batch(completions[:, prompt_len:].tolist())

    kept = []
    total_correct = 0
    total_incorrect = 0
    total_unparsed = 0
    for b in range(batch_size):
        group_completions = []
        for g in range(group_size):
            idx = b * group_size + g
            comp_str = completion_strs[idx]
            parsed = _parse_cot_and_answer(comp_str)
            if parsed is None:
                total_unparsed += 1
                continue
            cot, answer = parsed
            if not cot.strip():
                total_unparsed += 1
                continue

            extracted = extract_from_answer_tags(comp_str)
            is_correct = extracted is not None and _evaluate_and_verify_countdown(
                extracted,
                env_responses[b].data.numbers,
                env_responses[b].data.target,
            )
            if is_correct:
                total_correct += 1
            else:
                total_incorrect += 1
            group_completions.append(
                {"cot": cot, "answer": answer, "is_correct": is_correct}
            )

        n_pos = sum(1 for c in group_completions if c["is_correct"])
        n_neg = sum(1 for c in group_completions if not c["is_correct"])
        if n_pos >= n_pos_min and n_neg >= n_neg_min:
            kept.append(
                {
                    "prompt_str": prompts[b],
                    "numbers": env_responses[b].data.numbers,
                    "target": env_responses[b].data.target,
                    "completions": group_completions,
                }
            )

    return kept


def generate_inverse_cot_rollouts(
    *,
    rollout_gen_params: RolloutGenParams,
    countdown_params: CountdownParams,
    prompt_collection: PromptCollection,
):
    torch.manual_seed(rollout_gen_params.seed)

    model_info = MODEL_REGISTRY[rollout_gen_params.model_name]
    tokenizer = model_info.load_tokenizer()

    p = model_info.load_net(
        pretrained_weights=rollout_gen_params.start_ckpt_run is None
    )
    if rollout_gen_params.start_ckpt_run is not None:
        if rollout_gen_params.start_ckpt_step is None:
            raise ValueError("`start_ckpt_step` required when `start_ckpt_run` is set")
        project, run_name = rollout_gen_params.start_ckpt_run.split("/")
        p.load_state_dict(
            extty.load_checkpoint_from(
                project=project,
                run_name=run_name,
                step=rollout_gen_params.start_ckpt_step,
            )["model_state_dict"]
        )

    device = get_default_device()
    p = p.to(device)
    p.requires_grad_(False)
    p.eval()

    state_to_str = get_state_to_str(
        format_messages=model_info.format_messages,
        system_prompt=prompt_collection.system_prompt,
        assistant_prefill=prompt_collection.assistant_prefill,
    )

    env = CountdownEnv(
        seed=rollout_gen_params.seed,
        n_larges=countdown_params.n_larges,
        n_total=countdown_params.n_total,
        n_ops=countdown_params.n_ops,
        prompt_template=prompt_collection.env_prompt,
    )

    n_kept = 0
    n_generated = 0

    tmpfile = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, prefix="inverse_cot_rollouts_"
    )
    print(f"Writing rollouts to {tmpfile.name}")
    pbar = tqdm(total=rollout_gen_params.n_prompts, desc="Generating rollouts")
    try:
        while n_kept < rollout_gen_params.n_prompts:
            current_batch = min(
                rollout_gen_params.batch_size,
                rollout_gen_params.n_prompts - n_kept,
            )
            kept = _generate_rollouts_for_batch(
                p=p,
                env=env,
                state_to_str=state_to_str,
                tokenizer=tokenizer,
                eos_token_id=model_info.eos_token_id,
                pad_token_id=model_info.pad_token_id,
                batch_size=current_batch,
                group_size=rollout_gen_params.group_size,
                temperature=rollout_gen_params.temperature,
                max_tokens_generated=rollout_gen_params.max_tokens_generated,
                use_bf16=rollout_gen_params.use_bf16,
                n_pos_min=rollout_gen_params.n_pos_min,
                n_neg_min=rollout_gen_params.n_neg_min,
            )
            n_generated += current_batch
            for entry in kept[: rollout_gen_params.n_prompts - n_kept]:
                tmpfile.write(json.dumps(entry) + "\n")
                tmpfile.flush()
            n_kept += len(kept)
            pbar.update(len(kept))
            pbar.set_postfix(generated=n_generated, kept=n_kept)
    finally:
        tmpfile.close()
        pbar.close()

        from datetime import datetime

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        artifact_name = (
            f"inverse-cot-rollouts-{rollout_gen_params.model_name}-"
            f"g{rollout_gen_params.group_size}-n{n_kept}-{ts}"
        )
        meta = extty.save_artifact(
            name=artifact_name,
            path=tmpfile.name,
            description=f"Inverse CoT rollouts: {n_kept} prompts, group_size={rollout_gen_params.group_size}",
            metadata={
                "rollout_gen_params": dataclasses.asdict(rollout_gen_params),
                "countdown_params": dataclasses.asdict(countdown_params),
                "prompt_collection": dataclasses.asdict(prompt_collection),
            },
        )
        os.unlink(tmpfile.name)
        print(f"Uploaded artifact: {meta}")


if __name__ == "__main__":
    run_experiments_parser(
        [
            Experiment(
                env_name="countdown",
                fn=generate_inverse_cot_rollouts,
                include_prompt_collection_id=True,
            ),
        ]
    )
