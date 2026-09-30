# Reproducing the paper

These are the commands behind the experiments in *Inverse Reasoning Models for Efficient Post-Training and Verification from Answer-Only Data*. Each launcher logs the extty names it creates, and the commands pass those names along in shell variables. The paper ran each job on one 96 GB GPU (RTX PRO 6000), except q training, which used 2 GPUs. On machines without NVLink, q training needs `NCCL_P2P_DISABLE=1`.

## Shared flags

```bash
M=dialectic.experiments.launchers
GRPO=(--train_params.model-name qwen3-0.6b-thinking --train_params.seed 20
  --train_params.lr 1e-5 --train_params.batch-size 2 --train_params.accumulation-steps 16
  --train_params.max-grad-norm 1.0 --train_params.temperature 1.0 --train_params.use-bf16
  --train_params.val-freq 50 --train_params.val-batch-size 32 --train_params.save-ckpt-freq 100
  --train_params.max-episodes 1000000 --grpo_params.group-size 8 --grpo_params.beta 0.0
  --grpo_params.normalize-advantages --grpo_params.normalize-by-sequence-length
  --reward_params.answer-tags-weight 0.1 --reward_params.think-tags-weight 0.0)
ROLLOUT=(--rollout_gen_params.model-name qwen3-0.6b-thinking --rollout_gen_params.group-size 32
  --rollout_gen_params.temperature 1.0 --rollout_gen_params.seed 20 --rollout_gen_params.batch-size 16
  --rollout_gen_params.max-tokens-generated 600 --rollout_gen_params.use-bf16)
Q=(--prompt-collection-id 2 --train_params.model-name qwen3-0.6b-thinking --train_params.seed 20
  --train_params.batch-size 32 --train_params.lr 2e-4 --train_params.accumulation-steps 1
  --train_params.max-grad-norm 10.0 --train_params.max-episodes 1000000 --train_params.temperature 1.0
  --train_params.use-bf16 --train_params.val-freq 500 --train_params.val-batch-size 4
  --train_params.val-episodes 300 --train_params.max-tokens-generated 500 --train_params.save-ckpt-freq 1000
  --inverse_cot_params.contrastive-weight 1.0 --inverse_cot_params.train-group-size 8
  --inverse_cot_params.normalize-by-sequence-length --inverse_cot_params.max-cot-tokens 500
  --inverse_cot_params.gradient-checkpointing --inverse_cot_params.lora-rank 32
  --inverse_cot_params.lora-alpha 64 --inverse_cot_params.lora-target-modules all)
SFT=(--sft_params.temperature 1.0 --sft_params.batch-size 16 --sft_params.lr 5e-4
  --sft_params.max-episodes 10000000 --sft_params.max-tokens-generated 600 --sft_params.save-ckpt-freq 1000
  --sft_params.lora-rank 16 --sft_params.lora-alpha 32 --sft_params.max-grad-norm 1.0)
EVAL=(--eval_params.n-samples 32 --eval_params.model-name qwen3-0.6b-thinking
  --eval_params.use-bf16 --eval_params.temperature 1.0)
```

## Countdown

```bash
# 1. Dataset: 50k problems, 80/10/10 split                           -> $DATASET
uv run python -m $M.generate_countdown_dataset \
  --dataset_gen_params.n-examples 50000 --dataset_gen_params.seed 12 \
  --dataset_gen_params.train-pct 0.8 --dataset_gen_params.val-pct 0.1 --dataset_gen_params.test-pct 0.1 \
  --countdown_params.n-larges 1,2 --countdown_params.n-total 4,4 --countdown_params.n-ops 3,3

# 2. GRPO the forward model p (paper uses step 700)                   -> $P_RUN
uv run --group vllm python -m $M.grpo countdown "${GRPO[@]}" \
  --dataset-glob $DATASET --prompt-collection-id 3 --train_params.max-tokens-generated 600

# 3. 32 rollouts per prompt from p                                    -> $ROLLOUTS
uv run --group vllm python -m $M.generate_inverse_cot_rollouts countdown "${ROLLOUT[@]}" \
  --dataset-glob $DATASET --prompt-collection-id 3 \
  --rollout_gen_params.start-ckpt-run $P_RUN --rollout_gen_params.start-ckpt-step 700

# 4. Train q on p's rollouts (paper uses step 1000)                   -> $Q_RUN
NCCL_P2P_DISABLE=1 uv run torchrun --nproc_per_node=2 -m $M.inverse_cot countdown "${Q[@]}" \
  --inverse_cot_params.p-rollout-artifact $ROLLOUTS \
  --inverse_cot_params.forward-ckpt-run $P_RUN --inverse_cot_params.forward-ckpt-step 700

# 5. Synthesize CoTs with q                                           -> $QCOT
uv run python -m $M.generate_q_cot countdown --gen_params.q-ckpt-run $Q_RUN \
  --gen_params.q-ckpt-step 1000 --gen_params.batch-size 64 --gen_params.temperature 1.0

# 6. Distill into p. Mix ratio = fraction of q-CoT examples (0 = rejection-sampling baseline)
for MIX in 0 0.25 0.5 0.75 1.0; do for SEED in 81 20; do
  uv run python -m $M.sft_inverse_cot countdown "${SFT[@]}" --sft_params.q-cot-artifact $QCOT \
    --sft_params.mix-ratio $MIX --sft_params.seed $SEED \
    --sft_params.val-episodes 100 --sft_params.val-freq 1000 --sft_params.val-pass-at-n 8
done; done

# 7. Evaluate each SFT run on the test split (and the baseline: --eval_params.ckpt-run $P_RUN --eval_params.ckpt-step 700)
uv run --group vllm python -m $M.eval_grpo countdown "${EVAL[@]}" --eval_params.split test \
  --eval_params.p-rollout-artifact $ROLLOUTS \
  --eval_params.ckpt-run <sft run> --eval_params.ckpt-step 1000..121000..6000

# 8. FCR of q-synthesized CoTs
uv run --group vllm python -m $M.eval_inverse_cot countdown --eval_params.q-ckpt-run $Q_RUN \
  --eval_params.q-ckpt-step 1000 --eval_params.n-samples 32 --eval_params.batch-size 96 \
  --eval_params.temperature 1.0 --eval_params.max-tokens-generated 600 --eval_params.seed 16 --eval_params.split val

# 9. q as a verifier, then the offline budget/Pareto analysis (no GPU)
uv run --group eval-q python -m $M.eval_q_verifier countdown --eval_params.q-ckpt-run $Q_RUN \
  --eval_params.q-ckpt-step 1000 --eval_params.split test --eval_params.dump-scores
uv run --group eval-q python scripts/q_verifier_pareto.py --artifact <scores artifact from step 9>
```

### Fresh-data experiment

This experiment asks whether q works on prompts that neither p nor q saw in training. All arms start from p at step 700, and all are evaluated with the original `$ROLLOUTS` so they share the same test prompts and hard buckets.

```bash
# 1. 40k new train-only problems, deduplicated against $DATASET       -> $FRESH
uv run python -m $M.generate_countdown_dataset \
  --dataset_gen_params.n-examples 40000 --dataset_gen_params.seed 13 \
  --dataset_gen_params.train-pct 1 --dataset_gen_params.val-pct 0 --dataset_gen_params.test-pct 0 \
  --dataset_gen_params.exclude-artifacts $DATASET \
  --countdown_params.n-larges 1,2 --countdown_params.n-total 4,4 --countdown_params.n-ops 3,3

# 2. 32 rollouts per fresh prompt from p                               -> $FRESH_ROLLOUTS
uv run --group vllm python -m $M.generate_inverse_cot_rollouts countdown "${ROLLOUT[@]}" \
  --dataset-glob $FRESH --prompt-collection-id 3 \
  --rollout_gen_params.start-ckpt-run $P_RUN --rollout_gen_params.start-ckpt-step 700

# 3. Continued-GRPO arm: resume p from step 700 on the fresh prompts
for SEED in 20 81; do
  uv run --group vllm python -m $M.grpo countdown "${GRPO[@]}" --train_params.seed $SEED \
    --dataset-glob $FRESH --prompt-collection-id 3 --train_params.max-tokens-generated 600 \
    --train_params.start-ckpt-run $P_RUN --train_params.start-ckpt-step 700
done

# 4. Synthesize CoTs with q for the fresh prompts                      -> $FRESH_QCOT
uv run python -m $M.generate_q_cot countdown --gen_params.q-ckpt-run $Q_RUN \
  --gen_params.q-ckpt-step 1000 --gen_params.batch-size 64 --gen_params.temperature 1.0 \
  --gen_params.p-rollout-artifact $FRESH_ROLLOUTS

# 5. Distill into p. Pass both artifacts: otherwise the rollout artifact is inferred as the original $ROLLOUTS
for MIX in 0 0.25 0.5 0.75 1.0; do for SEED in 81 20; do
  uv run python -m $M.sft_inverse_cot countdown "${SFT[@]}" \
    --sft_params.q-cot-artifact $FRESH_QCOT --sft_params.p-rollout-artifact $FRESH_ROLLOUTS \
    --sft_params.mix-ratio $MIX --sft_params.seed $SEED
done; done

# 6. Select each run's checkpoint by hard_pass_at_8 on the original test prompts
#    (grid: 1000..101000..6000 for SFT runs, 100..2000..100 for GRPO runs)
uv run --group vllm python -m $M.eval_grpo countdown "${EVAL[@]}" --eval_params.split test \
  --eval_params.p-rollout-artifact $ROLLOUTS --eval_params.ckpt-run <run> --eval_params.ckpt-step <grid>

# 7. Report each selected checkpoint, plus the baseline ($P_RUN at step 700), on val
uv run --group vllm python -m $M.eval_grpo countdown "${EVAL[@]}" --eval_params.split val \
  --eval_params.p-rollout-artifact $ROLLOUTS --eval_params.ckpt-run <run> --eval_params.ckpt-step <selected step>
```

## GSM8K

The GSM8K pipeline runs steps 2–9 with `gsm8k` in place of `countdown`, with the differences below. Here the `val` split is the official GSM8K test set, and training never touches it. It is the default split for every GSM8K eval.

- **Data:** replace `--dataset-glob $DATASET` with `--gsm8k_params.train-path data/gsm8k/train.jsonl --gsm8k_params.val-path data/gsm8k/test.jsonl`, and use `--prompt-collection-id 2`.
- **GRPO:** `--train_params.max-tokens-generated 500`. The paper uses p at step 3600.
- **q:** the paper uses q at step 25000 for q-CoT synthesis, FCR and the verifier.
- **SFT:** drop the three `val-*` flags. Validation uses a stratified holdout from train.
- **Evals:** drop `--eval_params.split`. `eval_grpo` also takes `--eval_params.max-tokens-generated 600`. `eval_inverse_cot` drops `--eval_params.seed`.
