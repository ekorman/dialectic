# Hybrid Soft/Hard Token Reasoning with RL

## Overview

We extend a working GRPO implementation to use a hybrid reasoning architecture that mixes **internal soft tokens** (continuous, non-output vectors) with **hard tokens** (discrete vocabulary tokens) in the reasoning trace. The goal is to create reasoning traces where the hard token subsequence forms a clean, transferable, interpretable chain of intermediate results — suitable for offline and off-policy RL — while the soft tokens handle unconstrained internal computation via backpropagation.

**First experiment: Maze navigation** (not Countdown). Maze navigation is ideal for fixed interleaving because each hard token is a single complete action (a directional move), unlike Countdown where intermediate steps are multi-token expressions. This lets us validate the core architecture without needing a learned switching mechanism.

**Second experiment: Countdown** with learned Bernoulli switching (future work).

## Task: Maze Navigation

### Why mazes?
- Each action is a **single token** (N/S/E/W or up/down/left/right), perfect for fixed interleaving
- Natural need for **planning between moves** (soft tokens have a clear role)
- Easy **format reward**: is each move valid (doesn't hit a wall)?
- Easy **correctness reward**: did you reach the goal?
- Procedurally generated with controllable difficulty (grid size)
- Direct comparison to **AlphaMaze** (Qwen + GRPO baseline on mazes, arXiv:2502.14669)
- Used by **HRM** paper as a primary benchmark (our architectural inspiration)
- Rich interpretability literature on maze-solving transformers

### Maze environment setup
- **Roll our own maze generator** (see `maze-generator-spec.md` for details). This gives us full control over generation, tokenization, and reward computation without external dependencies.
- Start with **5×5 to 7×7 grids** (solvable by small models, well-studied scale)
- Generation algorithm: randomized DFS (standard, produces acyclic mazes with unique solutions)
- Tokenized maze representation as prompt (coordinate + wall tokens, following established conventions)
- Action tokens: `<up>`, `<down>`, `<left>`, `<right>` (or N/S/E/W — match whatever tokenization the base model handles best)

### Reward structure
1. **Correctness reward**: +1 if the agent reaches the goal, 0 otherwise
2. **Validity reward**: `0.5 × (valid_moves / total_moves)` — proportional, not all-or-nothing. Denser signal helps early training when the model produces mostly invalid moves.
3. **Efficiency bonus** (optional): small reward for shorter paths, or penalty per step to encourage efficiency

### Baselines
1. **Direct generation baseline**: Standard GRPO with direct move-sequence generation (no thinking tokens at all)
2. **Compute-matched CoT baseline**: Standard GRPO where the model generates 4 discrete "thinking tokens" between each move. This gives the same number of forward passes per move as the soft token model, isolating the soft token mechanism from the extra compute.
3. **AlphaMaze comparison**: Compare to published AlphaMaze results if using similar maze sizes and tokenization

## Architecture

### Starting point

Standard autoregressive transformer (Qwen-2.5 based) trained with GRPO. Currently have a working GRPO + vanilla CoT implementation on Countdown; adapt the training loop for maze navigation.

### Key modification

Replace the reasoning trace with an interleaved sequence of soft and hard tokens:

```
maze_prompt, [s1, s2, s3, s4], RIGHT, [s5, s6, s7, s8], DOWN, ..., [s_k-3, s_k-2, s_k-1, s_k], RIGHT, <done>
```

Where:
- `s_i` are **soft tokens**: continuous vectors in embedding space, generated autoregressively by the model but never projected through the vocabulary head and never output. They are internal computation.
- `h_i` are **hard tokens**: standard discrete tokens sampled from the vocabulary via the LM head. These are the model's explicit, interpretable reasoning checkpoints.

### Fixed interleaving pattern (first experiment)

Use a fixed ratio: **4 soft tokens followed by 1 hard token**, repeated for up to N cycles. This removes the need for a learned switching mechanism and isolates the core question: can the model learn to use hard tokens as meaningful intermediate reasoning steps while doing internal computation in soft token blocks?

**Early stopping via `<done>` token**: The hard token vocabulary includes a special `<done>` token. When the model emits `<done>` at any hard token position, generation stops immediately. Reward is computed on the moves emitted so far. Remaining cycles are not generated — no padding needed since generation is sequential. This lets the model learn variable-length trajectories rather than always using the maximum number of cycles.

### Soft token generation (no vocab head)

For soft token positions, the model:
1. Runs a normal transformer forward pass to get the last hidden state `h_t`
2. Feeds `h_t` directly as the input embedding for the next position

This skips the vocabulary head entirely. Soft tokens can represent arbitrary points in embedding space, including things with no vocabulary equivalent. Since `d_model == d_embed` in Qwen, the hidden state is already in the right space — no projection needed and no new parameters.

If training struggles (the hidden state distribution differs too much from the embedding distribution), a learned `W_soft` projection can be added later for extra expressivity.

### Hard token generation (standard)

For hard token positions, the model:
1. Runs a normal transformer forward pass to get `h_t`
2. Projects through the standard vocabulary head to get logits
3. Samples a token from the categorical distribution
4. Embeds the sampled token via the standard embedding table → next position input

This is identical to standard autoregressive generation.

### Stochasticity

- **Hard tokens**: stochasticity comes from sampling from the categorical token distribution (standard)
- **Soft tokens**: deterministic (no noise). Optionally add fixed Gaussian noise to embeddings as in "Soft Tokens Hard Truths" for regularization, but this is not required for RL and can be added later if needed.

All RL-relevant stochasticity comes from hard token sampling. Soft tokens are internal deterministic computation.

## Training

### RL formulation

The policy's **action space** consists only of:
- Hard token selections (categorical over vocabulary)

Soft tokens are **not actions**. They are internal differentiable computation, analogous to hidden layers in a neural network. They do not appear in the RL objective.

### Loss function

The GRPO loss is computed only over hard tokens and answer tokens:

```
L = Σ_i A_i · log π(h_i | context_i) + Σ_j A · log π(a_j | context_j)
```

Where:
- `h_i` are the hard reasoning tokens
- `a_j` are the answer tokens
- `A_i` / `A` are advantages from GRPO (computed from reward on complete trajectories)
- `context_i` includes all preceding soft and hard tokens

### Gradient flow

Gradients from the hard token log-probabilities flow back through the preceding soft tokens via backpropagation. This is the key mechanism by which soft tokens learn: they are optimized to produce hidden states that lead to good hard token decisions.

### Gradient strategy for soft token blocks

**Default: full BPTT through all soft tokens in each block.**

Within each soft block of 4 tokens, all positions run with gradients enabled. Backprop flows from the hard token log-prob through all 4 preceding soft tokens. At block size 4, the memory overhead is negligible — gradient checkpointing at hard token boundaries keeps cost bounded.

Note: the DEQ / fixed-point analogy from HRM does not apply here. Soft blocks use causal attention with different contexts per position, so they are not recurrent iterations converging to a fixed point.

**Future optimization for large soft blocks (32+):** One-step approximation (only backprop through the last soft token) to reduce memory. Not needed at block size 4.

### Reward structure

Two reward components:

1. **Correctness reward**: Did the path reach the goal? (Binary, +1 / 0)
2. **Validity reward on hard tokens**: `0.5 × (valid_moves / total_moves)`. Proportional rather than all-or-nothing — a trajectory with 8/10 valid moves scores 0.4, not 0. This provides dense learning signal early in training when the model produces mostly invalid moves.

Optional: small per-step penalty to encourage efficiency (shorter paths).

**Future: per-move credit assignment.** The episode-level validity reward applies the same advantage to all hard tokens. A better signal is per-move rewards — each hard token gets its own validity signal (was *this* move valid?), and advantages propagate backward so that a valid move at step `t` improves advantages for hard tokens at steps `≤ t`. This gives proper credit assignment to the soft token blocks that actually produced useful planning. The `validate_path` API tracks per-move validity, so switching to per-move advantages only requires changing the advantage computation, not the reward infrastructure.

## Rollout / Generation Procedure

During rollout (generating trajectories for GRPO):

```
for cycle in range(max_cycles):
    # soft block
    for s in range(soft_block_size):
        h_t = transformer_forward(current_context)
        append h_t to context as next input embedding  # direct passthrough, no vocab head

    # hard token
    h_t = transformer_forward(current_context)
    logits = vocab_head(h_t)
    token = sample(logits)
    embed = embedding_table[token]
    append embed to context as next input embedding
    record (token, log_prob) for RL

    if token == <done>:
        break
```

After reasoning trace, generate answer tokens normally (all hard, standard autoregressive).

## Training Loop Modification

During the GRPO training update:

1. **Cannot teacher-force soft tokens.** Unlike standard GRPO where you recompute all log-probs in one parallel forward pass, soft tokens must be regenerated autoregressively (each depends on the current parameters and all previous tokens).

2. **Procedure for computing training loss on a trajectory:**
   - Forward through prompt (parallel, standard)
   - For each soft block: regenerate soft tokens sequentially (all with gradients enabled)
   - At each hard token: compute log π(h_i | context) using known h_i from rollout
   - Compute GRPO loss on hard token log-probs
   - Backprop

3. **Cost:** The main cost is **loss of parallelism**, not raw FLOPs. A 30-move trajectory requires ~150 sequential forward passes (4 soft + 1 hard per cycle × 30 cycles), compared to a single parallel forward pass in standard GRPO. This is the dominant bottleneck. **Mitigation:** Cache KV states at hard token boundaries — when regenerating soft blocks during training, we only need to recompute the soft tokens (the prompt and prior hard tokens are unchanged). This reduces sequential forward passes to just the soft block length (4) per cycle, with the rest served from cache.

## Implementation Plan

### New components to implement:
1. Maze environment: generation, tokenization, reward computation (see `maze-generator-spec.md`)
2. Modified generation loop that alternates soft/hard at fixed positions, with `<done>` early stopping
3. Modified training forward pass that regenerates soft tokens and computes hard token log-probs (with KV caching at hard token boundaries)
4. Validity reward function (proportional: `valid_moves / total_moves`)
5. Adjustment to GRPO loss to only include hard tokens (the directional moves)

### What stays the same:
- Base model (Qwen-2.5)
- GRPO algorithm (advantage computation, clipping, KL penalty)
- General training loop structure

### Hyperparameters for first experiment:
- Soft block length: 4
- Hard token frequency: every 5th position (1 hard per 4 soft)
- Number of reasoning cycles: ~10-30 (so 10-30 moves, enough for 5×5 to 7×7 mazes)
- Gradient strategy: full BPTT through all soft tokens (cheap at block size 4)
- Maze size: start with 5×5, scale to 7×7
- Validity reward: `0.5 × (valid_moves / total_moves)`

### Success criteria:
1. Model converges and produces valid directional moves (not garbage tokens)
2. Model solves mazes at a rate comparable to or better than standard CoT baseline
3. Soft tokens demonstrably contribute (ablation: compare to hard-tokens-only with same compute budget)
4. (Stretch) Hard token subsequences transfer to a different model via offline RL

## Key Design Decisions and Rationale

| Decision | Choice | Rationale |
|----------|--------|-----------|
| First task | Maze navigation (not Countdown) | Single-token actions fit fixed interleaving; Countdown needs multi-token expressions |
| Soft token representation | Direct hidden state passthrough (no projection, no vocab head) | Zero new parameters; `d_model == d_embed` in Qwen so hidden state is already in embedding space. Add `W_soft` later if needed. |
| Switching mechanism | Fixed interleaving (4:1) | Simplest possible; removes switching as a confound |
| Soft token stochasticity | None (deterministic) | All RL stochasticity from hard tokens; simpler |
| Gradient through soft blocks | Full BPTT (all 4 soft tokens) | Negligible memory at block size 4; DEQ analogy doesn't hold for causal attention |
| RL algorithm | GRPO (existing) | Already implemented; works well |
| Soft tokens in RL objective | Excluded | They are internal computation, not actions |
| Hard token reward | Move validity (doesn't hit wall) | Provides gradient anchors; trivially verifiable |

## Novelty and Contribution

The unique aspects of this approach relative to existing work:

1. **Offline RL motivation**: No existing paper frames hybrid soft/hard tokens as creating transferable intermediate states for offline/off-policy RL. The hard token subsequence defines a shared, interpretable MDP that different policies can contribute to.

2. **Learned Bernoulli switching + BPTT soft tokens + RL only on hard tokens**: This specific combination doesn't exist in the literature (the fixed interleaving is a stepping stone to learned switching).

3. **Terse checkpoint shaping**: Using format rewards to shape hard tokens into structured single-move checkpoints (valid directional moves in the maze) while soft tokens handle the planning computation between moves. The hard token subsequence is a clean, verifiable action trace.

If this first experiment works, the next steps are:
- **Countdown with learned switching**: Bernoulli gate for soft/hard, format rewards for arithmetic expressions
- **Offline RL transfer experiment**: extract hard token trajectories from model A, use to train model B
- Scale to harder mazes (10×10+), then to GSM8K / harder math problems

## Existing Work on Maze Navigation for LLMs

Key references to be aware of:

- **AlphaMaze** (arXiv:2502.14669): Qwen 1.5B + SFT + GRPO on tokenized mazes, 93% accuracy. Most direct baseline — same model family, same RL algorithm. Public dataset of 100k mazes on HuggingFace.
- **Dualformer** (Meta FAIR, arXiv:2410.09918): Trains encoder-decoder transformer on maze + Sokoban with A* search traces. Randomized trace-dropping for fast/slow thinking modes. 15M params for mazes. Uses Searchformer tokenization.
- **HRM** (arXiv:2506.21734): Hierarchical Reasoning Model, 27M params, solves 30×30 mazes optimally with ~1000 training examples. Architectural inspiration for the soft/hard token interleaving concept.
- **maze-dataset** library (arXiv:2309.10498, `pip install maze-dataset`): Configurable Python package for maze generation, solving, and tokenization. Used by multiple interpretability papers. Supports multiple generation algorithms and output formats.
- **Transformers Can Navigate Mazes With Multi-Step Prediction** (arXiv:2412.05117): Shows MLM-U objective (predict multiple steps ahead/back) dramatically improves maze navigation vs next-token prediction. 4x more data efficient.
- **MazeEval** (arXiv:2507.20395): Benchmark for LLM maze navigation with coordinate-based feedback, 5×5 to 15×15 grids.
- **Structured World Representations / Causal World Models in Mazes** (arXiv:2312.02566, arXiv:2412.11867): Interpretability work showing transformers learn linear representations of maze connectivity. Useful for understanding what the soft tokens might learn.
