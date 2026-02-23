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
- Use the `maze-dataset` library (`pip install maze-dataset`) for maze generation and tokenization
- Start with **5×5 to 7×7 grids** (solvable by small models, well-studied scale)
- Generation algorithm: randomized DFS (standard, produces acyclic mazes with unique solutions)
- Tokenized maze representation as prompt (coordinate + wall tokens, following established conventions)
- Action tokens: `<up>`, `<down>`, `<left>`, `<right>` (or N/S/E/W — match whatever tokenization the base model handles best)
- Alternatively, roll our own simple maze generator if the library is too heavyweight — the core requirement is just: generate random grid mazes, compute shortest path, tokenize as text

### Reward structure
1. **Correctness reward**: +1 if the agent reaches the goal, 0 otherwise
2. **Validity reward**: bonus (e.g. +0.5) if all moves are valid (no wall collisions)
3. **Efficiency bonus** (optional): small reward for shorter paths, or penalty per step to encourage efficiency

### Baseline
- Standard GRPO with discrete CoT on the same mazes (direct generation of move sequence, no soft tokens)
- Comparison to AlphaMaze results if using similar maze sizes and tokenization

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

Use a fixed ratio: **4 soft tokens followed by 1 hard token**, repeated for a fixed number of cycles. This removes the need for a learned switching mechanism and isolates the core question: can the model learn to use hard tokens as meaningful intermediate reasoning steps while doing internal computation in soft token blocks?

### Soft token generation (no vocab head)

For soft token positions, the model:
1. Runs a normal transformer forward pass to get the last hidden state `h_t`
2. Applies a **single learned linear projection** `W_soft` (d_model → d_embed) to map the hidden state to the input embedding space: `z_t = W_soft · h_t + b_soft`
3. Feeds `z_t` directly as the input embedding for the next position

This skips the vocabulary head entirely. Soft tokens can represent arbitrary points in embedding space, including things with no vocabulary equivalent. The only new parameters are `W_soft` and `b_soft`.

If d_model == d_embed (as in Qwen), this is a square matrix. Could even start with identity initialization.

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

**Start with one-step approximation (inspired by HRM / DEQ):**

Within each soft block of 4 tokens:
- Soft tokens 1-3: run in `torch.no_grad()` (forward only, no gradient storage)
- Soft token 4: run with gradients enabled
- Hard token: compute log-prob, apply RL loss, backprop through soft token 4

This gives O(1) memory per soft block. The justification is that short soft blocks approximate iterative refinement, and the one-step gradient captures the essential learning signal.

**Fallback if training is unstable:** Enable full BPTT through all 4 soft tokens in each block. Memory cost is still small (only 4 steps per block with gradient checkpointing at hard token boundaries).

**Future option:** Make soft blocks explicitly recurrent (same weights per step, iterating on a hidden state) to make the fixed-point / DEQ justification rigorous.

### Reward structure

Two reward components:

1. **Correctness reward**: Did the path reach the goal? (Binary, +1 / 0)
2. **Validity reward on hard tokens**: Is each move valid (doesn't go through a wall)? Bonus (e.g., +0.5) if all moves in the trajectory are valid. This provides learning signal even when the model doesn't reach the goal.

Optional: small per-step penalty to encourage efficiency (shorter paths).

## Rollout / Generation Procedure

During rollout (generating trajectories for GRPO):

```
for each position in reasoning trace:
    if position is a soft token slot:
        h_t = transformer_forward(current_context)
        z_t = W_soft @ h_t + b_soft          # linear projection, no vocab head
        append z_t to context as next input embedding
    elif position is a hard token slot:
        h_t = transformer_forward(current_context)
        logits = vocab_head(h_t)
        token = sample(logits)
        embed = embedding_table[token]
        append embed to context as next input embedding
        record (token, log_prob) for RL
```

After reasoning trace, generate answer tokens normally (all hard, standard autoregressive).

## Training Loop Modification

During the GRPO training update:

1. **Cannot teacher-force soft tokens.** Unlike standard GRPO where you recompute all log-probs in one parallel forward pass, soft tokens must be regenerated autoregressively (each depends on the current parameters and all previous tokens).

2. **Procedure for computing training loss on a trajectory:**
   - Forward through prompt (parallel, standard)
   - For each soft block: regenerate soft tokens sequentially (no_grad for first 3, grad for last 1)
   - At each hard token: compute log π(h_i | context) using known h_i from rollout
   - Compute GRPO loss on hard token log-probs
   - Backprop

3. **Cost:** Training forward pass is ~same cost as rollout generation (both sequential through soft tokens). Roughly 2x the cost of standard GRPO per step. Acceptable for a first experiment.

## Implementation Plan

### New components to implement:
1. `W_soft` linear layer (d_model → d_embed)
2. Maze environment: generation, tokenization, reward computation (use `maze-dataset` or roll own)
3. Modified generation loop that alternates soft/hard at fixed positions
4. Modified training forward pass that regenerates soft tokens and computes hard token log-probs
5. Validity reward function (check each move against maze walls)
6. Adjustment to GRPO loss to only include hard tokens (the directional moves)

### What stays the same:
- Base model (Qwen-2.5)
- GRPO algorithm (advantage computation, clipping, KL penalty)
- General training loop structure

### Hyperparameters for first experiment:
- Soft block length: 4
- Hard token frequency: every 5th position (1 hard per 4 soft)
- Number of reasoning cycles: ~10-30 (so 10-30 moves, enough for 5×5 to 7×7 mazes)
- `W_soft` initialization: identity (or near-identity with small noise)
- Gradient strategy: one-step approximation initially
- Maze size: start with 5×5, scale to 7×7
- Validity reward weight: 0.5 × correctness reward

### Success criteria:
1. Model converges and produces valid directional moves (not garbage tokens)
2. Model solves mazes at a rate comparable to or better than standard CoT baseline
3. Soft tokens demonstrably contribute (ablation: compare to hard-tokens-only with same compute budget)
4. (Stretch) Hard token subsequences transfer to a different model via offline RL

## Key Design Decisions and Rationale

| Decision | Choice | Rationale |
|----------|--------|-----------|
| First task | Maze navigation (not Countdown) | Single-token actions fit fixed interleaving; Countdown needs multi-token expressions |
| Soft token representation | Direct linear projection from hidden state, no vocab head | More expressive, cheaper, avoids vocab bottleneck |
| Switching mechanism | Fixed interleaving (4:1) | Simplest possible; removes switching as a confound |
| Soft token stochasticity | None (deterministic) | All RL stochasticity from hard tokens; simpler |
| Gradient through soft blocks | One-step approximation | O(1) memory; justified by HRM/DEQ analogy |
| RL algorithm | GRPO (existing) | Already implemented; works well |
| Soft tokens in RL objective | Excluded | They are internal computation, not actions |
| Hard token reward | Move validity (doesn't hit wall) | Provides gradient anchors; trivially verifiable |

## Novelty and Contribution

The unique aspects of this approach relative to existing work:

1. **Offline RL motivation**: No existing paper frames hybrid soft/hard tokens as creating transferable intermediate states for offline/off-policy RL. The hard token subsequence defines a shared, interpretable MDP that different policies can contribute to.

2. **Learned Bernoulli switching + BPTT soft tokens + RL only on hard tokens**: This specific combination doesn't exist in the literature (the fixed interleaving is a stepping stone to learned switching).

3. **Terse checkpoint shaping**: Using format rewards to shape hard tokens into structured intermediate results (arithmetic expressions) while soft tokens handle fuzzy reasoning.

If this first experiment works, the next steps are:
- **Countdown with learned switching**: Bernoulli gate for soft/hard, format rewards for arithmetic expressions
- **Offline RL transfer experiment**: extract hard token trajectories from model A, use to train model B
- Scale to harder mazes (10×10+), then to GSM8K / harder math problems

## Existing Work on Maze Navigation for LLMs

Key references to be aware of:

- **AlphaMaze** (arXiv:2502.14669): Qwen 1.5B + SFT + GRPO on tokenized mazes, 93% accuracy. Most direct baseline — same model family, same RL algorithm. Public dataset of 100k mazes on HuggingFace.
- **Dualformer** (Meta FAIR, arXiv:2410.09918): Trains encoder-decoder transformer on maze + Sokoban with A* search traces. Randomized trace-dropping for fast/slow thinking modes. 15M params for mazes. Uses Searchformer tokenization.
- **HRM** (arXiv:2506.21734): Hierarchical Reasoning Model, 27M params, solves 30×30 mazes optimally with ~1000 training examples. Our architectural inspiration for the one-step gradient approximation.
- **maze-dataset** library (arXiv:2309.10498, `pip install maze-dataset`): Configurable Python package for maze generation, solving, and tokenization. Used by multiple interpretability papers. Supports multiple generation algorithms and output formats.
- **Transformers Can Navigate Mazes With Multi-Step Prediction** (arXiv:2412.05117): Shows MLM-U objective (predict multiple steps ahead/back) dramatically improves maze navigation vs next-token prediction. 4x more data efficient.
- **MazeEval** (arXiv:2507.20395): Benchmark for LLM maze navigation with coordinate-based feedback, 5×5 to 15×15 grids.
- **Structured World Representations / Causal World Models in Mazes** (arXiv:2312.02566, arXiv:2412.11867): Interpretability work showing transformers learn linear representations of maze connectivity. Useful for understanding what the soft tokens might learn.
