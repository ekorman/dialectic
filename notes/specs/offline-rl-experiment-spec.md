# Offline RL Transfer Experiment Spec

## Overview

This document describes the experiment that constitutes the **core novel contribution**: demonstrating that hard token subsequences from the hybrid soft/hard architecture are more suitable for offline and off-policy RL than standard CoT traces. Specifically, we want to show that trajectories collected from one policy (model A) can be used to improve a different policy (model B) more effectively when those trajectories consist of structured hard token checkpoints than when they consist of model-specific CoT reasoning.

## The Core Claim

Standard CoT traces are **policy-specific**: they contain filler words, model-specific phrasing, and stylistic artifacts that make them poor training signal for a different model. Soft token representations are even worse — they're continuous vectors with no cross-model meaning. But hard token subsequences from our architecture are **structured, terse, and interpretable** (e.g., directional moves in mazes, arithmetic steps in math). They define a shared action space that any model can learn from, regardless of how its internal reasoning works.

## Experimental Design

### Phase 1: Collect Trajectories from Source Policies

Train multiple source policies using our hybrid architecture on the same maze distributions:

**Source Policy A**: Qwen-2.5-1.5B with hybrid soft/hard tokens, trained with GRPO on MEDIUM mazes
**Source Policy B**: Qwen-2.5-3B with hybrid soft/hard tokens, trained with GRPO on MEDIUM mazes
**Source Policy C** (optional): Different architecture entirely (e.g., Llama-3.2-1B) with hybrid soft/hard tokens

For each source policy, generate a large trajectory dataset by rolling out on a held-out set of mazes (not seen during training). For each trajectory, record:

```python
@dataclass
class HardTokenTrajectory:
    maze: Maze                        # the maze instance
    maze_prompt: str                  # tokenized maze prompt
    moves: list[str]                  # hard token sequence: ["right", "down", "down", "left", ...]
    reached_goal: bool                # did this trajectory solve the maze?
    path_valid: bool                  # were all moves wall-respecting?
    reward: float                     # total reward received
    move_log_probs: list[float]       # log π_source(move_i | maze, moves_<i) -- optional, for importance weighting
    source_model: str                 # identifier for which model generated this
```

Note: we **discard all soft token information**. The offline dataset contains only the maze prompt and the hard token moves. This is the whole point — the dataset is model-agnostic.

**Dataset sizes**: Aim for ~50k-100k trajectories per source policy, with a mix of successful and failed trajectories (important for offline RL to see both).

### Phase 2: Collect CoT Baseline Trajectories

For comparison, train standard CoT policies on the same mazes:

**CoT Policy A**: Qwen-2.5-1.5B with standard GRPO + CoT (full discrete reasoning trace before answer)
**CoT Policy B**: Qwen-2.5-3B with standard GRPO + CoT

Collect trajectories in the same way, but now the trajectory includes the full CoT reasoning:

```python
@dataclass
class CoTTrajectory:
    maze: Maze
    maze_prompt: str
    reasoning_trace: str              # full CoT text: "I need to go right because..."
    moves: list[str]                  # extracted move sequence from the CoT
    reached_goal: bool
    path_valid: bool
    reward: float
    token_log_probs: list[float]      # log probs over ALL tokens (reasoning + moves)
    source_model: str
```

### Phase 3: Offline RL Training

Now train **target policies** using only the collected offline datasets, with no online interaction with the maze environment.

#### Target models

Use models that are **different from the source policies** to test transfer:

- **Target 1**: Qwen-2.5-0.5B (smaller than any source)
- **Target 2**: Qwen-2.5-1.5B with different random seed / initialization
- **Target 3** (stretch): Llama-3.2-1B (different architecture entirely)

#### Offline RL methods

For each target model, train using several offline approaches:

**Method 1: Filtered Behavior Cloning (simplest)**

Only train on successful trajectories (reached_goal=True). Standard supervised learning:

- **Hard token BC**: Train target model to predict move sequence given maze prompt. Loss = cross-entropy on moves only. The model learns: `maze_prompt → move_1, move_2, ..., move_n`
- **CoT BC**: Train target model to predict full CoT + moves given maze prompt. Loss = cross-entropy on all tokens.
- **CoT-then-extract BC**: Train on CoT trajectories but only supervise on the extracted move tokens (ignore reasoning text). This isolates whether the CoT reasoning helps or hurts transfer.

**Method 2: Weighted Behavior Cloning / Reward-Weighted Regression**

Train on all trajectories (including failures), but weight each trajectory by its reward:

```
L = -Σ_traj reward(traj) * Σ_i log π_target(move_i | context_i)
```

This is the simplest offline RL method. Better trajectories get more weight. Same three variants as above (hard token, full CoT, CoT-extract).

**Method 3: Offline GRPO / Best-of-N**

Group trajectories by maze instance. For each maze, compute advantages across the collected trajectories (same as online GRPO but using pre-collected rollouts):

```
For maze m:
    trajectories = all trajectories for maze m (from any source policy)
    mean_reward = mean(rewards)
    advantages = rewards - mean_reward

    L = -Σ_traj A(traj) * Σ_i log π_target(move_i | context_i)
```

This requires multiple trajectories per maze, so the dataset needs to include multiple rollouts per maze instance across source policies.

**Method 4: Conservative Q-Learning / IQL (advanced, optional)**

Frame the hard token sequence as an MDP:
- State: (maze, moves_so_far) — or equivalently, current position in maze
- Action: next move (U/D/L/R)
- Reward: per-step validity + terminal goal reward
- Transition: deterministic (move in maze)

Apply IQL (Implicit Q-Learning) or CQL (Conservative Q-Learning) on the offline dataset. This works naturally on the hard token MDP because states and actions are discrete and shared across policies.

For CoT comparison, this is much harder — the "state" includes the full reasoning trace which is model-specific and high-dimensional. This is where the argument for hard tokens should be strongest.

### Phase 4: Evaluation

Evaluate all target models on a held-out test set of mazes (never seen during source policy training or trajectory collection).

#### Metrics

1. **Solve rate**: fraction of test mazes where the target model reaches the goal
2. **Valid move rate**: fraction of moves that don't hit walls
3. **Path efficiency**: optimal_length / actual_length (1.0 = optimal)
4. **Solve rate by difficulty**: break down by maze size / openness / solution length

#### Key comparisons

**Comparison 1: Hard tokens vs CoT for transfer (main result)**

Hold everything else constant (same source policy, same target model, same offline RL method):

| Condition | Training data | Expected result |
|-----------|--------------|-----------------|
| Hard token BC | Moves only from hybrid model | Best transfer |
| CoT BC | Full CoT from CoT model | Worse — model-specific reasoning hurts |
| CoT-extract BC | Moves extracted from CoT model | Comparable to hard token? (interesting either way) |

If hard token BC beats CoT BC, the architecture is doing its job. If hard token BC also beats CoT-extract BC, it suggests the hybrid training produces better move sequences (not just that we stripped the reasoning).

**Comparison 2: Cross-model transfer**

Train on trajectories from model A, evaluate on model B's architecture:

| Source → Target | Hard tokens | CoT |
|----------------|-------------|-----|
| Qwen-1.5B → Qwen-0.5B | ? | ? |
| Qwen-3B → Qwen-0.5B | ? | ? |
| Qwen-1.5B → Llama-1B | ? | ? |

Hard tokens should transfer across architectures; CoT should degrade because of model-specific reasoning patterns.

**Comparison 3: Multi-source pooling**

Pool trajectories from multiple source policies into a single offline dataset:

| Dataset composition | Hard tokens | CoT |
|--------------------|-------------|-----|
| Single source | Baseline | Baseline |
| Pooled (2 sources) | Should improve (more coverage) | May hurt (conflicting reasoning styles) |
| Pooled (3 sources) | Should improve more | May hurt more |

This is the strongest test of the offline RL motivation. If pooling hard token trajectories from diverse source policies improves the target, while pooling CoT trajectories degrades it, that's a compelling result.

**Comparison 4: Online vs offline**

Compare the offline-trained target model against:
- Target model trained with online GRPO (upper bound — full environment access)
- Target model with no training (lower bound — zero-shot on mazes)

This contextualizes how much of the online RL gap the offline approach closes.

## Practical Considerations

### Dataset collection efficiency

Each source policy rollout is cheap (just inference, no gradient computation). Collecting 50k trajectories from a 1.5B model on 5×5 mazes should take ~1-2 GPU hours.

### Multiple trajectories per maze

For offline GRPO (Method 3), we need multiple trajectories per maze instance. Strategy:
- Generate 1000-5000 unique mazes
- Roll out each source policy 10-20 times per maze (with different sampling temperatures)
- This gives 10-60 trajectories per maze across sources

### Trajectory quality distribution

Important: the offline dataset should contain a **range of qualities**, not just successes. Include:
- Successful optimal paths
- Successful but suboptimal paths (longer than necessary)
- Partial successes (valid moves but didn't reach goal)
- Failures (hit walls, went in circles)

This gives offline RL methods something to differentiate. If we only include successes, everything reduces to behavior cloning.

Control trajectory quality by varying sampling temperature during collection:
- Low temp (0.1-0.3): mostly optimal/near-optimal trajectories
- Medium temp (0.5-0.8): mix of good and mediocre
- High temp (1.0-1.5): lots of exploration, many failures

### Handling the CoT action extraction

For CoT trajectories, we need to extract the move sequence from the reasoning trace. This requires a parser that identifies directional tokens within the CoT text. This should be straightforward for mazes (grep for U/D/L/R or up/down/left/right tokens) but is a potential source of noise.

For Countdown (future work), extracting intermediate computation steps from CoT is much harder and more ambiguous — another argument for the hybrid architecture.

## What Would Make This a Strong Paper

### Minimum viable result
Hard token BC outperforms CoT BC for cross-model transfer on the same set of mazes. This alone validates the core claim.

### Strong result
Multi-source pooling of hard token trajectories improves target performance, while pooling CoT trajectories does not. This demonstrates the model-agnostic property of the hard token MDP.

### Home run result
Offline GRPO on pooled hard token trajectories from multiple source policies approaches the performance of online GRPO for the target model. This would show that our architecture enables genuine offline RL for LLM reasoning — the hard tokens create a shared state/action space that makes off-policy learning tractable.

### Important negative result (also publishable)
If CoT-extract BC (moves extracted from standard CoT, ignoring reasoning) performs as well as hard token BC, that would suggest the architecture isn't necessary — you could just extract actions from any CoT. But we hypothesize this won't happen because:
1. CoT-trained models produce move sequences entangled with their reasoning patterns (e.g., they might condition on having just said "I should go right" before actually going right)
2. The hybrid architecture is trained to make each hard token a good standalone decision, while CoT models treat moves as part of a larger text generation problem
3. At a minimum, the hybrid approach is cleaner and doesn't require a fragile extraction step

## Experiment Timeline

Assuming the hybrid architecture is working on mazes:

1. **Week 1**: Train source policies (hybrid + CoT baselines), collect trajectory datasets
2. **Week 2**: Implement offline training loop (BC + reward-weighted BC), run Comparison 1
3. **Week 3**: Cross-model and multi-source experiments (Comparisons 2 + 3)
4. **Week 4**: Offline GRPO experiments, analysis, ablations

Total compute estimate: ~$50-100 across all experiments (single A40/4090, small models, maze task is lightweight).

## Connection to Future Work

If this works on mazes, the path to Countdown / math is:
1. Replace fixed interleaving with learned Bernoulli switching
2. Hard tokens become arithmetic expressions (multi-token, requiring the switch)
3. Offline dataset is (problem, intermediate_computation_steps, answer) tuples
4. Same offline RL methods apply, but now the "action" is a structured expression rather than a single direction token

The maze experiment validates the infrastructure and the core claim. The math experiment would then show it generalizes to more complex, variable-length reasoning.
