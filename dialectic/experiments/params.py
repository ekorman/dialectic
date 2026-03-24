import sys
from dataclasses import dataclass
from typing import Literal


@dataclass
class TrainParams:
    model_name: str
    batch_size: int
    lr: float
    accumulation_steps: int
    max_grad_norm: float
    max_episodes: int
    max_tokens_generated: int  # seems optional for sum e.g. internal reasoning sftgert
    compile_model: bool
    use_bf16: bool
    seed: int
    temperature: float
    val_batch_size: int
    val_episodes: int
    val_freq: int
    start_ckpt_run: str | None = None
    start_ckpt_step: int | None = None
    save_ckpt_freq: int = sys.maxsize
    logprob_chunk_size: int = 64
    warmup_steps: int = 0


@dataclass
class GRPOParams:
    group_size: int
    advantage_fn_type: Literal["grpo", "rloo"]
    normalize_advantages: bool  # only used for grpo
    normalize_by_sequence_length: bool
    beta: float
    mu: int = 1
    eps: float | None = None
    update_ref_net_batch_cadence: int | None = None


@dataclass
class SoftGRPOParams:
    noise_std: float
    normalize_soft_pdf_by_dim: bool


@dataclass
class InternalReasoningParams:
    soft_block_size: int
    soft_bptt_window: int
    soft_projection: bool = False
    soft_projection_alpha_init: float | None = None
    soft_projection_rank: int | None = None


@dataclass
class HybridReasoningParams:
    soft_block_size: int
    soft_bptt_window: int
    max_cycles: int
    soft_projection: bool = False
    soft_projection_alpha_init: float | None = None
    soft_projection_rank: int | None = None
    max_tokens_per_cycle: int = 20
    think_token_id: int | None = None
    pass_at_k_samples: int = 0
    pass_at_k_temperature: float = 0.7


@dataclass
class RewardParams:
    answer_tags_weight: float
    think_tags_weight: float
    scratch_tags_weight: float = 0.0


@dataclass
class MazeRewardParams:
    validity_weight: float
    distance_weight: float


@dataclass
class CountdownParams:
    n_larges: int | list[int]
    n_total: int | list[int]
    n_ops: int | list[int]


@dataclass
class MathEnvParams:
    difficulty: Literal["trivial", "easy", "medium"]
    direct_arithmetic_prob: float = 0.2
    twostep_arithmetic_prob: float = 0.3
    word_problem_prob: float = 0.35
    number_properties_prob: float = 0.15


@dataclass
class SoftCyclingParams:
    min_soft_steps: int = 0
    use_gumbel: bool = False
    soft_bptt_window: int | None = None
    max_cycles: int = 20
    max_tokens_per_cycle: int = 30
    max_soft_steps_per_cycle: int | None = None
    min_cycles: int = 0
    evict_soft_kv: bool = False
    n_expert_trajectories: int = 0


@dataclass
class NoiseReasoningParams:
    n_noise_per_cycle: int = 8
    noise_std: float = 1.0
    min_cycles: int = 0
    max_cycles: int = 10
    max_tokens_per_cycle: int = 30
    evict_noise_kv: bool = False
    prompt_style: str = "scratch_tags"
    use_noise_adapter: bool = False
    adapter_d_ff: int = 256
    adapter_n_heads: int = 4
    freeze_base_model: bool = False


@dataclass
class SingleStepSFTParams:
    max_answer_tokens: int
    normalize_by_sequence_length: bool


@dataclass
class MultiStepSFTParams:
    normalize_by_sequence_length: bool
