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
    val_freq: int
    val_episodes: int | None = None
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
class RewardParams:
    answer_tags_weight: float
    think_tags_weight: float


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
class SingleStepSFTParams:
    max_answer_tokens: int
    normalize_by_sequence_length: bool


@dataclass
class MultiStepSFTParams:
    normalize_by_sequence_length: bool


@dataclass
class InverseCotParams:
    normalize_by_sequence_length: bool
    freeze_lm_head: bool
    gradient_checkpointing: bool = False
    contrastive_weight: float = 1.0
    contrastive_margin: float = 1.0
    train_group_size: int | None = None
    max_cot_tokens: int | None = None
    unfreeze_mlp: bool = False
    forward_ckpt_run: str | None = None
    forward_ckpt_step: int | None = None


@dataclass
class InverseCotEvalParams:
    model_name: str
    batch_size: int
    seed: int
    temperature: float
    max_tokens_generated: int
    q_ckpt_run: str
    q_ckpt_step: int
    dataset_artifact: str
    split: str
    q_max_tokens_generated: int
    use_bf16: bool
    start_ckpt_run: str | None = None
    start_ckpt_step: int | None = None


@dataclass
class DatasetGenParams:
    n_examples: int
    train_pct: float
    val_pct: float
    test_pct: float
    seed: int


@dataclass
class CombineJsonlParams:
    name_prefix: str


@dataclass
class RolloutGenParams:
    model_name: str
    batch_size: int
    seed: int
    temperature: float
    max_tokens_generated: int
    group_size: int
    n_pos_min: int
    n_neg_min: int
    use_bf16: bool
    n_shards: int = 1
    start_ckpt_run: str | None = None
    start_ckpt_step: int | None = None
