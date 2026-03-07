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
    max_tokens_generated: int
    val_freq: int
    val_episodes: int
    val_batch_size: int
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


@dataclass
class GRPOParams:
    group_size: int
    advantage_fn_type: Literal["grpo", "rloo"]
    normalize_advantages: bool
    normalize_by_sequence_length: bool

    beta: float
    mu: int = 1
    eps: float | None = None
    update_ref_net_batch_cadence: int | None = None


@dataclass
class SoftGRPOParams:
    noise_std: float
    normalize_soft_pdf_by_dim: bool


# rename to internal reasoning params or hybrid reasoning params
@dataclass
class HybridReasoningParams:
    soft_block_size: int
    soft_bptt_window: int
    max_cycles: int
    soft_projection_alpha_init: float | None = None
    soft_projection_rank: int | None = None
    soft_projection: bool = False
    think_token_id: int | None = None


@dataclass
class STHTParams:
    noise_std: float


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
class SingleStepSFTParams:
    max_answer_tokens: int
    normalize_by_sequence_length: bool


@dataclass
class MultiStepSFTParams:
    move_name_to_id: dict[str, int]
    valid_hard_token_ids: list[int]
    normalize_by_sequence_length: bool
    think_token_id: int | None
