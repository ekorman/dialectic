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
    normalize_advantages: bool
    normalize_by_sequence_length: bool
    beta: float
    mu: int = 1
    eps: float | None = None
    update_ref_net_batch_cadence: int | None = None


@dataclass
class RewardParams:
    answer_tags_weight: float
    think_tags_weight: float


@dataclass
class CountdownParams:
    n_larges: int | list[int]
    n_total: int | list[int]
    n_ops: int | list[int]


@dataclass
class GSM8kParams:
    train_path: str
    val_path: str | None = None


@dataclass
class MathEnvParams:
    difficulty: Literal["trivial", "easy", "medium"]
    direct_arithmetic_prob: float = 0.2
    twostep_arithmetic_prob: float = 0.3
    word_problem_prob: float = 0.35
    number_properties_prob: float = 0.15


@dataclass
class InverseCotParams:
    normalize_by_sequence_length: bool
    freeze_lm_head: bool
    gradient_checkpointing: bool = False
    contrastive_weight: float = 1.0
    contrastive_margin: float = 1.0
    train_group_size: int | None = None
    max_cot_tokens: int | None = None
    forward_ckpt_run: str | None = None
    forward_ckpt_step: int | None = None


@dataclass
class InverseCotEvalParams:
    model_name: str
    forward_ckpt_run: str
    forward_ckpt_step: int
    use_bf16: bool
    q_ckpt_run: str | None = None
    q_ckpt_step: int | None = None
    seed: int = 42
    split: str = "val"
    batch_size: int = 16
    max_tokens_generated: int = 500
    temperature: float = 0.7
    baseline_only: bool = False
    n_samples: int = 1
    lora_rank: int | None = None
    lora_alpha: float = 16.0
    lora_target_modules: str = "all"
    max_prompts: int | None = None


@dataclass
class GenerateQCotParams:
    model_name: str
    forward_ckpt_run: str
    forward_ckpt_step: int
    q_ckpt_run: str
    q_ckpt_step: int
    use_bf16: bool
    batch_size: int = 16
    max_tokens_generated: int = 500
    temperature: float = 0.3
    seed: int = 42


@dataclass
class SftParams:
    model_name: str
    start_ckpt_run: str
    start_ckpt_step: int
    use_bf16: bool
    batch_size: int
    lr: float
    max_episodes: int
    max_grad_norm: float = 10.0
    seed: int = 20
    temperature: float = 1.0
    val_freq: int = 500
    val_batch_size: int = 16
    val_episodes: int | None = None
    save_ckpt_freq: int = sys.maxsize
    accumulation_steps: int = 1
    compile_model: bool = False
    fcr_filter: bool = False
    use_rollout_data: bool = False
    mix_rollout_artifact: str | None = None
    mix_ratio: float = 0.5
    warmup_steps: int = 0
    max_tokens_generated: int = 500
    val_rollout_artifact: str | None = None
    val_pass_at_n: int = 1
    freeze_embeddings: bool = False
    q_ckpt_run: str | None = None
    q_ckpt_step: int | None = None
    lora_rank: int | None = None
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    lora_target_modules: str = "all"


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
