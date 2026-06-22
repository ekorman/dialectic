import sys
from dataclasses import dataclass, field

# ---------- Reusable mixins ------------------------------------------------
#
# All mixin classes are `@dataclass(kw_only=True)` so their fields land in
# the keyword-only section of any subclass's ``__init__``. This sidesteps
# the dataclass "required-before-default" ordering rule when a concrete
# class with required positional fields composes one or more mixins.
#
# arg_parser always constructs dataclasses via keyword arguments, so the
# kw_only marker has no effect on the CLI surface — every flag still maps
# to a field name the same way.


@dataclass(kw_only=True)
class _LoraMixin:
    """LoRA configuration for the trainable model.

    Concrete defaults — used by launchers that train or merge LoRA at known
    settings. Eval-side classes that want None-as-"auto-derive-from-config"
    semantics (e.g. ``EvalCommonParams``) declare their LoRA fields inline.
    """

    lora_rank: int | None = None
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    lora_target_modules: str = "all"


@dataclass(kw_only=True)
class _StartCkptMixin:
    """Resume-from-checkpoint reference."""

    start_ckpt_run: str | None = None
    start_ckpt_step: int | None = None


@dataclass(kw_only=True)
class _ForwardCkptMixin:
    """Reference to the forward (p) checkpoint that q is paired with."""

    forward_ckpt_run: str | None = None
    forward_ckpt_step: int | None = None


@dataclass(kw_only=True)
class _QCkptMixin:
    """Reference to a q checkpoint (optional — e.g. SFT importance weighting)."""

    q_ckpt_run: str | None = None
    q_ckpt_step: int | None = None


# ---------- Concrete param classes -----------------------------------------


@dataclass
class TrainParams(_StartCkptMixin):
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
    save_ckpt_freq: int = sys.maxsize
    logprob_chunk_size: int = 64
    warmup_steps: int = 0
    weight_decay: float = 0.01


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
class InverseCotParams(_LoraMixin, _ForwardCkptMixin):
    normalize_by_sequence_length: bool
    freeze_lm_head: bool
    # The p-rollout artifact q trains on. Required at runtime (validated in
    # the launcher); has no derivation anchor since q training is the root
    # of the inverse-CoT lineage.
    p_rollout_artifact: str | None = None
    gradient_checkpointing: bool = False
    contrastive_weight: float = 1.0
    contrastive_margin: float = 1.0
    train_group_size: int | None = None
    max_cot_tokens: int | None = None


@dataclass
class EvalCommonParams:
    """Shared CLI inputs for q-eval launchers.

    Only ``q_ckpt_run`` and ``q_ckpt_step`` are required from the CLI; every
    other field defaults to ``None`` and is auto-derived at launch time from
    the q run's extty config (which carries ``model_name``, ``use_bf16``, the
    q-LoRA settings, and the ``forward_ckpt_run`` / ``forward_ckpt_step`` of
    the p checkpoint). p's own LoRA settings are recursively pulled from the
    forward run's config when available. Any field passed on the CLI takes
    precedence over the derived value.

    LoRA / q-LoRA fields are declared inline (not via ``_LoraMixin`` /
    ``_QLoraMixin``) because they default to ``None`` to signal
    "auto-derive from config" rather than the concrete defaults used by
    training-side launchers.

    Launchers that need additional knobs (sampling temperature, vLLM
    settings, etc.) subclass this and add them.
    """

    q_ckpt_run: str
    q_ckpt_step: int
    model_name: str | None = None
    forward_ckpt_run: str | None = None
    forward_ckpt_step: int | None = None
    use_bf16: bool | None = None
    lora_rank: int | None = None
    lora_alpha: float | None = None
    lora_target_modules: str | None = None
    q_lora_rank: int | None = None
    q_lora_alpha: float | None = None
    q_lora_target_modules: str | None = None
    # The rollout artifact to read prompts/completions from. Auto-derived
    # from the q run's training config (the artifact q was trained on)
    # when not set.
    p_rollout_artifact: str | None = None
    seed: int = 42
    split: str = "val"
    max_prompts: int | None = None


@dataclass
class InverseCotEvalParams(EvalCommonParams):
    """FCR evaluator inputs — adds the vLLM-sampling-specific knobs that the
    q-verifier launcher doesn't need."""

    batch_size: int = 16
    max_tokens_generated: int = 500
    temperature: float = 0.7
    n_samples: int = 1
    gpu_memory_utilization: float = 0.45


@dataclass
class GrpoEvalParams(_LoraMixin):
    """GRPO p-evaluator inputs.

    Only one model is in scope here (the checkpoint being evaluated), so
    the field is called ``ckpt_run`` rather than ``forward_ckpt_run`` —
    "forward" only carries meaning in inverse-CoT contexts where q is
    paired with p.

    The minimal CLI invocation is
    ``--eval_params.p-rollout-artifact <name>``: the artifact's producing-run
    metadata is traced back to its ``rollout_gen_params.start_ckpt_run/step``,
    which gives the ckpt to evaluate. ``model_name`` and ``use_bf16`` are
    then pulled from that ckpt's ``train_params``. Any field passed on the
    CLI overrides the corresponding step in the derivation chain.
    """

    seed: int = 42
    split: str = "val"
    max_tokens_generated: int = 500
    temperature: float = 0.7
    n_samples: int = 8
    max_prompts: int | None = None
    gpu_memory_utilization: float = 0.90
    # Keyword-only + Optional so the launcher's resolver hook can fill
    # them in from the rollout artifact's producing run and the forward
    # checkpoint's training config before extty snapshots the kwargs.
    p_rollout_artifact: str | None = field(default=None, kw_only=True)
    model_name: str | None = field(default=None, kw_only=True)
    ckpt_run: str | None = field(default=None, kw_only=True)
    # Which checkpoint step(s) of ``ckpt_run`` to evaluate. A spec string:
    #   single   — "63000"
    #   range    — "1000..160000..1000" (start..stop..step, stop inclusive);
    #              or "a..b" for every saved checkpoint in [a, b]
    #   list     — "1000,2000,3000"
    #   omitted  — None: evaluate ALL saved checkpoints of ``ckpt_run``
    # Each evaluated step is logged at ``step=ckpt_step`` so extty's native
    # step-axis is the learning curve. Specs are intersected with the run's
    # actually-saved checkpoints.
    ckpt_step: str | None = field(default=None, kw_only=True)
    use_bf16: bool | None = field(default=None, kw_only=True)


@dataclass
class GenerateQCotParams(EvalCommonParams):
    """q-CoT generation inputs — adds the sampling knobs needed for offline
    data synthesis on top of the shared auto-derivable surface."""

    batch_size: int = 16
    max_tokens_generated: int = 500
    temperature: float = 0.3
    # Which split of the rollout artifact to generate q-CoTs for. Overrides
    # the EvalCommonParams default of "val": synthesis targets training
    # data — only the train split's q-CoTs are ever consumed (by SFT), so
    # generating for val/test burns GPU on artifacts nothing reads.
    split: str = "train"


@dataclass
class SftParams(_LoraMixin, _StartCkptMixin, _QCkptMixin):
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
    q_cot_artifact: str | None = None
    p_rollout_artifact: str | None = None
    mix_ratio: float = 1.0
    # Drop q-cot training examples whose prompt was ALL-correct in the
    # p-rollout artifact: p already solves those reliably, so q's CoT adds
    # no signal there and dilutes the hard/rescue prompts that SFT exists
    # to fix. Disable with --sft_params.no-drop-all-correct-q-cots.
    drop_all_correct_q_cots: bool = True
    warmup_steps: int = 0
    max_tokens_generated: int = 500
    val_pass_at_n: int = 1
    # Only used by envs whose datasets lack a genuine val split (gsm8k):
    # training-time validation is a random holdout of this many TRAIN
    # prompts, keeping the artifact's "val" slot (the official test set)
    # clean for final test metrics. **0 disables the holdout entirely** — all
    # train data is used and in-training val is turned off (evaluate
    # checkpoints separately on the test learning curve). Default is 0 since
    # the reporting protocol no longer selects checkpoints from a small val.
    # Envs with a real train/val/test split (countdown) validate against the
    # artifact's val split and ignore this.
    n_val_holdout: int = 0
    # Keyword-only + Optional so the launcher's resolver can fill them in
    # from the q checkpoint's training config when ``q_ckpt_run`` is set.
    # Validation enforces non-None after derivation.
    model_name: str | None = field(default=None, kw_only=True)
    use_bf16: bool | None = field(default=None, kw_only=True)


@dataclass
class DatasetGenParams:
    n_examples: int
    train_pct: float
    val_pct: float
    test_pct: float
    seed: int


@dataclass
class RolloutGenParams(_StartCkptMixin):
    """Note: rollout artifacts are COMPLETE — every prompt is emitted
    regardless of its correctness mix. Any filtering (mixed-correctness for
    q training, ``is_correct`` for SFT, …) happens explicitly at load time
    in the consumer.
    """

    model_name: str
    batch_size: int
    seed: int
    temperature: float
    max_tokens_generated: int
    group_size: int
    use_bf16: bool
