"""Validate GRPO implementation against TRL's GRPOTrainer on Countdown task.

This script trains a dialectic BaseTransformer using TRL's GRPOTrainer
to validate that our custom GRPO implementation produces similar results.

Based on TinyZero: https://github.com/Jiayi-Pan/TinyZero

Usage:
    uv run --group trl python train_scripts/grpo_countdown_trl.py
    uv run --group trl python train_scripts/grpo_countdown_trl.py --device cuda
    uv run --group trl python train_scripts/grpo_countdown_trl.py --max-steps 500

Requirements:
    Install TRL dependency group: uv sync --group trl
"""

import argparse
import random
import re
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from datasets import Dataset
from tokenizers import Tokenizer
from transformers import (
    GenerationConfig,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from trl import GRPOConfig, GRPOTrainer

from dialectic.llm.base import BaseTransformer
from dialectic.llm.components import KVCache
from dialectic.llm.qwen import load_qwen_06b
from dialectic.llm.templates import Message, get_qwen_input_text_from_messages
from dialectic.llm.utils import get_default_device
from dialectic.rl.env import CountdownEnv


# ==============================================================================
# BaseTransformer Wrapper for HuggingFace Compatibility
# ==============================================================================


@dataclass
class BaseTransformerConfig(PretrainedConfig):
    """Configuration for BaseTransformer wrapped as PreTrainedModel."""

    model_type: str = "base_transformer"

    d: int = 1024
    vocab_size: int = 151936
    n_decoder_layers: int = 28
    attn_head_d: int = 128
    attn_num_heads: int = 16
    attn_num_kv_heads: int = 8
    mlp_hidden_d: int = 3072
    rms_norm_eps: float = 1e-6
    rope_base_value: float = 1000000
    pad_token_id: int = 151643
    eos_token_id: int = 151645
    bos_token_id: int = 151643

    def __init__(self, **kwargs):
        # Extract our custom fields before passing to parent
        custom_fields = [
            "d",
            "vocab_size",
            "n_decoder_layers",
            "attn_head_d",
            "attn_num_heads",
            "attn_num_kv_heads",
            "mlp_hidden_d",
            "rms_norm_eps",
            "rope_base_value",
        ]
        for f in custom_fields:
            if f in kwargs:
                setattr(self, f, kwargs.pop(f))
        super().__init__(**kwargs)


class BaseTransformerForCausalLM(PreTrainedModel):
    """Wrapper to make BaseTransformer compatible with HuggingFace/TRL.

    This wrapper:
    - Wraps a dialectic BaseTransformer as a PreTrainedModel
    - Implements forward() returning CausalLMOutputWithPast
    - Implements prepare_inputs_for_generation() for HF generate()
    - Manages KV cache for efficient generation
    """

    config_class = BaseTransformerConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _supports_cache_class = False

    def __init__(self, config: BaseTransformerConfig):
        super().__init__(config)
        self.config = config

        # Import decoder layer factory (we use Qwen's)
        from dialectic.llm.qwen import create_qwen_decoder_layer

        # Create the actual BaseTransformer
        self.model = BaseTransformer(
            d=config.d,
            vocab_size=config.vocab_size,
            n_decoder_layers=config.n_decoder_layers,
            attn_head_d=config.attn_head_d,
            attn_num_heads=config.attn_num_heads,
            attn_num_kv_heads=config.attn_num_kv_heads,
            mlp_hidden_d=config.mlp_hidden_d,
            rms_norm_eps=config.rms_norm_eps,
            rope_base_value=config.rope_base_value,
            decoder_layer_factory=create_qwen_decoder_layer,
        )

        # KV cache for generation (created on-demand)
        self._kv_caches: list[KVCache] | None = None
        self._cache_seq_len: int = 0

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.model.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.model.lm_head = new_embeddings

    def _init_kv_cache(self, batch_size: int, max_seq_len: int):
        """Initialize KV cache for generation."""
        device = next(self.parameters()).device
        self._kv_caches = [
            KVCache(
                max_seq_len=max_seq_len,
                num_heads=self.config.attn_num_kv_heads,
                head_dim=self.config.attn_head_d,
                device=device,
            )
            for _ in range(self.config.n_decoder_layers)
        ]
        self._cache_seq_len = 0

    def _clear_kv_cache(self):
        """Clear KV cache."""
        self._kv_caches = None
        self._cache_seq_len = 0

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # Convert attention mask to boolean if provided (HF uses 1/0, we use True/False)
        bool_mask = None
        if attention_mask is not None:
            bool_mask = attention_mask.bool()

        # Determine if we're using KV cache for generation
        kv_caches = None
        if use_cache and self._kv_caches is not None:
            kv_caches = self._kv_caches

        # Forward through the model
        logits = self.model(
            input_ids,
            kv_caches=kv_caches,
            attention_mask=bool_mask,
            return_all_logits=True,
        )

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )

        if not return_dict:
            output = (logits,)
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values="kv_cache" if use_cache else None,
            hidden_states=None,
            attentions=None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Prepare inputs for generation step."""
        # Initialize cache on first call
        if past_key_values is None:
            self._init_kv_cache(
                batch_size=input_ids.shape[0],
                max_seq_len=input_ids.shape[1] + 2048,  # prompt + max gen
            )
        else:
            # After first token, only pass the new token
            input_ids = input_ids[:, -1:]
            if attention_mask is not None:
                attention_mask = attention_mask

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": True,
        }

    def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs: dict[str, Any],
        is_encoder_decoder: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """Update model kwargs for next generation step."""
        model_kwargs["past_key_values"] = outputs.past_key_values

        # Extend attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [
                    attention_mask,
                    attention_mask.new_ones((attention_mask.shape[0], 1)),
                ],
                dim=-1,
            )

        return model_kwargs

    @classmethod
    def from_dialectic_model(
        cls, base_model: BaseTransformer, config: BaseTransformerConfig | None = None
    ) -> "BaseTransformerForCausalLM":
        """Create wrapper from existing BaseTransformer instance."""
        if config is None:
            # Infer config from model
            config = BaseTransformerConfig(
                d=base_model.d,
                vocab_size=base_model.vocab_size,
                n_decoder_layers=len(base_model.layers),
                attn_head_d=base_model.attn_head_d,
                attn_num_heads=base_model.attn_num_heads,
                attn_num_kv_heads=base_model.attn_num_kv_heads,
            )

        wrapper = cls(config)
        # Copy weights from the original model
        wrapper.model.load_state_dict(base_model.state_dict())
        return wrapper


# ==============================================================================
# Tokenizer Wrapper
# ==============================================================================


def create_hf_tokenizer(tokenizers_tokenizer: Tokenizer) -> PreTrainedTokenizerFast:
    """Wrap tokenizers.Tokenizer as PreTrainedTokenizerFast for TRL compatibility."""
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizers_tokenizer,
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        bos_token="<|endoftext|>",
    )
    hf_tokenizer.padding_side = "left"
    return hf_tokenizer


# ==============================================================================
# Reward Function for TRL
# ==============================================================================


def evaluate_countdown_expression(expr: str, numbers: list[int], target: int) -> bool:
    """Evaluate if expression equals target using valid numbers."""
    used_numbers = [int(n) for n in re.findall(r"\d+", expr)]

    available = numbers.copy()
    for n in used_numbers:
        if n in available:
            available.remove(n)
        else:
            return False

    try:
        result = eval(expr, {"__builtins__": {}}, {})
        return abs(result - target) < 1e-6
    except Exception:
        return False


def extract_answer(text: str) -> str | None:
    """Extract content from <answer>...</answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def countdown_reward_fn(
    completions: list[str],
    prompts: list[str] | None = None,
    numbers: list[list[int]] | None = None,
    targets: list[int] | None = None,
    **kwargs,
) -> list[float]:
    """TRL-compatible reward function for Countdown task.

    TRL passes completions and any extra columns from the dataset as kwargs.
    We expect 'numbers' and 'targets' columns in the dataset.
    """
    rewards = []
    for i, completion in enumerate(completions):
        # Get the problem parameters
        nums = numbers[i] if numbers else []
        target = targets[i] if targets else 0

        # Extract answer from completion
        extracted = extract_answer(completion)

        if extracted is None:
            # No answer tags - small reward for trying
            if "<answer>" in completion:
                rewards.append(0.1)
            else:
                rewards.append(0.0)
            continue

        # Check if parseable
        try:
            eval(extracted, {"__builtins__": {}}, {})
        except Exception:
            rewards.append(0.1)  # Has tags but not parseable
            continue

        # Check if correct
        if evaluate_countdown_expression(extracted, nums, target):
            rewards.append(1.0)
        else:
            rewards.append(0.3)  # Parseable but wrong answer

    return rewards


# ==============================================================================
# Dataset Generation
# ==============================================================================


def generate_countdown_dataset(
    n_samples: int,
    env: CountdownEnv,
    enable_thinking: bool = False,
    seed: int | None = None,
) -> Dataset:
    """Generate a dataset of countdown problems for TRL."""
    prompts = []
    numbers_list = []
    targets = []

    if seed is not None:
        env.rng.seed(seed)

    for i in range(n_samples):
        response = env.reset(seed=seed + i if seed is not None else None)
        data = response.data

        # Format as chat prompt (same as dialectic script)
        reasoning_tag = "think" if enable_thinking else "reasoning"
        prompt = get_qwen_input_text_from_messages(
            [Message(role="user", content=data.prompt)],
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        prompt += f"Let me solve this step by step\n<{reasoning_tag}>"

        prompts.append(prompt)
        numbers_list.append(data.numbers)
        targets.append(data.target)

    return Dataset.from_dict(
        {
            "prompt": prompts,
            "numbers": numbers_list,
            "targets": targets,
        }
    )


# ==============================================================================
# Main Training Function
# ==============================================================================


def train(
    *,
    device: str | None = None,
    max_steps: int = 500,
    batch_size: int = 2,
    group_size: int = 8,
    max_completion_length: int = 1024,
    lr: float = 1e-5,
    beta: float = 0.04,
    weights_path: str = "qwen3/qwen3-0.6b.pth",
    tokenizer_path: str = "qwen3/tokenizer.json",
    n_larges: int = 2,
    n_total: int = 6,
    n_ops: int = 5,
    seed: int = 42,
    use_qwen_thinking: bool = False,
    gradient_accumulation_steps: int = 8,
    dataset_size: int = 10000,
    logging_steps: int = 1,
    save_steps: int = 100,
    output_dir: str = "outputs/grpo_countdown_trl",
):
    """Train BaseTransformer using TRL's GRPOTrainer for validation."""
    torch.manual_seed(seed)
    random.seed(seed)

    device = device or get_default_device()
    print(f"Using device: {device}")

    # Load dialectic model
    print("Loading model...")
    base_model = load_qwen_06b()
    base_model.load_state_dict(
        torch.load(weights_path, map_location=device, weights_only=True)
    )
    print(f"Model loaded: {sum(p.numel() for p in base_model.parameters()) / 1e6:.1f}M parameters")

    # Wrap as HuggingFace-compatible model
    config = BaseTransformerConfig(
        pad_token_id=151643,
        eos_token_id=151645,
        bos_token_id=151643,
    )
    model = BaseTransformerForCausalLM.from_dialectic_model(base_model, config)
    model = model.to(device)

    # Create generation config
    model.generation_config = GenerationConfig(
        max_new_tokens=max_completion_length,
        do_sample=True,
        temperature=0.7,
        pad_token_id=151643,
        eos_token_id=151645,
    )

    # Load and wrap tokenizer
    print("Loading tokenizer...")
    tokenizers_tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer = create_hf_tokenizer(tokenizers_tokenizer)

    # Create countdown environment
    reasoning_tag = "think" if use_qwen_thinking else "reasoning"
    prompt_template = (
        "Using the numbers {numbers}, create an equation that equals {target}. "
        "You can use basic arithmetic operations (+, -, *, /) and each number at most once. "
        f"Show your reasoning in <{reasoning_tag}></{reasoning_tag}> tags."
        "Put your final equation in <answer></answer> tags, for example <answer> (1 + 2) / 3 </answer>."
    )

    env = CountdownEnv(
        seed=seed,
        n_larges=n_larges,
        n_total=n_total,
        n_ops=n_ops,
        prompt_template=prompt_template,
    )

    # Generate training dataset
    print(f"Generating {dataset_size} training examples...")
    train_dataset = generate_countdown_dataset(
        n_samples=dataset_size,
        env=env,
        enable_thinking=use_qwen_thinking,
        seed=seed,
    )
    print(f"Dataset created with {len(train_dataset)} examples")

    # Configure TRL GRPO
    grpo_config = GRPOConfig(
        output_dir=output_dir,
        # Training
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=lr,
        # GRPO specific
        beta=beta,
        loss_type="grpo",  # Use standard GRPO loss
        num_generations=group_size,
        max_completion_length=max_completion_length,
        temperature=0.7,
        # Advantage normalization (matches dialectic's normalize_advantages=True)
        scale_rewards="group",
        # Clipping (matches dialectic's eps=0.2)
        epsilon=0.2,
        # Logging
        logging_steps=logging_steps,
        save_steps=save_steps,
        report_to="none",  # Disable wandb/tensorboard for validation
        # Other
        seed=seed,
        bf16=torch.cuda.is_available(),
        remove_unused_columns=False,  # Keep numbers/targets columns for reward fn
    )

    # Create trainer
    print("Initializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_funcs=countdown_reward_fn,
    )

    # Train
    print("Starting training...")
    print(f"  Max steps: {max_steps}")
    print(f"  Batch size: {batch_size}")
    print(f"  Group size (num_generations): {group_size}")
    print(f"  Gradient accumulation: {gradient_accumulation_steps}")
    print(f"  Effective batch: {batch_size * gradient_accumulation_steps}")
    print(f"  Beta (KL coefficient): {beta}")
    print(f"  Learning rate: {lr}")
    print(f"  Max completion length: {max_completion_length}")
    print()

    trainer.train()

    print("Training complete!")
    print(f"Model saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate GRPO with TRL on Countdown task"
    )
    parser.add_argument(
        "--device", default=None, help="Device (default: auto-detect)"
    )
    parser.add_argument(
        "--max-steps", type=int, default=500, help="Max training steps"
    )
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Batch size per device"
    )
    parser.add_argument(
        "--group-size", type=int, default=8, help="Group size (num_generations)"
    )
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=1024,
        help="Max tokens to generate",
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument(
        "--beta", type=float, default=0.04, help="KL penalty coefficient"
    )
    parser.add_argument(
        "--weights-path",
        default="qwen3/qwen3-0.6b.pth",
        help="Path to model weights",
    )
    parser.add_argument(
        "--tokenizer-path",
        default="qwen3/tokenizer.json",
        help="Path to tokenizer",
    )
    parser.add_argument(
        "--n-larges", type=int, default=2, help="Number of large numbers"
    )
    parser.add_argument(
        "--n-total", type=int, default=6, help="Total numbers in problem"
    )
    parser.add_argument(
        "--n-ops", type=int, default=5, help="Number of operations for target"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=random.randint(0, 999),
        help="Random seed",
    )
    parser.add_argument(
        "--use-qwen-thinking",
        action="store_true",
        default=False,
        help="Use Qwen thinking mode (<think> tags)",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=10000,
        help="Number of training examples to generate",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/grpo_countdown_trl",
        help="Output directory",
    )

    args = parser.parse_args()

    train(
        device=args.device,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        group_size=args.group_size,
        max_completion_length=args.max_completion_length,
        lr=args.lr,
        beta=args.beta,
        weights_path=args.weights_path,
        tokenizer_path=args.tokenizer_path,
        n_larges=args.n_larges,
        n_total=args.n_total,
        n_ops=args.n_ops,
        seed=args.seed,
        use_qwen_thinking=args.use_qwen_thinking,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataset_size=args.dataset_size,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
