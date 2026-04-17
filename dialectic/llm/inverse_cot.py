import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from torch import Tensor

from dialectic.llm.base import BaseTransformer
from dialectic.llm.components.attention import MHSA
from dialectic.llm.components.kv_cache import KVCache
from dialectic.llm.components.rms_norm import RMSNorm


class ReversedDecoderLayer(nn.Module):
    """MLP -> Attention decoder layer (reversed from standard Attention -> MLP).

    The MLP and its input norm are shared from a pretrained model and frozen.
    The attention and its input norm are fresh and trainable.
    """

    def __init__(
        self,
        *,
        shared_mlp: nn.Module,
        shared_pre_mlp_norm: RMSNorm,
        self_attn: MHSA,
        post_mlp_norm: RMSNorm,
    ):
        super().__init__()
        self.pre_mlp_norm = shared_pre_mlp_norm
        self.mlp = shared_mlp
        self.post_mlp_norm = post_mlp_norm
        self.self_attn = self_attn

    def forward(
        self,
        x: Float[Tensor, "B L D"],
        kv_cache: KVCache | None = None,
        attention_mask: Bool[Tensor, "B 1 L L"] | Bool[Tensor, "B L"] | None = None,
    ) -> Float[Tensor, "B L D"]:
        x = x + self.mlp(self.pre_mlp_norm(x))
        x = x + self.self_attn(
            self.post_mlp_norm(x), kv_cache=kv_cache, attention_mask=attention_mask
        )
        return x


class InverseCotModel(nn.Module):
    """Model that reconstructs CoT reasoning given (prompt, answer).

    Shares token embeddings and decoder MLPs with a frozen model `p`.
    Has fresh attention weights and LM head.
    Decoder layers are reversed: MLP before attention.
    """

    def __init__(self, p: BaseTransformer, unfreeze_mlp: bool = False):
        super().__init__()
        self.d = p.d
        self.vocab_size = p.vocab_size
        self.attn_head_d = p.attn_head_d
        self.attn_num_kv_heads = p.attn_num_kv_heads
        self.attn_num_heads = p.attn_num_heads

        self.embed_tokens = p.embed_tokens
        self.embed_tokens.requires_grad_(False)

        rms_norm_eps = p.norm.eps

        self.layers = nn.ModuleList()
        for p_layer in p.layers:
            use_rope = p_layer.self_attn.use_rope
            apply_rms = p_layer.self_attn.apply_rms_norm

            fresh_attn = MHSA(
                d=p.d,
                head_d=p.attn_head_d,
                num_heads=p.attn_num_heads,
                num_kv_heads=p.attn_num_kv_heads,
                causal=False,
                apply_rms_norm=apply_rms,
                rms_norm_eps=rms_norm_eps if apply_rms else None,
                rope_base_value=1.0 if use_rope else None,
                max_position_embeddings=(
                    p_layer.self_attn.rope_cos.shape[1] if use_rope else 8192
                ),
            )
            if use_rope:
                fresh_attn.rope_sin = p_layer.self_attn.rope_sin
                fresh_attn.rope_cos = p_layer.self_attn.rope_cos

            if unfreeze_mlp:
                import copy

                layer_mlp = copy.deepcopy(p_layer.mlp)
                layer_mlp.requires_grad_(True)
                layer_pre_mlp_norm = copy.deepcopy(p_layer.post_attention_layernorm)
                layer_pre_mlp_norm.requires_grad_(True)
            else:
                layer_mlp = p_layer.mlp
                layer_pre_mlp_norm = p_layer.post_attention_layernorm
                layer_mlp.requires_grad_(False)
                layer_pre_mlp_norm.requires_grad_(False)

            self.layers.append(
                ReversedDecoderLayer(
                    shared_mlp=layer_mlp,
                    shared_pre_mlp_norm=layer_pre_mlp_norm,
                    self_attn=fresh_attn,
                    post_mlp_norm=RMSNorm(p.d, eps=rms_norm_eps),
                )
            )

        self.norm = RMSNorm(p.d, eps=p.norm.eps)
        self.lm_head = nn.Linear(p.d, p.vocab_size, bias=False)
        self.lm_head.weight.data.copy_(p.lm_head.weight.data)

        # When True, each decoder layer's forward is run under
        # `torch.utils.checkpoint.checkpoint` so its activations are dropped
        # after the layer returns and recomputed during backward. Trades
        # ~30% step time for ~5-10× less activation memory — load-bearing
        # for the InfoNCE contrastive loss, where the per-step batch is
        # (1 + n_negatives) times the NLL-only case and the prefix-LM
        # attention mask forces SDPA's math backend (which materializes
        # the full [N, H, L, L] scores matrix). Enable this via
        # `InverseCotParams.gradient_checkpointing` in training.
        self.use_gradient_checkpointing = False

    def forward(
        self,
        x: Int[Tensor, "B L"],
        kv_caches: list[KVCache] | None = None,
        attention_mask: Bool[Tensor, "B 1 L L"] | Bool[Tensor, "B L"] | None = None,
        return_all_logits: bool = False,
        return_hidden_states: bool = False,
    ):
        x = self.embed_tokens(x)

        use_ckpt = (
            self.use_gradient_checkpointing
            and self.training
            and torch.is_grad_enabled()
        )
        for layer, kv_cache in zip(self.layers, kv_caches or [None] * len(self.layers)):
            if use_ckpt:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, kv_cache, attention_mask, use_reentrant=False
                )
            else:
                x = layer(x, kv_cache=kv_cache, attention_mask=attention_mask)

        x = self.norm(x)

        if return_hidden_states:
            return x

        if not return_all_logits:
            x = x[:, -1:]
        return self.lm_head(x)


def create_prefix_lm_mask(
    prefix_lengths: Int[Tensor, " B"],
    seq_len: int,
    device: torch.device,
) -> Bool[Tensor, "B 1 L L"]:
    """Create a prefix-LM attention mask.

    Within the prefix: bidirectional (all-to-all).
    CoT tokens: attend to all prefix tokens + causally to prior CoT tokens.
    Prefix tokens: cannot see CoT tokens.
    """
    positions = torch.arange(seq_len, device=device)

    prefix_mask = positions.unsqueeze(0) < prefix_lengths.unsqueeze(1)

    is_prefix_query = prefix_mask.unsqueeze(2)
    is_prefix_key = prefix_mask.unsqueeze(1)

    # prefix-to-prefix: True (bidirectional)
    pp = is_prefix_query & is_prefix_key
    # cot-to-prefix: True
    cp = ~is_prefix_query & is_prefix_key
    # cot-to-cot: causal (j <= i)
    causal = positions.unsqueeze(0) <= positions.unsqueeze(1)
    cc = ~is_prefix_query & ~is_prefix_key & causal.unsqueeze(0)

    mask = pp | cp | cc
    return mask.unsqueeze(1)
