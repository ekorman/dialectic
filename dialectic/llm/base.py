from typing import Callable

import torch
import torch.nn as nn
from jaxtyping import Float, Int

from dialectic.llm.components import KVCache, RMSNorm


class BaseTransformer(nn.Module):
    def __init__(
        self,
        d: int,
        vocab_size: int,
        n_decoder_layers: int,
        attn_head_d: int,
        attn_num_heads: int,
        attn_num_kv_heads: int,
        mlp_hidden_d: int,
        rms_norm_eps: float,
        rope_base_value: float,
        decoder_layer_factory: Callable,
        tie_weights: bool = False,
    ):
        super().__init__()
        self.d = d
        self.attn_num_heads = attn_num_heads
        self.attn_num_kv_heads = attn_num_kv_heads
        self.attn_head_d = attn_head_d
        self.vocab_size = vocab_size
        self.embed_tokens = nn.Embedding(vocab_size, d)
        self.layers = nn.ModuleList(
            [
                decoder_layer_factory(
                    d=d,
                    attn_head_d=attn_head_d,
                    attn_num_heads=attn_num_heads,
                    attn_num_kv_heads=attn_num_kv_heads,
                    mlp_hidden_d=mlp_hidden_d,
                    rope_base_value=rope_base_value,
                    rms_norm_eps=rms_norm_eps,
                )
                for _ in range(n_decoder_layers)
            ]
        )
        self.norm = RMSNorm(d, rms_norm_eps)
        self.lm_head = nn.Linear(d, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.embed_tokens.weight

        self.use_gradient_checkpointing = False

    def forward(
        self,
        x: Int[torch.Tensor, "B L"]
        | Float[torch.Tensor, "B L V"]
        | Float[torch.Tensor, "B L D"],
        kv_caches: list[KVCache] | None = None,
        attention_mask: torch.Tensor | None = None,
        return_all_logits: bool = False,
        return_hidden_states: bool = False,
    ):
        x = self.embed_tokens(x)  # [B, L, D]

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

        # just get last element of output sequence
        # important: if attention_mask is not None then we assume left padding!
        if not return_all_logits:
            x = x[:, -1:]
        return self.lm_head(x)
