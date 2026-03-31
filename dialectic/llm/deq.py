import torch
import torch.nn as nn
from jaxtyping import Float

from dialectic.llm.base import BaseTransformer


class DEQReasoning(nn.Module):
    """Deep Equilibrium reasoning with a small cross-attention adapter.

    A single reasoning vector z is iteratively refined by a small
    cross-attention + FFN module that attends to the transformer's hidden
    states. All projections are spectral-normalized for guaranteed
    contraction. The transformer runs once; the adapter iterates cheaply.

    Parameters
    ----------
    d_model
        Model hidden dimension.
    n_heads
        Number of cross-attention heads.
    d_ff
        Feed-forward hidden dimension.
    alpha
        Damping factor for fixed-point iteration.
    max_iter
        Maximum refinement iterations.
    tol
        Convergence tolerance.
    neumann_terms
        Number of Neumann series terms for implicit differentiation.
    anderson_m
        Anderson mixing window size. 0 = disabled.
    noise_std
        Standard deviation for random z initialization.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        d_ff: int = 256,
        alpha: float = 0.1,
        max_iter: int = 50,
        tol: float = 1e-3,
        neumann_terms: int = 5,
        anderson_m: int = 5,
        noise_std: float = 1.0,
    ):
        super().__init__()

        self.q_proj = nn.utils.spectral_norm(nn.Linear(d_model, d_model))
        self.k_proj = nn.utils.spectral_norm(nn.Linear(d_model, d_model))
        self.v_proj = nn.utils.spectral_norm(nn.Linear(d_model, d_model))
        self.out_proj = nn.utils.spectral_norm(nn.Linear(d_model, d_model))
        self.ff1 = nn.utils.spectral_norm(nn.Linear(d_model, d_ff))
        self.ff2 = nn.utils.spectral_norm(nn.Linear(d_ff, d_model))
        self.context_norm = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_model = d_model
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.neumann_terms = neumann_terms
        self.anderson_m = anderson_m
        self.noise_std = noise_std
        self.last_n_iter: int = 0
        self.last_diff: float = 0.0

    def _adapter_step(
        self,
        z: Float[torch.Tensor, "B D"],
        context: Float[torch.Tensor, "B L D"],
    ) -> Float[torch.Tensor, "B D"]:
        z_in = z.unsqueeze(1)
        ctx = context

        q = self.q_proj(z_in)
        k = self.k_proj(ctx)
        v = self.v_proj(ctx)

        B, _, D = q.shape
        L = k.shape[1]
        q = q.view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, 1, D)
        h = self.out_proj(attn).squeeze(1)

        h = self.norm1(h)
        h = self.norm2(self.ff2(torch.nn.functional.gelu(self.ff1(h))))
        return h

    def _f(
        self,
        z: Float[torch.Tensor, "B D"],
        context: Float[torch.Tensor, "B L D"],
    ) -> Float[torch.Tensor, "B D"]:
        return (1 - self.alpha) * z + self.alpha * self._adapter_step(z, context)

    def _find_fixed_point(
        self,
        z: Float[torch.Tensor, "B D"],
        context: Float[torch.Tensor, "B L D"],
    ) -> tuple[Float[torch.Tensor, "B D"], int, float]:
        if self.anderson_m > 0:
            return self._anderson_iteration(z, context)
        return self._simple_iteration(z, context)

    def _simple_iteration(
        self,
        z: Float[torch.Tensor, "B D"],
        context: Float[torch.Tensor, "B L D"],
    ) -> tuple[Float[torch.Tensor, "B D"], int, float]:
        diff = float("inf")
        n_iter = self.max_iter
        for i in range(self.max_iter):
            z_new = self._f(z, context)
            diff = (z_new - z).norm(dim=-1).max().item()
            z = z_new
            if diff < self.tol:
                n_iter = i + 1
                break
        return z, n_iter, diff

    def _anderson_iteration(
        self,
        z: Float[torch.Tensor, "B D"],
        context: Float[torch.Tensor, "B L D"],
    ) -> tuple[Float[torch.Tensor, "B D"], int, float]:
        m = self.anderson_m
        B = z.shape[0]

        z_history: list[Float[torch.Tensor, "B D"]] = []
        f_history: list[Float[torch.Tensor, "B D"]] = []

        diff = float("inf")
        n_iter = self.max_iter

        for i in range(self.max_iter):
            f_z = self._f(z, context)

            z_history.append(z)
            f_history.append(f_z)
            if len(z_history) > m:
                z_history.pop(0)
                f_history.pop(0)

            k = len(z_history)
            if k < 2:
                z_new = f_z
            else:
                R = torch.stack([f_history[j] - z_history[j] for j in range(k)], dim=-1)
                RtR = torch.bmm(R.transpose(1, 2), R)
                ones_k = torch.ones(B, k, 1, device=z.device, dtype=z.dtype)

                try:
                    RtR_reg = RtR + 1e-6 * torch.eye(
                        k, device=z.device, dtype=z.dtype
                    ).unsqueeze(0)
                    alphas = torch.linalg.solve(RtR_reg, ones_k)
                    alphas = alphas / alphas.sum(dim=1, keepdim=True)
                    alphas = alphas.squeeze(-1)

                    F_stack = torch.stack(f_history, dim=-1)
                    z_new = torch.bmm(F_stack, alphas.unsqueeze(-1)).squeeze(-1)
                except Exception:
                    z_new = f_z

            diff = (z_new - z).norm(dim=-1).max().item()
            z = z_new
            if diff < self.tol:
                n_iter = i + 1
                break

        return z, n_iter, diff

    def _implicit_backward(
        self,
        z_star: Float[torch.Tensor, "B D"],
        grad_output: Float[torch.Tensor, "B D"],
        context: Float[torch.Tensor, "B L D"],
    ) -> None:
        z_req = z_star.detach().requires_grad_(True)
        with torch.enable_grad():
            f_val = self._f(z_req, context.detach())

        v = grad_output.clone()
        jvp = grad_output.clone()
        for _ in range(self.neumann_terms):
            (jvp,) = torch.autograd.grad(
                f_val, z_req, jvp, retain_graph=True, create_graph=False
            )
            v = v + jvp

        torch.autograd.backward(f_val, v, inputs=list(self.parameters()))

    def forward(
        self,
        net: BaseTransformer,
        context_emb: Float[torch.Tensor, "B L D"],
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[Float[torch.Tensor, "B D"], int, float]:
        """Find fixed point z* and return (z*, n_iterations, final_diff).

        Parameters
        ----------
        net
            Pretrained transformer (used once to get hidden states).
        context_emb
            Context embeddings [B, L, D] (prompt + previous hard tokens).
        attention_mask
            Attention mask for context [B, L].

        Returns
        -------
        tuple[Tensor, int, float]
            (z* [B, D], number of iterations, final convergence diff)
        """
        B = context_emb.shape[0]

        with torch.no_grad():
            with torch.autocast(
                device_type=context_emb.device.type,
                dtype=torch.bfloat16,
                enabled=context_emb.is_cuda,
            ):
                hidden_states = net(
                    context_emb,
                    return_hidden_states=True,
                    attention_mask=attention_mask,
                )
            context = hidden_states.float()

            z = torch.randn(B, self.d_model, device=context.device) * self.noise_std
            z_star, n_iter, final_diff = self._find_fixed_point(z, context)

        self.last_n_iter = n_iter
        self.last_diff = final_diff

        if self.training:
            z_star = z_star.detach().float().requires_grad_(True)
            z_star.register_hook(
                lambda grad: self._implicit_backward(z_star, grad.float(), context)
            )

        return z_star, n_iter, final_diff
