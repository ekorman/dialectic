import torch
import torch.nn as nn
from jaxtyping import Float

from dialectic.llm.base import BaseTransformer


class DEQReasoning(nn.Module):
    """Deep Equilibrium reasoning module.

    Iteratively refines a single reasoning vector z by applying the
    pretrained transformer, then uses implicit differentiation for
    constant-memory backpropagation.

    Parameters
    ----------
    d_model
        Model hidden dimension.
    alpha
        Damping factor for fixed-point iteration. Smaller = more stable.
    max_iter
        Maximum refinement iterations.
    tol
        Convergence tolerance.
    neumann_terms
        Number of Neumann series terms for implicit differentiation.
    """

    def __init__(
        self,
        d_model: int,
        alpha: float = 0.1,
        max_iter: int = 20,
        tol: float = 1e-3,
        neumann_terms: int = 5,
    ):
        super().__init__()
        self.z0 = nn.Parameter(torch.randn(d_model) * 0.02)
        self.w_proj = nn.utils.spectral_norm(nn.Linear(d_model, d_model))
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.neumann_terms = neumann_terms
        self.last_n_iter: int = 0
        self.last_diff: float = 0.0

    def _f(
        self,
        z: Float[torch.Tensor, "B D"],
        net: BaseTransformer,
        context_emb: Float[torch.Tensor, "B L D"],
        attention_mask: torch.Tensor | None,
    ) -> Float[torch.Tensor, "B D"]:
        z_inp = z.unsqueeze(1)
        inp = torch.cat([context_emb.to(z_inp.dtype), z_inp], dim=1)
        if attention_mask is not None:
            ones = torch.ones(z.shape[0], 1, dtype=torch.bool, device=z.device)
            mask = torch.cat([attention_mask, ones], dim=1)
        else:
            mask = None
        with torch.autocast(
            device_type=z.device.type, dtype=torch.bfloat16, enabled=z.is_cuda
        ):
            h = net(inp, return_hidden_states=True, attention_mask=mask)
        h_last = h[:, -1]
        return (1 - self.alpha) * z + self.alpha * self.w_proj(h_last.to(z.dtype))

    def _find_fixed_point(
        self,
        net: BaseTransformer,
        context_emb: Float[torch.Tensor, "B L D"],
        attention_mask: torch.Tensor | None,
    ) -> tuple[Float[torch.Tensor, "B D"], int, float]:
        B = context_emb.shape[0]
        z = self.z0.unsqueeze(0).expand(B, -1).clone()

        diff = float("inf")
        n_iter = self.max_iter
        for i in range(self.max_iter):
            z_new = self._f(z, net, context_emb, attention_mask)
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
        net: BaseTransformer,
        context_emb: Float[torch.Tensor, "B L D"],
        attention_mask: torch.Tensor | None,
    ) -> None:
        """Compute implicit gradients and accumulate onto parameters."""
        z_req = z_star.detach().requires_grad_(True)
        with torch.enable_grad():
            f_val = self._f(z_req, net, context_emb.detach(), attention_mask)

        # Solve (I - J^T) v = grad_output via Neumann series
        # v = grad + J^T @ grad + (J^T)^2 @ grad + ...
        v = grad_output.clone()
        jvp = grad_output.clone()
        for _ in range(self.neumann_terms):
            (jvp,) = torch.autograd.grad(
                f_val, z_req, jvp, retain_graph=True, create_graph=False
            )
            v = v + jvp

        # Accumulate gradients onto trainable parameters
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
            Frozen pretrained transformer.
        context_emb
            Context embeddings [B, L, D] (prompt + previous hard tokens).
        attention_mask
            Attention mask for context [B, L].

        Returns
        -------
        tuple[Tensor, int, float]
            (z* [B, D], number of iterations, final convergence diff)
        """
        with torch.no_grad():
            z_star, n_iter, final_diff = self._find_fixed_point(
                net, context_emb, attention_mask
            )
        self.last_n_iter = n_iter
        self.last_diff = final_diff

        if self.training:
            z_star = z_star.detach().requires_grad_(True)
            z_star.register_hook(
                lambda grad: self._implicit_backward(
                    z_star, grad, net, context_emb, attention_mask
                )
            )

        return z_star, n_iter, final_diff
