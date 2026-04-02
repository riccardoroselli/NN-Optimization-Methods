"""
Huber smooth approximation of the L1 penalty  r(w) = λ ‖w‖₁.

The component-wise Huber function ψ_μ is:

    ψ_μ(t) = t² / (2μ)        if |t| ≤ μ
    ψ_μ(t) = |t| − μ/2        if |t| > μ

so that  r_μ(w) = λ Σ_i ψ_μ(w_i)  is a smooth lower bound of r(w).

Properties
----------
- Lipschitz constant of ∇r_μ :  L_{r,μ} = λ / μ
- Approximation error bound  :  0 ≤ r(w) − r_μ(w) ≤ λ μ dim(w) / 2
"""

import numpy as np


class HuberSmoothing:
    """
    Parameters
    ----------
    lam : float
        L1 regularisation parameter λ.
    mu : float
        Smoothing parameter μ > 0  (smaller ⇒ closer to true L1).
    dim : int
        Dimension of w (needed for the approximation-error bound).
    """

    def __init__(self, lam: float, mu: float, dim: int):
        if mu <= 0:
            raise ValueError(f"Smoothing parameter μ must be positive, got {mu}")
        self.lam = float(lam)
        self.mu = float(mu)
        self.dim = int(dim)

        self.L_mu = self.lam / self.mu                      # Lipschitz of ∇r_μ
        self.approx_error = self.lam * self.mu * self.dim / 2.0  # worst-case gap

    def value(self, w: np.ndarray) -> float:
        """Evaluate the smoothed penalty  r_μ(w) = λ Σ_i ψ_μ(w_i)."""
        a = np.abs(w)
        huber = np.where(
            a <= self.mu,
            w * w / (2.0 * self.mu),   # quadratic region
            a - self.mu / 2.0,         # linear region
        )
        return self.lam * float(np.sum(huber))

    def grad(self, w: np.ndarray) -> np.ndarray:
        """Gradient  ∇r_μ(w) = λ · ψ'_μ(w)  element-wise."""
        return self.lam * np.where(
            np.abs(w) <= self.mu,
            w / self.mu,        # slope 1/μ near zero
            np.sign(w),         # ±1 away from zero
        )
