"""
Lasso objective for a single output column:

    h(w) = g(w) + r(w)
    g(w) = ‖Hw − y‖²          (smooth, quadratic)
    r(w) = λ ‖w‖₁             (convex, non-smooth)

One instance per output column avoids Kronecker products and is
mathematically equivalent to the full multi-output formulation.
"""

import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator


class LassoObjective:
    """
    Manages evaluation, gradient, and proximal operator for the Lasso.

    Parameters
    ----------
    H : ndarray of shape (n, m)
        Hidden activation matrix (fixed).
    y : ndarray of shape (n,)
        Target vector for one output column.
    lam : float
        L1 regularisation strength (λ > 0).
    """

    def __init__(self, H: np.ndarray, y: np.ndarray, lam: float):
        self.H = np.asarray(H, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64).ravel()
        self.lam = float(lam)
        self.n, self.m = self.H.shape

        # Pre-compute H^T y for fast gradient evaluation
        self.Hty = self.H.T @ self.y  # (m,)

        # Compute Lipschitz constant L_g = 2 λ_max(H^T H)
        # and strong-convexity parameter μ_g = 2 λ_min(H^T H)
        self._compute_spectral_constants()

    # ------------------------------------------------------------------
    # Spectral constants
    # ------------------------------------------------------------------
    def _compute_spectral_constants(self):
        """Compute L_g, μ_g, and condition number κ from H."""
        if self.m <= 2000:
            # Safe to form H^T H explicitly
            HtH = self.H.T @ self.H
            eig_max = eigsh(HtH, k=1, which='LM', return_eigenvectors=False)[0]
            eig_min = eigsh(HtH, k=1, which='SM', return_eigenvectors=False)[0]
        else:
            # Matrix-free approach for large m
            op = LinearOperator(
                (self.m, self.m),
                matvec=lambda v: self.H.T @ (self.H @ v),
            )
            eig_max = eigsh(op, k=1, which='LM', return_eigenvectors=False)[0]
            eig_min = eigsh(op, k=1, which='SM', return_eigenvectors=False)[0]

        self.L_g = 2.0 * float(eig_max)
        self.mu_g = 2.0 * max(float(eig_min), 0.0)
        self.kappa = self.L_g / self.mu_g if self.mu_g > 0 else float('inf')

    # ------------------------------------------------------------------
    # Objective values
    # ------------------------------------------------------------------
    def f_smooth(self, w: np.ndarray) -> float:
        """Squared residual  g(w) = ‖Hw − y‖²."""
        r = self.H @ w - self.y
        return float(r @ r)

    def f_l1(self, w: np.ndarray) -> float:
        """L1 penalty  r(w) = λ ‖w‖₁."""
        return self.lam * float(np.sum(np.abs(w)))

    def f_total(self, w: np.ndarray) -> float:
        """Full composite objective  h(w) = g(w) + λ ‖w‖₁."""
        return self.f_smooth(w) + self.f_l1(w)

    # ------------------------------------------------------------------
    # Gradients / sub-gradients
    # ------------------------------------------------------------------
    def grad_smooth(self, w: np.ndarray) -> np.ndarray:
        """
        Gradient of the smooth part:  ∇g(w) = 2 H^T (Hw − y).

        Computed via two matrix-vector products (never forms H^T H).
        """
        return 2.0 * (self.H.T @ (self.H @ w) - self.Hty)

    def subgrad_l1(self, w: np.ndarray) -> np.ndarray:
        """
        Subgradient of λ ‖w‖₁.

        At w_i = 0, the minimum-norm subgradient 0 is chosen
        (np.sign already returns 0 for 0 input).
        """
        return self.lam * np.sign(w)

    def grad_composite(self, w: np.ndarray) -> np.ndarray:
        """Sum of smooth gradient and L1 subgradient."""
        return self.grad_smooth(w) + self.subgrad_l1(w)

    # ------------------------------------------------------------------
    # Proximal operator
    # ------------------------------------------------------------------
    def prox_l1(self, z: np.ndarray, step: float) -> np.ndarray:
        """
        Soft-thresholding (proximal operator of step · λ · ‖·‖₁):

            prox(z)_i = sign(z_i) · max(|z_i| − step·λ, 0)
        """
        tau = step * self.lam
        return np.sign(z) * np.maximum(np.abs(z) - tau, 0.0)
