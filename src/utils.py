"""
Shared utilities: optimisation logging, momentum schedules, stopping criteria.
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Optional, Callable


# ======================================================================
# Optimisation logger
# ======================================================================
@dataclass
class OptimizationLog:
    """Accumulates per-iteration diagnostics for later analysis."""

    iterations: List[int] = field(default_factory=list)
    f_total: List[float] = field(default_factory=list)
    f_smooth: List[float] = field(default_factory=list)
    grad_norm: List[float] = field(default_factory=list)
    time_elapsed: List[float] = field(default_factory=list)
    sparsity: List[float] = field(default_factory=list)
    w_change: List[float] = field(default_factory=list)

    def record(self, k: int, f_tot: float, f_sm: float, gn: float,
               t: float, w: np.ndarray, w_prev: Optional[np.ndarray] = None):
        self.iterations.append(k)
        self.f_total.append(f_tot)
        self.f_smooth.append(f_sm)
        self.grad_norm.append(gn)
        self.time_elapsed.append(t)
        self.sparsity.append(float(np.mean(np.abs(w) < 1e-8)))
        if w_prev is not None:
            norm_diff = np.linalg.norm(w - w_prev)
            if np.isfinite(norm_diff):
                denom = max(1.0, np.linalg.norm(w_prev))
                self.w_change.append(float(norm_diff / denom))
            else:
                self.w_change.append(float('inf'))
        else:
            self.w_change.append(float('inf'))

    def to_dict(self) -> dict:
        """Return a plain dict (easy to pass to pandas.DataFrame)."""
        return {
            'iteration': self.iterations,
            'f_total': self.f_total,
            'f_smooth': self.f_smooth,
            'grad_norm': self.grad_norm,
            'time': self.time_elapsed,
            'sparsity': self.sparsity,
            'w_change': self.w_change,
        }


# ======================================================================
# Momentum schedules
# ======================================================================
def sutskever_momentum_schedule(t: int, mu_max: float = 0.99) -> float:
    """
    Sutskever et al. (2013), Equation 5.

    μ_t = min(1 − 2^{−1 − log₂(⌊t/250⌋ + 1)},  μ_max)

    Starts at 0.5 and doubles the "closeness to 1" every 250 iterations.
    """
    exponent = -1.0 - np.log2(t // 250 + 1)
    return min(1.0 - 2.0 ** exponent, mu_max)


def annealing_momentum_schedule(
    max_iter: int,
    mu_high: float = 0.95,
    mu_low: float = 0.5,
    anneal_fraction: float = 0.2,
) -> Callable[[int], float]:
    """
    Momentum schedule that uses high β early and anneals to low β
    in the final fraction of iterations.

    Parameters
    ----------
    max_iter : int
        Total iteration budget.
    mu_high : float
        Momentum during the main phase.
    mu_low : float
        Momentum at the end of the annealing phase.
    anneal_fraction : float
        Fraction of max_iter over which to linearly anneal from
        mu_high to mu_low (e.g. 0.2 = last 20% of iterations).
    """
    anneal_start = int(max_iter * (1.0 - anneal_fraction))

    def _schedule(t: int) -> float:
        if t < anneal_start:
            return mu_high
        progress = (t - anneal_start) / max(1, max_iter - anneal_start)
        return mu_high + (mu_low - mu_high) * min(progress, 1.0)

    return _schedule


# ======================================================================
# Stopping criterion
# ======================================================================
def check_stopping(w_new: np.ndarray, w_old: np.ndarray, tol: float = 1e-6) -> bool:
    """True when the relative change ‖Δw‖ / max(1, ‖w_old‖) < tol."""
    norm_diff = np.linalg.norm(w_new - w_old)
    if not np.isfinite(norm_diff):
        return False  # diverged — don't stop
    return float(norm_diff / max(1.0, np.linalg.norm(w_old))) < tol


# ======================================================================
# Helpers
# ======================================================================
def polyak_optimal_params(L_g: float, mu_g: float):
    """
    Optimal Heavy Ball parameters from Polyak (1964).

    For a smooth strongly convex quadratic with Lipschitz constant L_g
    and strong convexity parameter mu_g:

        η* = 4 / (√L_g + √μ_g)²
        β* = ((√κ − 1) / (√κ + 1))²

    where κ = L_g / μ_g.

    Returns (None, None) when mu_g ≤ 0 (problem is not strongly convex).
    """
    if mu_g <= 0:
        return None, None
    sqrt_L = np.sqrt(L_g)
    sqrt_mu = np.sqrt(mu_g)
    eta_star = 4.0 / (sqrt_L + sqrt_mu) ** 2
    sqrt_kappa = sqrt_L / sqrt_mu
    beta_star = ((sqrt_kappa - 1.0) / (sqrt_kappa + 1.0)) ** 2
    return eta_star, beta_star
