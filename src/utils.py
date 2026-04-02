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
            denom = max(1.0, np.linalg.norm(w_prev))
            self.w_change.append(float(np.linalg.norm(w - w_prev) / denom))
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


def constant_momentum(mu: float = 0.9) -> Callable[[int], float]:
    """Return a schedule that always yields the same momentum β."""
    def _schedule(t: int) -> float:
        return mu
    return _schedule


# ======================================================================
# Stopping criterion
# ======================================================================
def check_stopping(w_new: np.ndarray, w_old: np.ndarray, tol: float = 1e-6) -> bool:
    """True when the relative change ‖Δw‖ / max(1, ‖w_old‖) < tol."""
    return float(np.linalg.norm(w_new - w_old) / max(1.0, np.linalg.norm(w_old))) < tol


# ======================================================================
# Helpers
# ======================================================================
def compute_optimality_gap(f_current: float, f_reference: float) -> float:
    """f(w_k) − f*, floored at 1e-16 for safe log-scale plotting."""
    return max(f_current - f_reference, 1e-16)
