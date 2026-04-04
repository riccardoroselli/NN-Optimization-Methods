"""
A2 — Nesterov Smoothed Gradient Method (FISTA acceleration)

Replace the non-smooth L1 penalty with its Huber smooth approximation,
then apply FISTA-style accelerated gradient descent on the fully smooth
surrogate objective:

    h_μ(w) = ‖Hw − y‖² + λ Σ_i ψ_μ(w_i)

Step size is automatic: 1 / L_total  where  L_total = L_g + λ/μ.

References:
  Nesterov (2005)  — smooth minimisation of non-smooth functions
  Beck & Teboulle (2009) — FISTA
"""

import numpy as np
import time
from .objective import LassoObjective
from .smoothing import HuberSmoothing
from .utils import OptimizationLog, check_stopping


def nesterov_smoothed(
    obj: LassoObjective,
    smoother: HuberSmoothing,
    w0: np.ndarray,
    max_iter: int = 10000,
    tol: float = 1e-7,
    log_every: int = 1,
    verbose: bool = False,
) -> tuple:
    """
    FISTA-accelerated gradient descent on the Huber-smoothed objective.

    Parameters
    ----------
    obj : LassoObjective
    smoother : HuberSmoothing  (provides smoothed L1 value/gradient)
    w0 : initial weights, shape (m,)
    max_iter : iteration budget
    tol : relative-change stopping tolerance
    log_every : record diagnostics every N iterations
    verbose : print progress

    Returns
    -------
    w : final weight vector
    history : OptimizationLog
    """
    L_total = obj.L_g + smoother.L_mu
    step = 1.0 / L_total

    w = w0.astype(np.float64, copy=True)
    w_prev = w.copy()
    tk = 1.0

    history = OptimizationLog()
    t0 = time.time()

    for k in range(max_iter):
        # ---- FISTA momentum ----
        tk_new = (1.0 + np.sqrt(1.0 + 4.0 * tk * tk)) / 2.0
        tau = (tk - 1.0) / tk_new

        # ---- extrapolation ----
        x = w + tau * (w - w_prev)

        # ---- full gradient of the smoothed objective at x ----
        grad = obj.grad_smooth(x) + smoother.grad(x)

        # ---- gradient step ----
        w_new = x - step * grad

        # ---- logging (always log the TRUE objective, not smoothed) ----
        if k % log_every == 0 or k == max_iter - 1:
            elapsed = time.time() - t0
            f_tot = obj.f_total(w_new)
            f_sm = obj.f_smooth(w_new)
            gn = float(np.linalg.norm(grad))
            history.record(k, f_tot, f_sm, gn, elapsed, w_new, w)

            if verbose and k % (10 * log_every) == 0:
                f_smoothed = f_sm + smoother.value(w_new)
                print(f"  A2 iter {k:6d} | f_true={f_tot:.6e} | "
                      f"f_smoothed={f_smoothed:.6e} | ‖g‖={gn:.6e}")

        # ---- stopping ----
        if check_stopping(w_new, w, tol):
            if k % log_every != 0:
                elapsed = time.time() - t0
                history.record(k, obj.f_total(w_new), obj.f_smooth(w_new),
                               float(np.linalg.norm(grad)), elapsed, w_new, w)
            if verbose:
                print(f"  A2 converged at iteration {k}")
            break

        w_prev = w.copy()
        w = w_new
        tk = tk_new

    return w_new, history
