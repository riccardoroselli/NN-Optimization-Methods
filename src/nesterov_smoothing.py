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


def nesterov_smoothed_decreasing(
    obj: LassoObjective,
    w0: np.ndarray,
    mu0: float = 0.1,
    max_iter: int = 10000,
    tol: float = 1e-7,
    log_every: int = 1,
    verbose: bool = False,
) -> tuple:
    """
    FISTA-accelerated gradient descent with decreasing smoothing parameter.

    Instead of a fixed Huber smoothing μ, uses the schedule μ_k = μ₀/(k+1)
    (Nesterov, 2005). This shrinks the approximation error over time, so
    the method converges on the TRUE (non-smooth) objective at rate O(1/k),
    better than subgradient methods' O(1/sqrt(k)).

    Parameters
    ----------
    obj : LassoObjective
    w0 : initial weights, shape (m,)
    mu0 : initial smoothing parameter μ₀ > 0
    max_iter : iteration budget
    tol : relative-change stopping tolerance
    log_every : record diagnostics every N iterations
    verbose : print progress

    Returns
    -------
    w : final weight vector
    history : OptimizationLog
    """
    lam = obj.lam
    m = obj.m
    mu_floor = 1e-8

    w = w0.astype(np.float64, copy=True)
    w_prev = w.copy()
    tk = 1.0

    history = OptimizationLog()
    t0 = time.time()

    for k in range(max_iter):
        # ---- decreasing smoothing parameter ----
        mu_k = max(mu0 / (k + 1), mu_floor)

        # ---- adaptive step size ----
        L_total = obj.L_g + lam / mu_k
        step = 1.0 / L_total

        # ---- FISTA momentum ----
        tk_new = (1.0 + np.sqrt(1.0 + 4.0 * tk * tk)) / 2.0
        tau = (tk - 1.0) / tk_new

        # ---- extrapolation ----
        x = w + tau * (w - w_prev)

        # ---- gradient of smoothed objective at x ----
        # Huber gradient inline: λ * (x_i/μ if |x_i|≤μ, else sign(x_i))
        abs_x = np.abs(x)
        huber_grad = lam * np.where(abs_x <= mu_k, x / mu_k, np.sign(x))
        grad = obj.grad_smooth(x) + huber_grad

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
                print(f"  A2-dec iter {k:6d} | f_true={f_tot:.6e} | "
                      f"mu_k={mu_k:.2e} | step={step:.2e} | ‖g‖={gn:.6e}")

        # ---- stopping ----
        if check_stopping(w_new, w, tol):
            if k % log_every != 0:
                elapsed = time.time() - t0
                history.record(k, obj.f_total(w_new), obj.f_smooth(w_new),
                               float(np.linalg.norm(grad)), elapsed, w_new, w)
            if verbose:
                print(f"  A2-dec converged at iteration {k}")
            break

        w_prev = w.copy()
        w = w_new
        tk = tk_new

    return w_new, history
