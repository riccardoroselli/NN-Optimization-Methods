"""
A1 — Heavy Ball Method (Polyak Momentum)

Two L1-handling strategies:
  'subgradient': include the L1 subgradient in the momentum update
  'proximal':    gradient step on smooth part only, then soft-thresholding

Two momentum variants:
  'CM':  Classical Momentum  — gradient evaluated at current point w_k
  'NAG': Nesterov Accelerated Gradient — gradient at lookahead w_k + β v_k

References:
  Polyak (1964), Sutskever et al. (2013)
"""

import numpy as np
import time
from .objective import LassoObjective
from .utils import OptimizationLog, check_stopping


def heavy_ball(
    obj: LassoObjective,
    w0: np.ndarray,
    lr: float,
    momentum: float = 0.9,
    max_iter: int = 10000,
    tol: float = 1e-7,
    variant: str = 'NAG',
    l1_handling: str = 'subgradient',
    momentum_schedule=None,
    log_every: int = 1,
    verbose: bool = False,
) -> tuple:
    """
    Heavy Ball optimiser for the Lasso objective.

    Parameters
    ----------
    obj : LassoObjective
    w0 : initial weights, shape (m,)
    lr : learning rate η.  Safe default: 1/L_g.
    momentum : constant β (ignored when momentum_schedule is given)
    max_iter : iteration budget
    tol : relative-change stopping tolerance
    variant : 'CM' or 'NAG'
    l1_handling : 'subgradient' or 'proximal'
    momentum_schedule : callable(t) → β_t, overrides constant momentum
    log_every : record diagnostics every N iterations
    verbose : print progress

    Returns
    -------
    w : final weight vector
    history : OptimizationLog
    """
    w = w0.astype(np.float64, copy=True)
    v = np.zeros_like(w)
    history = OptimizationLog()
    t0 = time.time()

    for t in range(max_iter):
        beta = momentum_schedule(t) if momentum_schedule else momentum

        # ---- choose gradient function and evaluation point ----
        if l1_handling == 'subgradient':
            grad_fn = obj.grad_composite
        elif l1_handling == 'proximal':
            grad_fn = obj.grad_smooth
        else:
            raise ValueError(f"Unknown l1_handling: {l1_handling!r}")

        if variant == 'NAG':
            point = w + beta * v
        elif variant == 'CM':
            point = w
        else:
            raise ValueError(f"Unknown variant: {variant!r}")

        g = grad_fn(point)

        # ---- velocity + weight update ----
        v_new = beta * v - lr * g

        if l1_handling == 'subgradient':
            w_new = w + v_new
        else:  # proximal
            w_new = obj.prox_l1(w + v_new, lr)

        # ---- logging ----
        if t % log_every == 0 or t == max_iter - 1:
            elapsed = time.time() - t0
            f_tot = obj.f_total(w_new)
            f_sm = obj.f_smooth(w_new)
            gn = float(np.linalg.norm(g))
            history.record(t, f_tot, f_sm, gn, elapsed, w_new, w)

            if verbose and t % (10 * log_every) == 0:
                sp = np.mean(np.abs(w_new) < 1e-8)
                print(f"  A1 iter {t:6d} | f={f_tot:.6e} | "
                      f"‖g‖={gn:.6e} | sparsity={sp:.2%}")

        # ---- stopping ----
        if check_stopping(w_new, w, tol):
            if t % log_every != 0:
                elapsed = time.time() - t0
                history.record(t, obj.f_total(w_new), obj.f_smooth(w_new),
                               float(np.linalg.norm(g)), elapsed, w_new, w)
            if verbose:
                print(f"  A1 converged at iteration {t}")
            break

        w = w_new
        v = v_new

    return w_new, history


def heavy_ball_two_phase(
    obj: LassoObjective,
    w0: np.ndarray,
    lr: float,
    max_iter_phase1: int = 2000,
    max_iter_phase2: int = 8000,
    momentum_phase1: float = 0.9,
    momentum_phase2: float = 0.0,
    tol: float = 1e-7,
    variant: str = 'NAG',
    log_every: int = 1,
    verbose: bool = False,
) -> tuple:
    """
    Two-phase proximal Heavy Ball: high momentum for fast approach,
    then low/zero momentum for accurate refinement.

    Phase 1 uses high momentum to reach the vicinity of the solution
    quickly (proximal-momentum plateau). Phase 2 restarts with low
    momentum (resetting velocity to zero), allowing the proximal
    operator to settle into a sparser, more accurate solution.

    Always uses l1_handling='proximal'.

    Returns
    -------
    w : final weight vector
    history : OptimizationLog  (combined log from both phases)
    """
    # --- Phase 1: fast approach with high momentum ---
    if verbose:
        print(f"  Phase 1: momentum={momentum_phase1}, max_iter={max_iter_phase1}")
    w1, hist1 = heavy_ball(
        obj, w0, lr=lr, momentum=momentum_phase1,
        max_iter=max_iter_phase1, tol=tol,
        variant=variant, l1_handling='proximal',
        log_every=log_every, verbose=verbose,
    )

    # --- Phase 2: refinement with low momentum (velocity reset) ---
    if verbose:
        print(f"  Phase 2: momentum={momentum_phase2}, max_iter={max_iter_phase2}")
    w2, hist2 = heavy_ball(
        obj, w1, lr=lr, momentum=momentum_phase2,
        max_iter=max_iter_phase2, tol=tol,
        variant=variant, l1_handling='proximal',
        log_every=log_every, verbose=verbose,
    )

    # Merge histories: offset phase 2 iterations and times
    combined = OptimizationLog()
    combined.iterations = list(hist1.iterations)
    combined.f_total = list(hist1.f_total)
    combined.f_smooth = list(hist1.f_smooth)
    combined.grad_norm = list(hist1.grad_norm)
    combined.time_elapsed = list(hist1.time_elapsed)
    combined.sparsity = list(hist1.sparsity)
    combined.w_change = list(hist1.w_change)

    iter_offset = hist1.iterations[-1] + 1 if hist1.iterations else 0
    time_offset = hist1.time_elapsed[-1] if hist1.time_elapsed else 0.0

    for i in range(len(hist2.iterations)):
        combined.iterations.append(hist2.iterations[i] + iter_offset)
        combined.f_total.append(hist2.f_total[i])
        combined.f_smooth.append(hist2.f_smooth[i])
        combined.grad_norm.append(hist2.grad_norm[i])
        combined.time_elapsed.append(hist2.time_elapsed[i] + time_offset)
        combined.sparsity.append(hist2.sparsity[i])
        combined.w_change.append(hist2.w_change[i])

    return w2, combined
