"""
Smoke test: compare fixed-μ and decreasing-μ Nesterov smoothing on a small problem.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.elm import ELM
from src.objective import LassoObjective
from src.smoothing import HuberSmoothing
from src.nesterov_smoothing import nesterov_smoothed, nesterov_smoothed_decreasing
from src.reference import solve_lasso_reference


def main():
    rng = np.random.default_rng(42)
    n, d, m = 100, 10, 20
    lam = 0.1

    elm = ELM(d=d, m=m, seed=42)
    X = rng.standard_normal((n, d))
    H = elm.compute_H(X)

    w_true = rng.standard_normal(m)
    w_true[rng.random(m) < 0.7] = 0.0
    y = H @ w_true + 0.05 * rng.standard_normal(n)

    obj = LassoObjective(H, y, lam=lam)
    w0 = np.zeros(m)
    _, f_ref = solve_lasso_reference(H, y, lam)

    print(f"Problem: n={n}, m={m}, lam={lam}")
    print(f"  L_g={obj.L_g:.4e}, kappa={obj.kappa:.1f}, f*={f_ref:.6e}\n")

    max_iter = 5000

    # Fixed-mu variants
    for mu in [1e-1, 1e-2, 1e-3]:
        smoother = HuberSmoothing(lam=lam, mu=mu, dim=m)
        w_fix, hist_fix = nesterov_smoothed(obj, smoother, w0, max_iter=max_iter, tol=1e-9)
        f_fix = obj.f_total(w_fix)
        gap_fix = f_fix - f_ref
        print(f"  Fixed   mu={mu:.0e}: f={f_fix:.6e}, gap={gap_fix:.2e}, "
              f"iters={hist_fix.iterations[-1]}")

    # Decreasing-mu
    for mu0 in [1e-1, 1.0]:
        w_dec, hist_dec = nesterov_smoothed_decreasing(
            obj, w0, mu0=mu0, max_iter=max_iter, tol=1e-9,
        )
        f_dec = obj.f_total(w_dec)
        gap_dec = f_dec - f_ref
        print(f"  Decreas mu0={mu0:.0e}: f={f_dec:.6e}, gap={gap_dec:.2e}, "
              f"iters={hist_dec.iterations[-1]}")

    print("\nSmoke test passed — both variants run without error.")


if __name__ == '__main__':
    main()
