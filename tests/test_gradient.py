"""
Gradient verification via central finite differences.

Tests:
  1. ∇g(w)   — smooth objective gradient
  2. ∇r_μ(w) — Huber-smoothed L1 gradient
  3. ∇h_μ(w) — combined smoothed objective gradient (used by A2)

Run BEFORE any optimisation to catch analytical-gradient bugs early.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.elm import ELM
from src.objective import LassoObjective
from src.smoothing import HuberSmoothing


def finite_difference_grad(f, w, eps=1e-5):
    """Central finite-difference gradient: O(eps²) error."""
    g = np.zeros_like(w)
    for i in range(len(w)):
        e = np.zeros_like(w)
        e[i] = eps
        g[i] = (f(w + e) - f(w - e)) / (2.0 * eps)
    return g


def _rel_error(a, b):
    denom = max(np.linalg.norm(a), np.linalg.norm(b), 1e-15)
    return np.linalg.norm(a - b) / denom


# ================================================================
# Test 1: smooth objective gradient  ∇g(w) = 2 H^T (Hw − y)
# ================================================================
def test_grad_smooth():
    print("=" * 60)
    print("TEST 1: ∇g(w) — smooth objective gradient")

    rng = np.random.default_rng(0)
    n, m = 50, 10
    H = rng.standard_normal((n, m))
    y = rng.standard_normal(n)
    obj = LassoObjective(H, y, lam=0.1)

    # Test at several random points (including near zero)
    for trial, scale in enumerate([1.0, 0.01, 10.0]):
        w = rng.standard_normal(m) * scale
        g_an = obj.grad_smooth(w)
        g_fd = finite_difference_grad(obj.f_smooth, w)
        err = _rel_error(g_an, g_fd)
        status = "OK" if err < 1e-5 else "FAIL"
        print(f"  trial {trial} (scale={scale:5.2f}): rel_err = {err:.2e}  [{status}]")
        assert err < 1e-5, f"Smooth gradient FAILED at trial {trial}: {err}"

    print("  PASSED")


# ================================================================
# Test 2: Huber gradient  ∇r_μ(w)
# ================================================================
def test_grad_huber():
    print("=" * 60)
    print("TEST 2: ∇r_μ(w) — Huber-smoothed L1 gradient")

    rng = np.random.default_rng(1)
    m = 20
    lam = 0.1

    for mu in [1e-1, 1e-2, 1e-3]:
        smoother = HuberSmoothing(lam=lam, mu=mu, dim=m)
        # Place test points well away from the kink |w_i| = mu
        # so that finite-difference stencil doesn't straddle the kink
        w = rng.standard_normal(m)
        w = np.where(np.abs(np.abs(w) - mu) < 3 * mu, w + 5 * mu * np.sign(w), w)
        eps_fd = min(mu / 100.0, 1e-6)

        g_an = smoother.grad(w)
        g_fd = finite_difference_grad(smoother.value, w, eps=eps_fd)
        err = _rel_error(g_an, g_fd)
        status = "OK" if err < 1e-4 else "FAIL"
        print(f"  mu={mu:.0e}: rel_err = {err:.2e}  [{status}]")
        assert err < 1e-4, f"Huber gradient FAILED for mu={mu}: {err}"

    print("  PASSED")


# ================================================================
# Test 3: combined smoothed gradient  ∇g(w) + ∇r_μ(w)
# ================================================================
def test_grad_combined():
    print("=" * 60)
    print("TEST 3: ∇h_μ(w) — full smoothed objective gradient (used by A2)")

    rng = np.random.default_rng(2)
    n, m = 60, 15
    H = rng.standard_normal((n, m))
    y = rng.standard_normal(n)
    lam = 0.05
    mu = 1e-2

    obj = LassoObjective(H, y, lam=lam)
    smoother = HuberSmoothing(lam=lam, mu=mu, dim=m)

    def h_smoothed(w):
        return obj.f_smooth(w) + smoother.value(w)

    w = rng.standard_normal(m) * 0.5
    g_an = obj.grad_smooth(w) + smoother.grad(w)
    g_fd = finite_difference_grad(h_smoothed, w, eps=1e-6)
    err = _rel_error(g_an, g_fd)
    print(f"  rel_err = {err:.2e}")
    assert err < 1e-5, f"Combined gradient FAILED: {err}"
    print("  PASSED")


# ================================================================
# Test 4: gradient at the ELM scale (n=200, m=50)
# ================================================================
def test_grad_elm_scale():
    print("=" * 60)
    print("TEST 4: gradients at ELM-scale problem (n=200, m=50)")

    rng = np.random.default_rng(42)
    n, d, m = 200, 20, 50
    lam = 0.01

    elm = ELM(d=d, m=m, seed=42)
    X = rng.standard_normal((n, d))
    H = elm.compute_H(X)
    w_true = rng.standard_normal(m)
    w_true[rng.random(m) < 0.7] = 0.0
    y = H @ w_true + 0.1 * rng.standard_normal(n)

    obj = LassoObjective(H, y, lam=lam)
    smoother = HuberSmoothing(lam=lam, mu=1e-3, dim=m)

    w = rng.standard_normal(m) * 0.1

    # smooth gradient
    g_an = obj.grad_smooth(w)
    g_fd = finite_difference_grad(obj.f_smooth, w, eps=1e-5)
    err_smooth = _rel_error(g_an, g_fd)

    # huber gradient
    gh_an = smoother.grad(w)
    gh_fd = finite_difference_grad(smoother.value, w, eps=1e-7)
    err_huber = _rel_error(gh_an, gh_fd)

    print(f"  smooth gradient rel_err = {err_smooth:.2e}")
    print(f"  Huber  gradient rel_err = {err_huber:.2e}")
    assert err_smooth < 1e-5
    assert err_huber < 1e-4
    print(f"  L_g = {obj.L_g:.4e}, mu_g = {obj.mu_g:.4e}, kappa = {obj.kappa:.1f}")
    print("  PASSED")


# ================================================================
if __name__ == '__main__':
    test_grad_smooth()
    test_grad_huber()
    test_grad_combined()
    test_grad_elm_scale()
    print("\n" + "=" * 60)
    print("ALL GRADIENT TESTS PASSED")
    print("=" * 60)
