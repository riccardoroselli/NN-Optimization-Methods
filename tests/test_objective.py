"""
Standalone tests for src/objective.py — LassoObjective.

Tests:
  1. Spectral constants: L_g > 0, mu_g >= 0, kappa consistent
  2. Objective decomposition: f_total = f_smooth + f_l1
  3. Gradient correctness via finite differences
  4. Proximal operator properties
  5. Subgradient at zero
  6. Lipschitz bound: gradient difference bounded by L_g * step
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.elm import ELM
from src.objective import LassoObjective


def _finite_diff_grad(f, w, eps=1e-5):
    g = np.zeros_like(w)
    for i in range(len(w)):
        e = np.zeros_like(w)
        e[i] = eps
        g[i] = (f(w + e) - f(w - e)) / (2.0 * eps)
    return g


# ================================================================
# Test 1: Spectral constants
# ================================================================
def test_spectral_constants():
    print("=" * 60)
    print("TEST 1: Spectral constants L_g, mu_g, kappa")

    rng = np.random.default_rng(0)
    n, m = 100, 20
    H = rng.standard_normal((n, m))
    y = rng.standard_normal(n)
    obj = LassoObjective(H, y, lam=0.1)

    assert obj.L_g > 0, f"L_g should be positive, got {obj.L_g}"
    assert obj.mu_g >= 0, f"mu_g should be non-negative, got {obj.mu_g}"
    assert np.isfinite(obj.L_g), "L_g should be finite"

    if obj.mu_g > 0:
        assert obj.kappa == obj.L_g / obj.mu_g, "kappa should be L_g / mu_g"
        assert obj.kappa >= 1.0, f"kappa should be >= 1, got {obj.kappa}"
    else:
        assert obj.kappa == float('inf'), "kappa should be inf when mu_g = 0"

    # Verify against numpy full eigendecomposition
    HtH = H.T @ H
    eigvals = np.linalg.eigvalsh(HtH)
    expected_L = 2.0 * eigvals[-1]
    expected_mu = 2.0 * max(eigvals[0], 0.0)
    assert abs(obj.L_g - expected_L) / expected_L < 1e-6, \
        f"L_g mismatch: {obj.L_g} vs {expected_L}"
    assert abs(obj.mu_g - expected_mu) / max(expected_mu, 1e-15) < 1e-4, \
        f"mu_g mismatch: {obj.mu_g} vs {expected_mu}"

    print(f"  L_g={obj.L_g:.4e}, mu_g={obj.mu_g:.4e}, kappa={obj.kappa:.1f}")
    print("  PASSED")


# ================================================================
# Test 2: Objective decomposition
# ================================================================
def test_objective_decomposition():
    print("=" * 60)
    print("TEST 2: f_total = f_smooth + f_l1")

    rng = np.random.default_rng(1)
    n, m = 50, 15
    H = rng.standard_normal((n, m))
    y = rng.standard_normal(n)

    for lam in [0.001, 0.1, 1.0]:
        obj = LassoObjective(H, y, lam=lam)
        w = rng.standard_normal(m)

        f_tot = obj.f_total(w)
        f_sm = obj.f_smooth(w)
        f_l1 = obj.f_l1(w)

        assert abs(f_tot - (f_sm + f_l1)) < 1e-12, \
            f"Decomposition failed for lam={lam}: {f_tot} != {f_sm} + {f_l1}"
        assert f_sm >= 0, f"f_smooth should be >= 0, got {f_sm}"
        assert f_l1 >= 0, f"f_l1 should be >= 0, got {f_l1}"

    print("  PASSED")


# ================================================================
# Test 3: Gradient correctness
# ================================================================
def test_gradient_smooth():
    print("=" * 60)
    print("TEST 3: grad_smooth matches finite differences")

    rng = np.random.default_rng(2)
    n, m = 80, 25
    H = rng.standard_normal((n, m))
    y = rng.standard_normal(n)
    obj = LassoObjective(H, y, lam=0.05)

    for trial in range(3):
        w = rng.standard_normal(m) * (0.1 * (10 ** trial))
        g_an = obj.grad_smooth(w)
        g_fd = _finite_diff_grad(obj.f_smooth, w)
        rel_err = np.linalg.norm(g_an - g_fd) / max(np.linalg.norm(g_an), 1e-15)
        status = "OK" if rel_err < 1e-5 else "FAIL"
        print(f"  trial {trial}: rel_err = {rel_err:.2e} [{status}]")
        assert rel_err < 1e-5, f"Gradient check failed at trial {trial}"

    print("  PASSED")


# ================================================================
# Test 4: Proximal operator properties
# ================================================================
def test_proximal_operator():
    print("=" * 60)
    print("TEST 4: Proximal operator (soft-thresholding)")

    rng = np.random.default_rng(3)
    obj = LassoObjective(rng.standard_normal((20, 10)),
                          rng.standard_normal(20), lam=0.5)

    z = np.array([1.0, -1.0, 0.3, -0.3, 0.0, 5.0, -0.1])
    step = 0.2
    p = obj.prox_l1(z, step)
    tau = step * obj.lam  # = 0.1

    # Shrinkage: |prox(z)_i| <= |z_i|
    assert np.all(np.abs(p) <= np.abs(z) + 1e-15), "Prox should shrink magnitudes"

    # Exact zeros for small components
    assert p[4] == 0.0, "prox(0) should be 0"

    # Fixed point: prox(0, any_step) = 0
    assert np.all(obj.prox_l1(np.zeros(5), 1.0) == 0.0), "prox(0) should be 0"

    # Sign preservation
    for i in range(len(z)):
        if p[i] != 0:
            assert np.sign(p[i]) == np.sign(z[i]), \
                f"Sign mismatch at index {i}: z={z[i]}, prox={p[i]}"

    # Thresholding: components with |z_i| < tau should be exactly zero
    assert p[6] == 0.0, f"|z[6]|={abs(z[6])}, tau={tau}, prox should be 0"

    print("  PASSED")


# ================================================================
# Test 5: Subgradient at zero
# ================================================================
def test_subgradient_at_zero():
    print("=" * 60)
    print("TEST 5: Subgradient of L1 at w=0 is 0 (minimum-norm choice)")

    rng = np.random.default_rng(4)
    obj = LassoObjective(rng.standard_normal((20, 10)),
                          rng.standard_normal(20), lam=0.5)

    w_zero = np.zeros(10)
    sg = obj.subgrad_l1(w_zero)
    assert np.all(sg == 0.0), f"subgrad(0) should be all zeros, got {sg}"

    # Mixed: some zero, some nonzero
    w = np.array([0.0, 1.0, -1.0, 0.0, 0.5])
    obj2 = LassoObjective(rng.standard_normal((20, 5)),
                           rng.standard_normal(20), lam=1.0)
    sg2 = obj2.subgrad_l1(w)
    assert sg2[0] == 0.0 and sg2[3] == 0.0, "subgrad at zero components should be 0"
    assert sg2[1] == 1.0, "subgrad of |w| at w=1 should be 1"
    assert sg2[2] == -1.0, "subgrad of |w| at w=-1 should be -1"

    print("  PASSED")


# ================================================================
# Test 6: Lipschitz bound on gradient
# ================================================================
def test_lipschitz_bound():
    print("=" * 60)
    print("TEST 6: Gradient Lipschitz bound ||grad(w1) - grad(w2)|| <= L_g * ||w1 - w2||")

    rng = np.random.default_rng(5)
    n, m = 60, 15
    H = rng.standard_normal((n, m))
    y = rng.standard_normal(n)
    obj = LassoObjective(H, y, lam=0.1)

    for _ in range(10):
        w1 = rng.standard_normal(m)
        w2 = rng.standard_normal(m)
        g1 = obj.grad_smooth(w1)
        g2 = obj.grad_smooth(w2)
        grad_diff = np.linalg.norm(g1 - g2)
        w_diff = np.linalg.norm(w1 - w2)
        ratio = grad_diff / w_diff

        assert ratio <= obj.L_g * (1 + 1e-10), \
            f"Lipschitz violated: ratio={ratio:.4e} > L_g={obj.L_g:.4e}"

    print(f"  Max ratio / L_g = {ratio / obj.L_g:.6f} (should be <= 1)")
    print("  PASSED")


# ================================================================
if __name__ == '__main__':
    test_spectral_constants()
    test_objective_decomposition()
    test_gradient_smooth()
    test_proximal_operator()
    test_subgradient_at_zero()
    test_lipschitz_bound()
    print("\n" + "=" * 60)
    print("ALL OBJECTIVE TESTS PASSED")
    print("=" * 60)
