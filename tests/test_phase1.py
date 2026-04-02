"""
Phase 1 verification: ELM, Objective, Smoothing, Utils.
Tests gradients via finite differences and checks basic properties.
"""

import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.elm import ELM
from src.objective import LassoObjective
from src.smoothing import HuberSmoothing
from src.utils import (
    OptimizationLog, sutskever_momentum_schedule,
    constant_momentum, check_stopping, compute_optimality_gap,
)


def finite_difference_grad(f, w, eps=1e-5):
    """Central finite-difference gradient."""
    grad = np.zeros_like(w)
    for i in range(len(w)):
        e = np.zeros_like(w)
        e[i] = eps
        grad[i] = (f(w + e) - f(w - e)) / (2.0 * eps)
    return grad


# ==============================================================
# 1. ELM
# ==============================================================
def test_elm():
    print("=" * 60)
    print("TEST: ELM")
    n, d, m = 100, 20, 30

    elm = ELM(d=d, m=m, activation='relu', seed=0)
    assert elm.W1.shape == (m, d), f"W1 shape mismatch: {elm.W1.shape}"

    X = np.random.default_rng(1).standard_normal((n, d))
    H = elm.compute_H(X)
    assert H.shape == (n, m), f"H shape mismatch: {H.shape}"
    assert H.dtype == np.float64, f"H dtype: {H.dtype}"
    assert np.all(H >= 0), "ReLU output should be non-negative"
    assert np.any(H > 0), "H is all zeros — something is wrong"

    # Sigmoid variant
    elm_sig = ELM(d=d, m=m, activation='sigmoid', seed=0)
    H_sig = elm_sig.compute_H(X)
    assert np.all((H_sig >= 0) & (H_sig <= 1)), "Sigmoid output out of [0,1]"

    # Reproducibility
    H2 = elm.compute_H(X)
    assert np.array_equal(H, H2), "compute_H not deterministic"

    print("  PASSED: shapes, activations, reproducibility")


# ==============================================================
# 2. LassoObjective
# ==============================================================
def test_objective():
    print("=" * 60)
    print("TEST: LassoObjective")
    rng = np.random.default_rng(42)
    n, m = 50, 15
    H = rng.standard_normal((n, m))
    y = rng.standard_normal(n)
    lam = 0.1

    obj = LassoObjective(H, y, lam=lam)

    # --- Spectral constants ---
    assert obj.L_g > 0, f"L_g should be positive, got {obj.L_g}"
    assert obj.mu_g >= 0, f"mu_g should be non-negative, got {obj.mu_g}"
    print(f"  L_g={obj.L_g:.4e}, mu_g={obj.mu_g:.4e}, kappa={obj.kappa:.1f}")

    # --- Objective values ---
    w = rng.standard_normal(m)
    f_sm = obj.f_smooth(w)
    f_l1 = obj.f_l1(w)
    f_tot = obj.f_total(w)
    assert abs(f_tot - (f_sm + f_l1)) < 1e-12, "f_total != f_smooth + f_l1"
    assert f_sm >= 0, "f_smooth must be non-negative"
    assert f_l1 >= 0, "f_l1 must be non-negative"
    print(f"  f_smooth={f_sm:.4e}, f_l1={f_l1:.4e}, f_total={f_tot:.4e}")

    # --- Gradient finite-difference check ---
    grad_an = obj.grad_smooth(w)
    grad_fd = finite_difference_grad(obj.f_smooth, w)
    rel_err = np.linalg.norm(grad_an - grad_fd) / np.linalg.norm(grad_an)
    print(f"  Smooth gradient rel error: {rel_err:.2e}")
    assert rel_err < 1e-5, f"Gradient check FAILED: {rel_err}"

    # --- Proximal operator properties ---
    z = rng.standard_normal(m)
    step = 0.01
    p = obj.prox_l1(z, step)
    # prox should reduce L1 norm or keep it the same
    assert np.all(np.abs(p) <= np.abs(z) + 1e-15), "Prox increased magnitude"
    # prox of zero vector should be zero
    assert np.allclose(obj.prox_l1(np.zeros(m), step), 0.0)

    # --- Subgradient at zero returns zero ---
    assert np.allclose(obj.subgrad_l1(np.zeros(m)), 0.0), "subgrad(0) should be 0"

    print("  PASSED: objectives, gradient, proximal, subgradient")


# ==============================================================
# 3. HuberSmoothing
# ==============================================================
def test_smoothing():
    print("=" * 60)
    print("TEST: HuberSmoothing")
    rng = np.random.default_rng(7)
    m = 20
    lam = 0.1
    mu = 0.01

    smoother = HuberSmoothing(lam=lam, mu=mu, dim=m)

    assert smoother.L_mu == lam / mu, f"L_mu mismatch: {smoother.L_mu}"
    assert smoother.approx_error == lam * mu * m / 2.0

    # --- Huber is a lower bound of true L1 ---
    w = rng.standard_normal(m)
    true_l1 = lam * np.sum(np.abs(w))
    huber_val = smoother.value(w)
    assert huber_val <= true_l1 + 1e-12, \
        f"Huber {huber_val} should be ≤ true L1 {true_l1}"
    assert true_l1 - huber_val <= smoother.approx_error + 1e-12, \
        f"Gap {true_l1 - huber_val} exceeds bound {smoother.approx_error}"

    # --- Gradient finite-difference check ---
    # Use w values that aren't exactly at the kink |w_i| = mu
    w_test = rng.standard_normal(m) * 0.1
    grad_an = smoother.grad(w_test)
    grad_fd = finite_difference_grad(smoother.value, w_test, eps=1e-7)
    rel_err = np.linalg.norm(grad_an - grad_fd) / max(np.linalg.norm(grad_an), 1e-10)
    print(f"  Huber gradient rel error: {rel_err:.2e}")
    assert rel_err < 1e-4, f"Huber gradient check FAILED: {rel_err}"

    # --- Gradient at exactly zero should be zero ---
    assert np.allclose(smoother.grad(np.zeros(m)), 0.0)

    print("  PASSED: bounds, gradient, zero-point")


# ==============================================================
# 4. Utils
# ==============================================================
def test_utils():
    print("=" * 60)
    print("TEST: Utils")

    # --- Sutskever momentum schedule ---
    # t in [0, 249] → 0.5
    assert abs(sutskever_momentum_schedule(0) - 0.5) < 1e-12
    assert abs(sutskever_momentum_schedule(249) - 0.5) < 1e-12
    # t in [250, 499] → 0.75
    assert abs(sutskever_momentum_schedule(250) - 0.75) < 1e-12
    # t in [500, 749]: floor(500/250)+1=3, exponent = -1 - log2(3) → μ ≈ 0.8333
    assert abs(sutskever_momentum_schedule(500) - (1.0 - 2.0**(-1.0 - np.log2(3)))) < 1e-12
    # monotonically non-decreasing
    vals = [sutskever_momentum_schedule(t) for t in range(2000)]
    assert all(vals[i] <= vals[i + 1] + 1e-15 for i in range(len(vals) - 1))
    # bounded by mu_max
    assert all(v <= 0.99 + 1e-12 for v in vals)

    # --- Constant momentum ---
    sched = constant_momentum(0.8)
    assert sched(0) == 0.8
    assert sched(999) == 0.8

    # --- Stopping criterion ---
    w_old = np.array([1.0, 2.0, 3.0])
    w_new = w_old + 1e-9 * np.ones(3)
    assert check_stopping(w_new, w_old, tol=1e-6) is True
    w_new2 = w_old + np.ones(3)
    assert check_stopping(w_new2, w_old, tol=1e-6) is False

    # --- Optimality gap ---
    assert compute_optimality_gap(10.0, 5.0) == 5.0
    assert compute_optimality_gap(5.0, 5.0) == 1e-16  # floored

    # --- OptimizationLog ---
    log = OptimizationLog()
    w = np.array([0.0, 1.0, 0.0])
    log.record(0, 10.0, 9.0, 5.0, 0.01, w)
    assert len(log.iterations) == 1
    assert log.sparsity[0] == 2.0 / 3.0  # two zeros out of three
    d = log.to_dict()
    assert set(d.keys()) == {'iteration', 'f_total', 'f_smooth', 'grad_norm',
                              'time', 'sparsity', 'w_change'}

    print("  PASSED: momentum schedules, stopping, gap, logger")


# ==============================================================
# Run all
# ==============================================================
if __name__ == '__main__':
    test_elm()
    test_objective()
    test_smoothing()
    test_utils()
    print("\n" + "=" * 60)
    print("ALL PHASE 1 TESTS PASSED")
    print("=" * 60)
