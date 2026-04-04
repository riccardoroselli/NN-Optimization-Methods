"""
Experiment 2: Parameter sensitivity analysis.

Sweeps:
  A1 — step size eta, momentum beta, schedule vs constant
  A2 — smoothing parameter mu
  Both — regularisation strength lambda
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from src.objective import LassoObjective
from src.smoothing import HuberSmoothing
from src.heavy_ball import heavy_ball, heavy_ball_two_phase
from src.nesterov_smoothing import nesterov_smoothed
from src.reference import solve_lasso_reference
from src.utils import sutskever_momentum_schedule
from data.generate_data import generate_single_column_problem
from experiments.plotting import setup_plotting, SWEEP_COLORS


def _make_problem(n=2000, d=50, m=200, lam=0.01, seed=42):
    """Create a standard test problem."""
    H, y, _ = generate_single_column_problem(n=n, d=d, m=m, seed=seed)
    obj = LassoObjective(H, y, lam=lam)
    w0 = np.zeros(obj.m)
    _, f_ref = solve_lasso_reference(H, y, lam)
    return obj, w0, f_ref


def _gap_series(hist, f_ref):
    return [max(f - f_ref, 1e-16) for f in hist.f_total]


# ======================================================================
# A1: Step-size sweep
# ======================================================================
def sweep_a1_lr(save_dir='results/plots', log_dir='results/logs'):
    setup_plotting()
    obj, w0, f_ref = _make_problem()
    L = obj.L_g
    max_iter = 5000
    log_every = 10

    lr_factors = [0.1, 0.5, 1.0, 1.5, 2.0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for mode_idx, (l1h, title) in enumerate([
        ('subgradient', 'A1-Sub-NAG'), ('proximal', 'A1-Prox-NAG')
    ]):
        ax = axes[mode_idx]
        for i, factor in enumerate(lr_factors):
            lr = factor / L
            label = rf'$\eta = {factor}/L_g$'
            try:
                _, hist = heavy_ball(
                    obj, w0, lr=lr, max_iter=max_iter, variant='NAG',
                    l1_handling=l1h, momentum=0.9, log_every=log_every, tol=1e-9,
                )
                gaps = _gap_series(hist, f_ref)
                ax.semilogy(hist.iterations, gaps, label=label,
                            color=SWEEP_COLORS[i], linewidth=1.5)
            except Exception as e:
                print(f"  {title} lr={factor}/L diverged: {e}")
        ax.set_xlabel('Iteration')
        ax.set_ylabel(r'$f(w_k) - f^*$')
        ax.set_title(f'{title}: Step-Size Sensitivity')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(f'{save_dir}/params_a1_lr_sweep.pdf')
    plt.close()
    print(f"  Saved {save_dir}/params_a1_lr_sweep.pdf")


# ======================================================================
# A1: Momentum sweep
# ======================================================================
def sweep_a1_momentum(save_dir='results/plots', log_dir='results/logs'):
    setup_plotting()
    obj, w0, f_ref = _make_problem()
    lr = 1.0 / obj.L_g
    max_iter = 5000
    log_every = 10

    betas = [0.0, 0.5, 0.9, 0.95, 0.99]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for mode_idx, (l1h, title) in enumerate([
        ('subgradient', 'A1-Sub-NAG'), ('proximal', 'A1-Prox-NAG')
    ]):
        ax = axes[mode_idx]
        for i, beta in enumerate(betas):
            label = rf'$\beta = {beta}$'
            _, hist = heavy_ball(
                obj, w0, lr=lr, max_iter=max_iter, variant='NAG',
                l1_handling=l1h, momentum=beta, log_every=log_every, tol=1e-9,
            )
            gaps = _gap_series(hist, f_ref)
            ax.semilogy(hist.iterations, gaps, label=label,
                        color=SWEEP_COLORS[i], linewidth=1.5)

        # Also add Sutskever schedule
        _, hist = heavy_ball(
            obj, w0, lr=lr, max_iter=max_iter, variant='NAG',
            l1_handling=l1h, momentum_schedule=sutskever_momentum_schedule,
            log_every=log_every, tol=1e-9,
        )
        gaps = _gap_series(hist, f_ref)
        ax.semilogy(hist.iterations, gaps, label='Sutskever sched.',
                    color=SWEEP_COLORS[len(betas)], linewidth=1.5, linestyle='--')

        # Two-phase (proximal only)
        if l1h == 'proximal':
            iter_p1 = max_iter // 5
            iter_p2 = max_iter - iter_p1
            _, hist = heavy_ball_two_phase(
                obj, w0, lr=lr,
                max_iter_phase1=iter_p1, max_iter_phase2=iter_p2,
                momentum_phase1=0.9, momentum_phase2=0.0,
                tol=1e-9, variant='NAG', log_every=log_every,
            )
            gaps = _gap_series(hist, f_ref)
            ax.semilogy(hist.iterations, gaps, label='Two-phase (0.9→0)',
                        color=SWEEP_COLORS[len(betas) + 1], linewidth=1.5, linestyle='-.')

        ax.set_xlabel('Iteration')
        ax.set_ylabel(r'$f(w_k) - f^*$')
        ax.set_title(f'{title}: Momentum Sensitivity')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/params_a1_momentum_sweep.pdf')
    plt.close()
    print(f"  Saved {save_dir}/params_a1_momentum_sweep.pdf")


# ======================================================================
# A2: Smoothing parameter mu sweep
# ======================================================================
def sweep_a2_mu(save_dir='results/plots', log_dir='results/logs'):
    setup_plotting()
    obj, w0, f_ref = _make_problem()
    max_iter = 5000
    log_every = 10

    mu_vals = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, mu in enumerate(mu_vals):
        smoother = HuberSmoothing(lam=obj.lam, mu=mu, dim=obj.m)
        _, hist = nesterov_smoothed(
            obj, smoother, w0, max_iter=max_iter,
            log_every=log_every, tol=1e-9,
        )
        gaps = _gap_series(hist, f_ref)
        ax.semilogy(hist.iterations, gaps,
                    label=rf'$\mu = {mu:.0e}$',
                    color=SWEEP_COLORS[i], linewidth=1.5)

    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'$f(w_k) - f^*$')
    ax.set_title(r'A2 (Nesterov Smoothing): $\mu$ Sensitivity')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/params_a2_mu_sweep.pdf')
    plt.close()
    print(f"  Saved {save_dir}/params_a2_mu_sweep.pdf")


# ======================================================================
# Both: Lambda sweep
# ======================================================================
def sweep_lambda(save_dir='results/plots', log_dir='results/logs'):
    setup_plotting()
    lam_vals = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    max_iter = 5000
    log_every = 10

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, lam in enumerate(lam_vals):
        obj, w0, f_ref = _make_problem(lam=lam)
        lr = 1.0 / obj.L_g

        # A1-Prox-NAG
        _, hist1 = heavy_ball(
            obj, w0, lr=lr, max_iter=max_iter, variant='NAG',
            l1_handling='proximal', momentum=0.9, log_every=log_every, tol=1e-9,
        )
        gaps1 = _gap_series(hist1, f_ref)
        axes[0].semilogy(hist1.iterations, gaps1,
                         label=rf'$\lambda = {lam:.0e}$',
                         color=SWEEP_COLORS[i], linewidth=1.5)

        # A2 with mu=1e-3
        smoother = HuberSmoothing(lam=lam, mu=1e-3, dim=obj.m)
        _, hist2 = nesterov_smoothed(
            obj, smoother, w0, max_iter=max_iter,
            log_every=log_every, tol=1e-9,
        )
        gaps2 = _gap_series(hist2, f_ref)
        axes[1].semilogy(hist2.iterations, gaps2,
                         label=rf'$\lambda = {lam:.0e}$',
                         color=SWEEP_COLORS[i], linewidth=1.5)

    axes[0].set_title(r'A1-Prox-NAG: $\lambda$ Sensitivity')
    axes[1].set_title(r'A2 ($\mu=10^{-3}$): $\lambda$ Sensitivity')
    for ax in axes:
        ax.set_xlabel('Iteration')
        ax.set_ylabel(r'$f(w_k) - f^*$')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/params_lambda_sweep.pdf')
    plt.close()
    print(f"  Saved {save_dir}/params_lambda_sweep.pdf")


# ======================================================================
# A1: Heatmap eta x beta
# ======================================================================
def heatmap_a1_lr_beta(save_dir='results/plots', log_dir='results/logs'):
    setup_plotting()
    obj, w0, f_ref = _make_problem()
    L = obj.L_g
    max_iter = 3000
    log_every = max_iter  # only need final value

    lr_factors = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0]
    betas = [0.0, 0.5, 0.8, 0.9, 0.95, 0.99]
    results = np.full((len(betas), len(lr_factors)), np.nan)

    for bi, beta in enumerate(betas):
        for li, factor in enumerate(lr_factors):
            lr = factor / L
            try:
                w_final, _ = heavy_ball(
                    obj, w0, lr=lr, max_iter=max_iter, variant='NAG',
                    l1_handling='proximal', momentum=beta,
                    log_every=max_iter, tol=1e-9,
                )
                gap = max(obj.f_total(w_final) - f_ref, 1e-16)
                results[bi, li] = np.log10(gap)
            except Exception:
                pass  # leave NaN for diverged

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(results, aspect='auto', cmap='viridis_r',
                   origin='lower')
    ax.set_xticks(range(len(lr_factors)))
    ax.set_xticklabels([f'{f}/L' for f in lr_factors])
    ax.set_yticks(range(len(betas)))
    ax.set_yticklabels([str(b) for b in betas])
    ax.set_xlabel(r'Step size $\eta$')
    ax.set_ylabel(r'Momentum $\beta$')
    ax.set_title(r'A1-Prox-NAG: $\log_{10}(f - f^*)$ after 3000 iters')
    plt.colorbar(im, ax=ax, label=r'$\log_{10}(f - f^*)$')

    # Annotate cells
    for bi in range(len(betas)):
        for li in range(len(lr_factors)):
            val = results[bi, li]
            if not np.isnan(val):
                ax.text(li, bi, f'{val:.1f}', ha='center', va='center',
                        fontsize=8, color='white' if val < np.nanmedian(results) else 'black')
            else:
                ax.text(li, bi, 'div', ha='center', va='center',
                        fontsize=8, color='red')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/params_a1_heatmap.pdf')
    plt.close()
    print(f"  Saved {save_dir}/params_a1_heatmap.pdf")


def run_all_param_experiments():
    print("=" * 60)
    print("Experiment 2: Parameter Sensitivity")
    print("=" * 60)

    print("\n--- A1 step-size sweep ---")
    sweep_a1_lr()

    print("\n--- A1 momentum sweep ---")
    sweep_a1_momentum()

    print("\n--- A2 mu sweep ---")
    sweep_a2_mu()

    print("\n--- Lambda sweep (both) ---")
    sweep_lambda()

    print("\n--- A1 heatmap (eta x beta) ---")
    heatmap_a1_lr_beta()

    print("\nExperiment 2 complete.")


if __name__ == '__main__':
    run_all_param_experiments()
