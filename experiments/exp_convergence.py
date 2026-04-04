"""
Experiment 1: Convergence behaviour of all algorithm variants.

Produces:
  - f(w_k) - f* vs iteration  (semi-log)
  - f(w_k) - f* vs wall-clock time  (semi-log)
  - Sparsity vs iteration
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
from experiments.plotting import setup_plotting, COLORS, SWEEP_COLORS


def run_convergence_experiment(
    n=2000, d=50, m=200,
    lam=0.01, max_iter=5000, seed=42,
    log_every=10,
):
    """Run all algorithm variants on a single output column."""
    H, y, _ = generate_single_column_problem(n=n, d=d, m=m, seed=seed)

    obj = LassoObjective(H, y, lam=lam)
    w0 = np.zeros(m)

    print(f"Problem: n={n}, m={m}, lam={lam}")
    print(f"  L_g = {obj.L_g:.4e}, mu_g = {obj.mu_g:.4e}, kappa = {obj.kappa:.1f}")

    # Reference solution
    w_ref, f_ref = solve_lasso_reference(H, y, lam)
    print(f"  f* (reference) = {f_ref:.6e}")

    results = {}
    lr = 1.0 / obj.L_g

    # --- A1 variants ---
    configs_a1 = [
        ('A1-Sub-CM',   {'variant': 'CM',  'l1_handling': 'subgradient', 'momentum': 0.9}),
        ('A1-Sub-NAG',  {'variant': 'NAG', 'l1_handling': 'subgradient', 'momentum': 0.9}),
        ('A1-Prox-CM',  {'variant': 'CM',  'l1_handling': 'proximal',   'momentum': 0.9}),
        ('A1-Prox-NAG', {'variant': 'NAG', 'l1_handling': 'proximal',   'momentum': 0.9}),
    ]

    for name, kwargs in configs_a1:
        print(f"\n  Running {name}...")
        w_final, hist = heavy_ball(
            obj, w0, lr=lr, max_iter=max_iter,
            log_every=log_every, verbose=True, tol=1e-8, **kwargs,
        )
        results[name] = hist
        f_final = obj.f_total(w_final)
        print(f"    Final f = {f_final:.6e}, gap = {f_final - f_ref:.2e}")

    # --- A1 with Sutskever schedule ---
    print(f"\n  Running A1-Prox-NAG-Sched...")
    w_final, hist = heavy_ball(
        obj, w0, lr=lr, max_iter=max_iter,
        variant='NAG', l1_handling='proximal',
        momentum_schedule=sutskever_momentum_schedule,
        log_every=log_every, verbose=True, tol=1e-8,
    )
    results['A1-Prox-NAG-Sched'] = hist
    f_final = obj.f_total(w_final)
    print(f"    Final f = {f_final:.6e}, gap = {f_final - f_ref:.2e}")

    # --- A1 two-phase: high momentum → zero momentum refinement ---
    print(f"\n  Running A1-Prox-2Phase...")
    iter_p1 = max_iter // 5       # 20% budget for fast approach
    iter_p2 = max_iter - iter_p1  # 80% budget for refinement
    w_final, hist = heavy_ball_two_phase(
        obj, w0, lr=lr,
        max_iter_phase1=iter_p1, max_iter_phase2=iter_p2,
        momentum_phase1=0.9, momentum_phase2=0.0,
        tol=1e-8, variant='NAG',
        log_every=log_every, verbose=True,
    )
    results['A1-Prox-2Phase'] = hist
    f_final = obj.f_total(w_final)
    print(f"    Final f = {f_final:.6e}, gap = {f_final - f_ref:.2e}")

    # --- A2: Nesterov Smoothing with different mu ---
    for mu_val in [1e-1, 1e-2, 1e-3, 1e-4]:
        name = f'A2-mu={mu_val:.0e}'
        print(f"\n  Running {name}...")
        smoother = HuberSmoothing(lam=lam, mu=mu_val, dim=m)
        w_final, hist = nesterov_smoothed(
            obj, smoother, w0, max_iter=max_iter,
            log_every=log_every, verbose=True, tol=1e-8,
        )
        results[name] = hist
        f_final = obj.f_total(w_final)
        print(f"    Final f = {f_final:.6e}, gap = {f_final - f_ref:.2e}")

    return results, f_ref


def plot_convergence(results, f_ref, save_dir='results/plots'):
    """Generate convergence plots."""
    setup_plotting()
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    colors = SWEEP_COLORS

    # --- Plot 1: gap vs iteration ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (name, hist) in enumerate(results.items()):
        gaps = [max(f - f_ref, 1e-16) for f in hist.f_total]
        ax.semilogy(hist.iterations, gaps, label=name,
                     color=colors[i % len(colors)], linewidth=1.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'$f(w_k) - f^*$')
    ax.set_title('Convergence: Optimality Gap vs Iteration')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/convergence_vs_iter.pdf')
    plt.close()
    print(f"  Saved {save_dir}/convergence_vs_iter.pdf")

    # --- Plot 2: gap vs time ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (name, hist) in enumerate(results.items()):
        gaps = [max(f - f_ref, 1e-16) for f in hist.f_total]
        ax.semilogy(hist.time_elapsed, gaps, label=name,
                     color=colors[i % len(colors)], linewidth=1.5)
    ax.set_xlabel('Wall-clock time (s)')
    ax.set_ylabel(r'$f(w_k) - f^*$')
    ax.set_title('Convergence: Optimality Gap vs Time')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/convergence_vs_time.pdf')
    plt.close()
    print(f"  Saved {save_dir}/convergence_vs_time.pdf")

    # --- Plot 3: sparsity vs iteration ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (name, hist) in enumerate(results.items()):
        ax.plot(hist.iterations, [s * 100 for s in hist.sparsity],
                label=name, color=colors[i % len(colors)], linewidth=1.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Sparsity (%)')
    ax.set_title('Sparsity of Iterates vs Iteration')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/sparsity_vs_iter.pdf')
    plt.close()
    print(f"  Saved {save_dir}/sparsity_vs_iter.pdf")


def save_logs(results, f_ref, save_dir='results/logs'):
    """Save convergence data as CSV."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    for name, hist in results.items():
        df = pd.DataFrame(hist.to_dict())
        df['gap'] = [max(f - f_ref, 1e-16) for f in hist.f_total]
        fname = name.replace('=', '').replace(' ', '_')
        df.to_csv(f'{save_dir}/convergence_{fname}.csv', index=False)
    print(f"  CSV logs saved to {save_dir}/")


if __name__ == '__main__':
    results, f_ref = run_convergence_experiment()
    plot_convergence(results, f_ref)
    save_logs(results, f_ref)
    print("\nExperiment 1 (convergence) complete.")
