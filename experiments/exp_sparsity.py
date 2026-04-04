"""
Experiment 5: Sparsity analysis.

For varying lambda:
  - Exact zeros (|w_i| == 0) for proximal method
  - Near-zeros (|w_i| < threshold) for all methods
  - Histogram of |w_i| values
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
from data.generate_data import generate_single_column_problem
from experiments.plotting import setup_plotting, SWEEP_COLORS


def run_sparsity_experiment(
    n=2000, d=50, m=200,
    lam_values=(1e-4, 1e-3, 1e-2, 1e-1, 1.0),
    max_iter=5000, seed=42,
    threshold=1e-6,
):
    """Measure sparsity for each algorithm across lambda values."""
    records = []
    weight_distributions = {}  # (algo, lam) -> |w| array

    for lam in lam_values:
        print(f"\n  lambda = {lam:.0e}")
        H, y, _ = generate_single_column_problem(n=n, d=d, m=m, seed=seed)
        obj = LassoObjective(H, y, lam=lam)
        w0 = np.zeros(m)
        w_ref, f_ref = solve_lasso_reference(H, y, lam)
        lr = 1.0 / obj.L_g

        # Reference sparsity
        exact_ref = float(np.mean(w_ref == 0.0))
        near_ref = float(np.mean(np.abs(w_ref) < threshold))
        records.append({
            'algorithm': 'sklearn-ref', 'lambda': lam,
            'exact_zeros': exact_ref, 'near_zeros': near_ref,
            'f_total': f_ref,
        })

        iter_p1 = max_iter // 5
        iter_p2 = max_iter - iter_p1
        configs = [
            ('A1-Sub-NAG', lambda: heavy_ball(
                obj, w0, lr=lr, max_iter=max_iter, variant='NAG',
                l1_handling='subgradient', momentum=0.9, log_every=max_iter, tol=1e-9)),
            ('A1-Prox-NAG', lambda: heavy_ball(
                obj, w0, lr=lr, max_iter=max_iter, variant='NAG',
                l1_handling='proximal', momentum=0.9, log_every=max_iter, tol=1e-9)),
            ('A1-Prox-2Phase', lambda: heavy_ball_two_phase(
                obj, w0, lr=lr,
                max_iter_phase1=iter_p1, max_iter_phase2=iter_p2,
                momentum_phase1=0.9, momentum_phase2=0.0,
                tol=1e-9, variant='NAG', log_every=max_iter)),
            ('A2-mu=1e-3', lambda: nesterov_smoothed(
                obj, HuberSmoothing(lam=lam, mu=1e-3, dim=m), w0,
                max_iter=max_iter, log_every=max_iter, tol=1e-9)),
        ]

        for name, run_fn in configs:
            w_final, _ = run_fn()
            exact_z = float(np.mean(w_final == 0.0))
            near_z = float(np.mean(np.abs(w_final) < threshold))
            f_tot = obj.f_total(w_final)

            records.append({
                'algorithm': name, 'lambda': lam,
                'exact_zeros': exact_z, 'near_zeros': near_z,
                'f_total': f_tot,
            })
            weight_distributions[(name, lam)] = np.abs(w_final)
            print(f"    {name}: exact={exact_z:.2%}, near={near_z:.2%}, f={f_tot:.4e}")

    df = pd.DataFrame(records)
    return df, weight_distributions


def plot_sparsity(df, weight_distributions, save_dir='results/plots'):
    setup_plotting()
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    algos = ['sklearn-ref', 'A1-Sub-NAG', 'A1-Prox-NAG', 'A1-Prox-2Phase', 'A2-mu=1e-3']
    lam_vals = sorted(df['lambda'].unique())

    # --- Plot 1: sparsity (%) vs lambda ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, algo in enumerate(algos):
        sub = df[df['algorithm'] == algo].sort_values('lambda')
        axes[0].semilogx(sub['lambda'], sub['exact_zeros'] * 100, 'o-',
                         label=algo, color=SWEEP_COLORS[i], linewidth=1.5)
        axes[1].semilogx(sub['lambda'], sub['near_zeros'] * 100, 'o-',
                         label=algo, color=SWEEP_COLORS[i], linewidth=1.5)

    axes[0].set_xlabel(r'$\lambda$')
    axes[0].set_ylabel('Exact zeros (%)')
    axes[0].set_title('Exact Sparsity ($w_i = 0$) vs $\\lambda$')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel(r'$\lambda$')
    axes[1].set_ylabel(r'Near-zeros (%) [$|w_i| < 10^{-6}$]')
    axes[1].set_title(r'Near Sparsity ($|w_i| < 10^{-6}$) vs $\lambda$')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/sparsity_vs_lambda.pdf')
    plt.close()
    print(f"  Saved {save_dir}/sparsity_vs_lambda.pdf")

    # --- Plot 2: histograms of |w_i| for fixed lambda = 0.01 ---
    target_lam = 1e-2
    hist_algos = ['A1-Sub-NAG', 'A1-Prox-NAG', 'A1-Prox-2Phase', 'A2-mu=1e-3']
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    for i, algo in enumerate(hist_algos):
        key = (algo, target_lam)
        if key in weight_distributions:
            w_abs = weight_distributions[key]
            # Log-scale histogram (clip zeros for log)
            w_nonzero = w_abs[w_abs > 0]
            if len(w_nonzero) > 0:
                axes[i].hist(np.log10(w_nonzero + 1e-16), bins=40,
                             color=SWEEP_COLORS[i], alpha=0.7, edgecolor='black')
            n_zero = np.sum(w_abs == 0.0)
            axes[i].set_title(f'{algo}\n(exact zeros: {n_zero}/{len(w_abs)})')
            axes[i].set_xlabel(r'$\log_{10}|w_i|$')
            axes[i].set_ylabel('Count')
            axes[i].grid(True, alpha=0.3)

    plt.suptitle(rf'Weight magnitude distribution ($\lambda = {target_lam}$)', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/sparsity_histograms.pdf')
    plt.close()
    print(f"  Saved {save_dir}/sparsity_histograms.pdf")


def run_all():
    print("=" * 60)
    print("Experiment 5: Sparsity Analysis")
    print("=" * 60)

    df, wdist = run_sparsity_experiment()

    log_dir = 'results/logs'
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(f'{log_dir}/sparsity_results.csv', index=False)
    print(f"\n  CSV saved to {log_dir}/sparsity_results.csv")

    print("\nSparsity summary:")
    print(df.to_string(index=False))

    plot_sparsity(df, wdist)
    print("\nExperiment 5 complete.")


if __name__ == '__main__':
    run_all()
