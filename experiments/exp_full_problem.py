"""
Experiment 6: Full multi-column problem (10,000+ weights).

Solves all p output columns independently using the best algorithm
configurations, demonstrating the required scale:
    n = 2000 samples, m = 200 hidden units, p = 50 outputs
    → p × m = 10,000 total weights

Produces:
  - Summary table: algorithm | total_time | mean_gap | total_f | sparsity
  - Per-column gap distribution (box plot)
  - Convergence overlay for a few representative columns
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from pathlib import Path

from src.elm import ELM
from src.objective import LassoObjective
from src.smoothing import HuberSmoothing
from src.heavy_ball import heavy_ball, heavy_ball_two_phase
from src.nesterov_smoothing import nesterov_smoothed
from src.reference import solve_lasso_reference
from data.generate_data import generate_synthetic_data
from experiments.plotting import setup_plotting, SWEEP_COLORS


def run_full_problem(
    n=2000, d=50, m=200, p=50,
    lam=0.01, max_iter=5000, seed=42,
):
    """Solve all p Lasso sub-problems with the best algorithm configs."""

    # --- Generate data using the shared module ---
    X, Y, W2_true = generate_synthetic_data(
        n=n, d=d, m=m, p=p, sparsity=0.7, noise_std=0.1, seed=seed,
    )
    elm = ELM(d=d, m=m, activation='relu', seed=seed)
    H = elm.compute_H(X)

    print(f"Full problem: n={n}, d={d}, m={m}, p={p}")
    print(f"  Total weights: {p * m}")

    algorithms = {
        'A1-Sub-NAG': {},
        'A1-Prox-NAG': {},
        'A1-Prox-2Phase': {},
        'A2-mu=1e-3': {},
    }
    # Per-column records
    all_records = []
    # Per-algorithm aggregates
    algo_W = {name: np.zeros((p, m)) for name in algorithms}

    for name in algorithms:
        t_total = 0.0
        print(f"\n  --- {name} ---")

        for j in range(p):
            y_j = Y[:, j]
            obj = LassoObjective(H, y_j, lam=lam)
            w0 = np.zeros(m)
            lr = 1.0 / obj.L_g

            t0 = time.time()

            if name == 'A1-Sub-NAG':
                w_j, _ = heavy_ball(
                    obj, w0, lr=lr, max_iter=max_iter, variant='NAG',
                    l1_handling='subgradient', momentum=0.9,
                    log_every=max_iter, tol=1e-8,
                )
            elif name == 'A1-Prox-NAG':
                w_j, _ = heavy_ball(
                    obj, w0, lr=lr, max_iter=max_iter, variant='NAG',
                    l1_handling='proximal', momentum=0.9,
                    log_every=max_iter, tol=1e-8,
                )
            elif name == 'A1-Prox-2Phase':
                iter_p1 = max_iter // 5
                iter_p2 = max_iter - iter_p1
                w_j, _ = heavy_ball_two_phase(
                    obj, w0, lr=lr,
                    max_iter_phase1=iter_p1, max_iter_phase2=iter_p2,
                    momentum_phase1=0.9, momentum_phase2=0.0,
                    tol=1e-8, variant='NAG', log_every=max_iter,
                )
            elif name == 'A2-mu=1e-3':
                smoother = HuberSmoothing(lam=lam, mu=1e-3, dim=m)
                w_j, _ = nesterov_smoothed(
                    obj, smoother, w0, max_iter=max_iter,
                    log_every=max_iter, tol=1e-8,
                )

            elapsed = time.time() - t0
            t_total += elapsed

            # Reference for this column
            _, f_ref_j = solve_lasso_reference(H, y_j, lam)
            f_j = obj.f_total(w_j)
            gap_j = f_j - f_ref_j
            sp_j = float(np.mean(np.abs(w_j) < 1e-8))

            algo_W[name][j, :] = w_j

            all_records.append({
                'algorithm': name, 'column': j,
                'f_total': f_j, 'f_ref': f_ref_j, 'gap': gap_j,
                'sparsity': sp_j, 'time': elapsed,
            })

        print(f"    Total time: {t_total:.2f}s")

    df = pd.DataFrame(all_records)
    return df, algo_W, H, Y, W2_true


def plot_full_problem(df, save_dir='results/plots'):
    setup_plotting()
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    algos = df['algorithm'].unique()

    # --- Plot 1: box plot of per-column gaps ---
    fig, ax = plt.subplots(figsize=(10, 5))
    data_for_box = [df[df['algorithm'] == a]['gap'].values for a in algos]
    bp = ax.boxplot(data_for_box, tick_labels=algos, patch_artist=True)
    for patch, color in zip(bp['boxes'], SWEEP_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel(r'Optimality gap $f_j - f_j^*$')
    ax.set_title(f'Per-Column Optimality Gap (p={df["column"].nunique()} columns, '
                 f'{df["column"].nunique() * 200} total weights)')
    ax.set_yscale('symlog', linthresh=1e-10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/full_problem_gap_boxplot.pdf')
    plt.close()
    print(f"  Saved {save_dir}/full_problem_gap_boxplot.pdf")

    # --- Plot 2: bar chart of total time ---
    fig, ax = plt.subplots(figsize=(8, 5))
    time_totals = df.groupby('algorithm')['time'].sum()
    bars = ax.bar(range(len(algos)), [time_totals[a] for a in algos],
                  color=SWEEP_COLORS[:len(algos)], alpha=0.8)
    ax.set_xticks(range(len(algos)))
    ax.set_xticklabels(algos, rotation=15)
    ax.set_ylabel('Total time (s)')
    ax.set_title(f'Total Wall-Clock Time to Solve All {df["column"].nunique()} Columns')
    ax.grid(True, alpha=0.3, axis='y')
    # Add value labels on bars
    for bar, a in zip(bars, algos):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{time_totals[a]:.1f}s', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/full_problem_time_bar.pdf')
    plt.close()
    print(f"  Saved {save_dir}/full_problem_time_bar.pdf")

    # --- Plot 3: sparsity distribution ---
    fig, ax = plt.subplots(figsize=(10, 5))
    data_for_sp = [df[df['algorithm'] == a]['sparsity'].values * 100 for a in algos]
    bp = ax.boxplot(data_for_sp, tick_labels=algos, patch_artist=True)
    for patch, color in zip(bp['boxes'], SWEEP_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Sparsity per column (%)')
    ax.set_title('Per-Column Sparsity Distribution')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/full_problem_sparsity_boxplot.pdf')
    plt.close()
    print(f"  Saved {save_dir}/full_problem_sparsity_boxplot.pdf")


def run_all():
    print("=" * 60)
    print("Experiment 6: Full Multi-Column Problem (10,000 weights)")
    print("=" * 60)

    df, algo_W, H, Y, W2_true = run_full_problem()

    # Save CSV
    log_dir = 'results/logs'
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(f'{log_dir}/full_problem_results.csv', index=False)
    print(f"\n  CSV saved to {log_dir}/full_problem_results.csv")

    # Summary
    print("\n" + "=" * 70)
    print("FULL PROBLEM SUMMARY (p=50 columns, 10,000 total weights)")
    print("=" * 70)
    summary = df.groupby('algorithm').agg({
        'gap': ['mean', 'median', 'max'],
        'time': 'sum',
        'sparsity': 'mean',
        'f_total': 'sum',
    })
    summary.columns = ['mean_gap', 'median_gap', 'max_gap',
                        'total_time', 'mean_sparsity', 'total_f']
    print(summary.to_string())

    plot_full_problem(df)
    print("\nExperiment 6 complete.")


if __name__ == '__main__':
    run_all()
