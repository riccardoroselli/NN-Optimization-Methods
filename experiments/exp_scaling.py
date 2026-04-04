"""
Experiment 3: Scalability with problem size.

Vary m (hidden units) and n (samples) and measure time / iterations
to reach a target optimality gap.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from pathlib import Path

from src.objective import LassoObjective
from src.smoothing import HuberSmoothing
from src.heavy_ball import heavy_ball, heavy_ball_two_phase
from src.nesterov_smoothing import nesterov_smoothed
from src.reference import solve_lasso_reference
from data.generate_data import generate_single_column_problem
from experiments.plotting import setup_plotting, SWEEP_COLORS


def run_scaling_experiment(
    m_values=(100, 200, 500, 1000),
    n_values=(1000, 2000, 5000),
    d=50,
    lam=0.01,
    max_iter=5000,
    seed=42,
    target_gap=1e-2,
):
    """
    For each (n, m), run best A1 and best A2 configs, measure:
      - total time
      - iterations to reach target_gap
      - per-iteration cost
    """
    records = []

    for n in n_values:
        for m in m_values:
            print(f"\n  (n={n}, m={m})")
            H, y, _ = generate_single_column_problem(n=n, d=d, m=m, seed=seed)
            obj = LassoObjective(H, y, lam=lam)
            w0 = np.zeros(m)
            _, f_ref = solve_lasso_reference(H, y, lam)
            lr = 1.0 / obj.L_g

            # --- A1-Prox-NAG ---
            t0 = time.time()
            w1, hist1 = heavy_ball(
                obj, w0, lr=lr, max_iter=max_iter, variant='NAG',
                l1_handling='proximal', momentum=0.9,
                log_every=1, tol=1e-9,
            )
            time_a1 = time.time() - t0
            f_final_a1 = obj.f_total(w1)
            iters_a1 = max_iter
            if np.isfinite(f_final_a1):
                for idx, f in enumerate(hist1.f_total):
                    if np.isfinite(f) and f - f_ref < target_gap:
                        iters_a1 = hist1.iterations[idx]
                        break
            per_iter_a1 = time_a1 / max(len(hist1.iterations), 1)

            records.append({
                'algorithm': 'A1-Prox-NAG', 'n': n, 'm': m,
                'total_time': time_a1, 'iters_to_gap': iters_a1,
                'per_iter_time': per_iter_a1,
                'final_gap': f_final_a1 - f_ref if np.isfinite(f_final_a1) else float('inf'),
                'L_g': obj.L_g, 'kappa': obj.kappa,
            })
            print(f"    A1-Prox-NAG: {time_a1:.2f}s, iters_to_gap={iters_a1}")

            # --- A1-Prox-2Phase ---
            iter_p1 = max_iter // 5
            iter_p2 = max_iter - iter_p1
            t0 = time.time()
            w1b, hist1b = heavy_ball_two_phase(
                obj, w0, lr=lr,
                max_iter_phase1=iter_p1, max_iter_phase2=iter_p2,
                momentum_phase1=0.9, momentum_phase2=0.0,
                tol=1e-9, variant='NAG', log_every=1,
            )
            time_a1b = time.time() - t0
            f_final_a1b = obj.f_total(w1b)
            iters_a1b = max_iter
            if np.isfinite(f_final_a1b):
                for idx, f in enumerate(hist1b.f_total):
                    if np.isfinite(f) and f - f_ref < target_gap:
                        iters_a1b = hist1b.iterations[idx]
                        break
            per_iter_a1b = time_a1b / max(len(hist1b.iterations), 1)

            records.append({
                'algorithm': 'A1-Prox-2Phase', 'n': n, 'm': m,
                'total_time': time_a1b, 'iters_to_gap': iters_a1b,
                'per_iter_time': per_iter_a1b,
                'final_gap': f_final_a1b - f_ref if np.isfinite(f_final_a1b) else float('inf'),
                'L_g': obj.L_g, 'kappa': obj.kappa,
            })
            print(f"    A1-Prox-2Ph: {time_a1b:.2f}s, iters_to_gap={iters_a1b}")

            # --- A2 mu=1e-3 ---
            smoother = HuberSmoothing(lam=lam, mu=1e-3, dim=m)
            t0 = time.time()
            w2, hist2 = nesterov_smoothed(
                obj, smoother, w0, max_iter=max_iter,
                log_every=1, tol=1e-9,
            )
            time_a2 = time.time() - t0
            f_final_a2 = obj.f_total(w2)
            iters_a2 = max_iter
            if np.isfinite(f_final_a2):
                for idx, f in enumerate(hist2.f_total):
                    if np.isfinite(f) and f - f_ref < target_gap:
                        iters_a2 = hist2.iterations[idx]
                        break
            per_iter_a2 = time_a2 / max(len(hist2.iterations), 1)

            records.append({
                'algorithm': 'A2-mu=1e-3', 'n': n, 'm': m,
                'total_time': time_a2, 'iters_to_gap': iters_a2,
                'per_iter_time': per_iter_a2,
                'final_gap': f_final_a2 - f_ref if np.isfinite(f_final_a2) else float('inf'),
                'L_g': obj.L_g, 'kappa': obj.kappa,
            })
            print(f"    A2-mu=1e-3:  {time_a2:.2f}s, iters_to_gap={iters_a2}")

    df = pd.DataFrame(records)
    return df


def plot_scaling(df, save_dir='results/plots'):
    setup_plotting()
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # --- Time vs m (for fixed n=2000) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    df_n2k = df[df['n'] == 2000]
    for i, algo in enumerate(df_n2k['algorithm'].unique()):
        sub = df_n2k[df_n2k['algorithm'] == algo].sort_values('m')
        axes[0].loglog(sub['m'], sub['total_time'], 'o-',
                       label=algo, color=SWEEP_COLORS[i], linewidth=1.5)
        axes[1].loglog(sub['m'], sub['per_iter_time'], 'o-',
                       label=algo, color=SWEEP_COLORS[i], linewidth=1.5)

    axes[0].set_xlabel('Hidden units $m$')
    axes[0].set_ylabel('Total time (s)')
    axes[0].set_title('Total Time vs $m$ (n=2000)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel('Hidden units $m$')
    axes[1].set_ylabel('Per-iteration time (s)')
    axes[1].set_title('Per-Iteration Cost vs $m$ (n=2000)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/scaling_vs_m.pdf')
    plt.close()
    print(f"  Saved {save_dir}/scaling_vs_m.pdf")

    # --- Time vs n (for fixed m=200) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    df_m200 = df[df['m'] == 200]
    for i, algo in enumerate(df_m200['algorithm'].unique()):
        sub = df_m200[df_m200['algorithm'] == algo].sort_values('n')
        axes[0].loglog(sub['n'], sub['total_time'], 'o-',
                       label=algo, color=SWEEP_COLORS[i], linewidth=1.5)
        axes[1].loglog(sub['n'], sub['iters_to_gap'], 'o-',
                       label=algo, color=SWEEP_COLORS[i], linewidth=1.5)

    axes[0].set_xlabel('Samples $n$')
    axes[0].set_ylabel('Total time (s)')
    axes[0].set_title('Total Time vs $n$ (m=200)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel('Samples $n$')
    axes[1].set_ylabel('Iterations to gap < 0.01')
    axes[1].set_title('Iterations to Target Gap vs $n$ (m=200)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/scaling_vs_n.pdf')
    plt.close()
    print(f"  Saved {save_dir}/scaling_vs_n.pdf")


def run_all():
    print("=" * 60)
    print("Experiment 3: Scalability")
    print("=" * 60)

    df = run_scaling_experiment()

    # Save CSV
    log_dir = 'results/logs'
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(f'{log_dir}/scaling_results.csv', index=False)
    print(f"\n  CSV saved to {log_dir}/scaling_results.csv")
    print("\nScaling summary:")
    print(df.to_string(index=False))

    plot_scaling(df)
    print("\nExperiment 3 complete.")


if __name__ == '__main__':
    run_all()
