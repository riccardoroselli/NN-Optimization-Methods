"""
Experiment 4: Head-to-head algorithm comparison.

Uses the best configuration for each algorithm family.
Runs on multiple random seeds for robustness.
Produces summary table and overlay convergence plot.
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


def run_comparison_experiment(
    n=2000, d=50, m=200, lam=0.01,
    max_iter=5000, seeds=(42, 123, 456, 789, 1024),
    log_every=10,
):
    """Run best A1 and A2 configs across multiple seeds."""
    records = []
    all_histories = {s: {} for s in seeds}

    for seed in seeds:
        print(f"\n  Seed {seed}:")
        H, y, _ = generate_single_column_problem(n=n, d=d, m=m, seed=seed)
        obj = LassoObjective(H, y, lam=lam)
        w0 = np.zeros(m)
        _, f_ref = solve_lasso_reference(H, y, lam)
        lr = 1.0 / obj.L_g

        # Best A1 configs (standard)
        configs = [
            ('A1-Sub-NAG',  {'variant': 'NAG', 'l1_handling': 'subgradient', 'momentum': 0.9}),
            ('A1-Prox-NAG', {'variant': 'NAG', 'l1_handling': 'proximal',    'momentum': 0.9}),
        ]

        for name, kwargs in configs:
            w_final, hist = heavy_ball(
                obj, w0, lr=lr, max_iter=max_iter,
                log_every=log_every, tol=1e-9, **kwargs,
            )
            f_final = obj.f_total(w_final)
            gap = f_final - f_ref
            sp = float(np.mean(np.abs(w_final) < 1e-8))
            n_iters = hist.iterations[-1] if hist.iterations else max_iter
            t_total = hist.time_elapsed[-1] if hist.time_elapsed else 0

            records.append({
                'algorithm': name, 'seed': seed,
                'f_final': f_final, 'f_ref': f_ref, 'gap': gap,
                'rel_gap': gap / max(abs(f_ref), 1e-16),
                'iterations': n_iters, 'time': t_total,
                'sparsity': sp,
            })
            all_histories[seed][name] = (hist, f_ref)
            print(f"    {name}: gap={gap:.2e}, time={t_total:.2f}s, sparsity={sp:.2%}")

        # A1 two-phase: high momentum → zero momentum refinement
        name = 'A1-Prox-2Phase'
        iter_p1 = max_iter // 5
        iter_p2 = max_iter - iter_p1
        w_final, hist = heavy_ball_two_phase(
            obj, w0, lr=lr,
            max_iter_phase1=iter_p1, max_iter_phase2=iter_p2,
            momentum_phase1=0.9, momentum_phase2=0.0,
            tol=1e-9, variant='NAG', log_every=log_every,
        )
        f_final = obj.f_total(w_final)
        gap = f_final - f_ref
        sp = float(np.mean(np.abs(w_final) < 1e-8))
        n_iters = hist.iterations[-1] if hist.iterations else max_iter
        t_total = hist.time_elapsed[-1] if hist.time_elapsed else 0
        records.append({
            'algorithm': name, 'seed': seed,
            'f_final': f_final, 'f_ref': f_ref, 'gap': gap,
            'rel_gap': gap / max(abs(f_ref), 1e-16),
            'iterations': n_iters, 'time': t_total,
            'sparsity': sp,
        })
        all_histories[seed][name] = (hist, f_ref)
        print(f"    {name}: gap={gap:.2e}, time={t_total:.2f}s, sparsity={sp:.2%}")

        # Best A2 configs
        for mu_val in [1e-2, 1e-3]:
            name = f'A2-mu={mu_val:.0e}'
            smoother = HuberSmoothing(lam=lam, mu=mu_val, dim=m)
            w_final, hist = nesterov_smoothed(
                obj, smoother, w0, max_iter=max_iter,
                log_every=log_every, tol=1e-9,
            )
            f_final = obj.f_total(w_final)
            gap = f_final - f_ref
            sp = float(np.mean(np.abs(w_final) < 1e-8))
            n_iters = hist.iterations[-1] if hist.iterations else max_iter
            t_total = hist.time_elapsed[-1] if hist.time_elapsed else 0

            records.append({
                'algorithm': name, 'seed': seed,
                'f_final': f_final, 'f_ref': f_ref, 'gap': gap,
                'rel_gap': gap / max(abs(f_ref), 1e-16),
                'iterations': n_iters, 'time': t_total,
                'sparsity': sp,
            })
            all_histories[seed][name] = (hist, f_ref)
            print(f"    {name}: gap={gap:.2e}, time={t_total:.2f}s, sparsity={sp:.2%}")

    df = pd.DataFrame(records)
    return df, all_histories


def plot_comparison(df, all_histories, save_dir='results/plots'):
    setup_plotting()
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # --- Summary table ---
    summary = df.groupby('algorithm').agg({
        'gap': ['mean', 'std'],
        'time': ['mean', 'std'],
        'iterations': ['mean', 'std'],
        'sparsity': ['mean', 'std'],
    }).round(6)
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY (mean +/- std across seeds)")
    print("=" * 70)
    print(summary.to_string())

    # --- Overlay convergence for one seed ---
    seed0 = list(all_histories.keys())[0]
    hists = all_histories[seed0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, (name, (hist, f_ref)) in enumerate(hists.items()):
        gaps = [max(f - f_ref, 1e-16) for f in hist.f_total]
        axes[0].semilogy(hist.iterations, gaps, label=name,
                         color=SWEEP_COLORS[i], linewidth=1.5)
        axes[1].semilogy(hist.time_elapsed, gaps, label=name,
                         color=SWEEP_COLORS[i], linewidth=1.5)

    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel(r'$f(w_k) - f^*$')
    axes[0].set_title(f'Best Configs Comparison (seed={seed0})')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Wall-clock time (s)')
    axes[1].set_ylabel(r'$f(w_k) - f^*$')
    axes[1].set_title(f'Best Configs Comparison (seed={seed0})')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/comparison_best_configs.pdf')
    plt.close()
    print(f"  Saved {save_dir}/comparison_best_configs.pdf")

    # --- Bar chart of mean gaps ---
    fig, ax = plt.subplots(figsize=(8, 5))
    mean_gaps = df.groupby('algorithm')['gap'].mean().sort_values()
    std_gaps = df.groupby('algorithm')['gap'].std()
    algos = mean_gaps.index.tolist()
    ax.bar(range(len(algos)), mean_gaps.values,
           yerr=std_gaps[algos].values, capsize=4,
           color=[SWEEP_COLORS[i] for i in range(len(algos))])
    ax.set_xticks(range(len(algos)))
    ax.set_xticklabels(algos, rotation=15)
    ax.set_ylabel(r'Mean optimality gap $f - f^*$')
    ax.set_title('Final Optimality Gap (5 seeds)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/comparison_bar_gap.pdf')
    plt.close()
    print(f"  Saved {save_dir}/comparison_bar_gap.pdf")


def run_all():
    print("=" * 60)
    print("Experiment 4: Algorithm Comparison")
    print("=" * 60)

    df, all_histories = run_comparison_experiment()

    log_dir = 'results/logs'
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(f'{log_dir}/comparison_results.csv', index=False)
    print(f"\n  CSV saved to {log_dir}/comparison_results.csv")

    plot_comparison(df, all_histories)
    print("\nExperiment 4 complete.")


if __name__ == '__main__':
    run_all()
