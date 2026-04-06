"""
Enhanced Convergence Experiment — theory-experiment connections.

Extends exp_convergence.py with:
  A) New variants: Polyak-optimal A1, decreasing-mu A2
  B) Numerically fitted convergence rates with R^2
  C) Five targeted plots

Outputs:
  results/plots/convergence_enhanced_vs_iter.pdf
  results/plots/convergence_enhanced_rates.pdf
  results/plots/convergence_enhanced_dynamic_mu.pdf
  results/plots/convergence_enhanced_polyak.pdf
  results/plots/convergence_enhanced_sparsity.pdf
  results/logs/convergence_enhanced_rates.csv
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
from src.nesterov_smoothing import nesterov_smoothed, nesterov_smoothed_decreasing
from src.reference import solve_lasso_reference
from src.utils import sutskever_momentum_schedule, polyak_optimal_params
from data.generate_data import generate_single_column_problem
from experiments.plotting import setup_plotting, SWEEP_COLORS


# ======================================================================
# PART A — run all variants
# ======================================================================
def run_all_variants(n=2000, d=50, m=200, lam=0.01, max_iter=5000,
                     seed=42, log_every=10):
    H, y, _ = generate_single_column_problem(n=n, d=d, m=m, seed=seed)
    obj = LassoObjective(H, y, lam=lam)
    w0 = np.zeros(m)
    _, f_ref = solve_lasso_reference(H, y, lam)
    lr = 1.0 / obj.L_g

    eta_star, beta_star = polyak_optimal_params(obj.L_g, obj.mu_g)

    print(f"Problem: n={n}, m={m}, lam={lam}")
    print(f"  L_g={obj.L_g:.4e}, mu_g={obj.mu_g:.4e}, kappa={obj.kappa:.1f}")
    print(f"  f*={f_ref:.6e}")
    if eta_star is not None:
        print(f"  Polyak: eta*={eta_star:.4e} ({eta_star*obj.L_g:.2f}/L_g), beta*={beta_star:.4f}")

    results = {}

    def _run_hb(name, **kw):
        print(f"  Running {name}...")
        w, hist = heavy_ball(obj, w0, max_iter=max_iter, log_every=log_every,
                             tol=1e-10, **kw)
        f = obj.f_total(w)
        print(f"    gap={f - f_ref:.2e}, iters={hist.iterations[-1]}")
        results[name] = hist

    # --- Standard A1 variants ---
    _run_hb('A1-Sub-CM',   lr=lr, variant='CM',  l1_handling='subgradient', momentum=0.9)
    _run_hb('A1-Sub-NAG',  lr=lr, variant='NAG', l1_handling='subgradient', momentum=0.9)
    _run_hb('A1-Prox-CM',  lr=lr, variant='CM',  l1_handling='proximal',   momentum=0.9)
    _run_hb('A1-Prox-NAG', lr=lr, variant='NAG', l1_handling='proximal',   momentum=0.9)

    # --- Sutskever schedule ---
    _run_hb('A1-Prox-NAG-Sched', lr=lr, variant='NAG', l1_handling='proximal',
            momentum_schedule=sutskever_momentum_schedule)

    # --- Two-phase ---
    print(f"  Running A1-Prox-2Phase...")
    iter_p1 = max_iter // 5
    iter_p2 = max_iter - iter_p1
    w, hist = heavy_ball_two_phase(
        obj, w0, lr=lr, max_iter_phase1=iter_p1, max_iter_phase2=iter_p2,
        momentum_phase1=0.9, momentum_phase2=0.0,
        tol=1e-10, variant='NAG', log_every=log_every,
    )
    f = obj.f_total(w)
    print(f"    gap={f - f_ref:.2e}, iters={hist.iterations[-1]}")
    results['A1-Prox-2Phase'] = hist

    # --- Polyak-optimal A1 variants ---
    # Polyak's formula assumes a smooth quadratic. Our objective has L1, so the
    # large eta* can cause divergence. We use beta* (which is safe) and cap eta
    # at 1/L_g so the step size stays in the stable region for the composite problem.
    if eta_star is not None:
        lr_polyak = min(eta_star, lr)
        print(f"  Polyak: using lr={lr_polyak:.4e} (capped), beta*={beta_star:.4f}")
        _run_hb('A1-Sub-NAG-Polyak',  lr=lr_polyak, variant='NAG',
                l1_handling='subgradient', momentum=beta_star)
        _run_hb('A1-Prox-NAG-Polyak', lr=lr_polyak, variant='NAG',
                l1_handling='proximal',    momentum=beta_star)

    # --- A2 fixed-mu variants ---
    for mu_val in [1e-1, 1e-2, 1e-3, 1e-4]:
        name = f'A2-mu={mu_val:.0e}'
        print(f"  Running {name}...")
        smoother = HuberSmoothing(lam=lam, mu=mu_val, dim=m)
        w, hist = nesterov_smoothed(obj, smoother, w0, max_iter=max_iter,
                                    log_every=log_every, tol=1e-10)
        f = obj.f_total(w)
        print(f"    gap={f - f_ref:.2e}, iters={hist.iterations[-1]}")
        results[name] = hist

    # --- A2 decreasing-mu ---
    print(f"  Running A2-decr-mu0=1e-1...")
    w, hist = nesterov_smoothed_decreasing(obj, w0, mu0=0.1, max_iter=max_iter,
                                           log_every=log_every, tol=1e-10)
    f = obj.f_total(w)
    print(f"    gap={f - f_ref:.2e}, iters={hist.iterations[-1]}")
    results['A2-decr-mu0=1e-1'] = hist

    return results, f_ref, obj


# ======================================================================
# PART B — fit convergence rates
# ======================================================================
def fit_rates(results, f_ref):
    """
    A1 variants: fit log(gap) ~ intercept + slope * k  (linear convergence)
    A2 variants: fit log(gap) ~ intercept + exponent * log(k)  (polynomial)
    """
    records = []

    for name, hist in results.items():
        iters = np.array(hist.iterations, dtype=float)
        gaps = np.array([max(f - f_ref, 1e-16) for f in hist.f_total])

        # Skip points at the floor or non-finite
        valid = (gaps > 1e-15) & np.isfinite(gaps)
        iters_v = iters[valid]
        log_gaps = np.log(gaps[valid])

        if len(iters_v) < 10 or not np.all(np.isfinite(log_gaps)):
            continue

        is_a2 = name.startswith('A2')

        if is_a2:
            # For A2 polynomial rate fitting, we need the asymptotic regime.
            # Exclude the steep initial transient (first 30%) and points where
            # the gap has nearly stalled (last 10%). Also require the gap to
            # still be actively decreasing (> 2x the final gap).
            n_pts = len(iters_v)
            final_gap = log_gaps[-1]
            # Find where active convergence happens: gap > 2x final value
            active = log_gaps > final_gap + np.log(2.0)
            if np.sum(active) < 10:
                active = np.ones(n_pts, dtype=bool)
            i_start = max(n_pts // 3, np.argmax(active))
            i_end = n_pts - 1 if np.sum(active) > 10 else n_pts
            # Further trim: use last 50% of active region
            active_indices = np.where(active)[0]
            if len(active_indices) > 20:
                i_start = active_indices[len(active_indices) // 2]
                i_end = active_indices[-1] + 1

            iters_fit = iters_v[i_start:i_end]
            log_gaps_fit = log_gaps[i_start:i_end]

            if len(iters_fit) < 5:
                continue

            # Polynomial fit: log(gap) ~ intercept + exponent * log(k)
            log_k = np.log(np.maximum(iters_fit, 1.0))
            if np.std(log_k) < 1e-12:
                continue
            A = np.vstack([np.ones_like(log_k), log_k]).T
            coef, res, _, _ = np.linalg.lstsq(A, log_gaps_fit, rcond=None)
            intercept, exponent = coef
            # R^2
            ss_res = np.sum((log_gaps_fit - A @ coef) ** 2)
            ss_tot = np.sum((log_gaps_fit - np.mean(log_gaps_fit)) ** 2)
            r2 = 1.0 - ss_res / max(ss_tot, 1e-30)

            theory = "-2.0 (O(1/k^2))" if 'decr' not in name else "-1.0 (O(1/k))"
            records.append({
                'algorithm': name, 'rate_type': 'polynomial',
                'rate_value': exponent, 'R2': r2,
                'theory': theory,
            })
        else:
            # A1 linear rate: use middle 60% of trajectory
            n_pts = len(iters_v)
            i_start = n_pts // 5
            i_end = int(n_pts * 0.8)
            if i_end - i_start < 5:
                i_start = 0
                i_end = n_pts

            iters_fit = iters_v[i_start:i_end]
            log_gaps_fit = log_gaps[i_start:i_end]

            # Linear fit: log(gap) ~ intercept + slope * k
            A = np.vstack([np.ones_like(iters_fit), iters_fit]).T
            coef, res, _, _ = np.linalg.lstsq(A, log_gaps_fit, rcond=None)
            intercept, slope = coef
            rho = np.exp(slope)  # contraction factor per iteration
            # R^2
            ss_res = np.sum((log_gaps_fit - A @ coef) ** 2)
            ss_tot = np.sum((log_gaps_fit - np.mean(log_gaps_fit)) ** 2)
            r2 = 1.0 - ss_res / max(ss_tot, 1e-30)

            records.append({
                'algorithm': name, 'rate_type': 'linear',
                'rate_value': rho, 'R2': r2,
                'theory': 'rho* = (sqrt(kappa)-1)/(sqrt(kappa)+1)',
            })

    return pd.DataFrame(records)


# ======================================================================
# PART C — Plots
# ======================================================================
def _gap_series(hist, f_ref):
    return [max(f - f_ref, 1e-16) for f in hist.f_total]


def plot_all_variants(results, f_ref, save_dir):
    """Plot 1: all variants on one semi-log plot."""
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, (name, hist) in enumerate(results.items()):
        gaps = _gap_series(hist, f_ref)
        ax.semilogy(hist.iterations, gaps, label=name,
                     color=SWEEP_COLORS[i % len(SWEEP_COLORS)], linewidth=1.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'$f(w_k) - f^*$')
    ax.set_title('Enhanced Convergence: All Variants')
    ax.legend(fontsize=7, ncol=3, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/convergence_enhanced_vs_iter.pdf')
    plt.close()
    print(f"  Saved convergence_enhanced_vs_iter.pdf")


def plot_rates(df_rates, obj, save_dir):
    """Plot 2: fitted convergence rates."""
    df_lin = df_rates[df_rates['rate_type'] == 'linear'].copy()
    df_poly = df_rates[df_rates['rate_type'] == 'polynomial'].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: linear rates (contraction factor rho)
    if not df_lin.empty:
        df_lin = df_lin.sort_values('rate_value')
        ax = axes[0]
        bars = ax.barh(range(len(df_lin)), df_lin['rate_value'].values,
                       color=[SWEEP_COLORS[i % len(SWEEP_COLORS)] for i in range(len(df_lin))],
                       alpha=0.8)
        ax.set_yticks(range(len(df_lin)))
        ax.set_yticklabels(df_lin['algorithm'].values, fontsize=8)
        ax.set_xlabel(r'Contraction factor $\rho$ per iteration')
        ax.set_title('A1: Linear Convergence Rates')
        # Theoretical optimal
        sqrt_kappa = np.sqrt(obj.kappa)
        rho_star = (sqrt_kappa - 1) / (sqrt_kappa + 1)
        ax.axvline(rho_star, color='red', linestyle='--', linewidth=1.5,
                   label=rf'$\rho^* = {rho_star:.4f}$')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='x')
        # Annotate R^2
        for idx, (_, row) in enumerate(df_lin.iterrows()):
            ax.text(row['rate_value'] + 0.001, idx,
                    f"R²={row['R2']:.3f}", va='center', fontsize=7)

    # Right: polynomial exponents
    if not df_poly.empty:
        df_poly = df_poly.sort_values('rate_value')
        ax = axes[1]
        colors = [SWEEP_COLORS[i % len(SWEEP_COLORS)] for i in range(len(df_poly))]
        ax.barh(range(len(df_poly)), df_poly['rate_value'].abs().values,
                color=colors, alpha=0.8)
        ax.set_yticks(range(len(df_poly)))
        ax.set_yticklabels(df_poly['algorithm'].values, fontsize=8)
        ax.set_xlabel(r'Exponent $|p|$ in $O(1/k^p)$')
        ax.set_title(r'A2: Polynomial Convergence Exponents')
        ax.axvline(2.0, color='red', linestyle='--', linewidth=1.5,
                   label=r'Theory: $O(1/k^2)$')
        ax.axvline(1.0, color='orange', linestyle='--', linewidth=1.5,
                   label=r'Decr-$\mu$ theory: $O(1/k)$')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='x')
        for idx, (_, row) in enumerate(df_poly.iterrows()):
            ax.text(abs(row['rate_value']) + 0.05, idx,
                    f"R²={row['R2']:.3f}", va='center', fontsize=7)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/convergence_enhanced_rates.pdf')
    plt.close()
    print(f"  Saved convergence_enhanced_rates.pdf")


def plot_dynamic_mu(results, f_ref, save_dir):
    """Plot 3: fixed-mu vs decreasing-mu."""
    fig, ax = plt.subplots(figsize=(10, 6))
    a2_names = [n for n in results if n.startswith('A2')]
    for i, name in enumerate(a2_names):
        hist = results[name]
        gaps = _gap_series(hist, f_ref)
        ls = '-' if 'decr' not in name else '-'
        lw = 2.5 if 'decr' in name else 1.5
        ax.semilogy(hist.iterations, gaps, label=name,
                     color=SWEEP_COLORS[i % len(SWEEP_COLORS)],
                     linewidth=lw, linestyle=ls)
    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'$f(w_k) - f^*$')
    ax.set_title(r'A2: Fixed $\mu$ vs Decreasing $\mu$ Schedule')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/convergence_enhanced_dynamic_mu.pdf')
    plt.close()
    print(f"  Saved convergence_enhanced_dynamic_mu.pdf")


def plot_polyak(results, f_ref, save_dir):
    """Plot 4: hand-tuned vs Polyak-optimal for Sub-NAG and Prox-NAG."""
    pairs = [
        ('Subgradient NAG', 'A1-Sub-NAG', 'A1-Sub-NAG-Polyak'),
        ('Proximal NAG',    'A1-Prox-NAG', 'A1-Prox-NAG-Polyak'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (title, name_ht, name_pk) in zip(axes, pairs):
        if name_ht in results:
            gaps = _gap_series(results[name_ht], f_ref)
            ax.semilogy(results[name_ht].iterations, gaps,
                        label=r'Hand-tuned ($\eta=1/L_g$, $\beta=0.9$)',
                        color=SWEEP_COLORS[0], linewidth=1.5)
        if name_pk in results:
            gaps = _gap_series(results[name_pk], f_ref)
            ax.semilogy(results[name_pk].iterations, gaps,
                        label=r'Polyak optimal ($\eta^*$, $\beta^*$)',
                        color=SWEEP_COLORS[1], linewidth=1.5)
        ax.set_xlabel('Iteration')
        ax.set_ylabel(r'$f(w_k) - f^*$')
        ax.set_title(f'{title}: Hand-Tuned vs Polyak Optimal')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/convergence_enhanced_polyak.pdf')
    plt.close()
    print(f"  Saved convergence_enhanced_polyak.pdf")


def plot_sparsity_high_lambda(save_dir, n=2000, d=50, m=200, lam=5.0,
                              max_iter=5000, seed=42, log_every=10):
    """Plot 5: sparsity vs iteration at lambda=0.1."""
    H, y, _ = generate_single_column_problem(n=n, d=d, m=m, seed=seed)
    obj = LassoObjective(H, y, lam=lam)
    w0 = np.zeros(m)
    lr = 1.0 / obj.L_g

    print(f"\n  Sparsity plot: lam={lam}, L_g={obj.L_g:.4e}")

    runs = {}

    # A1 subgradient
    _, hist = heavy_ball(obj, w0, lr=lr, max_iter=max_iter, variant='NAG',
                         l1_handling='subgradient', momentum=0.9,
                         log_every=log_every, tol=1e-10)
    runs['A1-Sub-NAG'] = hist

    # A1 proximal
    _, hist = heavy_ball(obj, w0, lr=lr, max_iter=max_iter, variant='NAG',
                         l1_handling='proximal', momentum=0.9,
                         log_every=log_every, tol=1e-10)
    runs['A1-Prox-NAG'] = hist

    # A1 two-phase
    iter_p1 = max_iter // 5
    iter_p2 = max_iter - iter_p1
    _, hist = heavy_ball_two_phase(
        obj, w0, lr=lr, max_iter_phase1=iter_p1, max_iter_phase2=iter_p2,
        momentum_phase1=0.9, momentum_phase2=0.0,
        tol=1e-10, variant='NAG', log_every=log_every,
    )
    runs['A1-Prox-2Phase'] = hist

    # A2 fixed
    smoother = HuberSmoothing(lam=lam, mu=1e-3, dim=m)
    _, hist = nesterov_smoothed(obj, smoother, w0, max_iter=max_iter,
                                log_every=log_every, tol=1e-10)
    runs['A2-mu=1e-3'] = hist

    # A2 decreasing
    _, hist = nesterov_smoothed_decreasing(obj, w0, mu0=0.1, max_iter=max_iter,
                                           log_every=log_every, tol=1e-10)
    runs['A2-decr-mu0=1e-1'] = hist

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (name, hist) in enumerate(runs.items()):
        ax.plot(hist.iterations, [s * 100 for s in hist.sparsity],
                label=name, color=SWEEP_COLORS[i % len(SWEEP_COLORS)], linewidth=1.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Sparsity (%)')
    ax.set_title(rf'Sparsity of Iterates vs Iteration ($\lambda = {lam}$)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/convergence_enhanced_sparsity.pdf')
    plt.close()
    print(f"  Saved convergence_enhanced_sparsity.pdf")


# ======================================================================
# Main
# ======================================================================
def run_all():
    setup_plotting()
    plot_dir = 'results/plots'
    log_dir = 'results/logs'
    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ENHANCED CONVERGENCE EXPERIMENT")
    print("=" * 70)

    # Part A
    print("\n--- Part A: Running all variants ---")
    results, f_ref, obj = run_all_variants()

    # Part B
    print("\n--- Part B: Fitting convergence rates ---")
    df_rates = fit_rates(results, f_ref)
    df_rates.to_csv(f'{log_dir}/convergence_enhanced_rates.csv', index=False)
    print(f"\n  Rates saved to {log_dir}/convergence_enhanced_rates.csv")
    print("\n" + "=" * 90)
    print("CONVERGENCE RATE SUMMARY")
    print("=" * 90)
    for _, row in df_rates.iterrows():
        if row['rate_type'] == 'linear':
            print(f"  {row['algorithm']:25s} | rho = {row['rate_value']:.6f} "
                  f"| R² = {row['R2']:.4f} | {row['theory']}")
        else:
            print(f"  {row['algorithm']:25s} | exp = {row['rate_value']:.3f} "
                  f"| R² = {row['R2']:.4f} | {row['theory']}")

    # Part C
    print("\n--- Part C: Generating plots ---")
    plot_all_variants(results, f_ref, plot_dir)
    plot_rates(df_rates, obj, plot_dir)
    plot_dynamic_mu(results, f_ref, plot_dir)
    plot_polyak(results, f_ref, plot_dir)
    plot_sparsity_high_lambda(plot_dir)

    print("\nEnhanced convergence experiment complete.")


if __name__ == '__main__':
    run_all()
