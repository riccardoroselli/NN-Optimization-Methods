"""
Master experiment runner — runs all experiments in sequence.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time

from experiments.exp_convergence import run_all as run_convergence
from experiments.exp_params import run_all_param_experiments
from experiments.exp_scaling import run_all as run_scaling
from experiments.exp_comparison import run_all as run_comparison
from experiments.exp_sparsity import run_all as run_sparsity
from experiments.exp_full_problem import run_all as run_full_problem


def main():
    t0 = time.time()

    # --- Experiment 1: Convergence (all variants, rates, Polyak, dynamic mu) ---
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: CONVERGENCE (ALL VARIANTS + RATES + POLYAK + DYNAMIC MU)")
    print("=" * 70)
    run_convergence()

    # --- Experiment 2: Parameter Sensitivity ---
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: PARAMETER SENSITIVITY")
    print("=" * 70)
    run_all_param_experiments()

    # --- Experiment 3: Scalability ---
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: SCALABILITY")
    print("=" * 70)
    run_scaling()

    # --- Experiment 4: Algorithm Comparison ---
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: ALGORITHM COMPARISON")
    print("=" * 70)
    run_comparison()

    # --- Experiment 5: Sparsity Analysis ---
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: SPARSITY ANALYSIS")
    print("=" * 70)
    run_sparsity()

    # --- Experiment 6: Full Multi-Column Problem ---
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: FULL MULTI-COLUMN PROBLEM (10,000 WEIGHTS)")
    print("=" * 70)
    run_full_problem()

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"ALL EXPERIMENTS COMPLETE in {elapsed:.1f}s")
    print("=" * 70)
    print("  Plots: results/plots/")
    print("  Logs:  results/logs/")


if __name__ == '__main__':
    main()
