# Optimization for Data Science — Project 12

First-order optimization of an L1-regularised Extreme Learning Machine output layer.

## Problem

Given a fixed ELM hidden-layer activation matrix `H = σ(X·W₁ᵀ)` (with `W₁` drawn once and never updated), solve

```
minimize  h(w) = ||Hw − y||² + λ||w||₁
```

The smooth part `g(w) = ||Hw − y||²` is convex and quadratic; the non-smooth part `r(w) = λ||w||₁` is handled by subgradient, proximal, or Huber-smoothing strategies, depending on the algorithm.

## Algorithms

| Name | Algorithm | L1 handling | Momentum |
|------|-----------|-------------|----------|
| A1-Sub-CM | Heavy Ball | Subgradient | Classical (β = 0.9) |
| A1-Sub-NAG | Heavy Ball | Subgradient | Nesterov lookahead (β = 0.9) |
| A1-Prox-CM | Heavy Ball | Proximal (soft-threshold) | Classical (β = 0.9) |
| A1-Prox-NAG | Heavy Ball | Proximal (soft-threshold) | Nesterov lookahead (β = 0.9) |
| A1-Prox-NAG-Sched | Heavy Ball | Proximal | Sutskever (2013) schedule |
| A1-Prox-2Phase | Heavy Ball | Proximal | Phase 1 β = 0.9 → Phase 2 β = 0 |
| A2 (fixed μ) | FISTA on Huber-smoothed h | Smoothed (Huber) | FISTA extrapolation |
| A2 (decreasing μ) | FISTA on Huber-smoothed h | Smoothed (Huber, μₖ = μ₀/(k+1)) | FISTA extrapolation |

## Project structure

```
src/                          Core library (7 modules)
  elm.py                      ELM model (W₁ generation, H computation)
  objective.py                Lasso objective: values, gradient, prox, L_g, μ_g
  smoothing.py                Huber smoothing of the L1 penalty
  heavy_ball.py               A1: Heavy Ball optimiser (+ two-phase variant)
  nesterov_smoothing.py       A2: FISTA on Huber-smoothed objective (fixed/decreasing μ)
  reference.py                sklearn Lasso reference solver (for f*)
  utils.py                    OptimizationLog, schedules, stopping criterion

data/
  generate_data.py            Synthetic ELM regression data generator

experiments/                  Experiment scripts (6 + shared utilities)
  run_all.py                  Master runner — runs all 6 experiments
  exp_convergence.py          Exp 1: convergence curves, fitted rates, Polyak params, dynamic μ
  exp_params.py               Exp 2: parameter sensitivity (η, β, μ, λ)
  exp_scaling.py              Exp 3: scalability vs problem size
  exp_comparison.py           Exp 4: multi-seed head-to-head comparison
  exp_sparsity.py             Exp 5: sparsity vs regularisation strength
  exp_full_problem.py         Exp 6: full 10,000-weight multi-column problem
  plotting.py                 Shared matplotlib configuration

tests/                        Verification (3 files)
  test_gradient.py            Finite-difference gradient checks
  test_objective.py           LassoObjective unit tests
  test_small_convergence.py   All variants vs sklearn reference on a small problem

results/
  plots/                      Generated PDF figures (21)
  logs/                       CSV experiment logs (18)

report/                       LaTeX report sources and figures
```

## Setup

Python 3.10+ required.

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `scipy`, `matplotlib`, `scikit-learn`, `pandas`.

## Reproducing results

### Step 1 — Run tests

```bash
python tests/test_gradient.py
python tests/test_objective.py
python tests/test_small_convergence.py
```

- `test_gradient.py` — finite-difference checks for ∇g and the Huber gradient.
- `test_objective.py` — spectral constants, composite decomposition, prox, subgradient.
- `test_small_convergence.py` — every A1/A2 variant converges to within ≤ 2 % of the sklearn reference on an `n=50, m=10` problem.

### Step 2 — Run all experiments

```bash
python -m experiments.run_all
```

Runs all **6 experiments** sequentially, saving plots to `results/plots/` and CSV logs to `results/logs/`.

To run experiments individually:

```bash
python -m experiments.exp_convergence          # Exp 1: convergence curves, fitted rates, Polyak params, dynamic μ
python -m experiments.exp_params               # Exp 2: parameter sensitivity (η, β, μ, λ)
python -m experiments.exp_scaling              # Exp 3: scalability vs problem size
python -m experiments.exp_comparison           # Exp 4: head-to-head comparison, 5 seeds
python -m experiments.exp_sparsity             # Exp 5: sparsity vs regularisation strength
python -m experiments.exp_full_problem         # Exp 6: full 10,000-weight multi-column problem
```

## Default configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| n | 2000 | Training samples |
| d | 50 | Input features |
| m | 200 | Hidden units (weights per output column) |
| p | 50 | Output columns |
| Total weights | 10,000 | p × m, minimum required by project spec |
| λ | 0.01 | L1 regularisation strength |
| Activation | ReLU | Hidden-layer non-linearity |
| seed | 42 | Global random seed |
| L_g | ≈ 2.69 × 10⁵ | Lipschitz constant of ∇g |
| f* | ≈ 19.005 | Reference objective (sklearn Lasso, tol = 1e-12) |

## Reference solution

`src/reference.py` wraps `sklearn.linear_model.Lasso` with `fit_intercept=False`, `tol=1e-12`, and the scaling conversion `α = λ / (2n)` required to match sklearn's internal `(1/(2n))·‖Hw−y‖² + α·‖w‖₁` formulation to our own `‖Hw−y‖² + λ·‖w‖₁`. The reported `f*` is evaluated on our objective (not sklearn's internal loss) and is used only to compute the optimality gap `f(w_k) − f*` for convergence plots — it is never used as a solver for the algorithms themselves.

## Credits

Chiara Capodagli — c.capodagli@studenti.unipi.it
Roselli Riccardo  — r.roselli1@studenti.unipi.it

Optimization for Data Science
Master Degree in Data Science and Business Informatics
Università di Pisa, 2026
