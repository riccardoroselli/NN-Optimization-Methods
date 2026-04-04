# Extreme Learning Neural Network Optimization Methods

Optimization for Data Science course project — Universita di Pisa, 2025.

## Problem

Train the output layer of an **Extreme Learning Machine (ELM)** by solving:

```
minimize  h(w) = ||Hw - y||^2 + lambda ||w||_1
```

where `H = sigma(X * W1^T)` is a fixed hidden-layer activation matrix.

Two first-order algorithms are implemented and compared:

- **A1 -- Heavy Ball (Polyak Momentum):** subgradient and proximal variants, Classical Momentum (CM) and Nesterov Accelerated Gradient (NAG), plus a two-phase warm-restart strategy.
- **A2 -- Nesterov Smoothing + FISTA:** replaces L1 with a Huber smooth approximation, then applies accelerated gradient descent.

Scale: 10,000+ weights, 1,000+ samples. No off-the-shelf solvers for the optimization itself.

## Project Structure

```
src/                        Core library
  elm.py                    ELM model (W1 generation, H computation)
  objective.py              Lasso objective: f(w), gradient, prox, Lipschitz
  smoothing.py              Huber smoothing of L1
  heavy_ball.py             A1: Heavy Ball optimizer (4 variants + two-phase)
  nesterov_smoothing.py     A2: Nesterov smoothed gradient (FISTA)
  reference.py              sklearn Lasso reference solver
  utils.py                  Schedules, logging, helpers

experiments/                Experiment scripts
  run_all.py                Master runner (all 5 experiments)
  exp_convergence.py        Convergence curves
  exp_params.py             Parameter sensitivity
  exp_scaling.py            Scalability analysis
  exp_comparison.py         Algorithm comparison (multi-seed)
  exp_sparsity.py           Sparsity analysis
  plotting.py               Shared plot config

tests/                      Verification
  test_gradient.py          Finite-difference gradient checks
  test_small_convergence.py Small-scale convergence vs sklearn

results/
  plots/                    Generated PDF figures
  logs/                     CSV experiment logs
```

## Setup

Python 3.10+ required.

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `scipy`, `matplotlib`, `scikit-learn`, `pandas`.

## Reproducing Results

### 1. Run tests (verify correctness)

```bash
# Gradient checks (must pass before any optimization)
python tests/test_gradient.py

# Small-scale convergence test (all variants vs sklearn reference)
python tests/test_small_convergence.py
```

### 2. Run all experiments

```bash
python -m experiments.run_all
```

This runs all 5 experiments sequentially and saves:
- **14 PDF plots** to `results/plots/`
- **12+ CSV logs** to `results/logs/`

To run individual experiments:

```bash
python -m experiments.exp_convergence
python -m experiments.exp_params
python -m experiments.exp_scaling
python -m experiments.exp_comparison
python -m experiments.exp_sparsity
```

### 3. Default configuration

| Parameter | Value | Meaning |
|-----------|-------|---------|
| n | 2000 | Samples |
| d | 50 | Input features |
| m | 200 | Hidden units (= weights per output) |
| p | 50 | Output columns (total weights: 10,000) |
| lambda | 0.01 | L1 regularization strength |
| activation | ReLU | Hidden layer non-linearity |
| seed | 42 | Random seed |

## Algorithm Variants

| Name | Description |
|------|-------------|
| A1-Sub-CM | Heavy Ball, subgradient L1, classical momentum |
| A1-Sub-NAG | Heavy Ball, subgradient L1, Nesterov lookahead |
| A1-Prox-CM | Heavy Ball, proximal L1, classical momentum |
| A1-Prox-NAG | Heavy Ball, proximal L1, Nesterov lookahead |
| A1-Prox-2Phase | Two-phase: high momentum then zero-momentum refinement |
| A2-mu=X | Nesterov smoothing with FISTA, smoothing parameter mu |

## Credits

Chiara Capodagli - c.capodagli@studenti.unipi.it

Roselli Riccardo - r.roselli1@studenti.unipi.it

<p align="center"> Master Degree in Data Science and Business Informatics </p>

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/it/e/e2/Stemma_unipi.svg" width="70"/>
</p>
