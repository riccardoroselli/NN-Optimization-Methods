"""
Generate synthetic data for ELM experiments.

Default configuration:
    n = 2000   (samples)
    d = 50     (input features)
    m = 200    (hidden units)
    p = 50     (output dimensions)
"""

import numpy as np
from src.elm import ELM


def generate_synthetic_data(
    n: int = 2000,
    d: int = 50,
    m: int = 200,
    p: int = 50,
    sparsity: float = 0.7,
    noise_std: float = 0.1,
    seed: int = 42,
):
    """
    Generate synthetic regression data with known sparse ground truth.

    Returns
    -------
    X : ndarray (n, d)
    Y : ndarray (n, p)
    W2_true : ndarray (p, m) — ground truth (for reference only)
    """
    rng = np.random.default_rng(seed)

    X = rng.standard_normal((n, d))

    # Ground truth W2 with controlled sparsity
    W2_true = rng.standard_normal((p, m))
    mask = rng.random((p, m)) < sparsity
    W2_true[mask] = 0.0

    # Hidden layer (same ELM seed as will be used in optimisation)
    elm = ELM(d=d, m=m, activation='relu', seed=seed)
    H = elm.compute_H(X)

    # Targets: Y = H @ W2_true.T + noise
    Y = H @ W2_true.T + noise_std * rng.standard_normal((n, p))

    return X, Y, W2_true


def generate_single_column_problem(
    n: int = 2000,
    d: int = 50,
    m: int = 200,
    sparsity: float = 0.7,
    noise_std: float = 0.1,
    seed: int = 42,
):
    """
    Generate a single-column Lasso problem (H, y, w_true).

    Used by single-column experiments (convergence, params, etc.).

    Returns
    -------
    H : ndarray (n, m)
    y : ndarray (n,)
    w_true : ndarray (m,)
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))

    elm = ELM(d=d, m=m, activation='relu', seed=seed)
    H = elm.compute_H(X)

    w_true = rng.standard_normal(m)
    w_true[rng.random(m) < sparsity] = 0.0
    y = H @ w_true + noise_std * rng.standard_normal(n)

    return H, y, w_true
