"""
Extreme Learning Machine with fixed random hidden-layer weights.

The ELM maps input X ∈ ℝ^(n×d) to hidden features H ∈ ℝ^(n×m) via:
    H = σ(X · W₁ᵀ)
where W₁ ∈ ℝ^(m×d) is drawn once at construction and never modified.
"""

import numpy as np


class ELM:
    """
    Fixed random feature extractor.

    Parameters
    ----------
    d : int
        Input dimension (number of features).
    m : int
        Number of hidden units.
    activation : {'relu', 'sigmoid'}
        Non-linearity applied to X @ W1.T.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, d: int, m: int, activation: str = 'relu', seed: int = 42):
        rng = np.random.default_rng(seed)

        if activation == 'relu':
            # He initialisation: Var = 2/d
            scale = np.sqrt(2.0 / d)
        elif activation == 'sigmoid':
            # Xavier initialisation: Var = 1/d
            scale = np.sqrt(1.0 / d)
        else:
            raise ValueError(f"Unsupported activation: {activation!r}. Use 'relu' or 'sigmoid'.")

        self.W1 = rng.standard_normal((m, d)) * scale
        self.activation = activation
        self.d = d
        self.m = m

    def compute_H(self, X: np.ndarray) -> np.ndarray:
        """
        Compute hidden features H = σ(X @ W1.T).

        This should be called **once** per dataset; the resulting H is then
        reused throughout optimisation.

        Parameters
        ----------
        X : ndarray of shape (n, d)
            Input data matrix.

        Returns
        -------
        H : ndarray of shape (n, m)
            Hidden-layer activations (float64).
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.d:
            raise ValueError(
                f"Expected X with shape (n, {self.d}), got {X.shape}"
            )

        Z = X @ self.W1.T  # (n, m)

        if self.activation == 'relu':
            return np.maximum(0.0, Z)

        # sigmoid with clipping to prevent overflow
        return 1.0 / (1.0 + np.exp(-np.clip(Z, -500, 500)))
