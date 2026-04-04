"""
Reference solver using sklearn's Lasso for validation.

NOT used as our optimisation algorithm — only for computing f* to
validate our methods and for convergence-gap plots.
"""

import numpy as np
from sklearn.linear_model import Lasso


def solve_lasso_reference(H, y, lam, max_iter=100000):
    """
    Solve  min_w ‖Hw − y‖² + λ‖w‖₁  via sklearn.

    sklearn minimises  (1/(2n)) ‖Hw − y‖² + α ‖w‖₁,
    so we set  α = λ / (2n)  to match our formulation.

    Parameters
    ----------
    H : ndarray (n, m)
    y : ndarray (n,)
    lam : float — our λ
    max_iter : int — sklearn iteration budget

    Returns
    -------
    w_ref : ndarray (m,)
    f_ref : float — our objective evaluated at w_ref
    """
    n = H.shape[0]
    alpha_sklearn = lam / (2.0 * n)

    model = Lasso(
        alpha=alpha_sklearn,
        fit_intercept=False,
        max_iter=max_iter,
        tol=1e-12,
    )
    model.fit(H, y)
    w_ref = model.coef_

    # Evaluate *our* objective at the reference solution
    r = H @ w_ref - y
    f_ref = float(r @ r) + lam * np.sum(np.abs(w_ref))

    return w_ref, f_ref
