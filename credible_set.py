
from pred_rule import TabPFNClassifierPPD, TabPFNRegressorPPD, assert_ppd_args_shape
import warnings
from typing import Callable
import torch

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.stats import norm
from scipy.stats import chi2
from scipy.special import gammaln

import numpy as np
from tqdm import trange


def compute_gn(clf, t, x_grid, x_prev, y_prev):
    # Compute g_n = P(Y <= t | X=x_grid, x_prev, y_prev) or P(Y = t | X=x_grid, x_prev, y_prev)
    # t: (p,) array
    # x_grid: (m, d) array
    # x_prev: (n, d) array
    # y_prev: (n,) array
    # Return a 2d array of shape (p, m)
    assert_ppd_args_shape(x_grid, x_prev, y_prev)

    t = np.atleast_1d(t)
    m = x_grid.shape[0]

    # Guard against degenerate or low data
    if isinstance(clf, TabPFNClassifierPPD):
        if np.min(y_prev) == np.max(y_prev):
            # if all y_prev are the same, then g_n = 1 if t == y_prev[0], 0 otherwise
            probs = (t == y_prev[0]).astype(np.float32)
            return np.broadcast_to(probs[:, None], (t.shape[0], m))

    # Guard against degenerate or low data
    if isinstance(clf, TabPFNRegressorPPD):
        if y_prev.shape[0] < 2 or np.unique(y_prev).size < 2:
            # if all y_prev are the same, then g_n = 1 if t >= y_prev[0], 0 otherwise
            probs = (t >= y_prev[0]).astype(np.float32)
            return np.broadcast_to(probs[:, None], (t.shape[0], m))

    return clf.predict_event(t=t, x_new=x_grid, x_prev=x_prev, y_prev=y_prev)


def sample_gn_plus_1(key, clf, t, x_grid, x_prev, y_prev, size=100):
    # Draw P(Y_{n+2} <= t | X=x_grid, X_{n+1}, Y_{n+1}, x_prev, y_prev) for computing g_{n+1}. X_{n+1} is Bayesian bootstrap. Y_{n+1} is sampled from the PPD.
    # Return a 3d array of shape (mc_samples, p, m)
    assert_ppd_args_shape(x_grid, x_prev, y_prev)
    key_choice, key_sample = jr.split(key)
    idx = jr.choice(key_choice, x_prev.shape[0], shape=(size,), replace=True)
    x_plus_1 = x_prev[idx]
    y_plus_1, _ = clf.sample(
        key=key_sample, x_new=x_plus_1, x_prev=x_prev, y_prev=y_prev, size=1
    )
    y_plus_1 = np.array(y_plus_1).squeeze()

    assert x_plus_1.shape[0] == y_plus_1.shape[0]
    assert x_plus_1.shape[0] == size
    assert y_plus_1.ndim == 1
    prob_event = [
        clf.predict_event(
            t=t,
            x_new=x_grid,
            x_prev=np.vstack([x_prev, x_plus_1[i : i + 1]]),
            y_prev=np.hstack([y_prev, y_plus_1[i : i + 1]]),
        )
        for i in trange(size)
    ]
    return np.stack(prob_event, axis=0)


def compute_g0_to_gn(clf, t, x_grid, x_prev, y_prev):
    """
    Construct the sequence k ↦ v_k(x) mirroring build_g_hat_logreg,
    but leveraging compute_gn for probability evaluation.

    Parameters
    ----------
    clf : TabPFNClassifierPPD or TabPFNRegressorPPD
        Must expose `fit` and `predict_event`.
    t : (p,) array
        Events of the PPD.
    x_grid : (m, d) array
        Grid of covariate values.
    x_prev : (n, d) array
        Historical covariates.
    y_prev : (n,) array
        Historical binary labels.

    Returns
    -------
    g0_to_gn : (n+1, p, m) array
        If clf is TabPFNClassifierPPD, g0_to_gn[i, j, k] = P(Y = t[j] | X=x_grid[k], z_{1:i}).
        If clf is TabPFNRegressorPPD, g0_to_gn[i, j, k] = P(Y <= t[j] | X=x_grid[k], z_{1:i}).
        g0_to_gn[0, :, :] = NaN.
    """
    assert_ppd_args_shape(x_grid, x_prev, y_prev)

    n = x_prev.shape[0]
    m = x_grid.shape[0]
    g0_to_gn = np.empty((n + 1, t.shape[0], m), dtype=np.float32)
    g0_to_gn[0, :, :] = np.nan

    for i in trange(1, n + 1):
        g0_to_gn[i, :, :] = compute_gn(
            clf=clf, t=t, x_grid=x_grid, x_prev=x_prev[:i], y_prev=y_prev[:i]
        )

    return g0_to_gn


def compute_un(gn, gn_plus_1, n, type="simultaneous"):
    # the red one
    assert gn_plus_1.ndim == 2, "gn_plus_1 must be 2D array (mc_samples, m)"
    assert gn.ndim == 1, "gn must be 1D array (m,)"
    assert gn_plus_1.shape[1] == gn.shape[0], "gn_plus_1 and gn shape mismatch"

    diff = gn_plus_1 - gn  # (mc_samples, m)
    if type == "pointwise":
        # return shape
        return np.mean(((n + 1) * diff) ** 2, axis=0)  # (m,)
    elif type == "simultaneous":
        outer = np.einsum("ij,ik->ijk", diff, diff)
        return np.mean((n + 1) ** 2 * outer, axis=0)  # (m, m)


def compute_vn(g0_to_gn, type="simultaneous"):
    # the original
    assert g0_to_gn.ndim == 2, "g0_to_gn must be 2D array (n+1, m)"

    n = g0_to_gn.shape[0] - 1
    delta = g0_to_gn[2:, :] - g0_to_gn[1:-1, :]  # (n-1, m)
    k = np.arange(2, n + 1)  # (n-1,)

    if type == "pointwise":
        # v_n(x_j) = (1/(n-1)) * sum k^2 * Δ_k(x_j)^2
        return np.mean((k[:, None] * delta) ** 2, axis=0)
    elif type == "simultaneous":
        # v_n(x) = (1/(n-1)) * sum k^2 * Δ_k(x) Δ_k(x)^T
        outer = np.einsum("ij,ik->ijk", delta, delta)  # (n-1, m, m)
        return np.mean(k[:, None, None] ** 2 * outer, axis=0)  # (m, m)


def compute_pointwise_coverage(true_curve, bands):
    # check if each point in the grid is covered, then average over grid points
    intervals = [(b["lower"], b["upper"]) for b in bands]
    for l, u in intervals:
        assert true_curve.shape == l.shape == u.shape

    if any(np.any(np.isnan(l)) or np.any(np.isnan(u)) for l, u in intervals):
        return np.nan

    is_covered = [(true_curve >= l) & (true_curve <= u) for (l, u) in intervals]
    return np.mean(np.asarray(is_covered))


def compute_simultaneous_coverage(true_curve, bands):
    # coverage of the entire curve
    intervals = [(b["lower"], b["upper"]) for b in bands]
    for l, u in intervals:
        assert true_curve.shape == l.shape == u.shape

    if any(np.any(np.isnan(l)) or np.any(np.isnan(u)) for l, u in intervals):
        return np.nan

    is_covered = [np.all((true_curve >= l) & (true_curve <= u)) for (l, u) in intervals]
    return np.mean(is_covered)


def build_pointwise_band(mean, cov, alpha: float = 0.05):
    se = np.sqrt(cov)
    z = norm.ppf(1 - alpha / 2)
    # lower = np.clip(mean - z * se, 0.0, 1.0)
    # upper = np.clip(mean + z * se, 0.0, 1.0)
    lower = mean - z * se
    upper = mean + z * se
    width = 2 * z * se
    return {"mean": mean, "lower": lower, "upper": upper, "se": se, "width": width}


def build_simultaneous_band(mean, cov, alpha: float = 0.05):
    # See Algorithm 1 of https://doi.org/10.1002/jae.2656
    se = np.sqrt(np.diag(cov))

    key = jr.key(501938)
    draws = jr.multivariate_normal(key, jnp.zeros_like(mean), cov, shape=(1000,))

    # Handle division by zero safely in JAX
    se_safe = jnp.where(se == 0, jnp.inf, se)

    Z = draws / se_safe[None, :]
    T = jnp.max(jnp.abs(Z), axis=1)
    c_alpha = jnp.quantile(T, 1 - alpha)
    lower = mean - c_alpha * se
    upper = mean + c_alpha * se
    width = jnp.mean(2 * c_alpha * se)

    return {
        "mean": mean,
        "lower": lower,
        "upper": upper,
        "c_alpha": c_alpha,
        "se": se,
        "draws": draws,
        "width": width,
    }


def compute_ellipsoid_log_volume(cov, radius):
    # compute log of the volume of a high-dimensional ellipsoid defined by radius^2 > x^T cov^{-1} x
    d = cov.shape[0]
    log_unit_ball = (d / 2) * np.log(np.pi) - gammaln(d / 2 + 1)

    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        return -np.inf  # Or raise error: Covariance must be positive definite

    return log_unit_ball + 0.5 * logdet + d * np.log(radius)


def build_ellipsoid_band(mean, cov, alpha: float = 0.05):
    # given a multivariate normal defined by mean and cov, compute the ellipsoid
    # such that the volume of the ellipsoid is 1-alpha

    d = mean.shape[0]
    # The squared radius corresponding to probability mass 1-alpha
    # is the quantile of the chi-squared distribution with d degrees of freedom.
    radius_sq = chi2.ppf(1 - alpha, df=d)
    radius = np.sqrt(radius_sq)

    log_vol = compute_ellipsoid_log_volume(cov, radius)

    # This is the projection of the ellipsoid to the coordinate axes
    se = np.sqrt(np.diag(cov))
    delta = se * radius
    lower = mean - delta
    upper = mean + delta

    return {
        "mean": mean,
        "lower": lower,
        "upper": upper,
        "radius": radius,
        "log_volume": log_vol,
    }


# def build_g_hat_logreg(clf, X, y, x_grid):
#     """
#     Construct the sequence k ↦ g_k(x) of predictive probabilities on a finite grid.

#     Returns
#     -------
#     g_hat : (n+1, m) array
#         Row k corresponds to prefix length k (training on the first k samples).
#         Column j corresponds to grid point x_j.
#         Entry g_hat[k, j] is g_k(x_j) = P(Y=1 | X=x_j, z_{1:k}).
#         Row 0 is set to NaN because g_0 is undefined in practice.

#     Notes
#     -----
#     - If all labels in the current prefix are identical (all 0s or all 1s),
#       we **do not call TabPFN**. Instead, we set a constant prediction at every
#       grid point: 0 or 1 respectively. This avoids fitting a classifier on a
#       single-class sample.
#     - Otherwise, TabPFN is fit at each k=1..n with the given `n_estimators`.

#     Parameters
#     ----------
#     clf: Initialized TabPFNClassifier
#     X : array-like of shape (n, d)
#         Training covariates.
#     y : array-like of shape (n,)
#         Binary training labels {0,1}.
#     x_grid : array-like of shape (m, d)
#         Grid of covariate values where predictions are evaluated.
#     """
#     assert np.issubdtype(
#         y.dtype, np.integer
#     ), "y must be integer array for binary classification"
#     assert set(np.unique(y)).issubset({0, 1}), "y must be binary {0,1}"
#     assert isinstance(clf, TabPFNClassifier), "clf must be TabPFNClassifier instance"

#     n, m = len(X), len(x_grid)
#     g_hat = np.empty((n + 1, m), np.float32)
#     g_hat[0, :] = np.nan  # explicit (k,j) indexing

#     for k in trange(1, n + 1):
#         Xi, yi = X[:k], y[:k]
#         if yi.min() == yi.max():
#             g_hat[k, :] = float(yi[0])  # constant 0 or 1 at all x_j
#         else:
#             clf.fit(Xi, yi)
#             g_hat[k, :] = clf.predict_proba(x_grid)[:, 1]  # extract P(Y=1 | X=x_j)

#     return g_hat


# def build_g_hat_linreg(clf, X, y, x_grid, y_star):
#     """
#     g_hat[k, j] = P(Y <= y_star | X = x_grid[j], data z_{1:k}) via TabPFNRegressor.

#     Strategy:
#       - For each prefix k = 1..n, fit TabPFNRegressor on (X[:k], y[:k]).
#       - Query the predictive bar distribution at x_grid.
#       - Use BarDistribution.cdf() to evaluate P(Y <= y_star | X = x_j).
#       - Guards: If k < 2 or y[:k] has no variation, fall back to empirical CDF.

#     Parameters
#     ----------
#     X : array-like, shape (n,d)
#     y : array-like, shape (n,)
#     x_grid : array-like, shape (m,d)
#     y_star : float
#         Threshold for event {Y <= y_star}
#     n_estimators : int, default=64
#     device : str, default=DEVICE
#     seed : int or None

#     Returns
#     -------
#     g_hat : (n+1, m) array
#         g_hat[k,j] = P(Y <= y_star | x_grid[j], z_{1:k}), with g_hat[0,:] = NaN
#     """
#     assert np.issubdtype(y.dtype, np.floating), "y must be float array for regression"
#     assert X.ndim == 2, "X must be 2D array"
#     assert x_grid.ndim == 2, "x_grid must be 2D array"
#     assert (
#         X.shape[1] == x_grid.shape[1]
#     ), "X and x_grid must have same number of features"
#     assert isinstance(clf, TabPFNRegressor), "clf must be TabPFNRegressor instance"

#     # coerce shapes
#     # X = np.asarray(X, np.float32); X = X[:, None] if X.ndim == 1 else X
#     # xg = np.asarray(x_grid, np.float32); xg = xg[:, None] if xg.ndim == 1 else xg
#     y = np.asarray(y, np.float32)
#     y_star = float(y_star)

#     n, m = len(X), len(x_grid)
#     g_hat = np.empty((n + 1, m), dtype=np.float32)
#     g_hat[0, :] = np.nan

#     def empirical_cdf(k: int) -> float:
#         return float(np.mean(y[:k] <= y_star)) if k > 0 else np.nan

#     for k in trange(1, n + 1):
#         # Guard for very small/degenerate prefixes
#         if k < 2 or np.unique(y[:k]).size < 2:
#             g_hat[k, :] = empirical_cdf(k)
#             continue

#         # Fit TabPFNRegressor
#         clf.fit(X[:k], y[:k])
#         out = clf.predict(x_grid, output_type="full")

#         logits = torch.as_tensor(
#             out["logits"], dtype=torch.float32, device=torch.device("cpu")
#         )  # (m,B)
#         bardist = out["criterion"]

#         # --- minimal device alignment for cdf ---
#         # cdf_dev = getattr(bardist.borders, "device", torch.device("cpu"))
#         # ys = torch.full((logits.shape[0], 1), float(y_star), dtype=torch.float32, device=cdf_dev)
#         bardist.borders = bardist.borders.to(torch.device("cpu"))
#         ys = torch.full((logits.shape[0], 1), float(y_star))
#         cdf_vals = bardist.cdf(logits, ys).squeeze(-1)

#         g_hat[k, :] = cdf_vals.numpy()

#     return g_hat

# %%
