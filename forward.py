# %%
from tabpfn import TabPFNClassifier, TabPFNRegressor
import warnings
from typing import Callable
import torch

import numpy as np
from tqdm import trange


# class TabPFNRegresorPPD(TabPFNRegressor):

#     def __init__(
#         self,
#         categorical_x: list[bool],
#         n_estimators: int = 8,  # this is the default in 2.1.3
#         average_before_softmax: bool = False,
#     ):
#         categorical_features_indices = [i for i, c in enumerate(categorical_x) if c]
#         super().__init__(
#             n_estimators=n_estimators,
#             average_before_softmax=average_before_softmax,
#             softmax_temperature=1.0,
#             categorical_features_indices=categorical_features_indices,
#             fit_mode="low_memory",
#         )

# def sample(
#     self, key: KeyArray, x_new: np.ndarray, x_prev: np.ndarray, y_prev: np.ndarray
# ) -> tuple[np.ndarray, dict]:
#     # Sample from predictive density
#     self.fit(x_prev, y_prev)
#     with warnings.catch_warnings():
#         warnings.filterwarnings(
#             "ignore",
#             message="overflow encountered in cast",
#             category=RuntimeWarning,
#         )
#         pred_output = self.predict(x_new, output_type="full")
#     bardist = pred_output["criterion"]
#     y_new = self.bardist_sample(key, bardist.icdf, pred_output["logits"])
#     del pred_output["criterion"]
#     pred_output["logits"] = pred_output["logits"].numpy()
#     return np.squeeze(y_new), pred_output

# def bardist_sample(
#     self, key: KeyArray, bardist_icdf: Callable, logits: np.ndarray, t: float = 1.0
# ) -> np.ndarray:
#     """Samples values from the bar distribution. A modified version of
#     https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/model/bar_distribution.py#L576

#     Temperature t.
#     """
#     p_cdf = jax.random.uniform(key, shape=logits.shape[:-1])
#     return np.array(
#         [bardist_icdf(logits[i, :] / t, p) for i, p in enumerate(p_cdf.tolist())],
#     )


# class TabPFNClassifierPPD(TabPFNClassifier):

#     def __init__(
#         self,
#         categorical_x: list[bool],
#         n_estimators: int = 8,  # this is the default in 2.1.3
#         average_before_softmax: bool = False,
#     ):
#         categorical_features_indices = [i for i, c in enumerate(categorical_x) if c]
#         super().__init__(
#             n_estimators=n_estimators,
#             average_before_softmax=average_before_softmax,
#             softmax_temperature=1.0,
#             categorical_features_indices=categorical_features_indices,
#             fit_mode="low_memory",
#         )

# def sample(
#     self, key: KeyArray, x_new: np.ndarray, x_prev: np.ndarray, y_prev: np.ndarray
# ) -> tuple[np.ndarray, dict]:
#     self.fit(x_prev, y_prev)
#     probs_new = self.predict_proba(x_new).squeeze()
#     idx_new = jax.random.choice(key, a=self.classes_.size, p=probs_new)
#     y_new = self.classes_[idx_new]

#     # we use jax to sample from a categorical distribution in the PPD
#     # resampling step.
#     y_new = y_new.squeeze() if isinstance(y_new, np.ndarray) else y_new
#     return y_new, {"probs": probs_new}


def build_g_hat_logreg(clf, X, y, x_grid):
    """
    Construct the sequence k ↦ g_k(x) of predictive probabilities on a finite grid.

    Returns
    -------
    g_hat : (n+1, m) array
        Row k corresponds to prefix length k (training on the first k samples).
        Column j corresponds to grid point x_j.
        Entry g_hat[k, j] is g_k(x_j) = P(Y=1 | X=x_j, z_{1:k}).
        Row 0 is set to NaN because g_0 is undefined in practice.

    Notes
    -----
    - If all labels in the current prefix are identical (all 0s or all 1s),
      we **do not call TabPFN**. Instead, we set a constant prediction at every
      grid point: 0 or 1 respectively. This avoids fitting a classifier on a
      single-class sample.
    - Otherwise, TabPFN is fit at each k=1..n with the given `n_estimators`.

    Parameters
    ----------
    clf: Initialized TabPFNClassifier
    X : array-like of shape (n, d)
        Training covariates.
    y : array-like of shape (n,)
        Binary training labels {0,1}.
    x_grid : array-like of shape (m, d)
        Grid of covariate values where predictions are evaluated.
    """
    assert np.issubdtype(
        y.dtype, np.integer
    ), "y must be integer array for binary classification"
    assert set(np.unique(y)).issubset({0, 1}), "y must be binary {0,1}"
    assert isinstance(clf, TabPFNClassifier), "clf must be TabPFNClassifier instance"

    n, m = len(X), len(x_grid)
    g_hat = np.empty((n + 1, m), np.float32)
    g_hat[0, :] = np.nan  # explicit (k,j) indexing

    for k in trange(1, n + 1):
        Xi, yi = X[:k], y[:k]
        if yi.min() == yi.max():
            g_hat[k, :] = float(yi[0])  # constant 0 or 1 at all x_j
        else:
            clf.fit(Xi, yi)
            g_hat[k, :] = clf.predict_proba(x_grid)[:, 1]  # extract P(Y=1 | X=x_j)

    return g_hat


def build_g_hat_linreg(clf, X, y, x_grid, y_star):
    """
    g_hat[k, j] = P(Y <= y_star | X = x_grid[j], data z_{1:k}) via TabPFNRegressor.

    Strategy:
      - For each prefix k = 1..n, fit TabPFNRegressor on (X[:k], y[:k]).
      - Query the predictive bar distribution at x_grid.
      - Use BarDistribution.cdf() to evaluate P(Y <= y_star | X = x_j).
      - Guards: If k < 2 or y[:k] has no variation, fall back to empirical CDF.

    Parameters
    ----------
    X : array-like, shape (n,d)
    y : array-like, shape (n,)
    x_grid : array-like, shape (m,d)
    y_star : float
        Threshold for event {Y <= y_star}
    n_estimators : int, default=64
    device : str, default=DEVICE
    seed : int or None

    Returns
    -------
    g_hat : (n+1, m) array
        g_hat[k,j] = P(Y <= y_star | x_grid[j], z_{1:k}), with g_hat[0,:] = NaN
    """
    assert np.issubdtype(y.dtype, np.floating), "y must be float array for regression"
    assert X.ndim == 2, "X must be 2D array"
    assert x_grid.ndim == 2, "x_grid must be 2D array"
    assert (
        X.shape[1] == x_grid.shape[1]
    ), "X and x_grid must have same number of features"
    assert isinstance(clf, TabPFNRegressor), "clf must be TabPFNRegressor instance"

    # coerce shapes
    # X = np.asarray(X, np.float32); X = X[:, None] if X.ndim == 1 else X
    # xg = np.asarray(x_grid, np.float32); xg = xg[:, None] if xg.ndim == 1 else xg
    y = np.asarray(y, np.float32)
    y_star = float(y_star)

    n, m = len(X), len(x_grid)
    g_hat = np.empty((n + 1, m), dtype=np.float32)
    g_hat[0, :] = np.nan

    def empirical_cdf(k: int) -> float:
        return float(np.mean(y[:k] <= y_star)) if k > 0 else np.nan

    for k in trange(1, n + 1):
        # Guard for very small/degenerate prefixes
        if k < 2 or np.unique(y[:k]).size < 2:
            g_hat[k, :] = empirical_cdf(k)
            continue

        # Fit TabPFNRegressor
        clf.fit(X[:k], y[:k])
        out = clf.predict(x_grid, output_type="full")

        logits = torch.as_tensor(
            out["logits"], dtype=torch.float32, device=torch.device("cpu")
        )  # (m,B)
        bardist = out["criterion"]

        # --- minimal device alignment for cdf ---
        # cdf_dev = getattr(bardist.borders, "device", torch.device("cpu"))
        # ys = torch.full((logits.shape[0], 1), float(y_star), dtype=torch.float32, device=cdf_dev)
        bardist.borders = bardist.borders.to(torch.device("cpu"))
        ys = torch.full((logits.shape[0], 1), float(y_star))
        cdf_vals = bardist.cdf(logits, ys).squeeze(-1)

        g_hat[k, :] = cdf_vals.numpy()

    return g_hat

