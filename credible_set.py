# %%
from tabpfn import TabPFNClassifier, TabPFNRegressor
import warnings
from typing import Callable
import torch

import numpy as np
from scipy.stats import norm
from numpy.random import Generator
from tqdm import trange


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


def assert_ppd_args_shape(x_new, x_prev, y_prev):
    assert x_new.ndim == 2, "x_new must be 2D array"
    assert x_prev.ndim == 2, "x_prev must be 2D array"
    assert y_prev.ndim == 1, "y_prev must be 1D array"
    assert (
        x_prev.shape[0] == y_prev.shape[0]
    ), "x_prev and y_prev must have same number of samples"
    assert (
        x_prev.shape[1] == x_new.shape[1]
    ), "x_prev and x_new must have same number of features"
    assert y_prev.ndim == 1, "y_prev must be 1D array"


class TabPFNRegressorPPD(TabPFNRegressor):

    def __init__(self, *args, y_star: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.y_star = y_star

    def sample(
        self,
        rng: Generator,
        x_new: np.ndarray,
        x_prev: np.ndarray,
        y_prev: np.ndarray,
        size: int = 1,
    ) -> tuple[np.ndarray, dict]:
        # Sample from predictive density
        assert_ppd_args_shape(x_new, x_prev, y_prev)
        self.fit(x_prev, y_prev)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="overflow encountered in cast",
                category=RuntimeWarning,
            )
            pred_output = self.predict(x_new, output_type="full")
        bardist = pred_output["criterion"]
        logits = pred_output["logits"]
        y_new = [self.bardist_sample(rng, bardist.icdf, logits) for _ in range(size)]
        y_new = np.stack(y_new, axis=0)  # (n, num_x_new)

        return y_new, {"bardist": bardist, "logits": logits.cpu().numpy()}

    def bardist_sample(
        self, rng: Generator, bardist_icdf: Callable, logits: torch.Tensor
    ) -> np.ndarray:
        """Samples values from the bar distribution. A modified version of
        https://github.com/PriorLabs/TabPFN/blob/1b786570f5d5da3f3b9b6179c3fa43faf0c77894/src/tabpfn/architectures/base/bar_distribution.py#L581
        Temperature t.
        """
        assert logits.ndim == 2, "logits must be 2D array (num_data, num_of_bins)"
        p_cdf = rng.uniform(size=(logits.shape[0],))
        return np.array(
            [bardist_icdf(logits[i, :], p).cpu() for i, p in enumerate(p_cdf.tolist())]
        )

    def predict_event(
        self,
        x_new: np.ndarray,
        x_prev: np.ndarray,
        y_prev: np.ndarray,
        # y_star: float = 3.0,
    ) -> np.ndarray:
        """
        Return P(Y <= y_star | X = x_new, prev data).

        Parameters
        ----------
        x_new : (m, d) array
            Query covariates.
        x_prev : (n, d) array
            Historical covariates.
        y_prev : (n,) array
            Historical targets.
        y_star : float, default=3.0
            Event threshold.
        """
        assert_ppd_args_shape(x_new, x_prev, y_prev)
        self.fit(x_prev, y_prev)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="overflow encountered in cast",
                category=RuntimeWarning,
            )
            pred_output = self.predict(x_new, output_type="full")

        logits = pred_output["logits"]
        bardist = pred_output["criterion"]

        # Evaluate the predictive CDF at y_star for each query point.
        ys = torch.full(
            (logits.shape[0], 1),
            float(self.y_star),
            dtype=torch.float32,
        )

        bardist.borders = bardist.borders.cpu()
        cdf_vals = bardist.cdf(logits.cpu(), ys).squeeze(-1)
        return cdf_vals.numpy()


# %%


class TabPFNClassifierPPD(TabPFNClassifier):

    def __init__(self, *args, y_star: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.y_star = y_star

    def sample(
        self,
        rng: Generator,
        x_new: np.ndarray,
        x_prev: np.ndarray,
        y_prev: np.ndarray,
        size: int = 1,
    ) -> tuple[np.ndarray, dict]:
        assert_ppd_args_shape(x_new, x_prev, y_prev)
        self.fit(x_prev, y_prev)
        probs_new = self.predict_proba(x_new)

        idx_new = [rng.choice(a=self.classes_.size, p=p, size=size) for p in probs_new]
        y_new = np.stack(
            [self.classes_[idx] for idx in idx_new], axis=1
        )  # (n, num_x_new)

        return y_new, {"probs": probs_new}

    def predict_event(
        self,
        x_new: np.ndarray,
        x_prev: np.ndarray,
        y_prev: np.ndarray,
    ) -> np.ndarray:
        """Return P(Y = 1 | X = x_new, prev data)."""
        assert_ppd_args_shape(x_new, x_prev, y_prev)
        self.fit(x_prev, y_prev)
        probs = self.predict_proba(x_new)

        if self.y_star in self.classes_:
            # Identify the column corresponding to the positive class label.
            class_idx = np.where(self.classes_ == self.y_star)[0]
            event_prob = probs[:, class_idx[0]].squeeze()
        else:
            # If y_star is not supported, return zero probability
            event_prob = np.zeros(probs.shape[0], dtype=probs.dtype)
        return event_prob


# %%


def compute_gn(clf, x_grid, x_prev, y_prev):
    # return gn
    # return a 1d array of shape (m,)
    assert_ppd_args_shape(x_grid, x_prev, y_prev)

    # Guard against degenerate or low data
    if isinstance(clf, TabPFNClassifierPPD):
        if np.min(y_prev) == np.max(y_prev):
            return float(y_prev[0])

    # Guard against degenerate or low data
    if isinstance(clf, TabPFNRegressorPPD):
        if y_prev.shape[0] < 2 or np.unique(y_prev).size < 2:
            return float(np.mean(y_prev <= clf.y_star))

    return clf.predict_event(x_new=x_grid, x_prev=x_prev, y_prev=y_prev)


def sample_gn_plus_1(rng, clf, x_grid, x_prev, y_prev, size=100):
    # draw g_{n+1}
    # return a 2d array of shape (mc_samples, m)
    assert_ppd_args_shape(x_grid, x_prev, y_prev)
    x_plus_1 = x_prev[rng.choice(x_prev.shape[0], size=size, replace=True)]
    y_plus_1, _ = clf.sample(
        rng=rng, x_new=x_plus_1, x_prev=x_prev, y_prev=y_prev, size=1
    )
    y_plus_1 = y_plus_1.squeeze()

    assert x_plus_1.shape[0] == y_plus_1.shape[0]
    assert x_plus_1.shape[0] == size
    assert y_plus_1.ndim == 1
    prob_event = [
        clf.predict_event(
            x_new=x_grid,
            x_prev=np.vstack([x_prev, x_plus_1[i : i + 1]]),
            y_prev=np.hstack([y_prev, y_plus_1[i : i + 1]]),
        )
        for i in trange(size)
    ]
    return np.stack(prob_event, axis=0)


def compute_g0_to_gn(clf, x_grid, x_prev, y_prev):
    """
    Construct the sequence k ↦ v_k(x) mirroring build_g_hat_logreg,
    but leveraging compute_gn for probability evaluation.

    Parameters
    ----------
    clf : TabPFNClassifierPPD-like
        Must expose `fit` and `predict_event`.
    x_grid : (m, d) array
        Grid of covariate values.
    x_prev : (n, d) array
        Historical covariates.
    y_prev : (n,) array
        Historical binary labels.

    Returns
    -------
    g0_to_gn : (n+1, m) array
        g0_to_gn[k, j] = P(Y=1 | X=x_grid[j], z_{1:k}), with g0_to_gn[0,:] = NaN.
    """
    assert_ppd_args_shape(x_grid, x_prev, y_prev)

    n = x_prev.shape[0]
    m = x_grid.shape[0]
    g0_to_gn = np.empty((n + 1, m), dtype=np.float32)
    g0_to_gn[0, :] = np.nan

    for k in trange(1, n + 1):
        if isinstance(clf, TabPFNClassifier):
            y_prefix = y_prev[:k]
            if np.min(y_prefix) == np.max(y_prefix):
                # if all labels are identical, set constant prediction
                g0_to_gn[k, :] = float(y_prefix[0])
                continue

        g0_to_gn[k, :] = compute_gn(
            clf=clf, x_grid=x_grid, x_prev=x_prev[:k], y_prev=y_prev[:k]
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


def build_pointwise_band(mean, cov, alpha: float = 0.05):
    se = np.sqrt(cov)
    z = norm.ppf(1 - alpha / 2)
    # lower = np.clip(mean - z * se, 0.0, 1.0)
    # upper = np.clip(mean + z * se, 0.0, 1.0)
    lower = mean - z * se
    upper = mean + z * se
    return {"mean": mean, "lower": lower, "upper": upper, "se": se}


def build_simultaneous_band(mean, cov, alpha: float = 0.05):
    se = np.sqrt(np.diag(cov))
    # multivariate CLT draws
    rng = np.random.default_rng(501938)
    draws = rng.multivariate_normal(mean, cov, size=1000)
    # sup-norm calibration
    se_safe = se.copy()
    se_safe[se_safe == 0] = np.inf
    Z = (draws - mean[None, :]) / se_safe[None, :]
    T = np.max(np.abs(Z), axis=1)
    c_alpha = float(np.quantile(T, 1 - alpha))
    # lower = np.clip(mean - c_alpha * se, 0.0, 1.0)
    # upper = np.clip(mean + c_alpha * se, 0.0, 1.0)
    lower = mean - c_alpha * se
    upper = mean + c_alpha * se
    return {
        "mean": mean,
        "lower": lower,
        "upper": upper,
        "c_alpha": c_alpha,
        "se": se,
        "draws": draws,
    }
