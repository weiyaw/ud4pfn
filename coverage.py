# %%
import argparse
from loguru import logger
import sys

import numpy as np
from numpy.random import Generator

from scipy.stats import norm
import torch
from timeit import default_timer as timer
from tabpfn import TabPFNRegressor
import utils
import forward
import os


def clt_pointwise_band(g_hat, alpha: float = 0.05):
    """
    Pointwise (univariate) CLT bands for \tilde g(x) at each grid point.

    Notation (matches LaTeX)
    ------------------------
    - g_hat[k, j] = g_k(x_j) ∈ [0,1], where k=0..n and j=1..m.
      Row k is the predictive probability after seeing the first k data points.
      Column j corresponds to grid point x_j.
    - Define increments Δ_k(x_j) = g_k(x_j) - g_{k-1}(x_j), for k = 2..n.
    - Pointwise variance estimator:
          v_n(x_j) = (1/(n-1)) * Σ_{k=2}^n  k^2 * {Δ_k(x_j)}^2
      (uses n-1 since there are n-1 observable increments).
    - CLT (per j):  \tilde g(x_j) | z_{1:n}  ≈  N( g_n(x_j), v_n(x_j) / n ).

    Parameters
    ----------
    g_hat : np.ndarray, shape (n+1, m)
        Predictive means on the grid for all prefixes (row 0 may be NaN).
    alpha : float, default=0.05
        Target miscoverage for (1-α) intervals.

    Returns
    -------
    mean  : np.ndarray, shape (m,)
        The final predictive mean g_n(x_j) at each grid point.
    lower : np.ndarray, shape (m,)
        Pointwise (1-α) lower bounds: g_n - z * sqrt(v_n / n), clipped to [0,1].
    upper : np.ndarray, shape (m,)
        Pointwise (1-α) upper bounds: g_n + z * sqrt(v_n / n), clipped to [0,1].
    se    : np.ndarray, shape (m,)
        Standard errors sqrt(v_n / n) used in the bands.

    Edge cases
    ----------
    If n < 2 (i.e., fewer than two prefixes), increments are unavailable.
    Returns mean, and [0,1] as trivial bounds with se = NaN.

    Notes
    -----
    - The estimator divides by (n-1), but the CLT scaling is v_n / n.
    - Outputs are clipped to [0,1] since these are probabilities.
    """
    n = g_hat.shape[0] - 1
    mean = g_hat[-1, :].copy()

    if n < 2:
        m = mean.shape[0]
        return mean, np.zeros(m), np.ones(m), np.full(m, np.nan)

    Delta = g_hat[2:, :] - g_hat[1:-1, :]  # (n-1, m)
    k_idx = np.arange(2, n + 1)  # (n-1,)

    # v_n(x_j) = (1/(n-1)) * sum k^2 * Δ_k(x_j)^2
    num = ((k_idx[:, None] ** 2) * (Delta**2)).sum(axis=0)
    v_n = num / (n - 1)

    se = np.sqrt(v_n / n)
    z = norm.ppf(1 - alpha / 2)
    # lower = np.clip(mean - z * se, 0.0, 1.0)
    # upper = np.clip(mean + z * se, 0.0, 1.0)
    lower = mean - z * se
    upper = mean + z * se
    return mean, lower, upper, se


def multivariate_clt_draws(
    g_hat, n_draws: int = 100, rng: Generator = np.random.default_rng(28)
):
    """
    Multivariate CLT draws for \tilde g on the finite grid {x_1,...,x_m}.

    Notation (matches LaTeX)
    ------------------------
    - Let g_n = (g_n(x_1), ..., g_n(x_m))^T.
    - Define Δ_k = g_k - g_{k-1} ∈ R^m (vector over grid points), k = 2..n.
    - Matrix variance estimator:
          V_n = (1/(n-1)) * Σ_{k=2}^n (k Δ_k)(k Δ_k)^T  ∈ R^{m×m}.
    - Multivariate CLT:
          \tilde g | z_{1:n}  ≈  N_m( g_n,  V_n / n ).

    Parameters
    ----------
    g_hat : np.ndarray, shape (n+1, m)
        Predictive means on the grid for all prefixes (row 0 may be NaN).
    n_draws : int, default=100
        Number of Gaussian samples to simulate from N_m(g_n, V_n / n).
    seed : int, default=0
        RNG seed for reproducibility.

    Returns
    -------
    mean  : np.ndarray, shape (m,)
        The final predictive mean g_n at each grid point.
    draws : np.ndarray | None, shape (n_draws, m)
        i.i.d. samples from N_m(g_n, V_n / n). Returns None if n<2 or n_draws=0.

    Implementation detail
    ---------------------
    We construct a factor B such that (V_n / n) = B B^T using an SVD of
        D = (k Δ_k) / sqrt(n (n-1))   ∈ R^{(n-1)×m}
    so that D^T D = V_n / n. Then a draw is:  g_n + Z B^T  with Z ~ N(0, I).

    Notes
    -----
    - Draws are clipped to [0,1] elementwise since these are probabilities.
    - This returns *joint* samples across the grid (preserving correlations).
      There is no canonical “lower/upper band” in multiple dimensions; use
      these draws to build ellipsoids or simultaneous bands if needed.
    """
    n = g_hat.shape[0] - 1
    mean = g_hat[-1, :].copy()

    if n < 2 or n_draws == 0:
        return mean, None

    Delta = g_hat[2:, :] - g_hat[1:-1, :]  # (n-1, m)
    k_idx = np.arange(2, n + 1)  # (n-1,)

    # D^T D = V_n / n with V_n = (1/(n-1)) Σ (k Δ_k)(k Δ_k)^T
    D = (k_idx[:, None] * Delta) / np.sqrt(n * (n - 1))  # (n-1, m)

    # Economy SVD: D = U diag(s) VT  ⇒  D^T D = VT^T diag(s^2) VT
    _, s, VT = np.linalg.svd(D, full_matrices=False)
    B = VT.T * s  # (m, r) factor of V_n / n

    Z = (
        rng.standard_normal(size=(n_draws, B.shape[1]))
        if B.size
        else np.zeros((n_draws, 0))
    )
    # draws = np.clip(mean + Z @ B.T, 0.0, 1.0)
    draws = mean + Z @ B.T
    return mean, draws


# --- Simultaneous (sup-norm) CLT band using existing helpers -----------------
def clt_simultaneous_band(
    g_hat,
    alpha: float = 0.05,
    n_draws: int = 1000,
    rng: Generator = np.random.default_rng(28),
):
    """
    Compute a *simultaneous* (1−α) confidence band for g(x) across the grid
    via sup-norm calibration from the multivariate CLT.

    Returns
    -------
    mean   : (m,)
    lower  : (m,)
    upper  : (m,)
    c_alpha: float
    se     : (m,)
    draws  : (n_draws, m) or None
    """
    # 1) pointwise mean and SE (SE = sqrt(v_n / n))
    mean, _, _, se = clt_pointwise_band(g_hat, alpha=alpha)
    se_safe = se.copy()
    se_safe[se_safe == 0] = np.inf  # avoid divide-by-zero in standardization

    # 2) multivariate CLT draws (joint uncertainty across grid points)
    _, draws = multivariate_clt_draws(g_hat, n_draws=n_draws, rng=rng)
    if draws is None or len(draws) == 0:
        raise ValueError("No draws from multivariate CLT draws.")
        # Fallback: pointwise band when no multivariate draws available
        # z = norm.ppf(1 - alpha / 2)
        # lower = np.clip(mean - z * se, 0.0, 1.0)
        # upper = np.clip(mean + z * se, 0.0, 1.0)
        # return mean, lower, upper, float(z), se, None

    # 3) sup-norm calibration: c_alpha = quantile_{1-α}( max_j |Z_j| )
    Z = (draws - mean[None, :]) / se_safe[None, :]
    T = np.max(np.abs(Z), axis=1)
    c_alpha = float(np.quantile(T, 1 - alpha))

    # lower = np.clip(mean - c_alpha * se, 0.0, 1.0)
    # upper = np.clip(mean + c_alpha * se, 0.0, 1.0)
    lower = mean - c_alpha * se
    upper = mean + c_alpha * se
    return mean, lower, upper, c_alpha, se, draws


# %%
# synthetic linear gaussian regression
savedir = "./outputs/coverage"
data0 = utils.read_from(
    f"{savedir}/syn-linear-gaussian-regression y_star=3.0 n=100 m=100 n_est=64 seed=1000/data.pickle"
)
true_curve = data0["true_curve"]

files = []
for root, dirs, filenames in os.walk(savedir):
    for filename in filenames:
        if "syn-linear-gaussian" in root and filename == "ghat.pickle":
            files.append(os.path.join(root, filename))


ghat = np.asarray([utils.read_from(f) for f in files])  # (n_est, n, m)
rng = np.random.default_rng(100)
pbands = [clt_pointwise_band(g, alpha=0.05) for g in ghat]
sbands = [clt_simultaneous_band(g, alpha=0.05, n_draws=1000, rng=rng) for g in ghat]


def is_covered(x, lower, upper):
    return np.all((x >= lower) & (x <= upper))


# coverage of pointwise band
np.mean([is_covered(true_curve, lower, upper) for (_, lower, upper, _) in pbands])

# coverage of simultaneous band
np.mean([is_covered(true_curve, lower, upper) for (_, lower, upper, _, _, _) in sbands])

# %%
import matplotlib.pyplot as plt


def plot_pointwise_band(
    g_hat,
    x_grid,
    title: str,
    *,
    alpha: float = 0.05,
    X_train=None,
    true_curve=None,
    figsize=(6, 3),
    ylim=(-0.05, 1.05),
    pad=0.3,
):
    """
    One-step: compute the pointwise (1−α) CLT band from g_hat and plot it.

    Parameters
    ----------
    g_hat   : (n+1, m) array
        Prefix-wise predictive probabilities; last row is g_n(x).
    x_grid  : (m, d) or (m,) array
        Grid points where g_n and the band are evaluated.
    title   : str
        Figure title.
    alpha   : float, default 0.05
        Miscoverage (band has nominal coverage 1−α).
    X_train : array-like, optional
        If provided, plot rug marks at training x locations.
    true_curve : (m,) array, optional
        Optional reference curve to overlay (dashed).
    figsize, ylim, pad : plotting options.
    """

    mean, lower, upper, se = clt_pointwise_band(g_hat, alpha=alpha)

    x_flat = np.asarray(x_grid).ravel()
    fig, ax = plt.subplots(figsize=figsize)

    # band
    ax.fill_between(x_flat, lower, upper, alpha=0.25, label=f"pointwise band")
    # mean
    ax.plot(x_flat, mean, "k", lw=1.5, label="hat g_n(x)")

    # references
    if true_curve is not None:
        ax.plot(x_flat, true_curve, "k--", lw=1, label="true g(x)")
    if X_train is not None:
        x_train = np.asarray(X_train).ravel()
        ax.plot(
            x_train,
            np.full_like(x_train, ylim[0] + 0.02),
            "|",
            color="grey",
            markersize=6,
            alpha=0.6,
        )

    ax.set_xlim(x_flat.min(), x_flat.max())
    ax.set_ylim(*ylim)
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout(pad=pad)
    plt.show()


def plot_draws_with_simultaneous_band(
    g_hat,
    x_grid,
    title: str,
    *,
    alpha: float = 0.05,
    n_draws: int = 1000,
    seed: int = 0,
    X_train=None,
    true_curve=None,
    figsize=(6, 3),
    ylim=(-0.05, 1.05),
    pad=0.3,
):
    """
    One-step: compute the simultaneous (1−α) CLT band from g_hat and plot it
    over x_grid together with multivariate CLT sample paths.

    Parameters
    ----------
    g_hat : (n+1, m) array
        Prefix-wise predictive probabilities; last row is g_n(x).
    x_grid : (m, d) or (m,) array
        Grid points where g_n and the band are evaluated.
    title : str
        Figure title.
    alpha : float, default 0.05
        Miscoverage (band has nominal coverage 1−α).
    n_draws : int, default 1000
        Number of multivariate CLT draws for calibration/overlay.
    seed : int, default 0
        RNG seed for multivariate CLT draws.
    X_train : array-like, optional
        If provided, plot rug marks at training x locations.
    true_curve : (m,) array, optional
        Optional reference curve to overlay (dashed).
    figsize, ylim, pad : plotting options.

    Effects
    -------
    Renders the plot and returns nothing.
    """
    # import numpy as np, matplotlib.pyplot as plt
    # plt.rcParams.update({
    #     "text.usetex": True,
    #     "font.family": "serif",
    #     "font.serif": ["Computer Modern Roman"],
    # })

    mean, lower, upper, c_alpha, se, draws = clt_simultaneous_band(
        g_hat, alpha=alpha, n_draws=n_draws
    )

    x_flat = np.asarray(x_grid).ravel()
    fig, ax = plt.subplots(figsize=figsize)

    # Band
    ax.fill_between(x_flat, lower, upper, alpha=0.25, label=f"simultaneous band")

    # Draws
    if draws is not None:
        for path in draws:
            ax.plot(x_flat, path, alpha=0.12, lw=1)

    # Mean + refs
    ax.plot(x_flat, mean, "k", lw=1.5, label="hat g_n(x)")
    if true_curve is not None:
        ax.plot(x_flat, true_curve, "k--", lw=1, label="true g(x)")
    if X_train is not None:
        x_train = np.asarray(X_train).ravel()
        ax.plot(
            x_train,
            np.full_like(x_train, ylim[0] + 0.02),
            "|",
            color="grey",
            markersize=6,
            alpha=0.6,
        )

    ax.set_xlim(x_flat.min(), x_flat.max())
    ax.set_ylim(*ylim)
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout(pad=pad)
    plt.show()


plot_pointwise_band(
    ghat[0],
    data0["x_grid"],
    title=f"{data0['title']} pointwise CLT band",
    alpha=0.05,
    X_train=data0["X"],
    true_curve=data0["true_curve"],
)

plot_draws_with_simultaneous_band(
    ghat[0],
    data0["x_grid"],
    title=f"{data0['title']} simultaneous CLT band",
    alpha=0.05,
    X_train=data0["X"],
    true_curve=data0["true_curve"],
)

# %%

# synthetic mixture probit
# synthetic linear gaussian regression
savedir = "./outputs/coverage"
data0 = utils.read_from(
    f"{savedir}/syn-mixture-probit n=100 m=100 n_est=64 seed=1000/data.pickle"
)
true_curve = data0["true_curve"]

files = []
for root, dirs, filenames in os.walk(savedir):
    for filename in filenames:
        if "syn-mixture-probit" in root and filename == "ghat.pickle":
            files.append(os.path.join(root, filename))


ghat = np.asarray([utils.read_from(f) for f in files])  # (n_est, n, m)
rng = np.random.default_rng(100)
pbands = [clt_pointwise_band(g, alpha=0.05) for g in ghat]
sbands = [clt_simultaneous_band(g, alpha=0.05, n_draws=1000, rng=rng) for g in ghat]


# coverage of pointwise band
np.mean([is_covered(true_curve, lower, upper) for (_, lower, upper, _) in pbands])

# coverage of simultaneous band
np.mean([is_covered(true_curve, lower, upper) for (_, lower, upper, _, _, _) in sbands])

# %%
