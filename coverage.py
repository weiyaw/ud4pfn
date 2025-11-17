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



# %%
def build_pointwise_band(mean, cov, alpha: float = 0.05):
    se = np.sqrt(cov)
    z = norm.ppf(1 - alpha / 2)
    # lower = np.clip(mean - z * se, 0.0, 1.0)
    # upper = np.clip(mean + z * se, 0.0, 1.0)
    lower = mean - z * se
    upper = mean + z * se
    return mean, lower, upper, se


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
    return mean, lower, upper, c_alpha, se, draws



def compute_pointwise_coverage(true_curve, pbands):
    # check if each point in the grid is covered, then average over grid points
    intervals = [(l, u) for (_, l, u, _) in pbands]
    is_covered = [(true_curve >= l) & (true_curve <= u) for (l, u) in intervals]
    return np.mean(np.asarray(is_covered))


def compute_simultaneous_coverage(true_curve, sbands):
    # coverage of the entire curve
    intervals = [(l, u) for (_, l, u, _, _, _) in sbands]
    is_covered = [np.all((true_curve >= l) & (true_curve <= u)) for (l, u) in intervals]
    return np.mean(is_covered)


# %%
# synthetic linear gaussian regression
savedir = "../outputs/coverage"
data0 = utils.read_from(
    f"{savedir}/setup=linreg y_star=3.0 n=100 m=100 n_est=64 seed=1000/data.pickle"
)
true_curve = data0["true_curve"]

g0_to_gn = []
gn = []
gn_plus_1 = []
for root, dirs, filenames in os.walk(savedir):
    for filename in filenames:
        if "setup=linreg" in root and filename == "g0_to_gn.pickle":
            g0_to_gn.append(utils.read_from(os.path.join(root, filename)))
        elif "setup=linreg" in root and filename == "gn.pickle":
            gn.append(utils.read_from(os.path.join(root, filename)))
        elif "setup=linreg" in root and filename == "gn_plus_1.pickle":
            gn_plus_1.append(utils.read_from(os.path.join(root, filename)))


# %%
from forward import compute_vn

g0_to_gn = np.asarray(g0_to_gn)  # (rep, n, m)
gn = np.asarray(gn)  # (rep, m)
gn_plus_1 = np.asarray(gn_plus_1)  # (rep, mc_samples, m)
n = g0_to_gn.shape[1] - 1

pbands = []
sbands = []
for g in g0_to_gn:
    clt_mean = g[-1]
    # pointwise covariance
    clt_cov = compute_vn(g, type="pointwise") / n
    pbands.append(build_pointwise_band(clt_mean, clt_cov))

    # simultaneous covariance
    clt_cov = compute_vn(g, type="simultaneous") / n
    sbands.append(build_simultaneous_band(clt_mean, clt_cov))


# %%
# coverage
print(compute_pointwise_coverage(true_curve, pbands))
print(compute_simultaneous_coverage(true_curve, sbands))

# %%
from forward import compute_un
gn = np.asarray(gn)  # (rep, m)
gn_plus_1 = np.asarray(gn_plus_1)  # (rep, mc_samples, m)

pbands = []
sbands = []
for g1, g2 in zip(gn, gn_plus_1):
    clt_mean = g1
    # pointwise covariance
    clt_cov = compute_un(g1, g2, n, type="pointwise") / n
    pbands.append(build_pointwise_band(clt_mean, clt_cov))

    # simultaneous covariance
    clt_cov = compute_un(g1, g2, n, type="simultaneous") / n
    sbands.append(build_simultaneous_band(clt_mean, clt_cov))

# %%
# coverage
print(compute_pointwise_coverage(true_curve, pbands))
print(compute_simultaneous_coverage(true_curve, sbands))

# %%
import matplotlib.pyplot as plt


def plot_pointwise_band(
    # g_hat,
    pband,
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

    # mean, lower, upper, se = clt_pointwise_band(g_hat, alpha=alpha)
    mean, lower, upper, se = pband

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
    # g_hat,
    sband,
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

    # mean, lower, upper, c_alpha, se, draws = clt_simultaneous_band(
    #     g_hat, alpha=alpha, n_draws=n_draws
    # )
    mean, lower, upper, c_alpha, se, draws = sband

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


# %%

plot_pointwise_band(
    pbands[0],
    data0["x_grid"],
    title=f"{data0['title']} pointwise CLT band",
    alpha=0.05,
    X_train=data0["X"],
    true_curve=data0["true_curve"],
)

plot_draws_with_simultaneous_band(
    sbands[0],
    data0["x_grid"],
    title=f"{data0['title']} simultaneous CLT band",
    alpha=0.05,
    X_train=data0["X"],
    true_curve=data0["true_curve"],
)

# %%

# synthetic mixture probit
# synthetic linear gaussian regression
savedir = "../outputs/coverage"
data0 = utils.read_from(
    f"{savedir}/setup=probit n=100 m=100 n_est=64 seed=1000/data.pickle"
)
true_curve = data0["true_curve"]

files = []
for root, dirs, filenames in os.walk(savedir):
    for filename in filenames:
        if "setup=probit" in root and filename == "g0_to_gn.pickle":
            files.append(os.path.join(root, filename))


# %%
from forward import compute_vn

g0_to_gn = np.asarray(g0_to_gn)  # (rep, n, m)
gn = np.asarray(gn)  # (rep, m)
gn_plus_1 = np.asarray(gn_plus_1)  # (rep, mc_samples, m)
n = g0_to_gn.shape[1] - 1

pbands = []
sbands = []
for g in g0_to_gn:
    clt_mean = g[-1]
    # pointwise covariance
    clt_cov = compute_vn(g, type="pointwise") / n
    pbands.append(build_pointwise_band(clt_mean, clt_cov))

    # simultaneous covariance
    clt_cov = compute_vn(g, type="simultaneous") / n
    sbands.append(build_simultaneous_band(clt_mean, clt_cov))


# %%
# coverage
print(compute_pointwise_coverage(true_curve, pbands))
print(compute_simultaneous_coverage(true_curve, sbands))

# %%
from forward import compute_un
gn = np.asarray(gn)  # (rep, m)
gn_plus_1 = np.asarray(gn_plus_1)  # (rep, mc_samples, m)

pbands = []
sbands = []
for g1, g2 in zip(gn, gn_plus_1):
    clt_mean = g1
    # pointwise covariance
    clt_cov = compute_un(g1, g2, n, type="pointwise") / n
    pbands.append(build_pointwise_band(clt_mean, clt_cov))

    # simultaneous covariance
    clt_cov = compute_un(g1, g2, n, type="simultaneous") / n
    sbands.append(build_simultaneous_band(clt_mean, clt_cov))

# %%
# coverage
print(compute_pointwise_coverage(true_curve, pbands))
print(compute_simultaneous_coverage(true_curve, sbands))






















# %%

# synthetic mixture probit
# synthetic linear gaussian regression
savedir = "../outputs/coverage/obsolete"
data0 = utils.read_from(
    # f"{savedir}/setup=syn-mixture-probit n=100 m=100 n_est=64 seed=1000/data.pickle"
    f"{savedir}/syn-mixture-probit n=100 m=100 n_est=64 seed=1000/data.pickle"
)
true_curve = data0["true_curve"]

files = []
for root, dirs, filenames in os.walk(savedir):
    for filename in filenames:
        if "syn-mixture-probit" in root and filename == "ghat.pickle":
            files.append(os.path.join(root, filename))




ghat = np.asarray([utils.read_from(f) for f in files])  # (rep, n, m)
rng = np.random.default_rng(100)
pbands = [clt_pointwise_band(g, alpha=0.05) for g in ghat]
sbands = [clt_simultaneous_band(g, alpha=0.05, n_draws=1000, rng=rng) for g in ghat]

pbands = []
sbands = []
for g in ghat:
    clt_mean = g[-1]
    # pointwise covariance
    clt_cov = compute_vn(g, type="pointwise") / n
    pbands.append(build_pointwise_band(clt_mean, clt_cov))

    # simultaneous covariance
    clt_cov = compute_vn(g, type="simultaneous") / n
    sbands.append(build_simultaneous_band(clt_mean, clt_cov))

# %%
# coverage of pointwise band
compute_pointwise_coverage(true_curve, pbands)
# coverage of simultaneous band
compute_simultaneous_coverage(true_curve, sbands)


# %%
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
