# %%
import re
import numpy as np
from scipy.stats import norm
import utils
from metrics import (
    build_pointwise_band,
    build_simultaneous_band,
    compute_pointwise_coverage,
    compute_simultaneous_coverage,
)
import metrics
import posterior
from posterior import compute_vn, compute_un
import os

import matplotlib.pyplot as plt
import pandas as pd

from constants import Y_STAR_MAP

# %load_ext autoreload
# %autoreload 2

id_dir = "../outputs/2026-01-31"
image_dir = "../paper/images"


def trapezoidal_cumsum(n, b):
    # n: (N, )
    # b: (N, )
    areas = (b[1:] + b[:-1]) / 2 * (n[1:] - n[:-1])
    return np.cumsum(np.concatenate(([0], areas)))


def plot_band(x_grid, ci_band, true_event, X, title):
    figsize = (6, 4)
    fig, ax = plt.subplots(figsize=figsize)
    x_grid = x_grid.squeeze()
    ax.fill_between(
        x_grid, ci_band["lower"], ci_band["upper"], alpha=0.25, label=f"95% band"
    )
    ax.plot(x_grid, ci_band["mean"], "k", lw=1.5, label="g_n(x)")
    ax.plot(x_grid, true_event.squeeze(), "k--", lw=1, label="true event")
    ax.scatter(
        X,
        np.zeros_like(X),
        marker="|",
        s=20,
        c="black",
        alpha=0.6,
        label="training data",
    )
    ax.set_xlim(-10, 10)
    ax.set_ylim(-0.1, 1.1)
    ax.title.set_text(title)
    ax.legend()


# %%
## QUASI-SIMPLE TEST


def get_exp_info(outdir):
    n0 = int(re.search(r"n0=([^\s]+)", outdir).group(1))
    x_rollout = re.search(r"x_rollout=([^\s]+)", outdir).group(1)
    x_design = re.search(r"x_design=([^\s]+)", outdir).group(1)
    x_rollout = x_design if x_rollout == "truth" else x_rollout
    setup = re.search(r"setup=([^\s]+)", outdir).group(1)
    n_est = int(re.search(r"n_est=([^\s]+)", outdir).group(1))
    return {"n0": n0, "x_rollout": x_rollout, "setup": setup, "n_est": n_est}


def compile_outer(outer_dir, include_idx):
    delta_paths = utils.get_matching_files(outer_dir, r"delta-\d+\.pickle")
    delta_paths.sort(key=lambda x: int(re.search(r"delta-(\d+)", x).group(1)))
    delta_paths = [
        p
        for p in delta_paths
        if int(re.search(r"delta-(\d+)", p).group(1)) in include_idx
    ]
    deltas_raw = [utils.read_from(p) for p in delta_paths]
    deltas = np.asarray([d["delta"] for d in deltas_raw])
    weights = np.array([d["weight"] for d in deltas_raw])
    n = np.asarray([d["n"] for d in deltas_raw])
    return deltas, weights, n


def fit_lm(x, y):
    (slope, const), pcov = np.polyfit(x, y, 1, cov=True)
    err = 1.96 * np.sqrt(pcov[0, 0])
    y_pred = slope * x + const
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
    return slope, const, (slope - err, slope + err), rmse, r2


def plot_power_law(ax, b, n, title_prefix=""):
    # fit a power law model on E[|b|]
    # b: (k, outer), n: (k, outer)
    log_n = np.log(n)
    log_Eb = np.log(np.mean(np.abs(b), axis=-1))
    slope, const, ci, rmse, r2 = fit_lm(log_n, log_Eb)
    ax.plot(log_n, log_Eb, marker=".", alpha=0.6)
    fit_label = (
        f"Slope: {slope:.2f}, $R^2$: {r2:.2f}, RMSE: {rmse:.2f}\n"
        f"95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]"
    )
    ax.plot(log_n, const + log_n * slope, "b--", lw=2, label=fit_label)
    ax.set_xlabel("log(n)")
    ax.set_ylabel("log( E[|b|] )")
    ax.legend(loc="lower left")


def get_bias_and_n(outer_dirs, include_idx):
    # returned b: (k, outer, t, x_new)
    # returned n: (k, )
    b_all = []
    n_all = []
    for d in outer_dirs:
        # deltas: (k, 8, t, x_new), weights: (k, 8), k: (k,)
        deltas, weights, n = compile_outer(d, include_idx)
        b = np.sum(deltas * weights[:, :, None, None], axis=1)  # (k, t, x_new)
        b_all.append(b)
        n_all.append(n)
    b_all = np.stack(b_all, axis=1)  # (k, outer, t, x_new)
    n_all = np.stack(n_all, axis=1)  # (k, outer)
    assert np.all(n_all == n_all[:, [0]])
    return b_all, n_all[:, 0]


# %%
n_est_list = [8, 16]
n0 = 25
t_idx = 1
x_new_idx = 2

# geom spaced points
n_geom_spec = np.rint(np.geomspace(n0 + 100, n0 + 1000, 100)).astype(int)
k_geom_spec = n_geom_spec - n0 - 1


# %%
# Power law for E[|b_n|]
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
for n_est, ax in zip(n_est_list, axes):
    outdir = utils.get_matching_dirs(id_dir, rf"quasi-simple.+n_est={n_est} ")
    outdir = outdir[0]
    outer_dirs = utils.get_matching_dirs(outdir, r"outer-\d+")
    b_geom, n_geom = get_bias_and_n(outer_dirs, n_geom_spec)
    outer0_delta0 = utils.read_from(
        utils.get_matching_files(outer_dirs[0], r"delta-\d+\.pickle")[0]
    )
    t = outer0_delta0["t"]
    x_new = outer0_delta0["x_new"]
    plot_power_law(ax, b_geom[..., t_idx, x_new_idx], n_geom)
    ax.set_title(f"n_estimators={n_est}")

# fig.suptitle(
#     rf"Power Law of $E[|b_n|]$\n"
#     rf"n0={n0}, x_new={x_new[x_new_idx]}, t={t[t_idx]}, n_outer={len(outer_dirs)}"
# )
fig.tight_layout()
fig.savefig(f"{image_dir}/quasi-bias-power-law.pdf")


# %%
# Partial sum of E[|b_n|] with trapezoidal approximation
fig, axes = plt.subplots(1, 2, figsize=(10, 3))

for n_est, ax in zip(n_est_list, axes):
    outdir = utils.get_matching_dirs(id_dir, rf"quasi-simple.+n_est={n_est} ")
    outdir = outdir[0]
    outer_dirs = utils.get_matching_dirs(outdir, r"outer-\d+")

    # b_geom: (k, outer, t, x_new), n_geom: (k, )
    b_geom, n_geom = get_bias_and_n(outer_dirs, n_geom_spec)

    # Eb = E[|b_n|], shape: (k, )
    Eb = np.mean(np.abs(b_geom), axis=1)[..., t_idx, x_new_idx]
    Eb_cumsum = trapezoidal_cumsum(n_geom, Eb)

    ax.plot(n_geom, Eb_cumsum, "-", label="Trapezoidal Sum")
    ax.set_xlabel("n")
    ax.set_ylabel(r"$\sum_{m<n} E[|b_m|]$")
    ax.set_title(f"n_estimators={n_est}")
    ax.grid(True)

# fig.suptitle(
#     f"Partial Sum of $E[|b_n|]$ (Trapezoidal Approx)\nx_new={x_new[x_new_idx]}, t={t[t_idx]}"
# )
fig.tight_layout()
fig.savefig(f"{image_dir}/quasi-bias-partial-sum.pdf")

# %%
# Partial sum of \sqrt(n) E[|b_n|] with trapezoidal approximation
fig, axes = plt.subplots(1, 2, figsize=(10, 3))

for n_est, ax in zip(n_est_list, axes):
    outdir = utils.get_matching_dirs(id_dir, rf"quasi-simple.+n_est={n_est} ")
    outdir = outdir[0]
    outer_dirs = utils.get_matching_dirs(outdir, r"outer-\d+")

    # b_geom: (k, outer, t, x_new), n_geom: (k, )
    b_geom, n_geom = get_bias_and_n(outer_dirs, n_geom_spec)

    # wEb = \sqrt(n) E[|b_n|], shape: (k, )
    wEb = np.sqrt(n_geom) * np.mean(np.abs(b_geom), axis=1)[..., t_idx, x_new_idx]
    wEb_cumsum = trapezoidal_cumsum(n_geom, wEb)

    ax.plot(n_geom, wEb_cumsum, "-", label="Trapezoidal Sum")
    ax.set_xlabel("n")
    ax.set_ylabel(r"$\sum_{m<n} \sqrt{m} \ E[|b_m|]$")
    ax.set_title(f"n_estimators={n_est}")
    ax.grid(True)

# fig.suptitle(
#     f"Partial Sum of $E[|b_n|]$ (Trapezoidal Approx)\nx_new={x_new[x_new_idx]}, t={t[t_idx]}"
# )
fig.tight_layout()
fig.savefig(f"{image_dir}/quasi-weighted-bias-partial-sum.pdf")

# %%
## Power law of \log E(\Delta_k^2 \mid Z{1:k-1}) of one outer rep (2026-01-31)
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
for n_est, ax in zip(n_est_list, axes):
    outdir = utils.get_matching_dirs(id_dir, rf"quasi-simple.+n_est={n_est} ")
    outdir = outdir[0]
    outer_dirs = utils.get_matching_dirs(outdir, r"outer-\d+")

    deltas, weights, n_geom = compile_outer(
        outer_dirs[0], n_geom_spec
    )  # (k, 8, t, x_new)
    delta2 = np.square(deltas)[..., t_idx, x_new_idx]  # (k, 8)
    Edelta2 = np.sum(delta2 * weights, axis=1)  # (k, )

    log_Edelta2 = np.log(Edelta2)
    log_n = np.log(n_geom)
    slope, const, ci, rmse, r2 = fit_lm(log_n, log_Edelta2)

    # Normalising constant sanity check
    ax.plot(log_n, log_Edelta2, marker=".", alpha=0.6)
    fit_label = (
        f"Slope: {slope:.2f}, $R^2$: {r2:.2f}, RMSE: {rmse:.2f}\n"
        f"95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]"
    )
    ax.plot(log_n, const + log_n * slope, "b--", lw=2, label=fit_label)
    ax.set_xlabel("log(n)")
    ax.set_ylabel(r"$E[\Delta_n^2 \mid Z_{1:n-1}]$")
    ax.set_title(f"n_estimators={n_est}")

fig.suptitle(
    rf"Power Law of $E[\Delta_n^2 \mid Z_{{1:n-1}}]$, x_new={x_new[x_new_idx]}, t={t[t_idx]}"
)
fig.tight_layout()


# %%
# Table of fit per rollout
df_rows = []
for n_est in n_est_list:
    outdir = utils.get_matching_dirs(id_dir, rf"quasi-simple.+n_est={n_est} ")
    assert len(outdir) == 1
    outdir = outdir[0]
    outer_dirs = utils.get_matching_dirs(outdir, r"outer-\d+")
    assert len(outer_dirs) == 100
    for d in outer_dirs:
        deltas, weights, n_geom = compile_outer(d, n_geom_spec)  # (k, 8, t, x_new)
        delta2 = np.square(deltas)[..., t_idx, x_new_idx]  # (k, 8)
        Edelta2 = np.sum(delta2 * weights, axis=1)  # (k, )
        log_Edelta2 = np.log(Edelta2)
        log_n = np.log(n_geom)
        slope, const, ci, rmse, r2 = fit_lm(log_n, log_Edelta2)
        df_rows.append(
            {
                "n_est": n_est,
                "slope": slope,
                "ci_upper": ci[1],
                "ci_lower": ci[0],
                "rmse": rmse,
                "r2": r2,
            }
        )
df = pd.DataFrame(df_rows)
# %%
# 95% CI of the slope per rollout
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
for n_est, ax in zip(n_est_list, axes):
    df_n_est = df[df["n_est"] == n_est]
    for i, (_, row) in enumerate(df_n_est.iterrows()):
        ax.vlines(i, -row["ci_lower"], -row["ci_upper"], color="k")
        ax.scatter(i, -row["slope"], c="k", s=10)
    ax.axhline(2, color="r", linestyle="--")
    ax.set_xlabel("Rollout index")
    ax.set_ylabel(r"95% CI of $\gamma_r$")
    ax.set_title(f"n_estimators={n_est}")

fig.tight_layout()
fig.savefig(f"{image_dir}/quasi-gamma-ci.pdf")

# %%
# Histogram of the slope per rollout
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
for n_est, ax in zip(n_est_list, axes):
    df_n_est = df[df["n_est"] == n_est]
    ax.hist(-df_n_est["slope"], bins=30)
    ax.set_xlabel("$\gamma_r$")
    ax.set_ylabel("Count")
    ax.set_title(f"n_estimators={n_est}")

fig.tight_layout()
fig.savefig(f"{image_dir}/quasi-gamma-histogram.pdf")

# %%
# Plot of n^(fitted) * E[Delta^2_n]
df_median = df.groupby("n_est").median()
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
for n_est, ax in zip(n_est_list, axes):
    outdir = utils.get_matching_dirs(id_dir, rf"quasi-simple.+n_est={n_est} ")
    outdir = outdir[0]
    outer_dirs = utils.get_matching_dirs(outdir, r"outer-\d+")
    for d in outer_dirs:
        deltas, weights, n_geom = compile_outer(d, n_geom_spec)  # (k, 8, t, x_new)
        delta2 = np.square(deltas)[..., t_idx, x_new_idx]  # (k, 8)
        fitted_power = df_median.loc[n_est, "slope"]
        factor = n_geom ** (-1 * fitted_power)
        n2Edelta2 = factor * np.sum(delta2 * weights, axis=1)  # (k, )
        ax.plot(n_geom, n2Edelta2, alpha=0.2)
        ax.set_yscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel(rf"$n^{{\widehat\gamma_r}} E[\Delta_n^2 \mid Z_{{1:n-1}}]$")
    ax.set_title(rf"n_estimators={n_est}, $\widehat\gamma_r$={-fitted_power:.2f}")
# fig.suptitle(
#     f"Power Law of E[|b_n|] with fitted normalising constant, x_new={x_new[x_new_idx]}, t={t[t_idx]}"
# )
fig.tight_layout()
fig.savefig(f"{image_dir}/quasi-gamma-constant.pdf")
# %%
