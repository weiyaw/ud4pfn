# %%
import re
import numpy as np
import scipy
from scipy.special import logsumexp
import utils

import matplotlib.pyplot as plt
import pandas as pd

from constants import Y_STAR_MAP

# %load_ext autoreload
# %autoreload 2
np.set_printoptions(formatter={"float_kind": "{:.2e}".format}, linewidth=200)
pd.set_option("display.max_rows", None)
id_dir = "../outputs/2026-01-04"
# image_dir = "../paper/images"


def trapezoidal_cumsum(n, b):
    # n: (N, )
    # b: (N, )
    assert len(n) == len(b)
    assert np.all(n == np.sort(n))
    assert n.ndim == 1 and b.ndim == 1
    areas = (b[1:] + b[:-1]) / 2 * (n[1:] - n[:-1])
    areas = np.concatenate(([0], areas))
    return np.cumsum(areas), areas


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
    deltas = np.asarray([d["delta"] for d in deltas_raw]) # (k, mc_samples, t, x_new)
    n = np.asarray([d["n"] for d in deltas_raw]) # (k,)
    return deltas, n


def fit_lm(x, y):
    (slope, const), pcov = np.polyfit(x, y, 1, cov=True)
    n = len(x)
    df = n - 2
    t_val = scipy.stats.t.ppf(0.975, df)
    err = t_val * np.sqrt(pcov[0, 0])
    y_pred = slope * x + const
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
    return slope, const, (slope - err, slope + err), rmse, r2


def fit_lm2(x, y):
    """
    Fits a linear model with a common slope but separate intercepts for each row.
    Model: y_{ij} = alpha_i + beta * x_{ij} + epsilon_{ij}
    
    x, y: (rows, cols) matrices
    """
    assert x.ndim == 2 and y.ndim == 2, "Input x and y must be 2D matrices"
    assert x.shape == y.shape
    n_rows, n_cols = x.shape
    
    # Row-wise centering
    y_mean = np.mean(y, axis=1, keepdims=True)
    x_mean = np.mean(x, axis=1, keepdims=True)
    
    y_centered = y - y_mean
    x_centered = x - x_mean
    
    # Common slope (regression through origin on centered data)
    num = np.sum(y_centered * x_centered)
    den = np.sum(x_centered ** 2)
    slope = num / den
    
    # Intercepts
    const = y_mean - slope * x_mean
    const = const.flatten() # (n_rows,)
        
    # Statistics
    y_pred = slope * x + const[:, None]
         
    resid = y - y_pred
    rss = np.sum(resid ** 2)
    
    # Degrees of freedom: Total observations - (1 slope + n_rows intercepts)
    df = x.size - (1 + n_rows)
    if df > 0:
        mse = rss / df
        slope_se = np.sqrt(mse / den)
        t_val = scipy.stats.t.ppf(0.975, df)
        err = t_val * slope_se
    else:
        err = np.inf
    
    rmse = np.sqrt(np.mean(resid ** 2))
    sst = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - rss / sst
    
    return slope, const, (slope - err, slope + err), rmse, r2


def plot_power_law(ax, b, n, title_prefix=""):
    # fit a power law model on E[|b|]
    # b: (outer, k), n: (k, )
    log_n = np.log(n)
    log_Eb = np.log(np.mean(np.abs(b), axis=0))
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
    # returned b: (outer, k, t, x_new)
    # returned n: (outer, k)
    b_all = []
    n_all = []
    for d in outer_dirs:
        # deltas: (k, mc_samples, t, x_new), n: (k,)
        deltas, n = compile_outer(d, include_idx)
        b = np.mean(deltas, axis=1)  # (k, t, x_new)
        b_all.append(b)
        n_all.append(n)
    b_all = np.stack(b_all, axis=0)  # (outer, k, t, x_new)
    n_all = np.stack(n_all, axis=0)  # (outer, k)
    assert np.all(n_all == n_all[[0], :])
    return b_all, n_all


# %%
n_est_list = [8, 16]
n0 = 25
t_idx = 1
x_new_idx = 2

# geom spaced points
n_tail_spec = np.rint(np.geomspace(n0 + 100, n0 + 1000, 30)).astype(int)
n_head_spec = np.arange(n0, n0 + 100, 5) 
n_all_spec = np.unique(np.concatenate([n_head_spec, n_tail_spec]))
k_tail_spec = n_tail_spec - n0 - 1



# %%
# Power law for E[|b_n|]
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
for n_est, ax in zip(n_est_list, axes):
    outdir = utils.get_matching_dirs(id_dir, rf"quasi setup=gaussian-linear.+n_est={n_est} .+fix=True")
    assert len(outdir) == 1
    outdir = outdir[0]
    outer_dirs = utils.get_matching_dirs(outdir, r"outer-\d+")
    b_all, n_all = get_bias_and_n(outer_dirs, n_tail_spec)
    outer0_delta0 = utils.read_from(
        utils.get_matching_files(outer_dirs[0], r"delta-\d+\.pickle")[0]
    )
    t = outer0_delta0["t"]
    x_new = outer0_delta0["x_new"]
    plot_power_law(ax, b_all[..., t_idx, x_new_idx], n_all[0, :])
    ax.set_title(f"n_estimators={n_est}")

# fig.suptitle(
#     rf"Power Law of $E[|b_n|]$\n"
#     rf"n0={n0}, x_new={x_new[x_new_idx]}, t={t[t_idx]}, n_outer={len(outer_dirs)}"
# )
fig.tight_layout()
# fig.savefig(f"{image_dir}/quasi-bias-power-law.pdf")

# %%
# Power law for |b_n|
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
for n_est, ax in zip(n_est_list, axes):
    outdir = utils.get_matching_dirs(id_dir, rf"quasi setup=gaussian-linear.+n_est={n_est} .+fix=True")
    assert len(outdir) == 1
    outdir = outdir[0]
    outer_dirs = utils.get_matching_dirs(outdir, r"outer-\d+")
    b_all, n_all = get_bias_and_n(outer_dirs, n_tail_spec)
    outer0_delta0 = utils.read_from(
        utils.get_matching_files(outer_dirs[0], r"delta-\d+\.pickle")[0]
    )
    t = outer0_delta0["t"]
    x_new = outer0_delta0["x_new"]

    log_n = np.log(n_all) # (outer, k)
    log_b = np.log(np.abs(b_all[..., t_idx, x_new_idx])) # (outer, k)
    slope, const, ci, rmse, r2 = fit_lm2(log_n, log_b)

    ax.scatter(log_n, log_b, alpha=0.6, s=1)
    fit_label = (
        f"Slope: {slope:.2f}, $R^2$: {r2:.2f}, RMSE: {rmse:.2f}\n"
        f"95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]"
    )

    y_fit = const[:, None] + log_n * slope
    ax.plot(log_n[0], y_fit[0], "b--", lw=2, label=fit_label)
    ax.plot(log_n.T, y_fit.T, "b--", lw=2, alpha=0.2)
    ax.set_xlabel(r"$\log(n)$")
    ax.set_ylabel(r"$\log(|b_n|)$")
    ax.legend(loc="lower left")
    ax.set_title(f"n_estimators={n_est}")

# fig.suptitle(
#     rf"Power Law of $E[|b_n|]$\n"
#     rf"n0={n0}, x_new={x_new[x_new_idx]}, t={t[t_idx]}, n_outer={len(outer_dirs)}"
# )
fig.tight_layout()


# %%
# Partial sum of E[|b_n|] with trapezoidal approximation
fig, axes = plt.subplots(1, 2, figsize=(10, 3))

for n_est, ax in zip(n_est_list, axes):
    outdir = utils.get_matching_dirs(id_dir, rf"quasi setup=gaussian-linear.+n_est={n_est} .+fix=True")
    assert len(outdir) == 1
    outdir = outdir[0]
    outer_dirs = utils.get_matching_dirs(outdir, r"outer-\d+")

    # b_all: (k, outer, t, x_new), n_all: (k, )
    b_all, n_all = get_bias_and_n(outer_dirs, n_all_spec)

    # Eb = E[|b_n|], shape: (k, )
    Eb = np.mean(np.abs(b_all), axis=0)[..., t_idx, x_new_idx]
    Eb_cumsum, areas = trapezoidal_cumsum(n_all[0, :], Eb)

    ax.plot(n_all[0, :], Eb_cumsum, "-", label="Trapezoidal Sum")
    ax.set_xlabel("n")
    ax.set_ylabel(r"$\sum_{m<n} E[|b_m|]$")
    ax.set_title(f"n_estimators={n_est}")
    ax.grid(True)

# fig.suptitle(
#     f"Partial Sum of $E[|b_n|]$ (Trapezoidal Approx)\nx_new={x_new[x_new_idx]}, t={t[t_idx]}"
# )
fig.tight_layout()
# fig.savefig(f"{image_dir}/quasi-bias-partial-sum.pdf")

# %%
# Partial sum of \sqrt(n) E[|b_n|] with trapezoidal approximation
fig, axes = plt.subplots(1, 2, figsize=(10, 3))

for n_est, ax in zip(n_est_list, axes):
    outdir = utils.get_matching_dirs(id_dir, rf"quasi setup=gaussian-linear.+n_est={n_est} .+fix=True")
    assert len(outdir) == 1
    outdir = outdir[0]
    outer_dirs = utils.get_matching_dirs(outdir, r"outer-\d+")

    # b_all: (k, outer, t, x_new), n_all: (k, )
    b_all, n_all = get_bias_and_n(outer_dirs, n_all_spec)

    # wEb = \sqrt(n) E[|b_n|], shape: (k, )
    wEb = np.sqrt(n_all[0, :]) * np.mean(np.abs(b_all), axis=0)[..., t_idx, x_new_idx]
    wEb_cumsum, areas = trapezoidal_cumsum(n_all[0, :], wEb)

    ax.plot(n_all[0, :], wEb_cumsum, "-", label="Trapezoidal Sum")
    ax.set_xlabel("n")
    ax.set_ylabel(r"$\sum_{m<n} \sqrt{m} \ E[|b_m|]$")
    ax.set_title(f"n_estimators={n_est}")
    ax.grid(True)

# fig.suptitle(
#     f"Partial Sum of $E[|b_n|]$ (Trapezoidal Approx)\nx_new={x_new[x_new_idx]}, t={t[t_idx]}"
# )
fig.tight_layout()
# fig.savefig(f"{image_dir}/quasi-weighted-bias-partial-sum.pdf")

# %%
## Power law of \log E(\Delta_k^2 \mid Z{1:k-1}) of one outer rep
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
for n_est, ax in zip(n_est_list, axes):
    outdir = utils.get_matching_dirs(id_dir, rf"quasi setup=gaussian-linear.+n_est={n_est} .+fix=True")
    outdir = outdir[0]
    outer_dirs = utils.get_matching_dirs(outdir, r"outer-\d+")

    # use just the 0th rep
    deltas, n_tail = compile_outer(outer_dirs[0], n_tail_spec)  # (k, mc_samples, t, x_new)
    log_delta2 = 2 * np.log(np.abs(deltas[..., t_idx, x_new_idx]))
    log_Edelta2 = logsumexp(log_delta2, axis=-1) - np.log(log_delta2.shape[-1])
    log_n = np.log(n_tail)
    slope, const, ci, rmse, r2 = fit_lm(log_n, log_Edelta2)

    # Normalising constant sanity check
    ax.plot(log_n, log_Edelta2, marker=".", alpha=0.6)
    fit_label = (
        f"Slope: {slope:.2f}, $R^2$: {r2:.2f}, RMSE: {rmse:.2f}\n"
        f"95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]"
    )

    ax.plot(log_n, const + log_n * slope, "b--", lw=2, label=fit_label)
    ax.set_xlabel("log(n)")
    ax.set_ylabel(r"$\log(E[\Delta_n^2 \mid Z_{1:n-1}])$")
    ax.set_title(f"n_estimators={n_est}")
    ax.legend(loc="lower left")

fig.suptitle(
    rf"Power Law of $E[\Delta_n^2 \mid Z_{{1:n-1}}]$, x_new={x_new[x_new_idx]}, t={t[t_idx]}"
)
fig.tight_layout()

# %%
## Power law of \log E(\Delta_k^2 \mid Z{1:k-1}) of all reps
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
for n_est, ax in zip(n_est_list, axes):
    outdir = utils.get_matching_dirs(id_dir, rf"quasi setup=gaussian-linear.+n_est={n_est} .+fix=True")
    outdir = outdir[0]
    outer_dirs = utils.get_matching_dirs(outdir, r"outer-\d+")

    # use all reps
    deltas_all = []
    # weights_all = []
    n_all = []
    for od in outer_dirs:
        deltas, n_tail = compile_outer(od, n_tail_spec)  # (k, mc_samples, t, x_new)
        deltas_all.append(deltas)
        # weights_all.append(weights)
        n_all.append(n_tail)

    deltas = np.stack(deltas_all, axis=0)  # (outer, k, mc_samples, t, x_new)
    # weights = np.stack(weights_all, axis=0)  # (outer, k, 8, t, x_new)
    n_tail = np.stack(n_all, axis=0)  # (outer, k)

    log_delta2 = 2 * np.log(np.abs(deltas[..., t_idx, x_new_idx]))
    log_Edelta2 = logsumexp(log_delta2, axis=-1) - np.log(log_delta2.shape[-1])
    log_n = np.log(n_tail)
    slope, const, ci, rmse, r2 = fit_lm2(log_n, log_Edelta2)

    # Normalising constant sanity check
    ax.scatter(log_n, log_Edelta2, alpha=0.6, s=1)
    fit_label = (
        f"Slope: {slope:.2f}, $R^2$: {r2:.2f}, RMSE: {rmse:.2f}\n"
        f"95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]"
    )

    y_fit = const[:, None] + log_n * slope
    ax.plot(log_n[0], y_fit[0], "b--", lw=2, label=fit_label)
    ax.plot(log_n.T, y_fit.T, "b--", lw=2, alpha=0.2)

    ax.set_xlabel("log(n)")
    ax.set_ylabel(r"$\log(E[\Delta_n^2 \mid Z_{1:n-1}])$")
    ax.set_title(f"n_estimators={n_est}")
    ax.legend(loc="lower left")

# fig.suptitle(
#     rf"Power Law of $E[\Delta_n^2 \mid Z_{{1:n-1}}]$, x_new={x_new[x_new_idx]}, t={t[t_idx]}"
# )
fig.tight_layout()


# %%
# Table of fit per rollout
df_rows = []
for n_est in n_est_list:
    outdir = utils.get_matching_dirs(id_dir, rf"quasi setup=gaussian-linear.+n_est={n_est} .+fix=True")
    assert len(outdir) == 1
    outdir = outdir[0]
    outer_dirs = utils.get_matching_dirs(outdir, r"outer-\d+")
    assert len(outer_dirs) == 100
    for d in outer_dirs:
        deltas, n_tail = compile_outer(d, n_tail_spec)  # (k, mc_samples, t, x_new)
        delta2 = np.square(deltas)[..., t_idx, x_new_idx]  # (k, mc_samples)
        Edelta2 = np.mean(delta2, axis=1)  # (k, )
        log_Edelta2 = np.log(Edelta2)
        log_n = np.log(n_tail)
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
df.groupby("n_est").mean()
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
# fig.savefig(f"{image_dir}/quasi-gamma-ci.pdf")

# %%
# Histogram of the slope per rollout
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
for n_est, ax in zip(n_est_list, axes):
    df_n_est = df[df["n_est"] == n_est]
    slopes = -df_n_est["slope"]
    ax.hist(slopes, bins=30)
    ax.axvline(slopes.mean(), color="r", linestyle="--", label="Mean")
    ax.axvline(slopes.median(), color="b", linestyle="-", label="Median")
    ax.set_xlabel(r"$\gamma_r$")
    ax.set_ylabel("Count")
    ax.set_title(f"n_estimators={n_est}")
    ax.legend()

fig.tight_layout()
# fig.savefig(f"{image_dir}/quasi-gamma-histogram.pdf")

# %%
# Plot of n^(fitted) * E[Delta^2_n]
df_median = df.groupby("n_est").median()
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
for n_est, ax in zip(n_est_list, axes):
    outdir = utils.get_matching_dirs(id_dir, rf"quasi setup=gaussian-linear.+n_est={n_est} .+fix=True")
    outdir = outdir[0]
    outer_dirs = utils.get_matching_dirs(outdir, r"outer-\d+")
    for d in outer_dirs:
        deltas, n_tail = compile_outer(d, n_tail_spec)  # (k, mc_samples, t, x_new)
        delta2 = np.square(deltas)[..., t_idx, x_new_idx]  # (k, mc_samples)
        fitted_power = df_median.loc[n_est, "slope"]
        factor = n_tail ** (-1 * fitted_power)
        n2Edelta2 = factor * np.mean(delta2, axis=1)  # (k, )
        ax.plot(n_tail, n2Edelta2, alpha=0.2)
        ax.set_yscale("log")
    ax.set_xlabel("n")
    # ax.set_ylabel(rf"$n^{{\widehat\gamma_{{med}}}} E[\Delta_n^2 \mid Z_{{1:n-1}}]$")
    ax.set_ylabel(rf"$n^{{\widehat\gamma_{{med}}}} b^\prime_n$")
    ax.set_title(rf"n_estimators={n_est}, $\widehat\gamma_{{med}}$={-fitted_power:.2f}")
# fig.suptitle(
#     f"Power Law of E[|b_n|] with fitted normalising constant, x_new={x_new[x_new_idx]}, t={t[t_idx]}"
# )
fig.tight_layout()
# fig.savefig(f"{image_dir}/quasi-gamma-constant.pdf")
# %%

# %%
