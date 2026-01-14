# %%
import numpy as np
import utils
import re
import os
import matplotlib.pyplot as plt


def compile_outer(outer_dir):
    # Compile and produce a single outer
    FnQn_paths = utils.get_matching_files(outer_dir, r"FnQn-\d+\.pickle")
    FnQn_paths.sort(key=lambda x: int(re.search(r"FnQn-(\d+)", x).group(1)))
    FnQn_raw = [utils.read_from(p) for p in FnQn_paths]
    F_n = np.asarray([d["F_n"] for d in FnQn_raw])
    Q_n = np.asarray([d["Q_n"] for d in FnQn_raw])
    n = np.asarray([d["n"] for d in FnQn_raw])
    return F_n, Q_n, n


def compile_all_outer(outer_dirs):
    # Compile all outer and average
    outer0_delta0 = utils.read_from(
        utils.get_matching_files(outer_dirs[0], r"FnQn-\d+\.pickle")[0]
    )
    t = outer0_delta0["t"]
    u = outer0_delta0["u"]
    x_new = outer0_delta0["x_new"]

    F_n_all = []
    Q_n_all = []
    n_all = []
    for d in outer_dirs:
        F_n, Q_n, n = compile_outer(d)
        F_n_all.append(F_n)
        Q_n_all.append(Q_n)
        n_all.append(n)
    F_n_all = np.mean(np.stack(F_n_all), axis=0)  # (n, t, x_new)
    Q_n_all = np.mean(np.stack(Q_n_all), axis=0)  # (n, t, x_new)
    n_all = np.stack(n_all)  # (outer, n)
    assert np.all(n_all == n_all[0])
    n = n_all[0]
    return F_n_all, Q_n_all, n, t, u, x_new


def plot_FF(ax, F_n, k):
    colors = plt.cm.viridis((k - k.min()) / (k.max() - k.min()))
    for i, color in enumerate(colors):
        ax.plot(F_n[0], F_n[i], color=color, alpha=0.5)
    ax.plot([F_n.min(), F_n.max()], [F_n.min(), F_n.max()], color="black", linestyle="--")
    ax.grid()
    ax.set_xlabel(r"$F_{n_0}(t)$")


def plot_QQ(ax, Q_n, k):
    colors = plt.cm.viridis((k - k.min()) / (k.max() - k.min()))
    for i, color in enumerate(colors):
        ax.plot(Q_n[0], Q_n[i], color=color, alpha=0.5)
    ax.plot([Q_n.min(), Q_n.max()], [Q_n.min(), Q_n.max()], color="black", linestyle="--")
    ax.grid()
    ax.set_xlabel(r"$F^{-1}_{n_0}(u)$")


# %%
n0 = 25
x_new_idx = 1

n_est_ls = [1, 8, 16, 64]
fig1, ax1 = plt.subplots(1, 4, figsize=(12, 3), constrained_layout=True)
fig2, ax2 = plt.subplots(1, 4, figsize=(12, 3), constrained_layout=True)

for n_est, a1, a2 in zip(n_est_ls, ax1, ax2):
    outdir = utils.get_matching_dirs("../outputs/2026-01-41/", rf"cid.+n_est={n_est} ")[0]
    print(outdir)
    outer_dirs = utils.get_matching_dirs(outdir, r"outer-\d+")
    F_n_all, Q_n_all, n, t, u, x_new = compile_all_outer(outer_dirs)
    F_n = F_n_all[..., x_new_idx]
    Q_n = Q_n_all[..., x_new_idx]
    k = n - n0
    plot_FF(a1, F_n, k)
    plot_QQ(a2, Q_n, k)
    a1.set_title(f"n_estimators = {n_est}")
    a2.set_title(f"n_estimators = {n_est}")

ax1[0].set_ylabel(r"$\bar{F}_{n_0 + k}(t)$")
ax2[0].set_ylabel(r"$\bar{F}^{-1}_{n_0 + k}(u)$")

fig1.colorbar(
    plt.cm.ScalarMappable(norm=plt.Normalize(k.min(), k.max()), cmap="viridis"),
    label="k",
    ax=ax1[3],
)
fig2.colorbar(
    plt.cm.ScalarMappable(norm=plt.Normalize(k.min(), k.max()), cmap="viridis"),
    label="k",
    ax=ax2[3],
)

if not os.path.exists("images"):
    os.makedirs("images")
fig1.savefig("images/cid-pp.pdf")
fig2.savefig("images/cid-qq.pdf")


# %%
