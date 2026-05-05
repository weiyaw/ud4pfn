# %%
import re

import matplotlib.pyplot as plt
import numpy as np

import utils
from constants import DEFAULT_T_IDX
from metrics import build_pointwise_band, build_simultaneous_band
from posterior import compute_un, compute_vn

# %load_ext autoreload
# %autoreload 2


def plot_band(ax, x_grid, ci_band, true_event, X):
    x_grid = x_grid.squeeze()
    ax.fill_between(
        x_grid, ci_band["lower"], ci_band["upper"], alpha=0.25, label=f"95% band"
    )
    ax.plot(x_grid, ci_band["mean"], "k", lw=1.5, label="Mean")
    ax.plot(x_grid, true_event.squeeze(), "k--", lw=1, label="True probability")
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


# %%
## BAND
id_dir = "../outputs/2026-01-12/"
image_dir = "../paper/neurips2026/images"

regressor = [
    "gaussian-linear",
    "gaussian-polynomial",
    "gaussian-linear-dependent-error",
    "gaussian-sine",
    "poisson-linear",
]

classifier = [
    "probit-mixture",
    "categorical-linear",
]

n_list = [200, 500, 1000]

for name in regressor + classifier:
    fig, axes = plt.subplots(5, len(n_list), figsize=(12, 14))
    for i, n in enumerate(n_list):
        outdir = utils.get_matching_dirs(id_dir, rf"{name} .+n={n} .+")
        assert len(outdir) == 1
        outdir = outdir[0]
        setup_name = re.search(r"setup=([^\s]+)", outdir).group(1)
        data = utils.read_from(f"{outdir}/data.pickle")
        x_prev = data["x_prev"]
        y_prev = data["y_prev"]
        x_grid = data["x_grid"]
        grid_shape = data["grid_shape"]

        t = data["t"]
        t_idx = DEFAULT_T_IDX[name]
        gn = utils.read_from(f"{outdir}/gn.pickle")[t_idx]
        g0_to_gn = utils.read_from(f"{outdir}/g0_to_gn.pickle")[:, t_idx]
        gn_plus_1 = utils.read_from(f"{outdir}/gn_plus_1.pickle")[:, t_idx]
        true_prob = data["true_prob"][t_idx]

        n = y_prev.size
        axes[0, i].scatter(x_prev, y_prev, color="black", s=10, alpha=0.5, label="data")
        axes[0, i].set_title(f"Data ($n={n}$)")

        clt_cov = compute_vn(g0_to_gn, type="pointwise") / n
        ci_band = build_pointwise_band(g0_to_gn[-1], clt_cov)
        plot_band(axes[1, i], x_grid, ci_band, true_prob, x_prev)
        axes[1, i].set_title("$V_n$ Pointwise")

        clt_cov = compute_un(gn, gn_plus_1, n, type="pointwise") / n
        ci_band = build_pointwise_band(gn, clt_cov)
        plot_band(axes[2, i], x_grid, ci_band, true_prob, x_prev)
        axes[2, i].set_title("$U_n$ Pointwise")

        clt_cov = compute_vn(g0_to_gn, type="simultaneous") / n
        ci_band = build_simultaneous_band(g0_to_gn[-1], clt_cov)
        plot_band(axes[3, i], x_grid, ci_band, true_prob, x_prev)
        axes[3, i].set_title("$V_n$ Simultaneous")

        clt_cov = compute_un(gn, gn_plus_1, n, type="simultaneous") / n
        ci_band = build_simultaneous_band(gn, clt_cov)
        plot_band(axes[4, i], x_grid, ci_band, true_prob, x_prev)
        axes[4, i].set_title("$U_n$ Simultaneous")

        for k in range(0, 5):
            axes[k, i].set_xlim(-10, 10)
            axes[k, i].set_xlabel("x")

        axes[0, i].set_ylabel("y")
        for k in range(1, 5):
            if name in regressor:
                axes[k, i].set_ylabel(f"$P(Y \\leq {int(t[t_idx])} | x)$")
            elif name in classifier:
                axes[k, i].set_ylabel(f"$P(Y = 1 | x)$")
            axes[k, i].set_ylim(-0.01, 1.01)

    axes[-1, 2].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{image_dir}/gap-{name}.pdf")


# %%
