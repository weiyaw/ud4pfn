# %%
import re

import matplotlib.pyplot as plt
import numpy as np

from experiments._shared.artifacts import find_run_directories, read_pickle
from experiments._shared.runtime import FIGURES_ROOT, OUTPUTS_ROOT
from predictive_clt import (
    build_pointwise_band,
    build_simultaneous_band,
    compute_un,
    compute_vn,
)
from .run import EXPERIMENT_DEFINITIONS

# %load_ext autoreload
# %autoreload 2


def plot_band(ax, x_grid, ci_band, true_event, X):
    x_grid = x_grid.squeeze()
    ax.fill_between(
        x_grid, ci_band["lower"], ci_band["upper"], alpha=0.25, label="95% band"
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


# Resolve all repo paths relative to the repo root (this file lives at the repo
# root), never relative to the current working directory.
OUTPUTS_DIR = OUTPUTS_ROOT
FIG_DIR = FIGURES_ROOT

# %%
## BAND
id_dir = str(OUTPUTS_DIR / "gap")
image_dir = str(FIG_DIR)

REGRESSION_EXPERIMENT_NAMES = [
    "gaussian-linear",
    "gaussian-polynomial",
    "gaussian-linear-dependent-error",
    "gaussian-sine",
    "poisson-linear",
]

CLASSIFICATION_EXPERIMENT_NAMES = [
    "probit-mixture",
    "categorical-linear",
]

n_list = [200, 500, 1000]

for name in REGRESSION_EXPERIMENT_NAMES + CLASSIFICATION_EXPERIMENT_NAMES:
    fig, axes = plt.subplots(5, len(n_list), figsize=(12, 14))
    for i, n in enumerate(n_list):
        outdir = find_run_directories(id_dir, rf"{name} .+n={n} .+")
        assert len(outdir) == 1
        outdir = outdir[0]
        experiment_name = re.search(r"setup=([^\s]+)", outdir.name).group(1)
        data = read_pickle(outdir / "data.pickle")
        x_prev = data["x_prev"]
        y_prev = data["y_prev"]
        x_grid = data["x_grid"]
        grid_shape = data["grid_shape"]

        t = data["t"]
        t_idx = EXPERIMENT_DEFINITIONS[experiment_name].default_event_index
        gn = read_pickle(outdir / "gn.pickle")[t_idx]
        g0_to_gn = read_pickle(outdir / "g0_to_gn.pickle")[:, t_idx]
        gn_plus_1 = read_pickle(outdir / "gn_plus_1.pickle")[:, t_idx]
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
            if name in REGRESSION_EXPERIMENT_NAMES:
                axes[k, i].set_ylabel(f"$P(Y \\leq {int(t[t_idx])} | x)$")
            elif name in CLASSIFICATION_EXPERIMENT_NAMES:
                axes[k, i].set_ylabel("$P(Y = 1 | x)$")
            axes[k, i].set_ylim(-0.01, 1.01)

    # Single figure-level legend below all panels (previously sat inside the
    # bottom-right panel and occluded the band/mean there).
    h_data, l_data = axes[0, 0].get_legend_handles_labels()
    h_band, l_band = axes[1, 0].get_legend_handles_labels()
    handles = h_data + h_band
    labels = l_data + l_band
    fig.tight_layout()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.savefig(f"{image_dir}/gap-{name}.pdf", bbox_inches="tight")


# %%
