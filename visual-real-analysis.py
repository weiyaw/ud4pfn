# %%
import re

import matplotlib.pyplot as plt
import numpy as np

import utils
from constants import DEFAULT_T_IDX
from metrics import (
    build_pointwise_band,
    build_simultaneous_band,
)
from posterior import compute_vn

image_dir = "../paper/neurips2026/images"

# %%
# Labour force
expdir = "../outputs/2026-01-51/"
outdir = utils.get_matching_dirs(expdir, r"labour-force.+n_est=64")
assert len(outdir) == 1
outdir = outdir[0]
setup_name = re.search(r"setup=([^\s]+)", outdir).group(1)
data = utils.read_from(f"{outdir}/data.pickle")
x_prev = data["x_prev"]
y_prev = data["y_prev"]
x_grid = data["x_grid"]
grid_shape = data["grid_shape"]

t_idx = DEFAULT_T_IDX[setup_name]
t = data["t"][t_idx]
gn = utils.read_from(f"{outdir}/gn.pickle")[t_idx]
g0_to_gn = utils.read_from(f"{outdir}/g0_to_gn.pickle")[:, t_idx]
gn_plus_1 = utils.read_from(f"{outdir}/gn_plus_1.pickle")[:, t_idx]

n = y_prev.size


def plot_band_lf(ax, x_grid, ci_band, X):
    x_grid = x_grid.squeeze()
    ax.fill_between(
        x_grid, ci_band["lower"], ci_band["upper"], alpha=0.25, label="95% band"
    )
    ax.plot(x_grid, ci_band["mean"], "k", lw=1.5, label=r"$g_n(x)$")
    ax.scatter(X, np.full_like(X, 0.12), marker="|", s=50, c="black", alpha=0.6)
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(0.1, 0.9)
    ax.legend(loc="upper right")


fig, axes = plt.subplots(1, 2, figsize=(12, 2.5))
clt_cov = compute_vn(g0_to_gn, type="pointwise") / n
ci_band = build_pointwise_band(g0_to_gn[-1], clt_cov)
plot_band_lf(axes[0], x_grid, ci_band, x_prev)
axes[0].set_title(r"Pointwise band with $V_n$")

clt_cov = compute_vn(g0_to_gn, type="simultaneous") / n
ci_band = build_simultaneous_band(g0_to_gn[-1], clt_cov)
plot_band_lf(axes[1], x_grid, ci_band, x_prev)
axes[1].set_title(r"Simultaneous band with $V_n$")

fig.tight_layout()
fig.savefig(f"{image_dir}/labour-force-vn.pdf")


# %%
# Fibre strength
expdir = "../outputs/2026-01-51/"
outdir = utils.get_matching_dirs(expdir, r"fibre-strength.+n_est=64")
assert len(outdir) == 1
outdir = outdir[0]
setup_name = re.search(r"setup=([^\s]+)", outdir).group(1)
data = utils.read_from(f"{outdir}/data.pickle")
x_prev = data["x_prev"]
y_prev = data["y_prev"]
x_grid = data["x_grid"]
grid_shape = data["grid_shape"]

t_idx = DEFAULT_T_IDX[setup_name]
t = data["t"][t_idx]
gn = utils.read_from(f"{outdir}/gn.pickle")[t_idx]
g0_to_gn = utils.read_from(f"{outdir}/g0_to_gn.pickle")[:, t_idx]
gn_plus_1 = utils.read_from(f"{outdir}/gn_plus_1.pickle")[:, t_idx]

n = y_prev.size


def plot_band_fs(ax, x_grid, ci_band, X):
    x_grid = x_grid.squeeze()
    ax.fill_between(
        x_grid, ci_band["lower"], ci_band["upper"], alpha=0.25, label="95% band"
    )
    ax.plot(x_grid, ci_band["mean"], "k", lw=1.5, label=r"$R(1.5 MPa \mid x)$")
    ax.scatter(X, np.full_like(X, 0.42), marker="|", s=50, c="black", alpha=0.6)
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(0.4, 1.05)
    ax.legend(loc="upper right")


fig, axes = plt.subplots(1, 2, figsize=(12, 3))

clt_cov = compute_vn(g0_to_gn, type="pointwise") / n
ci_band = build_pointwise_band(g0_to_gn[-1], clt_cov)
plot_band_fs(axes[0], x_grid, ci_band, x_prev)
axes[0].set_title(r"Pointwise band with $V_n$")

clt_cov = compute_vn(g0_to_gn, type="simultaneous") / n
ci_band = build_simultaneous_band(g0_to_gn[-1], clt_cov)
plot_band_fs(axes[1], x_grid, ci_band, x_prev)
axes[1].set_title(r"Simultaneous band with $V_n$")

fig.tight_layout()
fig.savefig(f"{image_dir}/fibre-strength-vn.pdf")
# %%
