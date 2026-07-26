# %%
import matplotlib.pyplot as plt
import numpy as np

from experiments._shared.artifacts import find_run_directories, read_pickle
from experiments._shared.runtime import FIGURES_ROOT, OUTPUTS_ROOT
from predictive_clt import compute_vn
from .entropy import (
    compute_aleatoric_entropy_binary,
    compute_aleatoric_entropy_multiclass,
    compute_total_entropy_binary,
    compute_total_entropy_multiclass,
)
from .run import EXPERIMENT_DEFINITIONS

# %load_ext autoreload
# %autoreload 2

# Resolve all repo paths relative to the repo root (this file lives at the repo
# root), never relative to the current working directory.
OUTPUTS_DIR = OUTPUTS_ROOT
FIG_DIR = FIGURES_ROOT

id_dir = str(OUTPUTS_DIR / "entropic-ud")
image_dir = str(FIG_DIR)

# %%
## 1D UQ decomposition at various x^*
n_list = [15, 50, 75, 150]
t_idx = EXPERIMENT_DEFINITIONS["logistic-linear"].default_event_index
fig, axes = plt.subplots(2, 2, figsize=(12, 6))

y_lo_all, y_hi_all = [], []
for n, ax in zip(n_list, axes.flatten()):
    outdir = find_run_directories(id_dir, rf"logistic-linear.+n={n} .+seed=1000")
    assert len(outdir) == 1
    outdir = outdir[0]
    data = read_pickle(f"{outdir}/data.pickle")
    x_prev = data["x_prev"]
    y_prev = data["y_prev"]
    x_grid = data["x_grid"]
    grid_shape = data["grid_shape"]
    t = data["t"]
    x_grid = data["x_grid"]

    gn = read_pickle(f"{outdir}/gn.pickle")[t_idx]
    g0_to_gn = read_pickle(f"{outdir}/g0_to_gn.pickle")[:, t_idx]
    true_prob = data["true_prob"][t_idx]

    clt_var = compute_vn(g0_to_gn, type="pointwise") / y_prev.size
    total_entropy = compute_total_entropy_binary(gn)
    assert total_entropy.shape == clt_var.shape == gn.shape
    alea_entropy = compute_aleatoric_entropy_binary(gn, clt_var)

    ax.plot(x_grid.squeeze(), total_entropy, label="Total Uncertainty")
    ax.plot(x_grid.squeeze(), alea_entropy, label="Aleatoric Uncertainty")
    ax.vlines(
        x_prev[y_prev == 0], 0, 1, "m", alpha=0.4, linestyle="--", label="Data (y=0)"
    )
    ax.vlines(
        x_prev[y_prev == 1], 0, 1, "c", alpha=0.4, linestyle="--", label="Data (y=1)"
    )
    y_lo_all.append(min(alea_entropy))
    y_hi_all.append(max(total_entropy))
    ax.set_xlabel("Test covariate $x^*$")
    ax.set_ylabel("Uncertainty")
    ax.set_title(f"n={n}")

# Shared y-limits across all panels so the curves are directly comparable.
y_lo = min(y_lo_all) * 0.98
y_hi = max(y_hi_all) * 1.02
for ax in axes.flatten():
    ax.set_ylim(y_lo, y_hi)

# Single figure-level legend placed above the panels, off the data.
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4)
fig.tight_layout()
fig.savefig(f"{image_dir}/ud-logreg-xstar.pdf", bbox_inches="tight")

# %%
## 1D UQ decomposition at various n
# The context-length sweep reads the `entropic-ud-vary-n` outputs. When those
# outputs are not materialised, skip this block (and its two figures) entirely.
if (OUTPUTS_DIR / "entropic-ud-vary-n").exists():
    n_list = range(75, 201, 5)
    t_idx = EXPERIMENT_DEFINITIONS["logistic-linear"].default_event_index
    x_grid_idx = [0, 25, 50, 75, 100, 125, 150]

    total_entropy_all = []
    alea_entropy_all = []
    for n in n_list:
        # use entropic-ud-vary-n
        outdir = find_run_directories(
            str(OUTPUTS_DIR / "entropic-ud-vary-n"), rf"logistic-linear.+n={n} .+"
        )
        assert len(outdir) == 50
        total_entropy_seeds = []
        alea_entropy_seeds = []
        for d in outdir:
            data = read_pickle(f"{d}/data.pickle")
            x_grid = data["x_grid"]
            gn = read_pickle(f"{d}/gn.pickle")[t_idx, x_grid_idx]
            g0_to_gn = read_pickle(f"{d}/g0_to_gn.pickle")[:, t_idx, x_grid_idx]
            true_prob = data["true_prob"][t_idx]

            clt_var = compute_vn(g0_to_gn, type="pointwise") / n
            total_entropy = compute_total_entropy_binary(gn)
            assert total_entropy.shape == clt_var.shape == gn.shape
            alea_entropy = compute_aleatoric_entropy_binary(gn, clt_var)
            total_entropy_seeds.append(total_entropy)
            alea_entropy_seeds.append(alea_entropy)
        total_entropy_all.append(np.stack(total_entropy_seeds))
        alea_entropy_all.append(np.stack(alea_entropy_seeds))
    total_entropy_all = np.stack(total_entropy_all)  # (n, rep, x_grid)
    alea_entropy_all = np.stack(alea_entropy_all)  # (n, rep, x_grid)

    total_entropy_avg = np.mean(total_entropy_all, axis=1)  # (n, x_grid)
    alea_entropy_avg = np.mean(alea_entropy_all, axis=1)  # (n, x_grid)
    epis_entropy_avg = total_entropy_avg - alea_entropy_avg

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    for i, x in enumerate(data["x_grid"][x_grid_idx]):
        total_entropy = total_entropy_avg[..., i]
        alea_entropy = alea_entropy_avg[..., i]
        epis_entropy = total_entropy - alea_entropy
        if x.item() > -1 and x.item() < 6:
            axes[0].plot(n_list, epis_entropy, label=f"x={x.item()}")
            axes[1].plot(n_list, alea_entropy, label=f"x={x.item()}")
        else:
            axes[0].plot(n_list, epis_entropy, "--", label=f"x={x.item()}")
            axes[1].plot(n_list, alea_entropy, "--", label=f"x={x.item()}")
    axes[0].legend(loc="upper right", ncol=2, fontsize=10)
    axes[0].set_ylabel("Entropy", fontsize=16)
    axes[0].set_xlabel("Dataset Size/Context Length", fontsize=16)
    axes[1].set_xlabel("Dataset Size/Context Length", fontsize=16)
    axes[0].set_title("Epistemic Uncertainty", fontsize=16)
    axes[1].set_title("Aleatoric Uncertainty", fontsize=16)
    fig.savefig(f"{image_dir}/ud-logreg-context-length.pdf")

    # # Sanity check the proportion of aleatoric uncertainty
    fig, axes2 = plt.subplots(2, 1, figsize=(5, 6.5), constrained_layout=True)
    for i, x in enumerate(data["x_grid"][x_grid_idx]):
        alea_prop = (
            np.mean(alea_entropy_all, axis=1)[..., i]
            / np.mean(total_entropy_all, axis=1)[..., i]
        )
        epis_prop = 1 - alea_prop
        if x.item() > -1 and x.item() < 6:
            axes2[0].plot(n_list, epis_prop, label=f"x={x.item()}")
            axes2[1].plot(n_list, alea_prop, label=f"x={x.item()}")
        else:
            axes2[0].plot(n_list, epis_prop, "--", label=f"x={x.item()}")
            axes2[1].plot(n_list, alea_prop, "--", label=f"x={x.item()}")
    axes2[0].legend(loc="upper right", ncol=2, fontsize=10)
    fig.supylabel("Proportion of Total Uncertainty", fontsize=14)
    axes2[1].set_xlabel("Dataset Size/Context Length", fontsize=14)
    axes2[0].set_title("Epistemic Uncertainty", fontsize=14)
    axes2[1].set_title("Aleatoric Uncertainty", fontsize=14)
    fig.savefig(f"{image_dir}/ud-logreg-context-length-prop.pdf")


# %%
## 2D UQ decomposition (two moons)
setup_regex_list = [
    ("two-moons-1.+n=30", "Moons 1, n=30"),
    ("two-moons-1.+n=100", "Moons 1, n=100"),
    ("two-moons-2.+n=30", "Moons 2, n=30"),
    ("two-moons-2.+n=100", "Moons 2, n=100"),
]
t_idx = EXPERIMENT_DEFINITIONS["two-moons-1"].default_event_index

fig, axes = plt.subplots(
    5, len(setup_regex_list), figsize=(18, 20), constrained_layout=True
)

markers = ["o", "^", "s", "D", "v", "P", "X"]
row_titles = [
    "$g_n$",
    "$v_n / n$",
    "Total Uncertainty",
    "Aleatoric Uncertainty",
    "Epistemic Uncertainty",
]

# First pass: gather every column's grid and the five quantity fields, so that
# each row can share one colour scale (vmin/vmax computed across the row).
cols = []
for setup_regex, title in setup_regex_list:
    outdir = find_run_directories(id_dir, setup_regex)
    assert len(outdir) == 1
    outdir = outdir[0]
    data = read_pickle(f"{outdir}/data.pickle")
    x_prev = data["x_prev"]
    y_prev = data["y_prev"]
    x_grid = data["x_grid"]
    grid_shape = data["grid_shape"]
    n = y_prev.size

    gn = read_pickle(f"{outdir}/gn.pickle")[t_idx]
    g0_to_gn = read_pickle(f"{outdir}/g0_to_gn.pickle")[:, t_idx]

    clt_var = compute_vn(g0_to_gn, type="pointwise") / n
    total_entropy = compute_total_entropy_binary(gn)
    alea_entropy = compute_aleatoric_entropy_binary(gn, clt_var)
    assert total_entropy.shape == clt_var.shape == gn.shape == alea_entropy.shape
    epis_entropy = total_entropy - alea_entropy

    X = x_grid[:, 0].reshape(*grid_shape)
    Y = x_grid[:, 1].reshape(*grid_shape)

    cols.append(
        {
            "title": title,
            "X": X,
            "Y": Y,
            "x_prev": x_prev,
            "y_prev": y_prev,
            "fields": [
                gn.reshape(*grid_shape),
                clt_var.reshape(*grid_shape),
                total_entropy.reshape(*grid_shape),
                alea_entropy.reshape(*grid_shape),
                epis_entropy.reshape(*grid_shape),
            ],
        }
    )

# Second pass: one shared colour scale and one colourbar per row.
for row in range(5):
    vmin = min(c["fields"][row].min() for c in cols)
    vmax = max(c["fields"][row].max() for c in cols)
    im = None
    for j, c in enumerate(cols):
        ax = axes[row, j]
        im = ax.pcolormesh(
            c["X"],
            c["Y"],
            c["fields"][row],
            shading="auto",
            edgecolors="face",
            linewidths=0,
            rasterized=True,
            vmin=vmin,
            vmax=vmax,
        )
        for i, y_val in enumerate(np.unique(c["y_prev"])):
            mask = c["y_prev"] == y_val
            ax.scatter(
                c["x_prev"][mask, 0],
                c["x_prev"][mask, 1],
                label=f"y={y_val}",
                marker=markers[i % len(markers)],
                s=30,
            )
        ax.set_xlim(c["X"].min(), c["X"].max())
        ax.set_ylim(c["Y"].min(), c["Y"].max())
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title(f"{row_titles[row]} ({c['title']})")
    fig.colorbar(im, ax=axes[row, :].tolist())

axes[0, 0].legend(loc="upper right")
fig.savefig(f"{image_dir}/ud-two-moons.pdf")

# %%
## 2D UQ decomposition (two moons + spiral)
fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
markers = ["o", "^", "s", "D", "v", "P", "X"]

# Row 0: two moons (binary)
t_idx = EXPERIMENT_DEFINITIONS["two-moons-1"].default_event_index
outdir = find_run_directories(id_dir, setup_regex_list[0][0])
assert len(outdir) == 1
outdir = outdir[0]
data = read_pickle(f"{outdir}/data.pickle")
moons_x_prev = data["x_prev"]
moons_y_prev = data["y_prev"]
x_grid = data["x_grid"]
grid_shape = data["grid_shape"]
n = moons_y_prev.size

gn = read_pickle(f"{outdir}/gn.pickle")[t_idx]
g0_to_gn = read_pickle(f"{outdir}/g0_to_gn.pickle")[:, t_idx]
clt_var = compute_vn(g0_to_gn, type="pointwise") / n
total_entropy = compute_total_entropy_binary(gn)
alea_entropy = compute_aleatoric_entropy_binary(gn, clt_var)
epis_entropy = total_entropy - alea_entropy
moons_X = x_grid[:, 0].reshape(*grid_shape)
moons_Y = x_grid[:, 1].reshape(*grid_shape)
moons_fields = [
    total_entropy.reshape(*grid_shape),
    alea_entropy.reshape(*grid_shape),
    epis_entropy.reshape(*grid_shape),
]

# Row 1: spiral (multiclass)
outdir = find_run_directories(id_dir, "spiral")
assert len(outdir) == 1
outdir = outdir[0]
data = read_pickle(f"{outdir}/data.pickle")
spiral_x_prev = data["x_prev"]
spiral_y_prev = data["y_prev"]
x_grid = data["x_grid"]
grid_shape = data["grid_shape"]
n = spiral_y_prev.size
K = np.unique(spiral_y_prev).size

gn = read_pickle(f"{outdir}/gn.pickle")  # (K, m)
g0_to_gn = read_pickle(f"{outdir}/g0_to_gn.pickle")  # (n+1, K, m)
clt_var = np.array([compute_vn(g0_to_gn[:, k], type="pointwise") / n for k in range(K)])
total_entropy = compute_total_entropy_multiclass(gn)  # (m,)
alea_entropy = compute_aleatoric_entropy_multiclass(gn, clt_var)  # (m,)
epis_entropy = total_entropy - alea_entropy  # (m,)
spiral_X = x_grid[:, 0].reshape(*grid_shape)
spiral_Y = x_grid[:, 1].reshape(*grid_shape)
spiral_fields = [
    total_entropy.reshape(*grid_shape),
    alea_entropy.reshape(*grid_shape),
    epis_entropy.reshape(*grid_shape),
]

col_titles = ["Total Uncertainty", "Aleatoric Uncertainty", "Epistemic Uncertainty"]
rows = [
    (moons_X, moons_Y, moons_x_prev, moons_y_prev, moons_fields),
    (spiral_X, spiral_Y, spiral_x_prev, spiral_y_prev, spiral_fields),
]


def plot_panel(ax, X, Y, Z, x_prev, y_prev, vmin, vmax):
    im = ax.pcolormesh(
        X,
        Y,
        Z,
        shading="auto",
        edgecolors="face",
        linewidths=0,
        rasterized=True,
        vmin=vmin,
        vmax=vmax,
    )
    for i, y_val in enumerate(np.unique(y_prev)):
        mask = y_prev == y_val
        ax.scatter(
            x_prev[mask, 0],
            x_prev[mask, 1],
            label=f"y={y_val}",
            marker=markers[i % len(markers)],
            s=30,
        )
    # Clamp to the evaluation-grid extent so no unrendered edge band appears.
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    return im


for r, (X, Y, x_prev, y_prev, fields) in enumerate(rows):
    # Shared colour scale across total / aleatoric / epistemic within the row.
    vmin = min(f.min() for f in fields)
    vmax = max(f.max() for f in fields)
    im = None
    for c in range(3):
        im = plot_panel(axes[r, c], X, Y, fields[c], x_prev, y_prev, vmin, vmax)
        axes[r, c].set_title(col_titles[c])
    axes[r, 0].legend(loc="upper right")
    fig.colorbar(im, ax=axes[r, :].tolist())

fig.savefig(f"{image_dir}/ud-two-moons-spiral.pdf")
# %%
