# %%
import matplotlib.pyplot as plt
import numpy as np

import utils
from constants import DEFAULT_T_IDX
from metrics import (
    compute_aleatoric_entropy_binary,
    compute_aleatoric_entropy_multiclass,
    compute_total_entropy_binary,
    compute_total_entropy_multiclass,
)
from posterior import compute_vn

# %load_ext autoreload
# %autoreload 2

id_dir = "../outputs/2026-01-61"
image_dir = "../paper/images"

# %%
## 1D UQ decomposition at various x^*
n_list = [15, 50, 75, 150]
t_idx = DEFAULT_T_IDX["logistic-linear"]
fig, axes = plt.subplots(2, 2, figsize=(12, 6))

for n, ax in zip(n_list, axes.flatten()):
    outdir = utils.get_matching_dirs(id_dir, rf"logistic-linear.+n={n} .+seed=1000")
    assert len(outdir) == 1
    outdir = outdir[0]
    data = utils.read_from(f"{outdir}/data.pickle")
    x_prev = data["x_prev"]
    y_prev = data["y_prev"]
    x_grid = data["x_grid"]
    grid_shape = data["grid_shape"]
    t = data["t"]
    x_grid = data["x_grid"]

    gn = utils.read_from(f"{outdir}/gn.pickle")[t_idx]
    g0_to_gn = utils.read_from(f"{outdir}/g0_to_gn.pickle")[:, t_idx]
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
    ax.set_ylim(min(alea_entropy) * 0.98, max(total_entropy) * 1.02)
    ax.set_xlabel("Test covariate $x^*$")
    ax.set_ylabel("Uncertainty")
    ax.set_title(f"n={n}")

axes[0, 1].legend(loc="upper right")
fig.tight_layout()
fig.savefig(f"{image_dir}/ud-logreg-xstar.pdf")

# %%
## 1D UQ decomposition at various n
n_list = range(75, 201, 5)
t_idx = DEFAULT_T_IDX["logistic-linear"]
x_grid_idx = [0, 25, 50, 75, 100, 125, 150]

total_entropy_all = []
alea_entropy_all = []
for n in n_list:
    # use 2026-01-62
    outdir = utils.get_matching_dirs(
        "../outputs/2026-01-62", rf"logistic-linear.+n={n} .+"
    )
    assert len(outdir) == 50
    total_entropy_seeds = []
    alea_entropy_seeds = []
    for d in outdir:
        data = utils.read_from(f"{d}/data.pickle")
        x_grid = data["x_grid"]
        gn = utils.read_from(f"{d}/gn.pickle")[t_idx, x_grid_idx]
        g0_to_gn = utils.read_from(f"{d}/g0_to_gn.pickle")[:, t_idx, x_grid_idx]
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
fig.tight_layout()

# # Sanity check the proportion of aleatoric uncertainty
fig, axes2 = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
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
axes2[1].legend(loc="lower right", ncol=2, fontsize=10)
axes2[0].set_ylabel("Proportion of Total Uncertainty", fontsize=14)
axes2[0].set_xlabel("Dataset Size/Context Length", fontsize=14)
axes2[1].set_xlabel("Dataset Size/Context Length", fontsize=14)
axes2[0].set_title("Epistemic Uncertainty", fontsize=14)
axes2[1].set_title("Aleatoric Uncertainty", fontsize=14)
fig.savefig(f"{image_dir}/ud-logreg-context-length-prop.pdf")
fig.tight_layout()


# %%
## 2D UQ decomposition (two moons)
setup_regex_list = [
    ("two-moons-1.+n=30", "Moons 1, n=30"),
    ("two-moons-1.+n=100", "Moons 1, n=100"),
    ("two-moons-2.+n=30", "Moons 2, n=30"),
    ("two-moons-2.+n=100", "Moons 2, n=100"),
]
t_idx = DEFAULT_T_IDX["two-moons-1"]

fig, axes = plt.subplots(5, len(setup_regex_list), figsize=(18, 20))

for j, (setup_regex, title) in enumerate(setup_regex_list):
    outdir = utils.get_matching_dirs(id_dir, setup_regex)
    assert len(outdir) == 1
    outdir = outdir[0]
    data = utils.read_from(f"{outdir}/data.pickle")
    x_prev = data["x_prev"]
    y_prev = data["y_prev"]
    x_grid = data["x_grid"]
    grid_shape = data["grid_shape"]
    n = y_prev.size

    t = data["t"][t_idx]
    grid_shape = data["grid_shape"]
    gn = utils.read_from(f"{outdir}/gn.pickle")[t_idx]
    g0_to_gn = utils.read_from(f"{outdir}/g0_to_gn.pickle")[:, t_idx]
    true_prob = data["true_prob"][t_idx]

    clt_var = compute_vn(g0_to_gn, type="pointwise") / n
    total_entropy = compute_total_entropy_binary(gn)
    alea_entropy = compute_aleatoric_entropy_binary(gn, clt_var)
    assert total_entropy.shape == clt_var.shape == gn.shape == alea_entropy.shape
    epis_entropy = total_entropy - alea_entropy

    # Plot
    X = x_grid[:, 0].reshape(*grid_shape)
    Y = x_grid[:, 1].reshape(*grid_shape)

    unique_ys = np.unique(y_prev)
    markers = ["o", "^", "s", "D", "v", "P", "X"]

    def plot_heatmap(ax, X, Y, Z):
        im = ax.pcolormesh(
            X, Y, Z, shading="auto", edgecolors="face", linewidths=0, rasterized=True
        )
        for i, y_val in enumerate(unique_ys):
            mask = y_prev == y_val
            ax.scatter(
                x_prev[mask, 0],
                x_prev[mask, 1],
                label=f"y={y_val}",
                marker=markers[i % len(markers)],
                s=30,
            )
        plt.colorbar(im, ax=ax)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")

    plot_heatmap(axes[0, j], X, Y, gn.reshape(*grid_shape))
    axes[0, j].set_title(f"$g_n$ ({title})")
    plot_heatmap(axes[1, j], X, Y, clt_var.reshape(*grid_shape))
    axes[1, j].set_title(f"$v_n / n$ ({title})")
    plot_heatmap(axes[2, j], X, Y, total_entropy.reshape(*grid_shape))
    axes[2, j].set_title(f"Total Uncertainty ({title})")
    plot_heatmap(axes[3, j], X, Y, alea_entropy.reshape(*grid_shape))
    axes[3, j].set_title(f"Aleatoric Uncertainty ({title})")
    plot_heatmap(axes[4, j], X, Y, epis_entropy.reshape(*grid_shape))
    axes[4, j].set_title(f"Epistemic Uncertainty ({title})")
    axes[0, j].legend(loc="upper right")


plt.tight_layout()
plt.savefig(f"{image_dir}/ud-two-moons.pdf")

# %%
## 2D UQ decomposition (two moons) - subset
setup_regex, title = setup_regex_list[0]
t_idx = DEFAULT_T_IDX["two-moons-1"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

outdir = utils.get_matching_dirs(id_dir, setup_regex)
assert len(outdir) == 1
outdir = outdir[0]
data = utils.read_from(f"{outdir}/data.pickle")
x_prev = data["x_prev"]
y_prev = data["y_prev"]
x_grid = data["x_grid"]
grid_shape = data["grid_shape"]
n = y_prev.size

t = data["t"][t_idx]
grid_shape = data["grid_shape"]
gn = utils.read_from(f"{outdir}/gn.pickle")[t_idx]
g0_to_gn = utils.read_from(f"{outdir}/g0_to_gn.pickle")[:, t_idx]
true_prob = data["true_prob"][t_idx]

clt_var = compute_vn(g0_to_gn, type="pointwise") / n
total_entropy = compute_total_entropy_binary(gn)
alea_entropy = compute_aleatoric_entropy_binary(gn, clt_var)
assert total_entropy.shape == clt_var.shape == gn.shape == alea_entropy.shape
epis_entropy = total_entropy - alea_entropy

# Plot
X = x_grid[:, 0].reshape(*grid_shape)
Y = x_grid[:, 1].reshape(*grid_shape)

unique_ys = np.unique(y_prev)
markers = ["o", "^", "s", "D", "v", "P", "X"]


def plot_heatmap(ax, X, Y, Z):
    im = ax.pcolormesh(
        X, Y, Z, shading="auto", edgecolors="face", linewidths=0, rasterized=True
    )
    for i, y_val in enumerate(unique_ys):
        mask = y_prev == y_val
        ax.scatter(
            x_prev[mask, 0],
            x_prev[mask, 1],
            label=f"y={y_val}",
            marker=markers[i % len(markers)],
            s=30,
        )
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

plot_heatmap(axes[0], X, Y, total_entropy.reshape(*grid_shape))
axes[0].set_title(f"Total Uncertainty")
plot_heatmap(axes[1], X, Y, alea_entropy.reshape(*grid_shape))
axes[1].set_title(f"Aleatoric Uncertainty")
plot_heatmap(axes[2], X, Y, epis_entropy.reshape(*grid_shape))
axes[2].set_title(f"Epistemic Uncertainty")
axes[0].legend(loc="upper right")

plt.tight_layout()
plt.savefig(f"{image_dir}/ud-two-moons-subset.pdf")


# %%
## 2D UQ decomposition (spiral)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

outdir = utils.get_matching_dirs(id_dir, "spiral")
assert len(outdir) == 1
outdir = outdir[0]
data = utils.read_from(f"{outdir}/data.pickle")
x_prev = data["x_prev"]
y_prev = data["y_prev"]
x_grid = data["x_grid"]
grid_shape = data["grid_shape"]
n = y_prev.size
K = np.unique(y_prev).size  # number of classes

t = data["t"]
grid_shape = data["grid_shape"]
gn = utils.read_from(f"{outdir}/gn.pickle")  # (K, m)
g0_to_gn = utils.read_from(f"{outdir}/g0_to_gn.pickle")  # (n+1, K, m)
true_prob = data["true_prob"]

clt_var = [compute_vn(g0_to_gn[:, k], type="pointwise") / n for k in range(K)]
clt_var = np.array(clt_var)  # (K, m)
assert gn.shape == clt_var.shape
assert gn.shape[0] == K
total_entropy = compute_total_entropy_multiclass(gn)  # (m,)
alea_entropy = compute_aleatoric_entropy_multiclass(gn, clt_var)  # (m,)
assert total_entropy.shape == alea_entropy.shape
epis_entropy = total_entropy - alea_entropy  # (m,)

# Plot
X = x_grid[:, 0].reshape(*grid_shape)
Y = x_grid[:, 1].reshape(*grid_shape)

unique_ys = np.unique(y_prev)
markers = ["o", "^", "s", "D", "v", "P", "X"]


def plot_heatmap(ax, X, Y, Z):
    im = ax.pcolormesh(
        X, Y, Z, shading="auto", edgecolors="face", linewidths=0, rasterized=True
    )
    for i, y_val in enumerate(unique_ys):
        mask = y_prev == y_val
        ax.scatter(
            x_prev[mask, 0],
            x_prev[mask, 1],
            label=f"y={y_val}",
            marker=markers[i % len(markers)],
            s=30,
        )
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")


plot_heatmap(axes[0], X, Y, total_entropy.reshape(*grid_shape))
axes[0].set_title(f"Total Uncertainty")
plot_heatmap(axes[1], X, Y, alea_entropy.reshape(*grid_shape))
axes[1].set_title(f"Aleatoric Uncertainty")
plot_heatmap(axes[2], X, Y, epis_entropy.reshape(*grid_shape))
axes[2].set_title(f"Epistemic Uncertainty")
axes[0].legend(loc="upper right")


plt.tight_layout()
plt.savefig(f"{image_dir}/ud-spiral.pdf")
# %%
