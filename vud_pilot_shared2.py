# %%
"""Shared-Z VUD variant on TabPFN — Spiral (3 classes) and LogisticLinear —
the same simplified instantiation as vud_pilot.py used for Two Moons:

  - Auxiliary covariates Z: one fixed lattice shared by every grid point
    (valid for any Z by VUD Theorem 3.1; skips only the per-point tightening).
    NOTE this is OUR simplification. The VUD reference code's own toy
    classification runs (run_toy_classification.py, README run commands) use
    per-query candidates: num_z=15 single points z ~ N(x*, 0.1 * feature std)
    with perturb_about_x=1 by default for ALL datasets including `spirals`
    and `logistic_regression`; the only coded alternative (perturb_about_x=0)
    draws z about the data feature means. That configuration is what
    vud_pilot_faithful2.py replicates.
  - Fantasy labels U ~ p(U | Z, D): sampled autoregressively from TabPFN's
    predictive, chaining each sampled label into the context (16 MC draws).
  - Aleatoric upper bound  Va(x*) = E_U[ H(p(y | x*, Z, U, D)) ]  (MC mean).
  - Epistemic lower bound  Ve(x*) = H(p(y | x*, D)) - Va(x*).

CLT side computed in-script with the paper's own pipeline, as in
vud_pilot_faithful2.py. Data and grids replicate the paper's entropic-ud
conventions (run-ghat.py; n values are the paper's own panel settings,
annotated there as matching Jayasekera et al 2025):
  spiral:          n=200, seed=1000, fix_data=False, grid [-4,4]^2, 60/axis
  logistic-linear: n=75 (panels 15/50/75/150), fix_data=True (key 6683),
                   x_design="gaussian:1.5:3.0", grid linspace(-15,15,151)
"""
import argparse
import sys
import warnings
from pathlib import Path
from timeit import default_timer as timer

import numpy as np

warnings.filterwarnings("ignore")
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import jax.random as jr
import torch

import data as data_mod
import posterior
from metrics import (
    compute_aleatoric_entropy_binary,
    compute_aleatoric_entropy_multiclass,
    compute_total_entropy_binary,
    compute_total_entropy_multiclass,
)
from pred_rule import TabPFNClassifierPPD

parser = argparse.ArgumentParser()
parser.add_argument("--setup", type=str, default="spiral",
                    choices=["spiral", "logistic-linear"])
parser.add_argument("--n", type=int, default=None)
parser.add_argument("--seed", type=int, default=1000)
parser.add_argument("--grid-axis", type=int, default=60)
parser.add_argument("--mc-u", type=int, default=16, help="MC draws of U")
parser.add_argument("--n-estimators", type=int, default=8)
parser.add_argument("--model-path", type=str,
                    default="tabpfn-model/tabpfn-v2.5-classifier-v2.5_default.ckpt")
parser.add_argument("--outdir", type=str, default="vud_pilot_outputs")
args, _ = parser.parse_known_args()

EPS = 1e-12
rng = np.random.default_rng(20260724)


def cat_entropy(p, axis=-1):
    p = np.clip(p, EPS, 1.0)
    return -np.sum(p * np.log(p), axis=axis)


# ------------------------------------------------- data + grid (paper-exact)
if args.setup == "spiral":
    n = args.n or 200
    key = jr.key(1907 + args.seed)
    key_others, key_setup = jr.split(key)
    setup = data_mod.Spiral(key_setup, n, True, None)
    t = np.array([0, 1, 2])
    C = 3
    gs = args.grid_axis
    lin1 = np.linspace(-4.0, 4.0, gs)
    X1, X2 = np.meshgrid(lin1, lin1, indexing="ij")
    x_grid = np.stack([X1, X2], axis=-1).reshape(-1, 2)
    z1 = np.linspace(-4.0, 4.0, 5)[1:-1]           # {-2, 0, 2}
    Z = np.stack([g.ravel() for g in np.meshgrid(z1, z1)], axis=-1)  # 9 points
    probes = {"arm 0 core": [2.0, 0.0], "arm 1 core": [-1.0, 1.732],
              "arm 2 core": [-1.0, -1.732], "inter-arm gap": [2.33, 0.0],
              "centre": [0.0, 0.0], "far corner": [3.8, 3.8]}
else:
    n = args.n or 75
    setup = data_mod.LogisticLinear(jr.key(6683), n, True, "gaussian:1.5:3.0")
    t = np.array([0, 1])
    C = 2
    x_grid = np.linspace(-15.0, 15.0, 151).reshape(-1, 1)
    Z = np.linspace(-15.0, 15.0, 11)[1:-1].reshape(-1, 1)  # 9 points, -12..12
    probes = {"far left": [-15.0], "left flank": [-10.0], "data edge L": [-4.5],
              "boundary x=2": [2.0], "data edge R": [7.5], "far right": [15.0]}

torch.manual_seed(8655 + args.seed)
x_prev, y_prev = np.asarray(setup.X), np.asarray(setup.y)
m = x_grid.shape[0]
m_aux = Z.shape[0]
print(f"setup={args.setup} n={n} classes={C} grid={m} aux lattice={m_aux} "
      f"mc_u={args.mc_u}")

clf = TabPFNClassifierPPD(n_estimators=args.n_estimators, softmax_temperature=1.0,
                          fit_mode="low_memory", model_path=args.model_path)

# ------------------------------------------------- CLT side (paper pipeline)
t0 = timer()
gn_all = posterior.compute_gn(clf, t, x_grid, x_prev, y_prev)          # (C, m)
g0_to_gn = posterior.compute_g0_to_gn(clf, t, x_grid, x_prev, y_prev)  # (n+1, C, m)
sigma2 = np.stack([
    posterior.compute_vn(g0_to_gn[:, c, :], type="pointwise") / n for c in range(C)
])
if C == 2:
    total_entropy = compute_total_entropy_binary(gn_all[1])
    alea_clt = compute_aleatoric_entropy_binary(gn_all[1], sigma2[1])
else:
    total_entropy = compute_total_entropy_multiclass(gn_all)
    alea_clt = compute_aleatoric_entropy_multiclass(gn_all, sigma2)
epis_clt = total_entropy - alea_clt
clt_seconds = timer() - t0
print(f"CLT side done in {clt_seconds:.1f}s")

# ------------------------------------------------- VUD side (shared lattice)
t0 = timer()
H_entropy_samples = np.empty((args.mc_u, m))
u_samples = np.empty((args.mc_u, m_aux), dtype=int)
for s in range(args.mc_u):
    xc, yc = x_prev.copy(), y_prev.copy()
    u = np.empty(m_aux, dtype=int)
    for j in range(m_aux):
        clf.fit(xc, yc)
        pj = clf.predict_proba(Z[j: j + 1])[0]
        u[j] = int(rng.choice(C, p=pj / pj.sum()))
        xc = np.vstack([xc, Z[j: j + 1]])
        yc = np.append(yc, u[j])
    u_samples[s] = u
    clf.fit(xc, yc)
    p_grid = clf.predict_proba(x_grid)                 # (m, C)
    H_entropy_samples[s] = cat_entropy(p_grid, axis=1)
    print(f"  U-draw {s + 1}/{args.mc_u}: U={u.tolist()}")

Va = H_entropy_samples.mean(axis=0)
Va_se = H_entropy_samples.std(axis=0, ddof=1) / np.sqrt(args.mc_u)
epis_vud = total_entropy - Va
epis_vud_clip = np.maximum(epis_vud, 0.0)
vud_seconds = timer() - t0
print(f"VUD side done in {vud_seconds:.1f}s")

# ------------------------------------------------- statistics
from scipy.stats import spearmanr

z_score = (Va - total_entropy) / np.maximum(Va_se, 1e-12)
summary = []


def log(msg):
    print(msg)
    summary.append(msg)


log(f"================ shared-Z VUD on TabPFN: {args.setup} ================")
log(f"n={n} classes={C} grid={m} aux lattice={m_aux} mc_u={args.mc_u} "
    f"est={args.n_estimators}")
log("NOTE: shared lattice Z is OUR simplification; the VUD reference toy "
    "classification runs use per-query perturbation about x* (see docstring).")
log(f"wall-clock: CLT side {clt_seconds:.0f}s, VUD side {vud_seconds:.0f}s")
log(f"fraction of grid with VUD epistemic LB < 0: {(epis_vud < 0).mean():.3f}")
log(f"fraction with Va > H_total at z>2: {(z_score > 2).mean():.3f}   "
    f"at z>3: {(z_score > 3).mean():.3f}")
log(f"Spearman(epis CLT, epis VUD raw)  = {spearmanr(epis_clt, epis_vud).statistic:.3f}")
log(f"Spearman(epis CLT, epis VUD clip) = {spearmanr(epis_clt, epis_vud_clip).statistic:.3f}")
log(f"Spearman(alea CLT, Va)            = {spearmanr(alea_clt, Va).statistic:.3f}")

hdr = (f"{'probe':>14} | {'pmax(cls)':>9} | {'total':>6} | {'episCLT':>7} | "
       f"{'Va':>6} | {'Va_se':>6} | {'episVUD':>7} | {'z':>6}")
log("")
log(hdr)
for name, xy in probes.items():
    i = int(np.argmin(((x_grid - np.array(xy)) ** 2).sum(1)))
    c_hat = int(np.argmax(gn_all[:, i]))
    log(f"{name:>14} | {gn_all[c_hat, i]:5.2f}({c_hat}) | {total_entropy[i]:6.3f} | "
        f"{epis_clt[i]:7.3f} | {Va[i]:6.3f} | {Va_se[i]:6.3f} | "
        f"{epis_vud[i]:7.3f} | {z_score[i]:6.1f}")

# ------------------------------------------------- save
outdir = REPO_ROOT / args.outdir
outdir.mkdir(parents=True, exist_ok=True)
slug = "spiral" if args.setup == "spiral" else "logistic"
stem = f"vud_shared_{slug}_n{n}_mc{args.mc_u}_est{args.n_estimators}"
np.savez(outdir / f"{stem}.npz",
         x_grid=x_grid, gn_all=gn_all, sigma2=sigma2, total_entropy=total_entropy,
         alea_clt=alea_clt, epis_clt=epis_clt, Z=Z, u_samples=u_samples,
         H_entropy_samples=H_entropy_samples, Va=Va, Va_se=Va_se,
         epis_vud=epis_vud, x_prev=x_prev, y_prev=y_prev,
         clt_seconds=clt_seconds, vud_seconds=vud_seconds)
(outdir / f"{stem}_table.txt").write_text("\n".join(summary) + "\n")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if args.setup == "spiral":
    fields = [("CLT epistemic", epis_clt, None), ("VUD epistemic LB (raw)", epis_vud, None),
              ("CLT aleatoric", alea_clt, None), ("VUD aleatoric UB (Va)", Va, None)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    for ax, (title, f, vmin) in zip(axes.ravel(), fields):
        im = ax.pcolormesh(X1, X2, f.reshape(gs, gs), shading="auto", vmin=vmin,
                           rasterized=True)
        ax.scatter(x_prev[:, 0], x_prev[:, 1], c=y_prev, cmap="viridis", s=8,
                   edgecolors="k", linewidths=0.2)
        ax.scatter(Z[:, 0], Z[:, 1], marker="x", c="r", s=40)
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
else:
    xg = x_grid[:, 0]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(xg, gn_all[1], label="p(y=1|x,D)")
    ax.plot(xg, setup.get_true_event(x_grid, 1), "--", label="true p(y=1|x)")
    ax.scatter(x_prev[:, 0], y_prev, s=10, c="k", alpha=0.4, label="data")
    ax.scatter(Z[:, 0], np.full(Z.shape[0], 0.5), marker="x", c="r", s=40,
               label="shared Z")
    ax.set_title("predictive vs truth")
    ax.legend()
    ax = axes[0, 1]
    ax.plot(xg, total_entropy, label="total")
    ax.plot(xg, alea_clt, label="CLT aleatoric")
    ax.plot(xg, epis_clt, label="CLT epistemic")
    ax.set_title("CLT decomposition")
    ax.legend()
    ax = axes[1, 0]
    ax.plot(xg, epis_clt, label="CLT epistemic")
    ax.plot(xg, epis_vud, label="VUD epis LB raw")
    ax.plot(xg, epis_vud_clip, label="VUD epis LB clip")
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_title("epistemic: CLT vs VUD")
    ax.legend()
    ax = axes[1, 1]
    ax.plot(xg, Va, label="Va")
    ax.plot(xg, total_entropy, label="H total")
    ax.plot(xg, z_score, label="z(Va>H)")
    ax.set_title("aleatoric bound vs total")
    ax.legend()
    for a in axes.ravel():
        a.set_xlabel("x")
fig.suptitle(f"Shared-Z VUD on TabPFN — {args.setup}, n={n}, "
             f"lattice {m_aux}, mc_u={args.mc_u}, est={args.n_estimators}")
fig.savefig(outdir / f"{stem}.png", dpi=150)
print(f"\nsaved {stem}.npz / _table.txt / .png to {outdir}")
