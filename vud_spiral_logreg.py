# %%
"""VUD on TabPFN — Spiral (3 classes) and LogisticLinear —
extending vud_two_moons.py (Two Moons) with the identical recipe from
github.com/jacobyhsi/VUD (run_toy_classification.py + src/utils.calculate_min_Va_by_KL_rank):

  - K = 15 single-point z candidates per test point, drawn by perturbation
    about x*: z ~ N(x*, (0.1 * per-feature std of D)^2)   [their defaults]
  - per candidate, EXACT enumeration over the fantasy label u in the C classes:
        Va_k = sum_u p(u|z_k,D) * H(p(y | x*, z_k, u, D))
  - forward KL coherence meter per candidate:
        KL( p(y|x*,D) || sum_u p(u|z_k,D) p(y|x*,z_k,u,D) )
  - their aggregation: keep the num_valid_Va=5 lowest-KL candidates,
    min Va among them, max_Ve = H(p(y|x*,D)) - min_Va (raw AND vmin=0 clip)

CLT side computed in-script with the paper's own pipeline (compute_gn,
compute_g0_to_gn, compute_vn; binary entropy functions for logistic-linear,
multiclass Dirichlet-matched entropy functions for the spiral).

Data and grid replicate the paper's entropic-ud conventions
(run-experiments.sh + run-ghat.py):
  spiral:          n=200, seed=1000, fix_data=False, x_design=None,
                   grid [-4,4]^2 (paper m=100 per axis; this script uses 60)
  logistic-linear: n=75 (paper panels 15/50/75/150), seed=1000, fix_data=True
                   (data key jr.key(6683)), x_design="gaussian:1.5:3.0",
                   grid np.linspace(-15,15,151)  [same as Jayasekera et al 2025]

Deviations, stated: no L-permutation ensembling (TabPFN is permutation
invariant over context rows), no LLM prompt serialisation, n_estimators=8
(the Two Moons setting; the paper's panels use 64).
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
parser.add_argument("--n", type=int, default=None,
                    help="context size; default: spiral 200, logistic-linear 75")
parser.add_argument("--seed", type=int, default=1000)
parser.add_argument("--grid-axis", type=int, default=60,
                    help="spiral: grid points per axis (paper uses 100)")
parser.add_argument("--sub-stride", type=int, default=None,
                    help="stride into the grid for the VUD side; "
                         "default: spiral 5 (12x12 points), logistic-linear 1 (all 151)")
parser.add_argument("--num-z", type=int, default=15)
parser.add_argument("--num-valid-va", type=int, default=5)
parser.add_argument("--perturbation-std", type=float, default=0.1)
parser.add_argument("--n-estimators", type=int, default=8)
parser.add_argument("--model-path", type=str,
                    default="tabpfn-model/tabpfn-v2.5-classifier-v2.5_default.ckpt")
parser.add_argument("--outdir", type=str, default="vud_outputs")
args, _ = parser.parse_known_args()

EPS = 1e-12
rng = np.random.default_rng(20260725)


def cat_entropy(p, axis=-1):
    p = np.clip(p, EPS, 1.0)
    return -np.sum(p * np.log(p), axis=axis)


def cat_kl(p, q, axis=-1):
    p = np.clip(p, EPS, 1.0)
    q = np.clip(q, EPS, 1.0)
    return np.sum(p * np.log(p / q), axis=axis)


# ------------------------------------------------- data + grid (paper-exact)
if args.setup == "spiral":
    n = args.n or 200
    # fix_data=False path of run-ghat.py
    key = jr.key(1907 + args.seed)
    key_others, key_setup = jr.split(key)
    setup = data_mod.Spiral(key_setup, n, True, None)
    t = np.array([0, 1, 2])
    C = 3
    lin1 = np.linspace(-4.0, 4.0, args.grid_axis)
    lin2 = np.linspace(-4.0, 4.0, args.grid_axis)
    X1, X2 = np.meshgrid(lin1, lin2, indexing="ij")
    x_grid = np.stack([X1, X2], axis=-1).reshape(-1, 2)
    sub_stride = args.sub_stride or 5
    gs = args.grid_axis
    idx2d = np.arange(gs * gs).reshape(gs, gs)
    sub_axes = np.arange(0, gs, sub_stride)
    sub_idx = idx2d[np.ix_(sub_axes, sub_axes)].ravel()
    # probes: arm interiors at mid radius, an inter-arm gap, the centre where
    # all arms meet, a corner beyond the spiral's maximum radius
    probes = {"arm 0 core": [2.0, 0.0], "arm 1 core": [-1.0, 1.732],
              "arm 2 core": [-1.0, -1.732], "inter-arm gap": [2.33, 0.0],
              "centre": [0.0, 0.0], "far corner": [3.8, 3.8]}
else:
    n = args.n or 75
    # fix_data=True path of run-ghat.py: data key fixed at 6683
    setup = data_mod.LogisticLinear(jr.key(6683), n, True, "gaussian:1.5:3.0")
    t = np.array([0, 1])
    C = 2
    x_grid = np.linspace(-15.0, 15.0, 151).reshape(-1, 1)
    sub_stride = args.sub_stride or 1
    sub_idx = np.arange(0, x_grid.shape[0], sub_stride)
    # probes: far edges, sparse-data flanks, the data-mass edges (mean +- 2 sd
    # of the gaussian:1.5:3.0 design), the true decision boundary x=2
    probes = {"far left": [-15.0], "left flank": [-10.0], "data edge L": [-4.5],
              "boundary x=2": [2.0], "data edge R": [7.5], "far right": [15.0]}

torch.manual_seed(8655 + args.seed)
x_prev, y_prev = np.asarray(setup.X), np.asarray(setup.y)
print(f"setup={args.setup} n={n} classes={C} "
      f"class counts={np.bincount(y_prev.astype(int), minlength=C).tolist()}")
x_sub = x_grid[sub_idx]
msub = x_sub.shape[0]
print(f"CLT grid {x_grid.shape[0]} points; VUD subgrid {msub} points; K={args.num_z}")

clf = TabPFNClassifierPPD(n_estimators=args.n_estimators, softmax_temperature=1.0,
                          fit_mode="low_memory", model_path=args.model_path)

# ------------------------------------------------- CLT side (paper pipeline)
t0 = timer()
gn_all = posterior.compute_gn(clf, t, x_grid, x_prev, y_prev)      # (C, m)
g0_to_gn = posterior.compute_g0_to_gn(clf, t, x_grid, x_prev, y_prev)  # (n+1, C, m)
sigma2 = np.stack([
    posterior.compute_vn(g0_to_gn[:, c, :], type="pointwise") / n for c in range(C)
])                                                                  # (C, m)
if C == 2:
    total_entropy = compute_total_entropy_binary(gn_all[1])
    alea_clt = compute_aleatoric_entropy_binary(gn_all[1], sigma2[1])
else:
    total_entropy = compute_total_entropy_multiclass(gn_all)
    alea_clt = compute_aleatoric_entropy_multiclass(gn_all, sigma2)
epis_clt = total_entropy - alea_clt
clt_seconds = timer() - t0
print(f"CLT side done in {clt_seconds:.1f}s")

p_ref = gn_all.T[sub_idx]            # (msub, C) reference predictive p(y|x*,D)
H_sub = total_entropy[sub_idx]
epis_clt_sub = epis_clt[sub_idx]

# ------------------------------------------------- VUD side (their recipe)
D_std = x_prev.std(axis=0, ddof=0)
d = x_prev.shape[1]
all_Z = np.empty((msub, args.num_z, d))
for i in range(msub):
    all_Z[i] = rng.normal(x_sub[i], args.perturbation_std * D_std,
                          size=(args.num_z, d))

t0 = timer()
clf.fit(x_prev, y_prev)
p_u_all = clf.predict_proba(all_Z.reshape(-1, d)).reshape(msub, args.num_z, C)

Va = np.empty((msub, args.num_z))
KLf = np.empty((msub, args.num_z))
for i in range(msub):
    for k in range(args.num_z):
        z = all_Z[i, k]
        pu = p_u_all[i, k]                          # (C,)
        p_y_given_u = np.empty((C, C))
        for u in range(C):
            clf.fit(np.vstack([x_prev, z[None, :]]), np.append(y_prev, u))
            p_y_given_u[u] = clf.predict_proba(x_sub[i:i + 1])[0]
        Va[i, k] = float(np.sum(pu * cat_entropy(p_y_given_u, axis=1)))
        marg = pu @ p_y_given_u                     # p(y|x*,z,D), (C,)
        KLf[i, k] = float(cat_kl(p_ref[i], marg))
    if (i + 1) % 10 == 0:
        el = timer() - t0
        print(f"  point {i + 1}/{msub}  ({el:.0f}s, {el / (i + 1):.1f}s/pt)")
vud_seconds = timer() - t0
print(f"VUD side done in {vud_seconds:.1f}s")

# ------------------------------------------------- their aggregation
order = np.argsort(KLf, axis=1)
valid = order[:, : args.num_valid_va]
minVa = np.take_along_axis(Va, valid, axis=1).min(axis=1)
maxVe_raw = H_sub - minVa
maxVe_clip = np.maximum(maxVe_raw, 0.0)

# ------------------------------------------------- statistics
from scipy.stats import spearmanr

viol_cand = (Va > H_sub[:, None])
summary = []


def log(msg):
    print(msg)
    summary.append(msg)


log(f"================ VUD on TabPFN: {args.setup} ================")
log(f"n={n} classes={C} CLT grid={x_grid.shape[0]} VUD subgrid={msub} "
    f"K={args.num_z} keep={args.num_valid_va} est={args.n_estimators}")
log(f"wall-clock: CLT side {clt_seconds:.0f}s, VUD side {vud_seconds:.0f}s")
log(f"candidates with Va > H_total (coherence violation): {viol_cand.mean():.3f}")
log(f"points where ALL {args.num_z} candidates violate:   {viol_cand.all(axis=1).mean():.3f}")
log(f"points with negative max_Ve AFTER their KL-rank+min aggregation: {(maxVe_raw < 0).mean():.3f}")
log(f"forward KL (coherence meter): median={np.median(KLf):.4f}  "
    f"90%={np.quantile(KLf, .9):.4f}  max={KLf.max():.4f}")
log(f"Spearman(epis CLT, max_Ve raw)  = {spearmanr(epis_clt_sub, maxVe_raw).statistic:.3f}")
log(f"Spearman(epis CLT, max_Ve clip) = {spearmanr(epis_clt_sub, maxVe_clip).statistic:.3f}")

hdr = (f"{'probe':>14} | {'pmax(cls)':>9} | {'total':>6} | {'episCLT':>7} | "
       f"{'minVa':>6} | {'maxVe':>7} | {'#viol/' + str(args.num_z):>8} | {'medKL':>6}")
log("")
log(hdr)
for name, xy in probes.items():
    i = int(np.argmin(((x_sub - np.array(xy)) ** 2).sum(1)))
    c_hat = int(np.argmax(p_ref[i]))
    log(f"{name:>14} | {p_ref[i, c_hat]:5.2f}({c_hat}) | {H_sub[i]:6.3f} | "
        f"{epis_clt_sub[i]:7.3f} | {minVa[i]:6.3f} | {maxVe_raw[i]:7.3f} | "
        f"{int(viol_cand[i].sum()):8d} | {np.median(KLf[i]):6.4f}")

# ------------------------------------------------- save
outdir = REPO_ROOT / args.outdir
outdir.mkdir(parents=True, exist_ok=True)
slug = "spiral" if args.setup == "spiral" else "logistic"
stem = f"vud_{slug}_n{n}_eval{msub}_est{args.n_estimators}"
np.savez(outdir / f"{stem}.npz",
         x_grid=x_grid, gn_all=gn_all, sigma2=sigma2, total_entropy=total_entropy,
         alea_clt=alea_clt, epis_clt=epis_clt, sub_idx=sub_idx, x_sub=x_sub,
         p_ref=p_ref, H_sub=H_sub, epis_clt_sub=epis_clt_sub, all_Z=all_Z,
         p_u_all=p_u_all, Va=Va, KLf=KLf, minVa=minVa, maxVe_raw=maxVe_raw,
         maxVe_clip=maxVe_clip, x_prev=x_prev, y_prev=y_prev,
         clt_seconds=clt_seconds, vud_seconds=vud_seconds)
(outdir / f"{stem}_table.txt").write_text("\n".join(summary) + "\n")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if args.setup == "spiral":
    g = len(sub_axes)
    X1s = x_sub[:, 0].reshape(g, g)
    X2s = x_sub[:, 1].reshape(g, g)
    panels = [("CLT epistemic (full grid)", epis_clt.reshape(gs, gs), X1, X2, None),
              ("VUD max_Ve (raw)", maxVe_raw.reshape(g, g), X1s, X2s, None),
              ("VUD max_Ve (their vmin=0 clip)", maxVe_clip.reshape(g, g), X1s, X2s, 0.0),
              ("candidate violation fraction", viol_cand.mean(axis=1).reshape(g, g), X1s, X2s, 0.0)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    for ax, (title, f, A, B, vmin) in zip(axes.ravel(), panels):
        im = ax.pcolormesh(A, B, f, shading="auto", vmin=vmin, rasterized=True)
        ax.scatter(x_prev[:, 0], x_prev[:, 1], c=y_prev, cmap="viridis", s=8,
                   edgecolors="k", linewidths=0.2)
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
else:
    xg = x_grid[:, 0]
    xs = x_sub[:, 0]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(xg, gn_all[1], label="p(y=1|x,D)")
    ax.plot(xg, setup.get_true_event(x_grid, 1), "--", label="true p(y=1|x)")
    ax.scatter(x_prev[:, 0], y_prev, s=10, c="k", alpha=0.4, label="data")
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
    ax.plot(xs, maxVe_raw, ".-", label="VUD max_Ve raw")
    ax.plot(xs, maxVe_clip, ".-", label="VUD max_Ve clip")
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_title("epistemic: CLT vs VUD")
    ax.legend()
    ax = axes[1, 1]
    ax.plot(xs, viol_cand.mean(axis=1), ".-", label="violation fraction")
    ax.plot(xs, np.median(KLf, axis=1), ".-", label="median KL")
    ax.set_title("coherence diagnostics")
    ax.legend()
    for a in axes.ravel():
        a.set_xlabel("x")
fig.suptitle(f"VUD on TabPFN — {args.setup}, n={n}, K={args.num_z}, "
             f"est={args.n_estimators}")
fig.savefig(outdir / f"{stem}.png", dpi=150)
print(f"\nsaved {stem}.npz / _table.txt / .png to {outdir}")
