# %%
"""Faithful VUD replication on TabPFN (Two Moons 1, n=100), following the
reference implementation at github.com/jacobyhsi/VUD (run_toy_classification.py
+ eval/eval_toy_2d_class.ipynb + src/utils.calculate_min_Va_by_KL_rank):

  - K = 15 single-point z candidates per test point, drawn by perturbation
    about x*: z ~ N(x*, (0.1 * per-feature std of D)^2)   [their defaults:
    num_z=15, num_bo_z=0, perturb_about_x, perturbation_std=0.1]
  - per candidate, EXACT enumeration over the fantasy label u:
        Va_k = sum_u p(u|z_k,D) * H(p(y | x*, z_k, u, D))
    (their code enumerates u; no Monte Carlo over U)
  - forward KL coherence meter per candidate:
        KL( p(y|x*,D) || sum_u p(u|z_k,D) p(y|x*,z_k,u,D) )
    (zero for an exactly coherent Bayesian model, by the tower property)
  - their aggregation: keep the num_valid_Va=5 lowest-KL candidates,
    min Va among them, max_Ve = H(p(y|x*,D)) - min_Va
    (their two-moons notebook then plots max_Ve with vmin=0, clipping
    negatives; we report raw AND clipped)

Deviations, stated: no L-permutation ensembling (TabPFN is permutation
invariant over context rows), no LLM prompt serialisation (numeric rows are
TabPFN's native interface).

CLT side is read from the shared-Z pilot's saved npz (same model settings)
by subsampling its grid.
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

from pred_rule import TabPFNClassifierPPD

parser = argparse.ArgumentParser()
parser.add_argument("--num-z", type=int, default=15)
parser.add_argument("--num-valid-va", type=int, default=5)
parser.add_argument("--perturbation-std", type=float, default=0.1)
parser.add_argument("--sub-stride", type=int, default=5, help="stride into the 60-grid")
parser.add_argument("--n-estimators", type=int, default=8)
parser.add_argument("--model-path", type=str,
                    default="tabpfn-model/tabpfn-v2.5-classifier-v2.5_default.ckpt")
parser.add_argument("--prev-npz", type=str,
                    default="outputs/vud-pilot/vud_pilot_n100_grid60_est8.npz")
parser.add_argument("--outdir", type=str, default="outputs/vud-pilot")
args, _ = parser.parse_known_args()

rng = np.random.default_rng(20260725)
EPS = 1e-12

def binary_entropy(p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))

def binary_kl(p, q):
    p = np.clip(p, EPS, 1 - EPS); q = np.clip(q, EPS, 1 - EPS)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

# ------------------------------------------------- load shared-Z pilot artifacts
prev = np.load(REPO_ROOT / args.prev_npz)
x_prev, y_prev = prev["x_prev"], prev["y_prev"]
n = y_prev.size
gs = int(np.sqrt(prev["x_grid"].shape[0]))
idx2d = np.arange(gs).reshape(gs, gs)  # row-major grid indices
sub_axes = np.arange(0, gs, args.sub_stride)
sub_idx = idx2d[np.ix_(sub_axes, sub_axes)].ravel()
x_sub = prev["x_grid"][sub_idx]
gn_sub = prev["gn"][sub_idx]                      # p(y=1 | x*, D), est=8
H_sub = prev["total_entropy"][sub_idx]            # H(p(y|x*,D))
epis_clt_sub = prev["epis_clt"][sub_idx]
m = x_sub.shape[0]
print(f"subgrid {len(sub_axes)}x{len(sub_axes)} = {m} points; n={n}; K={args.num_z}")

D_std = x_prev.std(axis=0, ddof=0)

clf = TabPFNClassifierPPD(n_estimators=args.n_estimators, softmax_temperature=1.0,
                          fit_mode="low_memory", model_path=args.model_path)

# ------------------------------------------------- p(u|z,D): one fit serves all z
all_Z = np.empty((m, args.num_z, 2), dtype=np.float64)
for i in range(m):
    all_Z[i] = rng.normal(x_sub[i], args.perturbation_std * D_std,
                          size=(args.num_z, 2))
clf.fit(x_prev, y_prev)
p_u_all = clf.predict_proba(all_Z.reshape(-1, 2))[:, 1].reshape(m, args.num_z)

# ------------------------------------------------- per (point, candidate, u): fit + predict x*
Va = np.empty((m, args.num_z))
KLf = np.empty((m, args.num_z))     # forward KL, their kl_pyx_pyxz
start = timer()
for i in range(m):
    for k in range(args.num_z):
        z = all_Z[i, k]
        pu1 = p_u_all[i, k]
        p_y_given_u = np.empty(2)   # u = 0, 1
        for u in (0, 1):
            clf.fit(np.vstack([x_prev, z[None, :]]), np.append(y_prev, u))
            p_y_given_u[u] = clf.predict_proba(x_sub[i : i + 1])[0, 1]
        Va[i, k] = (1 - pu1) * binary_entropy(p_y_given_u[0]) + pu1 * binary_entropy(p_y_given_u[1])
        marg = (1 - pu1) * p_y_given_u[0] + pu1 * p_y_given_u[1]   # p(y|x,z,D)
        KLf[i, k] = binary_kl(gn_sub[i], marg)
    if (i + 1) % 10 == 0:
        el = timer() - start
        print(f"  point {i + 1}/{m}  ({el:.0f}s, {el / (i + 1):.1f}s/pt)")

# ------------------------------------------------- their aggregation
order = np.argsort(KLf, axis=1)
valid = order[:, : args.num_valid_va]                    # 5 lowest-KL candidates
minVa = np.take_along_axis(Va, valid, axis=1).min(axis=1)
maxVe_raw = H_sub - minVa
maxVe_clip = np.maximum(maxVe_raw, 0.0)                  # their plot's vmin=0

# ------------------------------------------------- violation + coherence statistics
viol_cand = (Va > H_sub[:, None])                        # exact, no MC noise
from scipy.stats import spearmanr
print("\n================ faithful VUD on TabPFN ================")
print(f"candidates with Va > H_total (coherence violation): {viol_cand.mean():.3f}")
print(f"points where ALL 15 candidates violate:             {viol_cand.all(axis=1).mean():.3f}")
print(f"points with negative max_Ve AFTER their KL-rank+min aggregation: {(maxVe_raw < 0).mean():.3f}")
print(f"forward KL (coherence meter): median={np.median(KLf):.4f}  90%={np.quantile(KLf, .9):.4f}  max={KLf.max():.4f}")
print(f"Spearman(epis CLT, max_Ve raw)  = {spearmanr(epis_clt_sub, maxVe_raw).statistic:.3f}")
print(f"Spearman(epis CLT, max_Ve clip) = {spearmanr(epis_clt_sub, maxVe_clip).statistic:.3f}")

probes = {"moon A core": [0.0, 1.0], "moon B core": [1.0, -0.5],
          "overlap": [0.5, 0.25], "far corner NE": [2.4, 2.4], "far corner SW": [-1.4, -1.4]}
print(f"\n{'probe':>14} | {'g_n':>5} | {'total':>6} | {'episCLT':>7} | {'minVa':>6} | {'maxVe':>7} | {'#viol/15':>8} | {'medKL':>6}")
for name, xy in probes.items():
    i = int(np.argmin(((x_sub - np.array(xy)) ** 2).sum(1)))
    print(f"{name:>14} | {gn_sub[i]:5.2f} | {H_sub[i]:6.3f} | {epis_clt_sub[i]:7.3f} | "
          f"{minVa[i]:6.3f} | {maxVe_raw[i]:7.3f} | {viol_cand[i].sum():8d} | {np.median(KLf[i]):6.4f}")

outdir = REPO_ROOT / args.outdir
outdir.mkdir(parents=True, exist_ok=True)
np.savez(outdir / f"vud_faithful_n{n}_sub{len(sub_axes)}_est{args.n_estimators}.npz",
         x_sub=x_sub, gn_sub=gn_sub, H_sub=H_sub, epis_clt_sub=epis_clt_sub,
         all_Z=all_Z, p_u_all=p_u_all, Va=Va, KLf=KLf,
         minVa=minVa, maxVe_raw=maxVe_raw, maxVe_clip=maxVe_clip)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
g = len(sub_axes)
X1s = x_sub[:, 0].reshape(g, g); X2s = x_sub[:, 1].reshape(g, g)
panels = [("CLT epistemic", epis_clt_sub, None),
          ("VUD max_Ve (raw)", maxVe_raw, None),
          ("VUD max_Ve (their vmin=0 clip)", maxVe_clip, 0.0),
          ("candidate violation fraction", viol_cand.mean(axis=1), 0.0)]
fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
for ax, (title, f, vmin) in zip(axes.ravel(), panels):
    im = ax.pcolormesh(X1s, X2s, f.reshape(g, g), shading="auto", vmin=vmin, rasterized=True)
    ax.scatter(x_prev[:, 0], x_prev[:, 1], c=y_prev, cmap="coolwarm", s=8,
               edgecolors="k", linewidths=0.2)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
fig.suptitle(f"Faithful VUD on TabPFN — Two Moons 1, n={n}, K={args.num_z}, est={args.n_estimators}")
fig.savefig(outdir / f"vud_faithful_n{n}_sub{len(sub_axes)}_est{args.n_estimators}.png", dpi=150)
print(f"\nsaved to {outdir}")
