# %%
"""Pilot: VUD (Jayasekera et al., arXiv:2509.02327) instantiated on TabPFN,
compared with the predictive-CLT entropic decomposition, on Two Moons 1.

VUD side (shared-Z variant):
  - Auxiliary covariates Z: one fixed lattice shared by every grid point
    (valid for any Z by VUD Theorem 3.1; skips only the per-point tightening).
  - Fantasy labels U ~ p(U | Z, D): sampled autoregressively from TabPFN's
    predictive, chaining each sampled label into the context.
  - Aleatoric upper bound  Va(x*) = E_U[ H(p(y | x*, Z, U, D)) ]  (MC mean).
  - Epistemic lower bound  Ve(x*) = H(p(y | x*, D)) - Va(x*).

CLT side: the paper's own pipeline (compute_g0_to_gn -> compute_vn ->
moment-matched Beta entropies), gamma = 1.

Setting: two-moons-1 (sigma = 0.1), n = 100, seed = 1000, exactly as
run-experiments.sh line 48; grid as in Appendix L.2.2.
"""
import argparse
import os
import sys
import warnings
from pathlib import Path
from timeit import default_timer as timer

import jax.random as jr
import numpy as np

warnings.filterwarnings("ignore")
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import data as data_mod
import posterior
from metrics import compute_aleatoric_entropy_binary, compute_total_entropy_binary
from pred_rule import TabPFNClassifierPPD

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=100)
parser.add_argument("--seed", type=int, default=1000)
parser.add_argument("--grid", type=int, default=100, help="grid points per axis")
parser.add_argument("--n-estimators", type=int, default=8)
parser.add_argument("--aux-lattice", type=int, default=3, help="Z lattice per axis")
parser.add_argument("--mc-u", type=int, default=16, help="MC draws of U")
parser.add_argument("--model-path", type=str,
                    default="tabpfn-model/tabpfn-v2.5-classifier-v2.5_default.ckpt")
parser.add_argument("--outdir", type=str, default="outputs/vud-pilot")
args, _ = parser.parse_known_args()

rng = np.random.default_rng(20260724)

# ---------------------------------------------------------------- data (paper-exact)
key = jr.key(1907 + args.seed)
key_others, key_setup = jr.split(key)  # fix_data=False path of run-ghat.py
setup = data_mod.TwoMoons1(key_setup, args.n, True, None)
x_prev, y_prev = np.asarray(setup.X), np.asarray(setup.y)
n = y_prev.size
print(f"data: two-moons-1 n={n} class balance={y_prev.mean():.2f}")

# grid per Appendix L.2.2 (Moons 1: both axes on [-1.5, 2.5])
gs = args.grid
ax1 = np.linspace(-1.5, 2.5, gs)
ax2 = np.linspace(-1.5, 2.5, gs)
X1, X2 = np.meshgrid(ax1, ax2)
x_grid = np.column_stack([X1.ravel(), X2.ravel()])
m = x_grid.shape[0]

t = np.array([0, 1])
T_IDX = 1  # DEFAULT_T_IDX for binary classification

mp = args.model_path if os.path.exists(args.model_path) else None
clf_kwargs = dict(n_estimators=args.n_estimators, softmax_temperature=1.0,
                  fit_mode="low_memory")
if mp is not None:
    clf_kwargs["model_path"] = mp
print(f"TabPFN: n_estimators={args.n_estimators} model_path={mp or 'package default'}")
clf = TabPFNClassifierPPD(**clf_kwargs)

# ---------------------------------------------------------------- CLT side (paper code)
start = timer()
gn = posterior.compute_gn(clf, t, x_grid, x_prev, y_prev)[T_IDX]
g0_to_gn = posterior.compute_g0_to_gn(clf, t, x_grid, x_prev, y_prev)[:, T_IDX]
clt_var = posterior.compute_vn(g0_to_gn, type="pointwise") / n
total_entropy = compute_total_entropy_binary(gn)
alea_clt = compute_aleatoric_entropy_binary(gn, clt_var)
epis_clt = total_entropy - alea_clt
print(f"CLT side done in {timer() - start:.1f}s")

# ---------------------------------------------------------------- VUD side
def binary_entropy(p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))

la = args.aux_lattice
z1 = np.linspace(-1.5, 2.5, la + 2)[1:-1]
Z = np.column_stack([g.ravel() for g in np.meshgrid(z1, z1)])  # (m_aux, 2)
m_aux = Z.shape[0]

start = timer()
H_entropy_samples = np.empty((args.mc_u, m))
u_samples = np.empty((args.mc_u, m_aux), dtype=int)
for s in range(args.mc_u):
    # autoregressive draw of U ~ p(U | Z, D)
    xc, yc = x_prev.copy(), y_prev.copy()
    u = np.empty(m_aux, dtype=int)
    for j in range(m_aux):
        clf.fit(xc, yc)
        pj = clf.predict_proba(Z[j : j + 1])[0, 1]
        u[j] = int(rng.random() < pj)
        xc = np.vstack([xc, Z[j : j + 1]])
        yc = np.append(yc, u[j])
    u_samples[s] = u
    # H(p(y | x*, Z, U, D)) over the whole grid, one batched call
    clf.fit(xc, yc)
    p_grid = clf.predict_proba(x_grid)[:, 1]
    H_entropy_samples[s] = binary_entropy(p_grid)
    print(f"  U-draw {s + 1}/{args.mc_u}: U={u.tolist()}")

Va = H_entropy_samples.mean(axis=0)            # aleatoric upper bound
epis_vud = total_entropy - Va                  # epistemic lower bound
print(f"VUD side done in {timer() - start:.1f}s")

# ---------------------------------------------------------------- compare + save
from scipy.stats import spearmanr

rho_epis = spearmanr(epis_clt, epis_vud).statistic
rho_alea = spearmanr(alea_clt, Va).statistic
neg_frac = (epis_vud < 0).mean()
print(f"Spearman(epistemic CLT, epistemic VUD-LB) = {rho_epis:.3f}")
print(f"Spearman(aleatoric CLT, Va)               = {rho_alea:.3f}")
print(f"fraction of grid with VUD epistemic LB < 0: {neg_frac:.3f}")

# probe points: moon cores, class-overlap region, far corners
probes = {
    "moon A core": [0.0, 1.0], "moon B core": [1.0, -0.5],
    "overlap": [0.5, 0.25], "far corner NE": [2.4, 2.4],
    "far corner SW": [-1.4, -1.4],
}
print(f"\n{'probe':>14} | {'g_n':>5} | {'total':>6} | {'aleaCLT':>7} | {'episCLT':>7} | {'Va':>6} | {'episVUD':>7}")
rows = []
for name, xy in probes.items():
    i = int(np.argmin(((x_grid - np.array(xy)) ** 2).sum(1)))
    rows.append((name, gn[i], total_entropy[i], alea_clt[i], epis_clt[i], Va[i], epis_vud[i]))
    print(f"{name:>14} | {gn[i]:5.2f} | {total_entropy[i]:6.3f} | {alea_clt[i]:7.3f} | "
          f"{epis_clt[i]:7.3f} | {Va[i]:6.3f} | {epis_vud[i]:7.3f}")

outdir = REPO_ROOT / args.outdir
outdir.mkdir(parents=True, exist_ok=True)
np.savez(
    outdir / f"vud_pilot_n{n}_grid{gs}_est{args.n_estimators}.npz",
    x_grid=x_grid, gn=gn, total_entropy=total_entropy, clt_var=clt_var,
    alea_clt=alea_clt, epis_clt=epis_clt, Va=Va, epis_vud=epis_vud,
    Z=Z, u_samples=u_samples, x_prev=x_prev, y_prev=y_prev,
    spearman_epis=rho_epis, spearman_alea=rho_alea,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fields = [
    ("CLT epistemic", epis_clt), ("VUD epistemic LB", epis_vud),
    ("CLT aleatoric", alea_clt), ("VUD aleatoric UB (Va)", Va),
]
fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
for ax, (title, f) in zip(axes.ravel(), fields):
    im = ax.pcolormesh(X1, X2, f.reshape(gs, gs), shading="auto", rasterized=True)
    ax.scatter(x_prev[:, 0], x_prev[:, 1], c=y_prev, cmap="coolwarm", s=8,
               edgecolors="k", linewidths=0.2)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
fig.suptitle(f"Two Moons 1, n={n}, TabPFN est={args.n_estimators}; "
             f"Spearman(epis)={rho_epis:.3f}")
fig.savefig(outdir / f"vud_pilot_n{n}_grid{gs}_est{args.n_estimators}.png", dpi=150)
print(f"saved outputs to {outdir}")
