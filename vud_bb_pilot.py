# %%
"""Pilot: VUD (Jayasekera et al., arXiv:2509.02327) instantiated on the
covariate-free Beta-Bernoulli BFT, compared against the analytic Beta
posterior ground truth and the predictive-CLT entropic decomposition.

VUD instantiation for a covariate-free exchangeable binary model
----------------------------------------------------------------
The BFT maps a binary prefix y_{1:n} to g_n = P(Y_{n+1}=1 | y_{1:n}).
VUD's auxiliary inputs Z are covariates, which do not exist here, so the
fantasy-conditioning step degenerates to the single available probe:
extend the context with m fantasy labels U = u_{1:m} drawn from the
model's own predictive, autoregressively. Then

    Va(m) = E_U[ H(p(y | y_{1:n}, U)) ]        (aleatoric upper bound,
                                                VUD Thm 3.1 with Z := next
                                                m slots of the sequence)
    epis_LB(m) = H(p(y | y_{1:n})) - Va(m)     (epistemic lower bound)

Deviations from the VUD reference implementation, all forced or noted:
  1. No auxiliary covariates Z — the probe is fantasy labels on the
     exchangeable sequence itself; the entropy query point is the
     next-token predictive (the only predictive there is).
  2. U is enumerated exactly: all 2^m branches with exact autoregressive
     weights (product of branch predictives). No Monte Carlo anywhere,
     so any recorded violation Va > H_total is a genuine incoherence of
     the trained network. The PFN forward runs in float32 — the paper's
     own diagnostic dtype; on this torch version a float64
     nn.TransformerEncoder forward is silently wrong (outputs become
     order-dependent although the architecture is permutation-invariant,
     and disagree with the stored float32 diagnostics by up to 0.7 in
     probability). Weights/entropies accumulate in float64 from the f32
     forward outputs, so violations are exact properties of the
     f32-computed model; only |LB| > 1e-6 is flagged as a violation
     (comfortably above f32 roundoff through the <= 2^m-term sums).
  3. Their aggregation (candidate Z sets, KL-rank filtering, min-Va) has
     no analogue: with no Z to vary, the only remaining knob is the
     fantasy budget m. We report Va(m) for m in {1,2,4,8}; for an
     exactly coherent model Va(m) is non-increasing in m (tower property
     + Jensen), so min-Va aggregation corresponds to the largest m.
     Treating m as the candidate set is a design choice of this pilot.

Ground truth (paper definitions, exact Beta posterior a=1+s_n, b=1+n-s_n):
    total = h(a/(a+b)),   alea = E[h(theta) | y_{1:n}] via digamma,
    epis  = total - alea.

CLT arm (paper's own pipeline): prefix trajectory g_1..g_n ->
compute_vn(.)/n -> moment-matched Beta -> closed-form entropies.
g_0 is not computable for the PFN (forward requires >= 1 training token);
compute_vn never reads index 0, a placeholder is stored there.

An exact-Bayes oracle predictor is run through the identical VUD code
path: its epis_LB gap to the truth isolates the intrinsic looseness of
the truncated bound (m finite) from BFT approximation error.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
import torch
from scipy.special import digamma

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "beta_bernoulli"))

from metrics import compute_aleatoric_entropy_binary, compute_total_entropy_binary
from posterior import compute_vn
from beta_bernoulli.diagnostic import (
    BayesOraclePredictor,
    Predictor,
    load_pfn_predictor,
)

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt-600", type=str,
                    default="beta_bernoulli/checkpoints/seqlen1024_training600.pt")
parser.add_argument("--ckpt-50k", type=str,
                    default="beta_bernoulli/checkpoints/seqlen1024_training50k.pt")
parser.add_argument("--n-list", type=int, nargs="+", default=[10, 50, 200])
parser.add_argument("--theta-list", type=float, nargs="+", default=[0.2, 0.5, 0.8])
parser.add_argument("--m-report", type=int, nargs="+", default=[1, 2, 4, 8])
parser.add_argument("--outdir", type=str, default="vud_pilot_outputs")
args, _ = parser.parse_known_args()

M_MAX = max(args.m_report)
PFN_DTYPE = torch.float32     # f64 TransformerEncoder forward is broken on this torch version
ORACLE_DTYPE = torch.float64  # oracle is pure arithmetic, safe in f64
VIOL_TOL = 1e-6

# ---------------------------------------------------------------- helpers


def binary_entropy(p: np.ndarray, eps: float = 1e-300) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def beta_posterior_truth(s: int, n: int, alpha: float = 1.0, beta: float = 1.0):
    """Paper-exact decomposition of the analytic Beta(alpha+s, beta+n-s) posterior:
    total = h(posterior mean), alea = E[h(theta)] closed form, epis = total - alea."""
    a, b = alpha + s, beta + (n - s)
    ab = a + b
    g_true = a / ab
    total = float(binary_entropy(g_true))
    alea = float(-(a / ab) * digamma(a + 1) - (b / ab) * digamma(b + 1) + digamma(ab + 1))
    # identity check against the paper's moment-matching function fed the exact moments
    var = a * b / (ab**2 * (ab + 1))
    alea_mm = float(compute_aleatoric_entropy_binary(np.array([g_true]), np.array([var]))[0])
    assert abs(alea - alea_mm) < 1e-8, (alea, alea_mm)
    return g_true, total, alea, total - alea


def prefix_trajectory(predictor: Predictor, y_ctx: np.ndarray, dtype) -> np.ndarray:
    """g_k = P(Y_{k+1}=1 | y_{1:k}) for k=1..n. Index 0 holds a placeholder
    (the PFN forward needs >= 1 training token; compute_vn never reads it)."""
    n = y_ctx.size
    y = torch.zeros(n + 1, 1, dtype=dtype)
    y[:n, 0] = torch.from_numpy(y_ctx).to(dtype)
    g = np.empty(n + 1, dtype=np.float64)
    g[0] = 0.5  # placeholder, unused by compute_vn
    for k in range(1, n + 1):
        g[k] = float(predictor.predict(y, single_eval_pos=k)[0])
    return g


def vud_exact_tree(predictor: Predictor, y_ctx: np.ndarray, m_max: int, dtype):
    """Exact enumeration of the fantasy tree to depth m_max.

    Level j holds all 2^j fantasy prefixes u_{1:j}; one batched forward per
    level gives g_j[i] = P(next=1 | y_ctx, branch i), which is simultaneously
    (a) the branch-extension probability and (b) the predictive whose entropy
    is averaged at depth j:  Va(j) = sum_i w_j[i] * h(g_j[i]).
    Va(0) = h(g_n) = the model's total entropy.
    Returns Va (m_max+1,), and g_n = g_0[0].
    """
    n = y_ctx.size
    ctx = torch.from_numpy(y_ctx).to(dtype)
    w = np.ones(1, dtype=np.float64)
    bits = np.zeros((1, 0), dtype=np.float64)
    Va = np.empty(m_max + 1, dtype=np.float64)
    g_n = None
    for j in range(m_max + 1):
        R = w.size
        assert abs(w.sum() - 1.0) < 1e-10, w.sum()
        y = torch.zeros(n + j + 1, R, dtype=dtype)
        y[:n] = ctx.unsqueeze(1).expand(n, R)
        if j > 0:
            y[n:n + j] = torch.from_numpy(bits.T).to(dtype)
        g = predictor.predict(y, single_eval_pos=n + j).cpu().numpy().astype(np.float64)
        Va[j] = float(np.sum(w * binary_entropy(g)))
        if j == 0:
            g_n = float(g[0])
        if j < m_max:  # branch: child 2i has u_{j+1}=0, child 2i+1 has u_{j+1}=1
            w = np.stack([w * (1 - g), w * g], axis=1).reshape(-1)
            bits = np.repeat(bits, 2, axis=0)
            new_bit = np.tile(np.array([0.0, 1.0]), R)
            bits = np.concatenate([bits, new_bit[:, None]], axis=1)
    return Va, g_n


# ---------------------------------------------------------------- models

models: dict[str, tuple[Predictor, torch.dtype]] = {
    "pfn600": (load_pfn_predictor(str(REPO_ROOT / args.ckpt_600), dtype=PFN_DTYPE), PFN_DTYPE),
    "pfn50k": (load_pfn_predictor(str(REPO_ROOT / args.ckpt_50k), dtype=PFN_DTYPE), PFN_DTYPE),
    "oracle": (
        BayesOraclePredictor(
            torch.tensor(1.0, dtype=ORACLE_DTYPE), torch.tensor(1.0, dtype=ORACLE_DTYPE)
        ),
        ORACLE_DTYPE,
    ),
}

# startup sanity check: the PFN is permutation-invariant over training tokens
# by construction; a broken forward (e.g. the f64 path) fails this loudly.
for name, (pred, dt) in models.items():
    if name == "oracle":
        continue
    y_a = torch.zeros(201, 1, dtype=dt)
    y_a[:200, 0] = (torch.arange(200) % 5 == 0).to(dt)  # s=40, interleaved
    y_b = torch.zeros(201, 1, dtype=dt)
    y_b[:40, 0] = 1.0                                   # same multiset, sorted
    ga = float(pred.predict(y_a, single_eval_pos=200)[0])
    gb = float(pred.predict(y_b, single_eval_pos=200)[0])
    assert abs(ga - gb) < 1e-5, (name, ga, gb)
    print(f"[check] {name}: permutation invariance ok (g={ga:.5f})")

# ---------------------------------------------------------------- contexts
rng = np.random.default_rng(20260724)
contexts = []  # shared across models so rows are paired
for n in args.n_list:
    for th in args.theta_list:
        y_ctx = (rng.random(n) < th).astype(np.float64)
        contexts.append((n, th, y_ctx))

# ---------------------------------------------------------------- run
rows = []
start = timer()
for name, (pred, dt) in models.items():
    for n, th, y_ctx in contexts:
        s = int(y_ctx.sum())
        g_true, total_true, alea_true, epis_true = beta_posterior_truth(s, n)

        # paper's CLT pipeline on this model's own prefix trajectory
        g0_to_gn = prefix_trajectory(pred, y_ctx, dt)
        clt_var = float(compute_vn(g0_to_gn[:, None], type="pointwise")[0] / n)
        gn_traj = g0_to_gn[n]
        total_model = float(compute_total_entropy_binary(np.array([gn_traj]))[0])
        alea_clt = float(
            compute_aleatoric_entropy_binary(np.array([gn_traj]), np.array([clt_var]))[0]
        )
        epis_clt = total_model - alea_clt

        # VUD exact fantasy tree
        Va, g_n = vud_exact_tree(pred, y_ctx, M_MAX, dt)
        assert abs(g_n - gn_traj) < 1e-12
        assert abs(Va[0] - total_model) < 1e-9  # Va(0) == h(g_n) == total
        epis_lb = total_model - Va

        rows.append(dict(
            model=name, n=n, theta=th, s=s,
            g_true=g_true, total_true=total_true,
            alea_true=alea_true, epis_true=epis_true,
            g_model=g_n, total_model=total_model,
            clt_var=clt_var, alea_clt=alea_clt, epis_clt=epis_clt,
            Va=Va, epis_lb=epis_lb,
        ))
    print(f"[{name}] done ({timer() - start:.1f}s cumulative)")

# ---------------------------------------------------------------- table
mr = args.m_report
hdr = (f"{'model':>7} {'n':>4} {'th':>4} {'s':>4} | {'alea*':>7} {'epis*':>7} | "
       f"{'g_mod':>6} {'total':>7} {'aleaCLT':>7} {'episCLT':>7} | "
       + " ".join(f"{'Va(' + str(m) + ')':>7}" for m in mr) + " | "
       + " ".join(f"{'LB(' + str(m) + ')':>8}" for m in mr) + " | viol")
lines = [hdr, "-" * len(hdr)]
for r in rows:
    viol = any(r["epis_lb"][m] < -VIOL_TOL for m in mr)
    lines.append(
        f"{r['model']:>7} {r['n']:>4} {r['theta']:>4.1f} {r['s']:>4} | "
        f"{r['alea_true']:7.4f} {r['epis_true']:7.4f} | "
        f"{r['g_model']:6.3f} {r['total_model']:7.4f} {r['alea_clt']:7.4f} {r['epis_clt']:7.4f} | "
        + " ".join(f"{r['Va'][m]:7.4f}" for m in mr) + " | "
        + " ".join(f"{r['epis_lb'][m]:8.5f}" for m in mr)
        + f" | {'YES' if viol else '.':>4}"
    )
table = "\n".join(lines)
print("\n" + table)

# ---------------------------------------------------------------- analysis
print("\n=== analysis ===")
summary = [table, "", "=== analysis ==="]


def log(msg: str) -> None:
    print(msg)
    summary.append(msg)


for name in models:
    sub = [r for r in rows if r["model"] == name]
    log(f"\n[{name}]")
    # (a) validity of the epistemic lower bound against the analytic truth
    for m in mr:
        ok = sum(r["epis_lb"][m] <= r["epis_true"] + 1e-12 for r in sub)
        gaps = np.array([r["epis_true"] - r["epis_lb"][m] for r in sub])
        log(f"  m={m}: LB<=epis_true {ok}/{len(sub)} | gap epis_true-LB: "
            f"min={gaps.min():.5f} med={np.median(gaps):.5f} max={gaps.max():.5f}")
    # (b) looseness at the deepest m (the min-Va analogue)
    m8 = mr[-1]
    err_vud = np.array([abs(r["epis_lb"][m8] - r["epis_true"]) for r in sub])
    err_clt = np.array([abs(r["epis_clt"] - r["epis_true"]) for r in sub])
    closer = int((err_clt < err_vud).sum())
    log(f"  |epis err| mean: CLT={err_clt.mean():.5f} VUD(m={m8})={err_vud.mean():.5f}; "
        f"CLT closer on {closer}/{len(sub)} contexts")
    # coherence: exact Va > total occurrences, and monotonicity of Va in m
    n_viol = sum(any(r["epis_lb"][m] < -VIOL_TOL for m in range(1, M_MAX + 1)) for r in sub)
    worst = min(min(r["epis_lb"][1:]) for r in sub)
    mono = sum(bool(np.all(np.diff(r["Va"]) <= VIOL_TOL)) for r in sub)
    log(f"  Va>total violations (any m<= {M_MAX}): {n_viol}/{len(sub)} contexts, "
        f"worst LB={worst:.2e}; Va monotone nonincreasing in m on {mono}/{len(sub)}")

# truncation scaling: for the exact posterior, epis(n) ~ 1/(2n), so
# LB(m) = total(n) - E[total(n+m)] ~ epis(n) * m/(n+m): the bound can only
# certify the fraction m/(n+m) of the true epistemic uncertainty.
log("\n[truncation scaling, oracle rows]  LB(m)/epis_true  vs  m/(n+m)")
for n in args.n_list:
    sub = [r for r in rows if r["model"] == "oracle" and r["n"] == n]
    for m in mr:
        ratio = np.mean([r["epis_lb"][m] / r["epis_true"] for r in sub])
        log(f"  n={n:4d} m={m}: observed {ratio:.3f}  predicted {m / (n + m):.3f}")

# ---------------------------------------------------------------- save
outdir = REPO_ROOT / args.outdir
outdir.mkdir(parents=True, exist_ok=True)
stem = f"vud_bb_pilot_m{M_MAX}"
np.savez(
    outdir / f"{stem}.npz",
    model=np.array([r["model"] for r in rows]),
    n=np.array([r["n"] for r in rows]),
    theta=np.array([r["theta"] for r in rows]),
    s=np.array([r["s"] for r in rows]),
    g_true=np.array([r["g_true"] for r in rows]),
    total_true=np.array([r["total_true"] for r in rows]),
    alea_true=np.array([r["alea_true"] for r in rows]),
    epis_true=np.array([r["epis_true"] for r in rows]),
    g_model=np.array([r["g_model"] for r in rows]),
    total_model=np.array([r["total_model"] for r in rows]),
    clt_var=np.array([r["clt_var"] for r in rows]),
    alea_clt=np.array([r["alea_clt"] for r in rows]),
    epis_clt=np.array([r["epis_clt"] for r in rows]),
    Va=np.stack([r["Va"] for r in rows]),           # (rows, M_MAX+1), levels 0..M_MAX
    epis_lb=np.stack([r["epis_lb"] for r in rows]),
    m_levels=np.arange(M_MAX + 1),
)
(outdir / f"{stem}_table.txt").write_text("\n".join(summary) + "\n")
print(f"\nsaved {stem}.npz and {stem}_table.txt to {outdir}")
