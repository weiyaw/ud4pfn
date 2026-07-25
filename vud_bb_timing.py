"""Wall-clock cost of the predictive-CLT pipeline vs VUD exact enumeration
on the Beta-Bernoulli BFT (same instantiation, contexts, and code paths as
vud_bb_pilot.py, re-run with per-arm timers).

Per context y_{1:n}:
  CLT   prefix trajectory g_1..g_n (n forwards, batch 1)
        + compute_vn + closed-form entropy decomposition.
  VUD   exact fantasy tree to depth m: one batched forward per level j,
        batch 2^j, so 2^{m+1}-1 branch evaluations in m+1 forwards.

Wall-clock is averaged over the three theta contexts per n and over
--repeats timing repetitions. Forward-pass counts are recorded alongside.

Usage (from the repo root):
    python vud_bb_timing.py --out rebuttal/vud_bb_timing.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "beta_bernoulli"))

from metrics import compute_aleatoric_entropy_binary, compute_total_entropy_binary
from posterior import compute_vn
from beta_bernoulli.diagnostic import Predictor, load_pfn_predictor

PFN_DTYPE = torch.float32


def binary_entropy(p: np.ndarray, eps: float = 1e-300) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def prefix_trajectory(predictor: Predictor, y_ctx: np.ndarray, dtype) -> np.ndarray:
    """Identical to vud_bb_pilot.prefix_trajectory."""
    n = y_ctx.size
    y = torch.zeros(n + 1, 1, dtype=dtype)
    y[:n, 0] = torch.from_numpy(y_ctx).to(dtype)
    g = np.empty(n + 1, dtype=np.float64)
    g[0] = 0.5
    for k in range(1, n + 1):
        g[k] = float(predictor.predict(y, single_eval_pos=k)[0])
    return g


def vud_exact_tree(predictor: Predictor, y_ctx: np.ndarray, m_max: int, dtype):
    """Identical to vud_bb_pilot.vud_exact_tree."""
    n = y_ctx.size
    ctx = torch.from_numpy(y_ctx).to(dtype)
    w = np.ones(1, dtype=np.float64)
    bits = np.zeros((1, 0), dtype=np.float64)
    Va = np.empty(m_max + 1, dtype=np.float64)
    for j in range(m_max + 1):
        R = w.size
        y = torch.zeros(n + j + 1, R, dtype=dtype)
        y[:n] = ctx.unsqueeze(1).expand(n, R)
        if j > 0:
            y[n:n + j] = torch.from_numpy(bits.T).to(dtype)
        g = predictor.predict(y, single_eval_pos=n + j).cpu().numpy().astype(np.float64)
        Va[j] = float(np.sum(w * binary_entropy(g)))
        if j < m_max:
            w = np.stack([w * (1 - g), w * g], axis=1).reshape(-1)
            bits = np.repeat(bits, 2, axis=0)
            new_bit = np.tile(np.array([0.0, 1.0]), R)
            bits = np.concatenate([bits, new_bit[:, None]], axis=1)
    return Va


def clt_pipeline(predictor: Predictor, y_ctx: np.ndarray, dtype) -> None:
    n = y_ctx.size
    g0_to_gn = prefix_trajectory(predictor, y_ctx, dtype)
    clt_var = float(compute_vn(g0_to_gn[:, None], type="pointwise")[0] / n)
    gn = g0_to_gn[n]
    total = float(compute_total_entropy_binary(np.array([gn]))[0])
    alea = float(compute_aleatoric_entropy_binary(np.array([gn]), np.array([clt_var]))[0])
    _ = total - alea


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-600", type=str,
                   default="beta_bernoulli/checkpoints/seqlen1024_training600.pt")
    p.add_argument("--ckpt-50k", type=str,
                   default="beta_bernoulli/checkpoints/seqlen1024_training50k.pt")
    p.add_argument("--n-list", type=int, nargs="+", default=[10, 50, 200])
    p.add_argument("--theta-list", type=float, nargs="+", default=[0.2, 0.5, 0.8])
    p.add_argument("--m-report", type=int, nargs="+", default=[4, 8])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out", type=str, default="rebuttal/vud_bb_timing.md")
    args = p.parse_args()

    models = {
        "pfn-600": load_pfn_predictor(str(REPO_ROOT / args.ckpt_600), dtype=PFN_DTYPE),
        "pfn-50k": load_pfn_predictor(str(REPO_ROOT / args.ckpt_50k), dtype=PFN_DTYPE),
    }

    rng = np.random.default_rng(20260724)  # same context stream as vud_bb_pilot.py
    contexts: list[tuple[int, float, np.ndarray]] = []
    for n in args.n_list:
        for th in args.theta_list:
            contexts.append((n, th, (rng.random(n) < th).astype(np.float64)))

    m_max = max(args.m_report)
    header = (
        "| Model | n | CLT: prefix + v_n + entropies (s) | "
        + " | ".join(f"VUD exact tree m={m} (s)" for m in args.m_report)
        + " | CLT forwards | " + f"VUD forwards (m={m_max}) |"
    )
    lines = [header, "|" + "---|" * (4 + len(args.m_report) + 1)]

    for name, pred in models.items():
        # warm-up forward so first-call overhead is not billed to either arm
        clt_pipeline(pred, contexts[0][2], PFN_DTYPE)
        for n in args.n_list:
            ctxs = [c for c in contexts if c[0] == n]
            t_clt, t_vud = [], {m: [] for m in args.m_report}
            for _ in range(args.repeats):
                for _, _, y_ctx in ctxs:
                    t0 = perf_counter()
                    clt_pipeline(pred, y_ctx, PFN_DTYPE)
                    t_clt.append(perf_counter() - t0)
                    for m in args.m_report:
                        t0 = perf_counter()
                        vud_exact_tree(pred, y_ctx, m, PFN_DTYPE)
                        t_vud[m].append(perf_counter() - t0)
            cells = [f"{np.mean(t_clt):.3f} ± {np.std(t_clt):.3f}"]
            cells += [f"{np.mean(t_vud[m]):.3f} ± {np.std(t_vud[m]):.3f}"
                      for m in args.m_report]
            lines.append(
                f"| {name} | {n} | " + " | ".join(cells)
                + f" | {n} (batch 1) | {m_max + 1} (batch ≤ 2^{m_max}) |"
            )
            print(lines[-1], flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
