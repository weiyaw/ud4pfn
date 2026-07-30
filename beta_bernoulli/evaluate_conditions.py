"""Finite-horizon sup statistics and T-split condition summaries from saved
Beta-Bernoulli diagnostic dumps (diag_*.pt: per-rollout y, f_prev, b, delta).

(a) Sup statistic at gamma=1, per rollout:
        M_delta(N) = max_{k_min <= k <= N} k^{1/2} |Delta_k|
        M_b(N)     = max_{k_min <= k <= N} k^{1/2} |b_k|
    reported at N in {1024, 5000, 10001} (k_max = 10^4 is the largest stored
    probe, so N = 10001 uses every available k). If the max is attained at
    small k and never overtaken, the statistic is constant in N.

(b) Condition trajectories split at the training horizon n = T = 1024,
    at gamma = 1 (definitions exactly as plotted by plot.py /
    Theorem th:ascondmult, all tails truncated at the stored horizon
    N = k_max = 10^4):
        C1(n) = |sum_{k>n} b_k|
        C2(n) = n^{1/2} |sum_{k>n} b_k|
        C3(n) = n^{1/2} |b_n|
        C4(n) = n sum_{k>=n} b_k^2
        R(n)  = n sum_{k>n} Delta_k^2 / (1 - n/N)   (gamma=1 truncation-
                corrected, as in plot.py's residual scatter)
    Within-T summarises the trajectory on n in [32, 512]; beyond-T on
    n in [1200, 5000] (kept <= N/2 so the horizon edge does not bias the
    tails). Per rollout we record the median value of the statistic over
    the window and its log-log OLS slope; the tables aggregate
    min/median/max across rollouts.

Usage (from the repo root):
    python beta_bernoulli/evaluate_conditions.py \
        --out rebuttal/condition_tables.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

GAMMA = 1.0
T_SPLIT = 1024
N_SUP = (1024, 5000, 10001)

DIAGS = {
    "pfn-600": "beta_bernoulli/checkpoints/diag_seqlen1024_training600.pt",
    "pfn-50k": "beta_bernoulli/checkpoints/diag_seqlen1024_training50k.pt",
    "oracle": "beta_bernoulli/checkpoints/diag_oracle.pt",
}

CONDITIONS = ("C1", "C2", "C3", "C4", "R")


def load(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    d = torch.load(path, map_location="cpu", weights_only=False)
    b = d["b"].numpy().astype(np.float64)          # [K, R], k = k_min..k_max
    delta = d["delta"].numpy().astype(np.float64)  # [K, R]
    y = d["y"].numpy().astype(np.float64)          # [seq_len, R]
    return b, delta, y, int(d["k_min"])


def sup_stat(arr: np.ndarray, ks: np.ndarray, n_max: int) -> tuple[np.ndarray, np.ndarray]:
    """max_{k <= n_max} k^{gamma/2} |arr_k| per rollout -> values (R,), argmax k (R,)."""
    mask = ks <= n_max
    w = ks[mask].astype(np.float64) ** (GAMMA / 2.0)
    weighted = w[:, None] * np.abs(arr[mask])
    idx = weighted.argmax(axis=0)
    return weighted.max(axis=0), ks[mask][idx]


def condition_traj(
    b: np.ndarray, delta: np.ndarray, ks: np.ndarray, lo: int, hi: int
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Trajectories of C1..C4 and R on n in [lo, hi]; tails run to the full
    stored horizon N = ks[-1] (as plotted). Returns {name: (n, stat[K_w, R])}."""
    P = np.cumsum(b, axis=0)
    Q = np.cumsum(b * b, axis=0)
    S = np.cumsum(delta * delta, axis=0)
    R_ = b.shape[1]
    Q_prev = np.concatenate([np.zeros((1, R_)), Q[:-1]], axis=0)

    wmask = (ks >= lo) & (ks <= hi)
    n = ks[wmask].astype(np.float64)
    tail_gt = P[-1][None, :] - P[wmask]            # sum_{k>n} b_k
    sq_tail_ge = Q[-1][None, :] - Q_prev[wmask]    # sum_{k>=n} b_k^2
    d_tail_gt = S[-1][None, :] - S[wmask]          # sum_{k>n} delta_k^2
    trunc = np.maximum(1.0 - n / float(ks[-1]), 1e-12)

    n_half = n ** (GAMMA / 2.0)
    n_full = n ** GAMMA
    return {
        "C1": (n, np.abs(tail_gt)),
        "C2": (n, n_half[:, None] * np.abs(tail_gt)),
        "C3": (n, n_half[:, None] * np.abs(b[wmask])),
        "C4": (n, n_full[:, None] * sq_tail_ge),
        "R": (n, (n_full / trunc)[:, None] * d_tail_gt),
    }


def loglog_slope(n: np.ndarray, v: np.ndarray) -> float:
    m = np.isfinite(v) & (v > 0)
    if m.sum() < 3:
        return np.nan
    x = np.log(n[m])
    A = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(A, np.log(v[m]), rcond=None)
    return float(coef[1])


def mmm(v: np.ndarray) -> str:
    return f"{np.median(v):.3g} [{v.min():.3g}, {v.max():.3g}]"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="rebuttal/condition_tables.md")
    args = p.parse_args()

    lines: list[str] = []

    # ------------------------------------------------ (a) sup statistic
    lines.append("### (a) Finite-horizon sup statistic at gamma=1")
    lines.append("")
    lines.append(
        "| Model | Statistic | N=1024 | N=5000 | N=10001 |"
    )
    lines.append("|---|---|---|---|---|")
    for model, path in DIAGS.items():
        b, delta, y, k_min = load(path)
        ks = np.arange(k_min, k_min + b.shape[0])
        for stat_name, arr in (("sup k^{1/2}|Delta_k|", delta), ("sup k^{1/2}|b_k|", b)):
            cells = []
            argmax_full = None
            for N in N_SUP:
                vals, arg = sup_stat(arr, ks, N)
                cells.append(mmm(vals))
                argmax_full = arg
            lines.append(f"| {model} | {stat_name} | " + " | ".join(cells) + " |")
            lines.append(
                f"<!-- {model} {stat_name}: argmax k across rollouts "
                f"med={int(np.median(argmax_full))} "
                f"range=[{int(argmax_full.min())}, {int(argmax_full.max())}] -->"
            )
    lines.append("")

    # ------------------------------------------------ (b) T-split
    for model, path in DIAGS.items():
        b, delta, y, k_min = load(path)
        ks = np.arange(k_min, k_min + b.shape[0])
        within = condition_traj(b, delta, ks, lo=32, hi=512)
        beyond = condition_traj(b, delta, ks, lo=1200, hi=5000)

        lines.append(f"### (b) T-split at n = T = {T_SPLIT}, gamma=1 — {model}")
        lines.append("")
        lines.append(
            "| Cond | within-T value | within-T slope | beyond-T value | beyond-T slope |"
        )
        lines.append("|---|---|---|---|---|")
        for c in CONDITIONS:
            nw, vw = within[c]
            nb, vb = beyond[c]
            val_w = np.median(vw, axis=0)
            val_b = np.median(vb, axis=0)
            slope_w = np.array([loglog_slope(nw, vw[:, r]) for r in range(vw.shape[1])])
            slope_b = np.array([loglog_slope(nb, vb[:, r]) for r in range(vb.shape[1])])
            lines.append(
                f"| {c} | {mmm(val_w)} | {mmm(slope_w)} | {mmm(val_b)} | {mmm(slope_b)} |"
            )
        lines.append("")
        theta_emp = y.mean(axis=0)
        v_pred = theta_emp * (1 - theta_emp)
        lines.append(
            f"(R) coherent-model reference level theta_hat(1-theta_hat): {mmm(v_pred)}"
        )
        lines.append("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
