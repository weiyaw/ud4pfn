"""Predictive-CLT diagnostic plots for the Beta-Bernoulli testbed.

Two families of conditions are tested on the same stored per-rollout signed
drift tensor b[k, r]:

  "signed" -- the signed-tail predictive-CLT conditions (C1)--(C4).
              These are path-wise a.s. statements, so the plot shows
              every rollout individually with NO cross-rollout averaging.

  "qm"     -- the original strict quasi-martingale conditions
              (E|b_n| absolutely summable, sqrt-n-weighted version).
              These are about expectations, so the plot uses the
              cross-rollout mean hat_E|b_n| = (1/R) sum_r |b_n^(r)|,
              fits a power law, and shows the partial sums
                  S(n) = sum_{m<=n}         hat_E|b_m|
                  T(n) = sum_{m<=n} sqrt(m) hat_E|b_m|

See DIAGNOSTIC_MATH.md for the math behind each statistic and the caveat
about finite k_max.

Usage:
    beta_bernoulli/.venv/bin/python beta_bernoulli/plot.py \
        --diag beta_bernoulli/checkpoints/diag.pt \
        --out-dir beta_bernoulli/checkpoints \
        --conditions signed qm
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Signed-tail path-wise conditions (Theorem 1.2, sandra_new0804.tex)
# ---------------------------------------------------------------------------


def compute_signed(b: np.ndarray, k_min: int, gamma_values: list[float]) -> dict:
    """Per-rollout trajectories for the four signed-tail statistics.

    Uses the telescoping identity
        sum_{k>=n} b_k   = P_infty - P_{n-1}
        sum_{k>=n} b_k^2 = Q_infty - Q_{n-1}
    with P_infty := P_{k_max}, Q_infty := Q_{k_max} (finite-truncation proxies).
    """
    K, R = b.shape
    ks = np.arange(k_min, k_min + K)

    P = np.cumsum(b, axis=0)          # [K, R]
    Q = np.cumsum(b * b, axis=0)      # [K, R]
    Pinf = P[-1]
    Qinf = Q[-1]

    P_prev = np.concatenate([np.zeros((1, R)), P[:-1]], axis=0)
    Q_prev = np.concatenate([np.zeros((1, R)), Q[:-1]], axis=0)

    signed_tail_gt = Pinf[None, :] - P      # [K, R]   sum_{k > ks[i]} b_k
    squared_tail = Qinf[None, :] - Q_prev  # [K, R]  sum_{k >= ks[i]} b_k^2

    results: dict = {
        "n": ks,
        "P": P,
        "unweighted_tail": np.abs(signed_tail_gt),
        "per_gamma": {},
    }
    for gamma in gamma_values:
        n_half = ks.astype(np.float64) ** (gamma / 2.0)
        n_full = ks.astype(np.float64) ** gamma
        results["per_gamma"][gamma] = {
            "rate_residual_tail": n_half[:, None] * np.abs(signed_tail_gt),
            "rate_abs_bn": n_half[:, None] * np.abs(b),
            "rate_residual_squared": n_full[:, None] * squared_tail,
        }
    return results


def plot_signed(results: dict, out_path: str, title: str) -> None:
    """Per-rollout plot of P_n and the three rate-weighted residual panels."""
    gammas = sorted(results["per_gamma"].keys())
    n = results["n"]
    K, R = results["P"].shape
    cutoff = n[0] + (K // 2)

    fig = plt.figure(figsize=(13, 2.6 + 2.4 * len(gammas)))
    gs = fig.add_gridspec(
        nrows=1 + len(gammas),
        ncols=3,
        height_ratios=[1.6] + [1.0] * len(gammas),
        hspace=0.55,
        wspace=0.3,
    )

    ax_top = fig.add_subplot(gs[0, :])
    for r in range(R):
        ax_top.plot(
            n, results["unweighted_tail"][:, r],
            color="black", alpha=0.22, linewidth=0.8, rasterized=True,
        )
    ax_top.axvline(cutoff, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_top.set_xscale("log")
    ax_top.set_yscale("log")
    ax_top.set_xlabel(r"$n$")
    ax_top.set_ylabel(r"$|P_{k_{\max}}^{(r)} - P_n^{(r)}|$")
    ax_top.set_title(
        r"(C1): $|P_{k_{\max}}^{(r)} - P_n^{(r)}|$",
        fontsize=10,
    )
    ax_top.grid(True, which="both", alpha=0.25)

    stat_info = [
        ("rate_residual_tail",
         r"(C2): $n^{\gamma/2}\,|P_{k_{\max}}^{(r)} - P_n^{(r)}|$"),
        ("rate_abs_bn",
         r"(C3): $n^{\gamma/2}\,|b_n^{(r)}|$"),
        ("rate_residual_squared",
         r"(C4): $n^{\gamma}\,(Q^{(r)}_{k_{\max}} - Q^{(r)}_{n-1})$"),
    ]

    # Collect axes by column so we can share y-limits within each column.
    col_axes: list[list[plt.Axes]] = [[] for _ in stat_info]

    for row, gamma in enumerate(gammas):
        for col, (key, label) in enumerate(stat_info):
            ax = fig.add_subplot(gs[1 + row, col])
            col_axes[col].append(ax)
            arr = results["per_gamma"][gamma][key]
            for r in range(R):
                ax.plot(
                    n, arr[:, r],
                    color="black", alpha=0.18, linewidth=0.7, rasterized=True,
                )
            if key != "rate_abs_bn":
                ax.axvline(cutoff, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.set_xscale("log")
            ax.set_yscale("log")
            if row == 0:
                ax.set_title(label, fontsize=9)
            if col == 0:
                ax.set_ylabel(rf"$\gamma={gamma}$", fontsize=11)
            if row == len(gammas) - 1:
                ax.set_xlabel(r"$n$")
            ax.grid(True, which="both", alpha=0.25)

    # Share y-limits within each column across all gamma rows.
    for axes in col_axes:
        ymin = min(ax.get_ylim()[0] for ax in axes)
        ymax = max(ax.get_ylim()[1] for ax in axes)
        for ax in axes:
            ax.set_ylim(ymin, ymax)

    fig.suptitle(
        f"{title}\n"
        "red dashed = $k_{\\max}/2$ (beyond it, finite-truncation dominates)",
        fontsize=11,
    )
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"[signed] wrote {out_path}  ({R} per-rollout paths, no averaging)")


# ---------------------------------------------------------------------------
# Quasi-martingale conditions (strict, expectation-based)
# ---------------------------------------------------------------------------


def fit_power_law(n: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """OLS fit of log y = log C - beta * log n, returning beta, 95% CI, logC."""
    mask = (y > 0) & np.isfinite(y)
    logn = np.log(n[mask])
    logy = np.log(y[mask])
    X = np.column_stack([np.ones_like(logn), logn])
    coefs, *_ = np.linalg.lstsq(X, logy, rcond=None)
    logC, slope = float(coefs[0]), float(coefs[1])
    beta = -slope
    residuals = logy - X @ coefs
    sigma2 = float(np.sum(residuals**2) / max(1, len(logy) - 2))
    xtx_inv = np.linalg.inv(X.T @ X)
    slope_se = float(np.sqrt(sigma2 * xtx_inv[1, 1]))
    return beta, 1.96 * slope_se, logC


def compute_qm(b: np.ndarray, k_min: int) -> dict:
    """Cross-rollout aggregates for the strict quasi-martingale conditions.

    hat_E|b_n| = mean over rollouts of |b_n^(r)|
    S(n)  = sum_{m<=n}        hat_E|b_m|          (unweighted partial sum)
    T(n)  = sum_{m<=n} sqrt(m) hat_E|b_m|         (sqrt-n-weighted partial sum)
    """
    K, R = b.shape
    n = np.arange(k_min, k_min + K)
    abs_b = np.abs(b)
    E_abs = abs_b.mean(axis=1)
    q25 = np.quantile(abs_b, 0.25, axis=1)
    q75 = np.quantile(abs_b, 0.75, axis=1)
    S = np.cumsum(E_abs)
    T = np.cumsum(np.sqrt(n) * E_abs)
    return {
        "n": n,
        "E_abs": E_abs,
        "q25": q25,
        "q75": q75,
        "S": S,
        "T": T,
        "R": R,
    }


def plot_qm(
    results: dict,
    out_path: str,
    title: str,
    fit_n_min: int,
    fit_n_max: int,
) -> None:
    """Cross-rollout E|b_n| with power-law fit, plus partial sums S(n), T(n)."""
    n = results["n"]
    E_abs = results["E_abs"]
    q25 = results["q25"]
    q75 = results["q75"]
    S = results["S"]
    T = results["T"]
    R = results["R"]

    fit_mask = (n >= fit_n_min) & (n <= fit_n_max)
    n_fit = n[fit_mask]
    y_fit = E_abs[fit_mask]
    beta, ci, logC = fit_power_law(n_fit, y_fit)
    fit_curve = np.exp(logC) * n_fit.astype(np.float64) ** (-beta)

    print(f"[qm] R={R} rollouts, probe range k in [{n[0]}, {n[-1]}]")
    print(
        f"[qm] Power-law fit on n in [{fit_n_min}, {fit_n_max}]: "
        f"beta_hat = {beta:.3f}  (95% CI +/- {ci:.3f})"
    )
    print(f"[qm]   quasi-martingale  (beta > 1)   : "
          f"{'holds' if beta - ci > 1.0 else 'not clearly'}")
    print(f"[qm]   sqrt-n weighted   (beta > 1.5) : "
          f"{'holds' if beta - ci > 1.5 else 'not clearly'}")
    print(f"[qm]   final S(n_max) = {S[-1]:.4g}")
    print(f"[qm]   final T(n_max) = {T[-1]:.4g}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    ax = axes[0]
    ax.plot(n, E_abs, color="C0", label=r"$\widehat{\mathbb{E}}|b_n|$ (mean)")
    ax.fill_between(n, q25, q75, color="C0", alpha=0.15, label="rollout IQR")
    ax.plot(
        n_fit, fit_curve, color="red", linestyle="--",
        label=rf"fit: $\widehat\beta = {beta:.2f}\pm{ci:.2f}$",
    )
    ax.axvspan(fit_n_min, fit_n_max, color="grey", alpha=0.08)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$n$")
    ax.set_title(r"$\widehat{\mathbb{E}}|b_n|$ and fit $Cn^{-\beta}$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    ax = axes[1]
    ax.plot(n, S, color="C1")
    ax.set_xscale("log")
    ax.set_xlabel(r"$n$")
    ax.set_title(r"$S(n) = \sum_{m\leq n}\widehat{\mathbb{E}}|b_m|$")
    ax.grid(True, which="both", alpha=0.3)
    ax.annotate(
        "flattens iff\nquasi-martingale ($\\beta>1$)",
        xy=(0.98, 0.05), xycoords="axes fraction",
        ha="right", va="bottom", fontsize=8, alpha=0.7,
    )

    ax = axes[2]
    ax.plot(n, T, color="C2")
    ax.set_xscale("log")
    ax.set_xlabel(r"$n$")
    ax.set_title(r"$T(n) = \sum_{m\leq n}\sqrt{m}\,\widehat{\mathbb{E}}|b_m|$")
    ax.grid(True, which="both", alpha=0.3)
    ax.annotate(
        "flattens iff\n$\\sqrt{n}$-weighted ($\\beta>1.5$)",
        xy=(0.98, 0.05), xycoords="axes fraction",
        ha="right", va="bottom", fontsize=8, alpha=0.7,
    )

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"[qm] wrote {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--diag", type=str, default="beta_bernoulli/checkpoints/diag.pt")
    p.add_argument("--out-dir", type=str, default="beta_bernoulli/checkpoints")
    p.add_argument("--stem", type=str, default=None,
                   help="Output file stem; defaults to the diag filename without extension")
    p.add_argument(
        "--conditions", nargs="+", default=["signed", "qm"],
        choices=["signed", "qm"],
    )
    p.add_argument("--gammas", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    p.add_argument("--title", type=str,
                   default="Beta-Bernoulli PFN")
    p.add_argument("--fit-n-min", type=int, default=10)
    p.add_argument("--fit-n-max", type=int, default=300)
    p.add_argument("--plot-k-max", type=int, default=None,
                   help="Truncate b to this many steps before plotting. "
                        "Useful for restricting to the in-distribution range.")
    args = p.parse_args()

    data = torch.load(args.diag, map_location="cpu", weights_only=False)
    b = data["b"].numpy()
    k_min = int(data["k_min"])

    if args.plot_k_max is not None:
        keep = args.plot_k_max - k_min + 1
        if keep < b.shape[0]:
            b = b[:keep]
            print(f"[truncate] k_max -> {args.plot_k_max} (kept {keep} of {data['b'].shape[0]} steps)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.stem or Path(args.diag).stem

    if "signed" in args.conditions:
        results = compute_signed(b, k_min=k_min, gamma_values=args.gammas)
        plot_signed(
            results,
            out_path=str(out_dir / f"{stem}_signed.pdf"),
            title=f"{args.title} — signed-tail conditions, path-wise",
        )

    if "qm" in args.conditions:
        results = compute_qm(b, k_min=k_min)
        plot_qm(
            results,
            out_path=str(out_dir / f"{stem}_qm.pdf"),
            title=f"{args.title} — strict quasi-martingale conditions",
            fit_n_min=args.fit_n_min,
            fit_n_max=args.fit_n_max,
        )


if __name__ == "__main__":
    main()
