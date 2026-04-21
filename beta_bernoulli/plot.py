"""Predictive-CLT diagnostic plots for the Beta-Bernoulli testbed.

Two families of conditions are tested on the same stored per-rollout signed
drift tensor b[k, r]:

  "signed" -- the pathwise signed-tail conditions (C1)--(C4), certifying a
              predictive CLT at rate n^{gamma/2} (Theorem th:ascondmult in
              sandra_new0804.tex). (C1) does not involve gamma. (C2), (C3),
              (C4) are swept over a dense gamma grid (default 10 rows from
              gamma=0.1 to 1.0); the reader eyeballs whether the per-rollout
              trajectories decay to zero. No automated pass/fail.

  "qm"     -- Sandra's general-gamma quasi-martingale condition (Q_gamma):
              sum_{k>=1} k^{gamma/2} E|b_k| < +oo  (eq:rootn_qm in
              sandra_internal_work.tex). Expectation-based, so it uses the
              cross-rollout mean hat_E|b_n|. Fit a power law hat_E|b_n| ~
              C n^{-beta}; sufficient condition is beta > 1 + gamma/2,
              giving gamma_Q_star = min(1, 2(beta-1)). We also plot
              U_gamma(n) = sum_{m<=n} m^{gamma/2} hat_E|b_m| for several
              gamma; U_gamma flattens iff (Q_gamma) holds.

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
# Signed-tail pathwise conditions (Theorem th:ascondmult, sandra_new0804.tex)
# ---------------------------------------------------------------------------


def compute_signed(b: np.ndarray, k_min: int, gammas: np.ndarray) -> dict:
    """Per-rollout trajectories for the four signed-tail statistics at each gamma.

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
    Q_prev = np.concatenate([np.zeros((1, R)), Q[:-1]], axis=0)

    signed_tail_gt = Pinf[None, :] - P      # sum_{k > n} b_k   (for C1, C2)
    squared_tail = Qinf[None, :] - Q_prev    # sum_{k >= n} b_k^2 (for C4)

    results: dict = {
        "n": ks,
        "unweighted_tail": np.abs(signed_tail_gt),
        "per_gamma": {},
    }
    for gamma in gammas:
        nf = ks.astype(np.float64)
        n_half = nf ** (gamma / 2.0)
        n_full = nf ** gamma
        results["per_gamma"][float(gamma)] = {
            "rate_residual_tail": n_half[:, None] * np.abs(signed_tail_gt),
            "rate_abs_bn": n_half[:, None] * np.abs(b),
            "rate_residual_squared": n_full[:, None] * squared_tail,
        }
    return results


def plot_signed(
    results: dict,
    out_path: str,
    title: str,
    c2_xlim: tuple[int, int] | None = (2000, 5000),
) -> None:
    """Per-rollout plot of P_n and the three rate-weighted residual panels.

    Top row: (C1) per-rollout trajectories (no gamma), full width.
    Below: one row per gamma, three columns for (C2), (C3), (C4).
    ``c2_xlim`` restricts the x-axis of the (C2) column only; (C3) and
    (C4) keep the full range.
    """
    gammas = sorted(results["per_gamma"].keys())
    n = results["n"]
    K, R = results["unweighted_tail"].shape
    cutoff = n[0] + (K // 2)

    fig = plt.figure(figsize=(13, 2.2 + 1.3 * len(gammas)))
    gs = fig.add_gridspec(
        nrows=1 + len(gammas),
        ncols=3,
        height_ratios=[1.6] + [0.85] * len(gammas),
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
        r"(C1): $|P_{k_{\max}}^{(r)} - P_n^{(r)}|$ (no $\gamma$)",
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

    if c2_xlim is not None:
        c2_mask = (n >= c2_xlim[0]) & (n <= c2_xlim[1])
    else:
        c2_mask = np.ones_like(n, dtype=bool)

    # Which column is C2?
    c2_col = next(i for i, (k, _) in enumerate(stat_info)
                  if k == "rate_residual_tail")

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
            if col == c2_col and c2_xlim is not None:
                ax.set_xlim(*c2_xlim)
                vis = arr[c2_mask].ravel()
                vis = vis[np.isfinite(vis) & (vis > 0)]
                if vis.size > 0:
                    ax.set_ylim(float(vis.min()) * 0.85,
                                float(vis.max()) * 1.15)
            if row == 0:
                ax.set_title(label, fontsize=9)
            if col == 0:
                ax.set_ylabel(rf"$\gamma={gamma:g}$", fontsize=10)
            if row == len(gammas) - 1:
                ax.set_xlabel(r"$n$")
            ax.grid(True, which="both", alpha=0.25)

    # C3 and C4 keep the full x-range; share their y-axis across gamma rows
    # (same behaviour as before the C2 windowing was introduced).
    for col, axes in enumerate(col_axes):
        if col == c2_col:
            continue
        ymin = min(ax.get_ylim()[0] for ax in axes)
        ymax = max(ax.get_ylim()[1] for ax in axes)
        for ax in axes:
            ax.set_ylim(ymin, ymax)

    fig.suptitle(
        f"{title}\n"
        "red dashed = $k_{\\max}/2$ (beyond it, finite-truncation dominates)",
        fontsize=10,
    )
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[signed] wrote {out_path}  ({R} per-rollout paths, "
          f"{len(gammas)} gamma rows)")


# ---------------------------------------------------------------------------
# Quasi-martingale condition (general gamma, Sandra eq:rootn_qm)
# ---------------------------------------------------------------------------


def fit_power_law(n: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """OLS fit of log y = log C - beta * log n. Returns beta, 95% CI, logC."""
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


def compute_qm(
    b: np.ndarray,
    k_min: int,
    gammas_U: tuple = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> dict:
    """Cross-rollout aggregates for Sandra's general-gamma (Q_gamma).

    hat_E|b_n| = mean_r |b_n^(r)|.
    U_gamma(n) = sum_{m<=n} m^{gamma/2} hat_E|b_m|.
    Under the power-law ansatz hat_E|b_n| ~ C n^{-beta}, (Q_gamma) holds iff
    beta > 1 + gamma/2, so gamma_Q_star = min(1, 2(beta - 1)).
    """
    K, R = b.shape
    n = np.arange(k_min, k_min + K)
    abs_b = np.abs(b)
    E_abs = abs_b.mean(axis=1)
    q25 = np.quantile(abs_b, 0.25, axis=1)
    q75 = np.quantile(abs_b, 0.75, axis=1)
    nf = n.astype(np.float64)
    U = {float(g): np.cumsum((nf ** (g / 2.0)) * E_abs) for g in gammas_U}
    return {
        "n": n,
        "E_abs": E_abs,
        "q25": q25,
        "q75": q75,
        "U": U,
        "gammas_U": tuple(float(g) for g in gammas_U),
        "R": R,
    }


def plot_qm(
    results: dict,
    out_path: str,
    title: str,
    fit_n_min: int,
    fit_n_max: int,
) -> None:
    """QM figure: hat_E|b_n| + power-law fit, and U_gamma overlay."""
    n = results["n"]
    E_abs = results["E_abs"]
    q25 = results["q25"]
    q75 = results["q75"]
    U = results["U"]
    gammas_U = results["gammas_U"]
    R = results["R"]

    fit_mask = (n >= fit_n_min) & (n <= fit_n_max)
    beta, ci, logC = fit_power_law(n[fit_mask], E_abs[fit_mask])
    n_fit = n[fit_mask]
    fit_curve = np.exp(logC) * n_fit.astype(np.float64) ** (-beta)
    gamma_Q_star = max(0.0, min(1.0, 2.0 * (beta - 1.0)))

    print(f"[qm] R={R} rollouts, n in [{n[0]}, {n[-1]}]")
    print(f"[qm] beta_hat = {beta:.3f} (95% CI +/- {ci:.3f}),  "
          f"gamma_Q* = min(1, 2(beta-1)) = {gamma_Q_star:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.plot(n, E_abs, color="C0", label=r"$\widehat{\mathbb{E}}|b_n|$")
    ax.fill_between(n, q25, q75, color="C0", alpha=0.15, label="rollout IQR")
    ax.plot(n_fit, fit_curve, color="red", linestyle="--",
            label=rf"fit: $\widehat\beta={beta:.3f}\pm{ci:.3f}$")
    ax.axvspan(fit_n_min, fit_n_max, color="grey", alpha=0.08)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$n$")
    ax.set_title(r"$\widehat{\mathbb{E}}|b_n|$ and OLS fit $C n^{-\beta}$",
                 fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    ax = axes[1]
    cmap = plt.get_cmap("viridis")
    for i, g in enumerate(gammas_U):
        color = cmap(i / max(1, len(gammas_U) - 1))
        passes = beta > 1.0 + g / 2.0
        style = "-" if passes else "--"
        label = rf"$\gamma={g}$" + ("" if passes else "  (fail)")
        ax.plot(n, U[g], color=color, linestyle=style, linewidth=1.3, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$n$")
    ax.set_title(
        r"$U_\gamma(n)=\sum_{m\leq n} m^{\gamma/2}\widehat{\mathbb{E}}|b_m|$"
        "\n(solid = bounded under fit, dashed = diverges)",
        fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="best")

    fig.suptitle(
        f"{title}\n"
        rf"$\widehat\beta={beta:.3f}$;  "
        rf"$\gamma_Q^*=\min(1,2(\widehat\beta-1))={gamma_Q_star:.3f}$",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
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
    p.add_argument(
        "--gammas", type=float, nargs="+", default=None,
        help="Gamma values (one trajectory row per gamma) for the signed panel. "
             "Default: {0.1, 0.2, ..., 1.0}.")
    p.add_argument("--gammas-U", type=float, nargs="+",
                   default=[0.0, 0.25, 0.5, 0.75, 1.0],
                   help="Gamma values for the U_gamma(n) overlay in the QM panel.")
    p.add_argument("--title", type=str, default="Beta-Bernoulli predictor")
    p.add_argument("--fit-n-min", type=int, default=10)
    p.add_argument("--fit-n-max", type=int, default=2000)
    p.add_argument("--plot-k-max", type=int, default=None,
                   help="Truncate b to this many steps before plotting.")
    p.add_argument("--c2-xlim", type=int, nargs=2, default=[2000, 5000],
                   metavar=("N_MIN", "N_MAX"),
                   help="x-axis window for the (C2) column only. "
                        "(C1 top row, (C3) and (C4) keep the full range.) "
                        "Pass e.g. 0 0 to disable.")
    args = p.parse_args()

    gammas = (np.linspace(0.1, 1.0, 10) if args.gammas is None
              else np.asarray(args.gammas, dtype=np.float64))

    data = torch.load(args.diag, map_location="cpu", weights_only=False)
    b = data["b"].numpy()
    k_min = int(data["k_min"])

    if args.plot_k_max is not None:
        keep = args.plot_k_max - k_min + 1
        if keep < b.shape[0]:
            b = b[:keep]
            print(f"[truncate] k_max -> {args.plot_k_max} "
                  f"(kept {keep} of {data['b'].shape[0]} steps)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.stem or Path(args.diag).stem

    if "signed" in args.conditions:
        results = compute_signed(b, k_min=k_min, gammas=gammas)
        xlim = tuple(args.c2_xlim) if args.c2_xlim[1] > args.c2_xlim[0] else None
        plot_signed(
            results,
            out_path=str(out_dir / f"{stem}-signed.pdf"),
            title=f"{args.title} — pathwise signed conditions (C1)-(C4)",
            c2_xlim=xlim,
        )

    if "qm" in args.conditions:
        results = compute_qm(b, k_min=k_min, gammas_U=tuple(args.gammas_U))
        plot_qm(
            results,
            out_path=str(out_dir / f"{stem}-qm.pdf"),
            title=rf"{args.title} — general-$\gamma$ QM condition $(Q_\gamma)$",
            fit_n_min=args.fit_n_min,
            fit_n_max=args.fit_n_max,
        )


if __name__ == "__main__":
    main()
