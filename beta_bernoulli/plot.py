"""Predictive-CLT diagnostic plots for the Beta-Bernoulli testbed.

Two families of conditions are tested on the stored per-rollout drift tensor
b[k, r] and increment tensor delta[k, r]:

  "signed" -- the pathwise signed-tail conditions (C1)--(C4) on the
              conditional drift b_k together with the residual condition
              (R) on the realised increments delta_k, certifying a
              predictive CLT at rate n^{gamma/2} (Theorem th:ascondmult).
              (C1)--(C4) should decay to zero; (R) should stabilise at a
              positive finite limit. (C1) does not involve gamma;
              (C2)--(C4) and (R) are swept over a dense gamma grid
              (default 10 rows from gamma=0.1 to 1.0). The certified rate
              is the largest gamma at which (C2)--(C4) decay to zero and
              (R) stabilises at a positive finite value on every rollout.

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

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Signed-tail pathwise conditions (Theorem th:ascondmult, sandra_new0804.tex)
# ---------------------------------------------------------------------------


def compute_signed(
    b: np.ndarray,
    k_min: int,
    gammas: np.ndarray,
    delta: np.ndarray | None = None,
) -> dict:
    """Per-rollout trajectories for the signed-tail statistics at each gamma.

    Computes, by cumulative-sum telescoping over k >= k_min:
        sum_{k=n}^{k_max} b_k         (signed tail of conditional drift)
        sum_{k=n}^{k_max} b_k^2       (squared tail of conditional drift)
        sum_{k=n+1}^{k_max} delta_k^2 (residual quadratic variation, if delta given)

    where k_max is the largest probe index in the supplied tensors (a
    finite-N truncation of the theoretical sum to infinity). These are the
    quantities entering (C1), (C2), (C4), and (R) of Appendix H.

    If ``delta`` is supplied, also computes the per-gamma weighted residual
    n^gamma * sum_{k>n} delta_k^2 underlying (R) (condition (iii) of
    Theorem th:ascondmult). Unlike (C2)--(C4) which decay to zero, (R)
    stabilises at a positive finite limit at the correct gamma.
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

    if delta is not None:
        if delta.shape != b.shape:
            raise ValueError(f"delta shape {delta.shape} != b shape {b.shape}")
        S = np.cumsum(delta * delta, axis=0)        # [K, R]
        Sinf = S[-1]
        delta_squared_tail_gt = Sinf[None, :] - S    # sum_{k > n} delta_k^2 (for C5)
    else:
        delta_squared_tail_gt = None

    results: dict = {
        "n": ks,
        "unweighted_tail": np.abs(signed_tail_gt),
        "per_gamma": {},
    }
    for gamma in gammas:
        nf = ks.astype(np.float64)
        n_half = nf ** (gamma / 2.0)
        n_full = nf ** gamma
        per = {
            "rate_residual_tail": n_half[:, None] * np.abs(signed_tail_gt),
            "rate_abs_bn": n_half[:, None] * np.abs(b),
            "rate_residual_squared": n_full[:, None] * squared_tail,
        }
        if delta_squared_tail_gt is not None:
            per["rate_delta_squared_tail"] = n_full[:, None] * delta_squared_tail_gt
        results["per_gamma"][float(gamma)] = per
    return results


def _residual_scatter_data(
    delta: np.ndarray,
    y_labels: np.ndarray,
    n: np.ndarray,
    gammas: np.ndarray,
    n_eval: int,
) -> tuple[dict[float, np.ndarray], np.ndarray, float, int]:
    """Per-rollout (V_emp, V_pred) at each gamma for the (R) scatter column.

    For Beta-Bernoulli the oracle increment satisfies
    Delta_k = (Y_k - g_{k-1})/(k+2), so at the certified rate gamma=1
        n * sum_{k>n} Delta_k^2  ->  theta(1-theta)  a.s.,
    finite-N truncated at index N-1: n*sum_{k=n+1}^{N-1}Delta_k^2
        ~ theta(1-theta) * (1 - n/(N-1)).
    We evaluate the rate-weighted truncated sum at a single probe index
    n_eval and divide by Cbar1 := 1 - n_eval/(N-1), the gamma=1 oracle
    truncation factor. The same factor is applied at every gamma row; the
    diagonal y=x in each panel is the gamma=1 oracle prediction.
    Rows at gamma<1 show V_emp dropping toward zero (martingale-variance
    dominated), gamma>1 would blow up.

    Returns
    -------
    V_emp_per_gamma : dict[float, np.ndarray of shape (R,)]
        n_eval^gamma * sum_{k>n_eval}(Delta_k^{(r)})^2 / Cbar1, per rollout.
    V_pred : np.ndarray of shape (R,)
        Per-rollout gamma=1 theoretical limit theta_emp(1-theta_emp).
    Cbar1 : float
        The gamma=1 finite-N truncation factor (1 - n_eval/(N-1)).
    n_eval_actual : int
        The actual probe index used (closest available index to n_eval).
    """
    K, R = delta.shape
    S = np.cumsum(delta * delta, axis=0)
    Sinf = S[-1]
    tail = Sinf - S  # tail[i, r] = sum_{k>n[i]} delta_k^2
    Nm1 = float(n[-1])
    idx = int(np.argmin(np.abs(n - n_eval)))
    n_eval_actual = int(n[idx])
    Cbar1 = max(1.0 - n_eval_actual / Nm1, 1e-12)

    V_emp_per_gamma: dict[float, np.ndarray] = {}
    for gamma in gammas:
        V_raw = (float(n_eval_actual) ** gamma) * tail[idx]  # shape (R,)
        V_emp_per_gamma[float(gamma)] = V_raw / Cbar1

    y_arr = y_labels
    if y_arr.shape[0] == R and y_arr.shape[1] != R:
        y_arr = y_arr.T
    theta_emp = y_arr.mean(axis=0)
    V_pred = theta_emp * (1.0 - theta_emp)
    return V_emp_per_gamma, V_pred, Cbar1, n_eval_actual


def plot_signed(
    results: dict,
    out_path: str,
    title: str,
    x_max_trunc: int | None = 5000,
    c2_xlim: tuple[int, int] | None = (2000, 5000),
    y_labels: np.ndarray | None = None,
    delta: np.ndarray | None = None,
    n_eval: int = 2000,
) -> None:
    """Per-rollout signed-tail panels with rate sweep over gamma.

    Top row: (C1) per-rollout trajectories of |sum_{k>=n} b_k|.
    Below: one row per gamma. Columns (C2), (C3), (C4) sweep the rate-
    weighted drift conditions; the (R) column shows a per-gamma scatter
    of n_eval^gamma * sum_{k>n_eval}(Delta_k)^2 / Cbar1 against
    theta_emp(1-theta_emp), with the diagonal y=x as the gamma=1 oracle
    reference.

    ``x_max_trunc`` caps the x-axis of the C1 top row and the C2, C4
    columns; (C3) keeps the full probe range. The histogram requires both
    ``delta`` and ``y_labels``; if either is missing, the (R) column is
    omitted and the figure has only three sweep columns.
    """
    gammas = sorted(results["per_gamma"].keys())
    n = results["n"]
    K, R = results["unweighted_tail"].shape

    show_residual = (
        "rate_delta_squared_tail" in next(iter(results["per_gamma"].values()))
        and y_labels is not None
        and delta is not None
    )
    n_cols = 4 if show_residual else 3

    fig = plt.figure(figsize=(2.5 * n_cols, 1.6 + 1.0 * len(gammas)))
    gs = fig.add_gridspec(
        nrows=1 + len(gammas),
        ncols=n_cols,
        height_ratios=[1.6] + [1.0] * len(gammas),
        hspace=0.7,
        wspace=0.35,
    )

    if x_max_trunc is not None and x_max_trunc > n[0]:
        trunc_mask = n <= x_max_trunc
    else:
        trunc_mask = np.ones_like(n, dtype=bool)

    ax_top = fig.add_subplot(gs[0, :])
    for r in range(R):
        ax_top.plot(
            n, results["unweighted_tail"][:, r],
            color="black", alpha=0.22, linewidth=0.8, rasterized=True,
        )
    ax_top.set_xscale("log")
    ax_top.set_yscale("log")
    ax_top.set_xlabel(r"$n$", fontsize=11, labelpad=2)
    ax_top.set_title("(C1)", fontsize=12)
    ax_top.tick_params(axis="both", labelsize=9)
    ax_top.grid(True, which="both", alpha=0.25)
    if x_max_trunc is not None:
        ax_top.set_xlim(float(n[0]), float(x_max_trunc))
        vis = results["unweighted_tail"][trunc_mask].ravel()
        vis = vis[np.isfinite(vis) & (vis > 0)]
        if vis.size > 0:
            ax_top.set_ylim(float(vis.min()) * 0.85, float(vis.max()) * 1.15)
        ax_top.set_xticks([1e1, 1e2, 1e3])
        ax_top.set_xticklabels([r"$10^1$", r"$10^2$", r"$10^3$"])

    # key, label, yscale, truncate_x  --- the three drift sweep columns
    stat_info = [
        ("rate_residual_tail", "(C2)", "log", True),
        ("rate_abs_bn",        "(C3)", "log", False),
        ("rate_residual_squared", "(C4)", "log", True),
    ]

    col_axes: list[list[plt.Axes]] = [[] for _ in stat_info]
    c2_mask = (n >= c2_xlim[0]) & (n <= c2_xlim[1]) if c2_xlim else None

    for row, gamma in enumerate(gammas):
        for col, (key, label, yscale, truncate) in enumerate(stat_info):
            ax = fig.add_subplot(gs[1 + row, col])
            col_axes[col].append(ax)
            arr = results["per_gamma"][gamma][key]
            for r in range(R):
                ax.plot(
                    n, arr[:, r],
                    color="black", alpha=0.18, linewidth=0.7, rasterized=True,
                )
            ax.set_xscale("log")
            ax.set_yscale(yscale)

            # C2 column gets its own zoom into the asymptotic strip.
            if key == "rate_residual_tail" and c2_xlim is not None:
                ax.set_xlim(*c2_xlim)
                vis = arr[c2_mask].ravel()
                vis = vis[np.isfinite(vis) & (vis > 0)]
                if vis.size > 0:
                    ax.set_ylim(float(vis.min()) * 0.85,
                                float(vis.max()) * 1.15)
                ax.set_xticks([c2_xlim[0], c2_xlim[1]])
                ax.set_xticklabels(
                    [f"$\\!{int(c2_xlim[0])}$", f"$\\!{int(c2_xlim[1])}$"]
                )
                ax.minorticks_off()
            elif truncate and x_max_trunc is not None:
                ax.set_xlim(float(n[0]), float(x_max_trunc))
                vis = arr[trunc_mask].ravel()
                if yscale == "log":
                    vis = vis[np.isfinite(vis) & (vis > 0)]
                else:
                    vis = vis[np.isfinite(vis)]
                if vis.size > 0:
                    if yscale == "linear":
                        lo, hi = float(vis.min()), float(vis.max())
                        span = hi - lo if hi > lo else max(abs(hi), 1e-12)
                        ax.set_ylim(lo - 0.05 * span, hi + 0.05 * span)
                    else:
                        ax.set_ylim(float(vis.min()) * 0.85,
                                    float(vis.max()) * 1.15)
            if row == 0:
                ax.set_title(label, fontsize=12)
            if col == 0:
                ax.set_ylabel(rf"$\gamma={gamma:g}$", fontsize=12)
            if row == len(gammas) - 1:
                ax.set_xlabel(r"$n$", fontsize=11)
            ax.tick_params(axis="both", labelsize=9)
            ax.grid(True, which="both", alpha=0.25)

    # Columns whose x-window is independently set per row keep per-row ylims;
    # remaining columns share their y-axis across rows.
    independent_keys = {"rate_residual_tail", "rate_residual_squared"}
    for col, (key, _, _, _) in enumerate(stat_info):
        if key in independent_keys:
            continue
        axes = col_axes[col]
        ymin = min(ax.get_ylim()[0] for ax in axes)
        ymax = max(ax.get_ylim()[1] for ax in axes)
        for ax in axes:
            ax.set_ylim(ymin, ymax)

    if show_residual:
        V_emp_per_gamma, V_pred, _, n_eval_used = _residual_scatter_data(
            delta, y_labels, n, np.asarray(gammas, dtype=np.float64),
            n_eval=n_eval,
        )
        x_lim = 0.27
        for row, gamma in enumerate(gammas):
            ax = fig.add_subplot(gs[1 + row, 3])
            V_emp_g = V_emp_per_gamma[float(gamma)]
            v_max = float(np.nanmax(V_emp_g)) if V_emp_g.size else x_lim
            y_lim = max(x_lim, v_max * 1.1)
            diag_to = min(x_lim, y_lim)
            ax.plot([0.0, diag_to], [0.0, diag_to],
                    color="black", lw=0.8, linestyle="--", zorder=2)
            ax.scatter(V_pred, V_emp_g, color="C1", s=18, alpha=0.85,
                       edgecolor="black", linewidth=0.3, zorder=3)
            ax.set_xlim(0.0, x_lim)
            ax.set_ylim(0.0, y_lim)
            ax.tick_params(axis="both", labelsize=9)
            ax.grid(True, alpha=0.25)
            if row == 0:
                ax.set_title("(R)", fontsize=12)
            if row == len(gammas) - 1:
                ax.set_xlabel(r"$\hat\theta_r(1{-}\hat\theta_r)$",
                              fontsize=11)

    fig.suptitle(title, fontsize=12)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    extra = " + (R) histogram" if show_residual else ""
    print(f"[signed] wrote {out_path}  ({R} per-rollout paths, "
          f"{len(gammas)} gamma rows, 3 sweep cols{extra})")


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

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))

    ax = axes[0]
    ax.plot(n, E_abs, color="C0", label=r"$\widehat{\mathbb{E}}|b_n|$")
    ax.fill_between(n, q25, q75, color="C0", alpha=0.15, label="rollout IQR")
    ax.plot(n_fit, fit_curve, color="red", linestyle="--",
            label=rf"fit: $\widehat\beta={beta:.3f}\pm{ci:.3f}$")
    ax.axvspan(fit_n_min, fit_n_max, color="grey", alpha=0.08)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$n$", fontsize=11)
    ax.set_title(r"$\widehat{\mathbb{E}}|b_n|$ and OLS fit $C n^{-\beta}$",
                 fontsize=12)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=10, loc="best")

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
    ax.set_xlabel(r"$n$", fontsize=11)
    ax.set_title(
        r"$U_\gamma(n)=\sum_{m\leq n} m^{\gamma/2}\widehat{\mathbb{E}}|b_m|$",
        fontsize=12,
    )
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="best")

    fig.suptitle(
        f"{title}\n"
        rf"$\widehat\beta={beta:.3f}$;  "
        rf"$\gamma_Q^*=\min(1,2(\widehat\beta-1))={gamma_Q_star:.3f}$",
        fontsize=12,
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
    p.add_argument("--x-max-trunc", type=int, default=5000,
                   help="Cap the x-axis at this n on the C1 top row and on "
                        "the C2, C4, and (R) columns; (C3) keeps the full "
                        "range. Pass 0 to disable.")
    args = p.parse_args()

    gammas = (np.linspace(0.1, 1.0, 10) if args.gammas is None
              else np.asarray(args.gammas, dtype=np.float64))

    data = torch.load(args.diag, map_location="cpu", weights_only=False)
    b = data["b"].numpy()
    delta = data["delta"].numpy() if "delta" in data and data["delta"] is not None else None
    y_labels = data["y"].numpy() if "y" in data and data["y"] is not None else None
    k_min = int(data["k_min"])

    if args.plot_k_max is not None:
        keep = args.plot_k_max - k_min + 1
        if keep < b.shape[0]:
            b = b[:keep]
            if delta is not None:
                delta = delta[:keep]
            print(f"[truncate] k_max -> {args.plot_k_max} "
                  f"(kept {keep} of {data['b'].shape[0]} steps)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.stem or Path(args.diag).stem

    if "signed" in args.conditions:
        results = compute_signed(b, k_min=k_min, gammas=gammas, delta=delta)
        x_max_trunc = args.x_max_trunc if args.x_max_trunc and args.x_max_trunc > 0 else None
        has_R = (delta is not None) and (y_labels is not None)
        col_label = "(C1)-(C4) and (R)" if has_R else "(C1)-(C4)"
        plot_signed(
            results,
            out_path=str(out_dir / f"{stem}-signed.pdf"),
            title=f"{args.title} — pathwise signed conditions {col_label}",
            x_max_trunc=x_max_trunc,
            y_labels=y_labels,
            delta=delta,
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
