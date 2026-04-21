"""Generate the two-panel Figure 1 for the NeurIPS paper intro.

Panel (a): CLT Gaussian approximation (blue) vs analytic Beta posterior
           (orange) for the trained 50k-step Beta-Bernoulli PFN on a
           fixed sequence y_{1:n}.

Panel (b): Schematic showing how V_n is accumulated from the predictive
           updates Delta_k = g_k - g_{k-1} along the same sequence.

Usage (from repo root, with the beta_bernoulli venv active):

    beta_bernoulli/.venv/bin/python beta_bernoulli/plot_intro_fig.py \\
        --checkpoint beta_bernoulli/checkpoints/seqlen1024_training50k.pt \\
        --out ud4pfn_writing/neurips2026/images/beta-bernoulli-intro.pdf
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import math

import matplotlib.pyplot as plt
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent


def _log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def beta_pdf(t: np.ndarray, a: float, b: float) -> np.ndarray:
    """Beta(a, b) pdf on (0, 1) — avoids a scipy dependency."""
    log_pdf = (a - 1.0) * np.log(t) + (b - 1.0) * np.log(1.0 - t) - _log_beta(a, b)
    return np.exp(log_pdf)
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from diagnostic import load_pfn_predictor  # noqa: E402


@torch.no_grad()
def compute_g_sequence(
    predictor, y_col: torch.Tensor, n: int, g0: float,
) -> np.ndarray:
    """Return g_k = PFN.predict(y_{1:k}) for k = 0, 1, ..., n as a length-(n+1) array.

    ``y_col`` has shape [seq_len, 1] where seq_len >= n+1. The PFN requires at
    least one context token, so ``g_0`` (the prior predictive) is supplied
    externally; remaining entries are evaluated by the model.
    """
    gs = np.empty(n + 1, dtype=np.float64)
    gs[0] = g0
    for k in range(1, n + 1):
        f = predictor.predict(y_col, single_eval_pos=k)  # [R=1]
        gs[k] = float(f[0])
    return gs


def build_panel_a(ax, y_bits: np.ndarray, g_n: float, V_n: float, alpha: float, beta: float) -> None:
    n = len(y_bits)
    ones = int(y_bits.sum())
    zeros = n - ones

    a_post = alpha + ones
    b_post = beta + zeros
    t = np.linspace(0.001, 0.999, 400)
    pdf_beta = beta_pdf(t, a_post, b_post)

    sigma2 = V_n / n
    sigma = float(np.sqrt(max(sigma2, 1e-12)))
    pdf_gauss = np.exp(-0.5 * ((t - g_n) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))

    ax.fill_between(t, pdf_beta, color="C1", alpha=0.25, linewidth=0)
    ax.plot(t, pdf_beta, color="C1", linewidth=1.8,
            label="Beta posterior")
    ax.plot(t, pdf_gauss, color="C0", linewidth=1.8, linestyle="--",
            label=r"Predictive CLT")
    ax.set_xlim(max(0.0, g_n - 4 * sigma), min(1.0, g_n + 4 * sigma))
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("density")
    ax.set_title(rf"(a) Posterior of $\theta \mid y_{{1:{n}}}$", fontsize=10)
    ax.legend(fontsize=7, loc="upper right", frameon=False,
              handlelength=1.1, borderaxespad=0.3, labelspacing=0.25)


def build_panel_b(ax_top, ax_bot, gs: np.ndarray, n_show: int, n_panel_a: int) -> None:
    ks = np.arange(n_show + 1)
    g = gs[: n_show + 1]
    deltas = np.diff(g)
    k_int = np.arange(1, n_show + 1)
    contrib = (k_int.astype(np.float64) ** 2) * deltas ** 2 / float(n_panel_a)

    # -- top: g_k trajectory --------------------------------------------------
    ax_top.plot(ks, g, color="black", linewidth=0.9, marker="o",
                markersize=2.0, markerfacecolor="white",
                markeredgewidth=0.5, zorder=3)
    ax_top.set_xlim(-2, n_show + 2)
    pad = max(0.03, 0.2 * (g.max() - g.min()))
    ax_top.set_ylim(g.min() - pad, g.max() + pad)
    ax_top.set_ylabel(r"$g_k$")
    ax_top.set_title(r"(b) $V_n=\frac{1}{n}\sum_{k=1}^{n} k^{2}\,\Delta_k^{2}$,"
                     r"  $\Delta_k:=g_k-g_{k-1}$",
                     fontsize=10)
    ax_top.tick_params(labelbottom=False)

    # -- bottom: per-step contribution k^2 Delta_k^2 / n ---------------------
    ax_bot.vlines(k_int, 0.0, contrib, color="C0", linewidth=1.0, alpha=0.85)
    ax_bot.set_xlim(-2, n_show + 2)
    ax_bot.set_ylim(0.0, contrib.max() * 1.1 if contrib.max() > 0 else 1.0)
    ax_bot.set_xlabel(r"$k$")
    ax_bot.set_ylabel(r"$k^{2}\Delta_k^{2}/n$")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str,
                   default="beta_bernoulli/checkpoints/seqlen1024_training50k.pt")
    p.add_argument("--out", type=str,
                   default="ud4pfn_writing/neurips2026/images/beta-bernoulli-intro.pdf")
    p.add_argument("--n", type=int, default=200,
                   help="Context length for the CLT comparison (panel a).")
    p.add_argument("--theta-star", type=float, default=0.3,
                   help="True Bernoulli probability generating the demo context.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--n-schematic", type=int, default=None,
                   help="Number of steps shown in the schematic (panel b). "
                        "Defaults to --n so panels (a) and (b) agree.")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    args = p.parse_args()
    if args.n_schematic is None:
        args.n_schematic = args.n

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = load_pfn_predictor(args.checkpoint, dtype=torch.float32, device=device)

    n_total = max(args.n, args.n_schematic) + 1
    y_bits = (rng.random(n_total) < args.theta_star).astype(np.float32)
    y = torch.from_numpy(y_bits).to(device).view(n_total, 1)

    g0 = args.alpha / (args.alpha + args.beta)
    gs = compute_g_sequence(predictor, y, n=args.n, g0=g0)

    deltas = np.diff(gs)
    ks = np.arange(1, args.n + 1, dtype=np.float64)
    V_n = float(np.sum((ks ** 2) * deltas ** 2) / args.n)
    g_n = float(gs[-1])

    print(f"n={args.n} ones={int(y_bits[:args.n].sum())} "
          f"g_n={g_n:.4f} V_n={V_n:.4f} sigma={np.sqrt(V_n/args.n):.4f}")

    fig = plt.figure(figsize=(8.5, 3.0))
    gs_layout = fig.add_gridspec(
        nrows=2, ncols=2,
        height_ratios=[2.1, 1.0],
        width_ratios=[1.0, 1.1],
        hspace=0.15, wspace=0.28,
    )
    ax_a = fig.add_subplot(gs_layout[:, 0])
    ax_b_top = fig.add_subplot(gs_layout[0, 1])
    ax_b_bot = fig.add_subplot(gs_layout[1, 1], sharex=ax_b_top)

    build_panel_a(ax_a, y_bits[:args.n], g_n, V_n, args.alpha, args.beta)
    build_panel_b(ax_b_top, ax_b_bot, gs, args.n_schematic, args.n)

    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
