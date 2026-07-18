"""Variance-only diagnostic of the Gaussian approximation.

The asymptotic Gaussian approximation
$\\tilde F - F_n \\approx \\mathcal N(\\sum b_k, \\sum \\Delta_k^2)$ is a
good fit when the bias is small relative to the standard deviation,
$(\\sum b_k)^2 \\ll \\sum \\Delta_k^2$, and the drift's contribution to
the variance is dominated by the martingale variance,
$\\sum b_k^2 \\ll \\sum \\Delta_k^2$. The $n^\\gamma$ in the rate-
weighted certified-rate formulation cancels in the asymptotic shape; it
re-enters only through the variance estimator $\\hat V_n / n^\\gamma$.

This script plots three per-rollout curves per predictive rule on a
single set of log-log axes:
  (a) $(\\sum_{k>=n} b_k)^2$           squared cumulative bias
  (b) $\\sum_{k>=n} b_k^2$             drift's contribution to variance
  (c) $\\sum_{k>=n} \\Delta_k^2$        martingale variance

For an asymptotically Gaussian predictive rule, (a) and (b) sit far below
(c) over the visible probe range.

Usage (from beta_bernoulli/):
    .venv/bin/python plot_variance_only.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


CHECKPOINTS = Path("checkpoints")


def load_abc(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (n, a, b, c) for the diagnostic file at `path`.

    Each of a, b, c has shape (K, R): one curve per rollout.
    """
    d = torch.load(path, map_location="cpu", weights_only=False)
    b_drift = d["b"].numpy()           # conditional drifts, (K, R)
    delta = d["delta"].numpy()         # increments, (K, R)
    K, R = b_drift.shape
    n = np.arange(int(d["k_min"]), int(d["k_min"]) + K)

    P = np.cumsum(b_drift, axis=0)
    Pinf = P[-1]
    bias_tail = Pinf[None, :] - P                    # sum_{k>n} b_k
    a = bias_tail ** 2

    Q = np.cumsum(b_drift * b_drift, axis=0)
    Qinf = Q[-1]
    Q_prev = np.concatenate([np.zeros((1, R)), Q[:-1]], axis=0)
    b_var = Qinf[None, :] - Q_prev                   # sum_{k>=n} b_k^2

    S = np.cumsum(delta * delta, axis=0)
    Sinf = S[-1]
    c = Sinf[None, :] - S                            # sum_{k>n} delta_k^2

    return n, a, b_var, c


def draw_panel(ax: plt.Axes, n: np.ndarray, a: np.ndarray, b: np.ndarray,
               c: np.ndarray, title: str, *, with_legend: bool = False,
               with_xlabel: bool = False, with_ylabel: bool = False,
               x_max: int = 5000) -> None:
    R = a.shape[1]
    for r in range(R):
        ax.plot(n, c[:, r], color="tab:blue",   alpha=0.30, lw=0.7, rasterized=True)
        ax.plot(n, b[:, r], color="tab:orange", alpha=0.30, lw=0.7, rasterized=True)
        ax.plot(n, a[:, r], color="tab:red",    alpha=0.30, lw=0.7, rasterized=True)
    if with_legend:
        ax.plot([], [], color="tab:blue",
                label=r"(c) $\sum_{k>n}\Delta_k^2$")
        ax.plot([], [], color="tab:orange",
                label=r"(b) $\sum_{k\geq n} b_k^2$")
        ax.plot([], [], color="tab:red",
                label=r"(a) $(\sum_{k\geq n} b_k)^2$")
        ax.legend(fontsize=8, loc="lower left")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(float(n[0]), float(x_max))
    ax.set_title(title, fontsize=11)
    ax.tick_params(axis="both", labelsize=9)
    if with_xlabel:
        ax.set_xlabel(r"$n$", fontsize=11)
    if with_ylabel:
        ax.set_ylabel("magnitude", fontsize=10)
    ax.grid(True, which="both", alpha=0.25)


def figure_three(diag_paths: list[Path], titles: list[str], out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), sharey=True)
    for i, (path, title) in enumerate(zip(diag_paths, titles)):
        n, a, b, c = load_abc(path)
        draw_panel(
            axes[i], n, a, b, c, title,
            with_legend=(i == 0), with_xlabel=True,
            with_ylabel=(i == 0),
        )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def figure_eight(diag_paths: list[Path], titles: list[str], out: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.4), sharey=True)
    for i, (path, title) in enumerate(zip(diag_paths, titles)):
        ax = axes[i // 4, i % 4]
        n, a, b, c = load_abc(path)
        draw_panel(
            ax, n, a, b, c, title,
            with_legend=(i == 0),
            with_xlabel=(i // 4 == 1),
            with_ylabel=(i % 4 == 0),
        )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    bft_paths = [
        CHECKPOINTS / "diag_oracle.pt",
        CHECKPOINTS / "diag_seqlen1024_training600.pt",
        CHECKPOINTS / "diag_seqlen1024_training50k.pt",
    ]
    bft_titles = ["Bayes oracle", "600-step BFT", "50k-step BFT"]
    figure_three(
        bft_paths, bft_titles,
        CHECKPOINTS / "bb-variance-only-bfts.pdf",
    )

    sweep = CHECKPOINTS / "corrupt_sweeps"
    corrupt_paths = [
        sweep / "diag_noise_eps1e-3.pt",
        sweep / "diag_noise_eps1e-2.pt",
        sweep / "diag_noise_eps1e-1.pt",
        sweep / "diag_decay_p0.25.pt",
        sweep / "diag_decay_p0.5.pt",
        sweep / "diag_decay_p1.0.pt",
        sweep / "diag_decay_p1.5.pt",
        sweep / "diag_decay_p2.0.pt",
    ]
    corrupt_titles = [
        r"noise $\varepsilon=10^{-3}$",
        r"noise $\varepsilon=10^{-2}$",
        r"noise $\varepsilon=10^{-1}$",
        r"decay $p=0.25$",
        r"decay $p=0.5$",
        r"decay $p=1.0$",
        r"decay $p=1.5$",
        r"decay $p=2.0$",
    ]
    figure_eight(
        corrupt_paths, corrupt_titles,
        CHECKPOINTS / "bb-variance-only-corrupt.pdf",
    )


if __name__ == "__main__":
    main()
