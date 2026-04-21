"""Plot per-rollout (C2) diagnostics for corrupted-oracle sweeps.

One figure per sweep (noise / decay). For each sweep, a grid:
  columns: corruption levels (eps or p)
  rows:    gamma in {0.5, 1.0}
Each panel shows per-rollout trajectories of n^{gamma/2} * |P_{k_max} - P_n|,
the (C2) diagnostic.

Usage (from beta_bernoulli dir):
    .venv/bin/python plot_corrupt_sweep.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


SWEEP_DIR = Path("checkpoints/corrupt_sweeps")
OUT_DIR = Path("checkpoints")


def load(path: Path) -> dict:
    return torch.load(path, weights_only=False)


def c2_trajectory(b: np.ndarray, k_min: int, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (n_grid, [K,R] trajectory of n^{gamma/2} * |P_kmax - P_n|)."""
    K, R = b.shape
    n = np.arange(k_min, k_min + K, dtype=np.float64)
    P = np.cumsum(b, axis=0)
    Pinf = P[-1]
    tail = Pinf[None, :] - P
    weight = n ** (gamma / 2.0)
    return n, weight[:, None] * np.abs(tail)


def plot_sweep(
    diag_paths: list[Path],
    labels: list[str],
    title: str,
    out_path: Path,
    gammas=(0.5, 1.0),
) -> None:
    ncols = len(diag_paths)
    nrows = len(gammas)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(2.9 * ncols, 2.3 * nrows),
        squeeze=False,
        sharex=True,
    )
    for col, (path, label) in enumerate(zip(diag_paths, labels)):
        d = load(path)
        b = d["b"].numpy()
        k_min = d["k_min"]
        K, R = b.shape
        cutoff = k_min + K // 2
        for row, gamma in enumerate(gammas):
            ax = axes[row, col]
            n, traj = c2_trajectory(b, k_min, gamma)
            for r in range(R):
                ax.plot(n, traj[:, r], lw=0.6, alpha=0.6)
            ax.axvline(cutoff, color="red", lw=0.8, ls="--", alpha=0.6)
            ax.set_xscale("log")
            ax.set_yscale("log")
            if row == 0:
                ax.set_title(label, fontsize=10)
            if row == nrows - 1:
                ax.set_xlabel("$n$")
            if col == 0:
                ax.set_ylabel(f"$n^{{\\gamma/2}}|P_\\infty - P_n|$, $\\gamma={gamma}$")
    fig.suptitle(title, y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def mean_abs_bn_overlay(
    diag_paths: list[Path],
    labels: list[str],
    title: str,
    out_path: Path,
) -> None:
    """Overlay mean_r |b_n| on log-log axes for context (one curve per setting)."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for path, label in zip(diag_paths, labels):
        d = load(path)
        b = d["b"].numpy()
        k_min = d["k_min"]
        K, _ = b.shape
        n = np.arange(k_min, k_min + K, dtype=np.float64)
        mean_abs = np.mean(np.abs(b), axis=1)
        ax.plot(n, mean_abs, lw=0.8, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$n$")
    ax.set_ylabel(r"$\widehat{\mathbb{E}}|b_n|$")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    noise_epsilons = ["1e-3", "1e-2", "1e-1"]
    noise_paths = [SWEEP_DIR / f"diag_noise_eps{e}.pt" for e in noise_epsilons]
    noise_labels = [f"iid noise, $\\epsilon={e}$" for e in noise_epsilons]
    plot_sweep(
        noise_paths, noise_labels,
        "Experiment (i): corrupted oracle, iid logit-scale Gaussian noise. (C2) at two $\\gamma$ values.",
        OUT_DIR / "diag_corrupt_noise_c2.pdf",
    )
    mean_abs_bn_overlay(
        noise_paths, noise_labels,
        r"Mean $|b_n|$ for noise sweep",
        OUT_DIR / "diag_corrupt_noise_bn.pdf",
    )

    decay_ps = ["0.25", "0.5", "1.0", "1.5", "2.0"]
    decay_paths = [SWEEP_DIR / f"diag_decay_p{p}.pt" for p in decay_ps]
    decay_labels = [f"decay, $\\epsilon=0.5$, $p={p}$" for p in decay_ps]
    plot_sweep(
        decay_paths, decay_labels,
        "Experiment (ii): corrupted oracle, envelope $\\epsilon\\,n^{-p}$. (C2) at two $\\gamma$ values.",
        OUT_DIR / "diag_corrupt_decay_c2.pdf",
    )
    mean_abs_bn_overlay(
        decay_paths, decay_labels,
        r"Mean $|b_n|$ for decay sweep",
        OUT_DIR / "diag_corrupt_decay_bn.pdf",
    )


if __name__ == "__main__":
    main()
