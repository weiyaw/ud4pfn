# W5 extrapolation profile: |g_n - (1+S)/(n+2)| for the trained Beta-Bernoulli
# BFT, evaluated on Bernoulli(theta) prefixes across n on both sides of the
# training horizon n=1024. The predictive is a deterministic map of the prefix,
# so no rollouts are involved.
#
# Usage: <beta_bernoulli venv python> rebuttal/extrapolation_profile.py \
#            --checkpoint <path to seqlen1024_training50k.pt>
import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "beta_bernoulli"))

from diagnostic import load_pfn_predictor

N_GRID = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 10000]
THETAS = [0.1, 0.3, 0.5, 0.7, 0.9]
PREFIXES_PER_THETA = 20


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=90210)
    args = parser.parse_args()

    dtype = torch.float64
    predictor = load_pfn_predictor(args.checkpoint, dtype=dtype)
    gen = torch.Generator().manual_seed(args.seed)

    n_max = max(N_GRID)
    # Prefix columns: PREFIXES_PER_THETA independent Bernoulli(theta) sequences
    # per theta, shared across all n so the profile is a refinement over n.
    cols = []
    for theta in THETAS:
        cols.append(
            torch.bernoulli(
                torch.full((n_max + 1, PREFIXES_PER_THETA), theta, dtype=dtype),
                generator=gen,
            )
        )
    y = torch.cat(cols, dim=1)  # [n_max+1, R]

    print(f"{'n':>6} {'median |g_n - oracle|':>22} {'max |g_n - oracle|':>20}")
    rows = []
    for n in N_GRID:
        with torch.no_grad():
            g = predictor.predict(y, single_eval_pos=n)  # [R]
        S = y[:n].sum(dim=0)
        oracle = (1.0 + S) / (n + 2.0)
        err = (g.to(dtype) - oracle).abs()
        rows.append((n, err.median().item(), err.max().item()))
        print(f"{n:>6} {rows[-1][1]:>22.5f} {rows[-1][2]:>20.5f}")

    out = Path(__file__).parent / "extrapolation_profile.txt"
    with open(out, "w") as f:
        f.write("n\tmedian_abs_err\tmax_abs_err\n")
        for n, med, mx in rows:
            f.write(f"{n}\t{med:.6f}\t{mx:.6f}\n")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
