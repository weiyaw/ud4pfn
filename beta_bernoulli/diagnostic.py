"""Compute predictive-CLT diagnostic quantities on Beta-Bernoulli data.

At each probe step k along a rollout we need three one-step-ahead predictive
probabilities from the model:

    F_{k-1}       = P(Y_k=1 | Y_{1:k-1})
    F_k | Y_k=0  = P(Y_{k+1}=1 | Y_{1:k-1}, Y_k=0)
    F_k | Y_k=1  = P(Y_{k+1}=1 | Y_{1:k-1}, Y_k=1)

From these we compute the two-point conditional expectation

    b_k = F_{k-1} * F_k(Y_{1:k-1},1) + (1-F_{k-1}) * F_k(Y_{1:k-1},0) - F_{k-1}
        = E[Delta_k | Y_{1:k-1}]   (exactly, because Y_k is binary).

We also record the realized update Delta_k = F_k(Y_{1:k-1}, Y_k) - F_{k-1}.

The "predictor" abstraction means this loop is identical for a trained PFN
and for the exact Beta-Bernoulli Bayes PPD. The only difference is which
Predictor subclass is plugged in. Compare:

    pfn    : predictor = PFNPredictor(trained_pfn)
    oracle : predictor = BayesOraclePredictor(alpha, beta)

Usage (from the repo root):

    # PFN version
    beta_bernoulli/.venv/bin/python beta_bernoulli/diagnostic.py \
        --mode pfn --checkpoint beta_bernoulli/checkpoints/pfn.pt \
        --out beta_bernoulli/checkpoints/diag.pt \
        --num-rollouts 32 --seq-len 1024 --k-min 2 --k-max 600

    # Oracle version (exact Bayes PPD; dtype for precision floor studies)
    beta_bernoulli/.venv/bin/python beta_bernoulli/diagnostic.py \
        --mode oracle --dtype float64 \
        --out beta_bernoulli/checkpoints/diag_oracle_f64.pt \
        --num-rollouts 32 --seq-len 20001 --k-min 2 --k-max 20000
"""
from __future__ import annotations

import argparse
from abc import ABC, abstractmethod

import torch
from torch import Tensor

from data import sample_batch
from pfn import PFN


# ---------------------------------------------------------------------------
# Predictor abstraction
# ---------------------------------------------------------------------------


class Predictor(ABC):
    """One-step-ahead predictive probability for Y_{single_eval_pos+1} given
    the first `single_eval_pos` tokens of each rollout in `y` ([seq_len, R])."""

    @abstractmethod
    def predict(self, y: Tensor, single_eval_pos: int) -> Tensor:
        ...


class PFNPredictor(Predictor):
    def __init__(self, model: PFN) -> None:
        self.model = model

    @torch.no_grad()
    def predict(self, y: Tensor, single_eval_pos: int) -> Tensor:
        # Input truncation: the prediction at single_eval_pos only needs the
        # first single_eval_pos training tokens plus one query token, so we
        # pass a y of length single_eval_pos + 1 regardless of y's full length.
        # Cost per forward is then O((single_eval_pos + 1)^2) instead of
        # O(seq_len^2), which is the main enabler for large k_max diagnostics.
        needed = single_eval_pos + 1
        full_len, R = y.shape
        if needed < full_len:
            y_trunc = y[:needed]
        else:
            y_trunc = y
        x = torch.zeros(y_trunc.shape[0], R, 1, device=y.device, dtype=y.dtype)
        logits = self.model(x, y_trunc, single_eval_pos=single_eval_pos)
        return torch.sigmoid(logits[0, :, 0])


class BayesOraclePredictor(Predictor):
    """Exact Beta-Bernoulli Bayes PPD: F_k = (alpha + S_k) / (alpha + beta + k)."""

    def __init__(self, alpha: Tensor, beta: Tensor) -> None:
        self.alpha = alpha
        self.beta = beta

    @torch.no_grad()
    def predict(self, y: Tensor, single_eval_pos: int) -> Tensor:
        if single_eval_pos == 0:
            return self.alpha / (self.alpha + self.beta)
        S = y[:single_eval_pos].sum(dim=0)
        return (self.alpha + S) / (self.alpha + self.beta + single_eval_pos)


# ---------------------------------------------------------------------------
# Core diagnostic loop (predictor-agnostic)
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_b_and_delta(
    predictor: Predictor,
    y: Tensor,        # [seq_len, R]
    k_min: int,
    k_max: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """For each probe k in [k_min, k_max] compute b_k, Delta_k, F_{k-1}.

    b_k     = F_{k-1} * F_k(Y_{1:k-1},1) + (1-F_{k-1}) * F_k(Y_{1:k-1},0) - F_{k-1}
    Delta_k = F_k(Y_{1:k-1}, Y_k) - F_{k-1}

    Returns three tensors of shape [K, R] where K = k_max - k_min + 1.
    """
    seq_len, R = y.shape
    assert k_min >= 1 and k_max < seq_len

    import time
    K = k_max - k_min + 1
    b = torch.empty(K, R, dtype=y.dtype, device=y.device)
    delta = torch.empty(K, R, dtype=y.dtype, device=y.device)
    f_prev = torch.empty(K, R, dtype=y.dtype, device=y.device)
    print(f"[probe] k in [{k_min}, {k_max}], R={R}, dtype={y.dtype}", flush=True)
    t0 = time.time()

    for i, k in enumerate(range(k_min, k_max + 1)):
        # F_{k-1}: first query token is position k-1 => P(Y_k=1 | Y_{1:k-1}).
        f_km1 = predictor.predict(y, single_eval_pos=k - 1)

        # F_k(Y_{1:k-1}, 0) and F_k(Y_{1:k-1}, 1): override y[k-1] and
        # query at M=k (first query token is position k, i.e. Y_{k+1}).
        y0 = y.clone()
        y1 = y.clone()
        y0[k - 1] = 0.0
        y1[k - 1] = 1.0
        f_k_y0 = predictor.predict(y0, single_eval_pos=k)
        f_k_y1 = predictor.predict(y1, single_eval_pos=k)

        y_k = y[k - 1]
        f_k_actual = torch.where(y_k > 0.5, f_k_y1, f_k_y0)

        b[i] = f_km1 * f_k_y1 + (1.0 - f_km1) * f_k_y0 - f_km1
        delta[i] = f_k_actual - f_km1
        f_prev[i] = f_km1

        if (k % 200 == 0) or (k == k_max):
            elapsed = time.time() - t0
            print(
                f"[probe] k={k}/{k_max}  elapsed={elapsed:.1f}s",
                flush=True,
            )

    return b, delta, f_prev


# ---------------------------------------------------------------------------
# Predictor factories
# ---------------------------------------------------------------------------


def load_pfn_predictor(checkpoint_path: str, dtype: torch.dtype) -> PFNPredictor:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = ckpt["args"]
    model = PFN(
        x_dim=1,
        d_model=args["d_model"],
        nhead=args["nhead"],
        nlayers=args["nlayers"],
        dim_feedforward=args["dim_feedforward"],
        dropout=0.0,
    ).to(dtype)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return PFNPredictor(model)


@torch.no_grad()
def sample_pfn_induced_rollouts(
    predictor: PFNPredictor,
    seq_len: int,
    num_rollouts: int,
    dtype: torch.dtype,
    generator: torch.Generator,
    log_every: int = 200,
) -> Tensor:
    """Generate rollouts by iterating the trained PFN's own predictive rule.

    Y_1 is seeded uniformly (Bernoulli(0.5)) because the PFN's forward pass
    requires at least one training token (single_eval_pos >= 1). For n >= 2,
    Y_n is drawn from Bernoulli(g_{n-1}(Y_{1:n-1})) where g_{n-1} is the PFN's
    one-step-ahead probability given the prefix Y_{1:n-1}.

    Returns a tensor of shape [seq_len, num_rollouts] in the given dtype.
    """
    import time
    R = num_rollouts
    device = torch.device("cpu")
    y = torch.zeros(seq_len, R, dtype=dtype, device=device)
    # Seed Y_1 ~ Bernoulli(0.5) for each rollout.
    y[0] = torch.bernoulli(
        torch.full((R,), 0.5, dtype=dtype, device=device),
        generator=generator,
    )
    print(f"[rollout] seq_len={seq_len} R={R} dtype={dtype}", flush=True)
    t0 = time.time()
    # Iterate the PFN's predictive rule for n >= 2.
    for n in range(2, seq_len + 1):
        # predictor.predict internally handles input truncation: it passes the
        # first (single_eval_pos + 1) tokens of y. Here that is Y_{1:n-1}
        # as training tokens plus one query token at position n-1.
        g = predictor.predict(y, single_eval_pos=n - 1)  # [R]
        sampled = torch.bernoulli(g.to(dtype), generator=generator)
        y[n - 1] = sampled
        if n % log_every == 0 or n == seq_len:
            elapsed = time.time() - t0
            print(
                f"[rollout] n={n}/{seq_len}  elapsed={elapsed:.1f}s",
                flush=True,
            )
    return y


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["pfn", "oracle"], required=True)
    p.add_argument("--checkpoint", type=str,
                   default="beta_bernoulli/checkpoints/pfn.pt",
                   help="PFN checkpoint (only used in --mode pfn)")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--num-rollouts", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--k-min", type=int, default=2)
    p.add_argument("--k-max", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32",
                   help="float64 lowers the oracle noise floor by ~9 orders of "
                        "magnitude; use it for oracle precision-floor studies. "
                        "For PFN use float32 to match training precision.")
    args = p.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    torch.manual_seed(args.seed)
    torch.set_default_dtype(dtype)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)

    # Build predictor and draw rollouts.
    #
    # --mode pfn:    rollouts are generated by iterating the trained PFN's
    #                own one-step predictive rule. The conditions of
    #                Theorem 1 are pathwise statements under the law of the
    #                sequence, so to test them on the PFN we sample from the
    #                PFN's induced law, not from the underlying
    #                Beta-Bernoulli hyperprior. (alpha, beta, theta) are not
    #                defined in this case.
    #
    # --mode oracle: rollouts are drawn from the Beta-Bernoulli hyperprior,
    #                which IS the law induced by the exact Bayes predictor.
    #                Here b_k = 0 analytically, so the oracle run traces the
    #                numerical precision floor of the diagnostic.
    alpha: Tensor | None
    beta: Tensor | None
    theta: Tensor | None
    if args.mode == "pfn":
        predictor = load_pfn_predictor(args.checkpoint, dtype=dtype)
        y = sample_pfn_induced_rollouts(
            predictor=predictor,
            seq_len=args.seq_len,
            num_rollouts=args.num_rollouts,
            dtype=dtype,
            generator=gen,
        )
        alpha = None
        beta = None
        theta = None
    else:
        batch = sample_batch(seq_len=args.seq_len, batch_size=args.num_rollouts)
        y = batch.y.to(dtype)
        alpha = batch.alpha.to(dtype)
        beta = batch.beta.to(dtype)
        theta = batch.theta
        predictor = BayesOraclePredictor(alpha=alpha, beta=beta)

    # Run the probe loop.
    k_max = args.k_max if args.k_max is not None else args.seq_len - 1
    b, delta, f_prev = compute_b_and_delta(
        predictor=predictor, y=y, k_min=args.k_min, k_max=k_max,
    )

    torch.save(
        {
            "b": b.cpu(),
            "delta": delta.cpu(),
            "f_prev": f_prev.cpu(),
            "alpha": None if alpha is None else alpha.cpu(),
            "beta": None if beta is None else beta.cpu(),
            "theta": None if theta is None else theta.cpu(),
            "y": y.cpu(),
            "k_min": args.k_min,
            "k_max": k_max,
            "num_rollouts": args.num_rollouts,
            "seq_len": args.seq_len,
            "mode": args.mode,
            "dtype": args.dtype,
            "checkpoint": args.checkpoint if args.mode == "pfn" else "oracle-bayes-ppd",
        },
        args.out,
    )
    print(f"[done] mode={args.mode} dtype={args.dtype}  saved to {args.out}")
    print(f"  b shape {tuple(b.shape)}")
    print(f"  |b|     max={b.abs().max().item():.3e}  mean={b.abs().mean().item():.3e}")
    print(f"  |delta| max={delta.abs().max().item():.3e}  mean={delta.abs().mean().item():.3e}")


if __name__ == "__main__":
    main()
