"""Beta-Bernoulli batch sampler.

Each batch element is an i.i.d. Bernoulli(theta) sequence of length seq_len,
with theta drawn from Beta(alpha, beta). (alpha, beta) are FIXED constants
across all tasks and all predictors in Appendix C -- there is no hyperprior.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.distributions import Beta, Bernoulli


@dataclass
class Batch:
    x: Tensor          # [seq, batch, 1]  (dummy zeros)
    y: Tensor          # [seq, batch]     in {0, 1}
    single_eval_pos: int
    alpha: Tensor      # [batch]  Beta prior concentration (fixed, broadcast)
    beta: Tensor       # [batch]  Beta prior concentration (fixed, broadcast)
    theta: Tensor      # [batch]  draw from Beta(alpha, beta)


def sample_batch(
    seq_len: int,
    batch_size: int,
    alpha: float = 1.0,
    beta: float = 1.0,
    single_eval_pos: int | None = None,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
) -> Batch:
    """Draw a meta-training batch.

    theta ~ Beta(alpha, beta), Y_1..Y_seq iid Bernoulli(theta).
    (alpha, beta) are fixed scalars; every batch element uses the same pair.
    """
    device = device or torch.device("cpu")
    alpha_t = torch.full((batch_size,), float(alpha), device=device)
    beta_t = torch.full((batch_size,), float(beta), device=device)
    theta = Beta(alpha_t, beta_t).sample()  # [batch]

    probs = theta.unsqueeze(0).expand(seq_len, -1)  # [seq, batch]
    y = Bernoulli(probs=probs).sample()             # float tensor of 0/1
    x = torch.zeros(seq_len, batch_size, 1, device=device)

    if single_eval_pos is None:
        single_eval_pos = int(torch.randint(1, seq_len, (1,), generator=generator).item())

    return Batch(
        x=x, y=y, single_eval_pos=single_eval_pos,
        alpha=alpha_t, beta=beta_t, theta=theta,
    )
