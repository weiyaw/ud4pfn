"""Beta-Bernoulli batch sampler.

Each batch element is an i.i.d. Bernoulli(theta) sequence of length seq_len,
with theta drawn from Beta(alpha, beta) and (alpha, beta) drawn from a
Gamma hyperprior.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.distributions import Beta, Gamma, Bernoulli


@dataclass
class Batch:
    x: Tensor          # [seq, batch, 1]  (dummy zeros)
    y: Tensor          # [seq, batch]     in {0, 1}
    single_eval_pos: int
    alpha: Tensor      # [batch]  Beta prior concentration (latent)
    beta: Tensor       # [batch]  Beta prior concentration (latent)
    theta: Tensor      # [batch]  draw from Beta(alpha, beta)


def sample_batch(
    seq_len: int,
    batch_size: int,
    hyperprior_concentration: float = 2.0,
    hyperprior_rate: float = 2.0,
    single_eval_pos: int | None = None,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
) -> Batch:
    """Draw a meta-training batch.

    (alpha, beta) ~ Gamma(conc, rate), theta ~ Beta(alpha, beta),
    Y_1..Y_seq iid Bernoulli(theta).
    """
    device = device or torch.device("cpu")
    g = Gamma(
        torch.full((batch_size,), hyperprior_concentration, device=device),
        torch.full((batch_size,), hyperprior_rate, device=device),
    )
    alpha = g.sample() + 1e-3
    beta = g.sample() + 1e-3
    theta = Beta(alpha, beta).sample()  # [batch]

    probs = theta.unsqueeze(0).expand(seq_len, -1)  # [seq, batch]
    y = Bernoulli(probs=probs).sample()             # float tensor of 0/1
    x = torch.zeros(seq_len, batch_size, 1, device=device)

    if single_eval_pos is None:
        single_eval_pos = int(torch.randint(1, seq_len, (1,), generator=generator).item())

    return Batch(
        x=x, y=y, single_eval_pos=single_eval_pos,
        alpha=alpha, beta=beta, theta=theta,
    )
