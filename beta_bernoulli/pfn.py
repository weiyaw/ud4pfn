"""Minimal PFN (Prior-fitted Network) for the Beta-Bernoulli testbed.

Follows the automl/PFNs architecture (Mueller et al., ICLR 2022):
- TransformerEncoder stack, no positional encoding (permutation-equivariant).
- Training tokens combine x and y via additive fusion: token = x_enc(x) + y_enc(y).
- Query tokens use only x_enc(x); their y is masked out.
- Attention mask: training tokens attend among themselves; each query token
  attends to all training tokens plus itself (efficient eval masking).

For Beta-Bernoulli we have no covariates, so x is a dummy constant and the
transformer is effectively an exchangeable binary-sequence model.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class PFN(nn.Module):
    def __init__(
        self,
        x_dim: int = 1,
        d_model: int = 64,
        nhead: int = 4,
        nlayers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.x_encoder = nn.Linear(x_dim, d_model)
        self.y_encoder = nn.Linear(1, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=False,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

        self.head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, 1),
        )

        # Initialize attention out_proj and FFN linear2 near zero so the net
        # starts near identity, per the PFN paper's recommendation.
        for layer in self.transformer.layers:
            nn.init.zeros_(layer.self_attn.out_proj.weight)
            nn.init.zeros_(layer.self_attn.out_proj.bias)
            nn.init.zeros_(layer.linear2.weight)
            nn.init.zeros_(layer.linear2.bias)

    @staticmethod
    def _build_mask(seq_len: int, single_eval_pos: int, device: torch.device) -> Tensor:
        """Attention mask shape [seq, seq], float with 0=keep, -inf=block.

        Training tokens (positions 0..M-1) see training tokens only.
        Query tokens (positions M..seq-1) see training tokens + themselves.
        """
        M = single_eval_pos
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        # Training block: train tokens see all other train tokens.
        mask[:M, :M] = 0.0
        # Query block: each query token sees all training tokens...
        mask[M:, :M] = 0.0
        # ...plus only itself among the query tokens.
        idx = torch.arange(M, seq_len, device=device)
        mask[idx, idx] = 0.0
        return mask

    def forward(self, x: Tensor, y: Tensor, single_eval_pos: int) -> Tensor:
        """Run the PFN and return query-token logits.

        Args:
            x: covariates, shape [seq, batch, x_dim].
            y: targets, shape [seq, batch]. Values at positions >= single_eval_pos
               are ignored.
            single_eval_pos: number of training tokens M (>= 1).

        Returns:
            logits at query positions, shape [seq - M, batch, 1].
        """
        seq_len, batch = y.shape
        M = int(single_eval_pos)
        assert 1 <= M < seq_len

        x_src = self.x_encoder(x)  # [seq, batch, d_model]
        y_src = self.y_encoder(y.unsqueeze(-1).to(self.y_encoder.weight.dtype))  # [seq, batch, d_model]

        tokens = x_src.clone()
        tokens[:M] = tokens[:M] + y_src[:M]

        mask = self._build_mask(seq_len, M, x.device)
        out = self.transformer(tokens, mask=mask)  # [seq, batch, d_model]

        logits = self.head(out[M:])  # [seq - M, batch, 1]
        return logits
