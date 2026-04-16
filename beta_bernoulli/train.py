"""Meta-train a small PFN on Beta-Bernoulli sequences.

Usage (from repo root):
    beta_bernoulli/.venv/bin/python beta_bernoulli/train.py \
        --seq-len 1024 --steps 600 --batch-size 32 --d-model 64

A checkpoint is written to beta_bernoulli/checkpoints/pfn.pt.
"""
from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from data import sample_batch
from pfn import PFN


def cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return LambdaLR(optimizer, lr_lambda)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--nlayers", type=int, default=2)
    p.add_argument("--dim-feedforward", type=int, default=128)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="beta_bernoulli/checkpoints/pfn.pt")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    model = PFN(
        x_dim=1,
        d_model=args.d_model,
        nhead=args.nhead,
        nlayers=args.nlayers,
        dim_feedforward=args.dim_feedforward,
        dropout=0.0,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"[model] PFN params={num_params:,} d_model={args.d_model} layers={args.nlayers}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = cosine_with_warmup(optimizer, args.warmup_steps, args.steps)
    loss_fn = nn.BCEWithLogitsLoss(reduction="mean")

    t0 = time.time()
    model.train()
    running = 0.0
    for step in range(1, args.steps + 1):
        batch = sample_batch(
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            device=device,
        )

        logits = model(batch.x, batch.y, batch.single_eval_pos)  # [seq-M, B, 1]
        targets = batch.y[batch.single_eval_pos:].unsqueeze(-1)  # [seq-M, B, 1]
        loss = loss_fn(logits, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        running += loss.item()

        if step % args.log_every == 0 or step == 1:
            avg = running / (args.log_every if step != 1 else 1)
            elapsed = time.time() - t0
            print(
                f"[step {step:4d}/{args.steps}] loss={avg:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e} elapsed={elapsed:.1f}s"
            )
            running = 0.0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "args": vars(args),
        },
        out,
    )
    print(f"[done] total={time.time()-t0:.1f}s checkpoint={out}")


if __name__ == "__main__":
    main()
