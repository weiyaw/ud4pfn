# Beta-Bernoulli predictive-CLT diagnostic testbed

A small meta-trained PFN on Beta-Bernoulli binary sequences, plus the exact
Beta-Bernoulli Bayes posterior predictive distribution (PPD) as a
precision-floor reference. The testbed evaluates the predictive-CLT
sufficient conditions (C1)--(C4) and (R), and the quasi-martingale
condition (Q$_\gamma$), against a controlled, analytically-tractable predictor.

## Source files

| file | purpose |
|------|---------|
| `pfn.py` | minimal `TransformerEncoder`-based PFN (Mueller et al., 2022), no positional encoding |
| `data.py` | Beta-Bernoulli sampler: Gamma hyperprior on $(\alpha,\beta)$, iid Bernoulli$(\theta)$ with $\theta\sim\text{Beta}(\alpha,\beta)$ |
| `train.py` | meta-training loop (AdamW + cosine, BCE on query tokens) |
| `diagnostic.py` | computes per-rollout $b_k$, $\Delta_k$, $g_{k-1}$. `--mode pfn` or `--mode oracle` |
| `plot.py` | two plot families from the same `b` tensor: `--conditions signed` and `--conditions qm` (defaults to both) |

## Notation

For a rollout $Y_{1:N}\in\{0,1\}^N$, let $g_k$ be the predictor's
one-step-ahead estimate of $\Pr(Y_{k+1}{=}1\mid Y_{1:k})$.
For the Bayes PPD, $g_k = (\alpha + \sum_{i=1}^k Y_i)/(\alpha+\beta+k)$;
for the PFN, $g_k$ is read out from a forward pass.

The conditional drift $b_k := \mathbb{E}[\Delta_k \mid Y_{1:k-1}]$ is computed
exactly via a two-point average (three forward passes per $k$):

$$
b_k = g_{k-1}\cdot g_k(Y_{1:k-1}, 1) + (1-g_{k-1})\cdot g_k(Y_{1:k-1}, 0) - g_{k-1}.
$$

## Reproduce

```bash
# train
.venv/bin/python train.py --seq-len 1024 --steps 600 --batch-size 16 \
    --d-model 64 --nhead 4 --nlayers 2 --dim-feedforward 128 \
    --warmup-steps 50 --out checkpoints/seqlen1024_training600.pt
.venv/bin/python train.py --seq-len 1024 --steps 50000 --batch-size 16 \
    --d-model 64 --nhead 4 --nlayers 2 --dim-feedforward 128 \
    --warmup-steps 1000 --out checkpoints/seqlen1024_training50k.pt

# diagnose (PFN-induced rollouts, float32; oracle in float64)
.venv/bin/python diagnostic.py --mode pfn \
    --checkpoint checkpoints/seqlen1024_training600.pt \
    --out checkpoints/diag_seqlen1024_training600.pt \
    --num-rollouts 16 --seq-len 10001 --k-min 2 --k-max 10000 \
    --dtype float32 --seed 0
.venv/bin/python diagnostic.py --mode pfn \
    --checkpoint checkpoints/seqlen1024_training50k.pt \
    --out checkpoints/diag_seqlen1024_training50k.pt \
    --num-rollouts 16 --seq-len 10001 --k-min 2 --k-max 10000 \
    --dtype float32 --seed 0
.venv/bin/python diagnostic.py --mode oracle \
    --out checkpoints/diag_oracle.pt \
    --num-rollouts 16 --seq-len 10001 --k-min 2 --k-max 10000 \
    --dtype float64 --seed 1

# plot
.venv/bin/python plot.py --diag checkpoints/diag_seqlen1024_training600.pt \
    --out-dir checkpoints --stem diag_600 --fit-n-min 10 --fit-n-max 512
.venv/bin/python plot.py --diag checkpoints/diag_seqlen1024_training50k.pt \
    --out-dir checkpoints --stem diag_50k --fit-n-min 10 --fit-n-max 512
.venv/bin/python plot.py --diag checkpoints/diag_oracle.pt \
    --out-dir checkpoints --stem diag_oracle --fit-n-min 10 --fit-n-max 512
```

If you have diagnostic data at a larger `k_max` (e.g., from a long run), use
`--plot-k-max 1024` to truncate to the in-distribution range before plotting.

## Environment

Python 3.14, torch 2.11, numpy 2.4, matplotlib 3.10. Install:
```bash
python3 -m venv .venv && .venv/bin/python -m pip install torch numpy matplotlib
```
