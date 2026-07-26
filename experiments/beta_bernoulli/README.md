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
| `data.py` | Beta-Bernoulli sampler: $\theta\sim\text{Beta}(\alpha,\beta)$ with FIXED $\alpha=\beta=1$ (i.e. Beta$(1,1)$, uniform) across all tasks, then iid Bernoulli$(\theta)$. There is no hyperprior on $(\alpha,\beta)$. |
| `train.py` | meta-training loop (AdamW + cosine, BCE on query tokens) |
| `diagnostic.py` | computes per-rollout $b_k$, $\Delta_k$, $g_{k-1}$. `--mode pfn`, `--mode oracle`, or `--mode corrupt` |
| `plot.py` | two plot families from the same `b` tensor: `--conditions signed` and `--conditions qm` (defaults to both) |
| `plot_variance_only.py` | variance-only diagnostic panels for the BFT and corrupted-oracle sweeps |
| `plot_corrupt_sweep.py` | per-rollout (C2) diagnostics for the corrupted-oracle noise/decay sweeps |
| `plot_intro_fig.py` | the two-panel intro figure (CLT Gaussian vs Beta posterior; $V_n$ schematic) |

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

All commands are run from inside `experiments/beta_bernoulli/`. Each
`diagnostic.py` run
takes roughly 25 min on an H100 (the pfn/oracle/corrupt rollouts dominate);
plotting from the cached `.pt` files is a few seconds on CPU. The trained
checkpoints and diagnostic tensors are already shipped under `checkpoints/`,
so plotting can be reproduced without re-running training or diagnostics.

### 1. Train (batch size 16, matching the shipped checkpoints)

```bash
.venv/bin/python train.py --seq-len 1024 --steps 600 --batch-size 16 \
    --d-model 64 --nhead 4 --nlayers 2 --dim-feedforward 128 \
    --warmup-steps 50 --out checkpoints/seqlen1024_training600.pt
.venv/bin/python train.py --seq-len 1024 --steps 50000 --batch-size 16 \
    --d-model 64 --nhead 4 --nlayers 2 --dim-feedforward 128 \
    --warmup-steps 1000 --out checkpoints/seqlen1024_training50k.pt
```

### 2. Diagnose (PFN-induced rollouts float32; oracle in float64)

```bash
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
```

### 3. Corrupted-oracle sweeps (8 settings)

The corrupted-oracle diagnostics feed `plot_corrupt_sweep.py` and the
`bb-variance-only-corrupt`/`bb-corrupt-*` figures. Noise sweep perturbs the
oracle logit by iid Gaussian noise of amplitude $\varepsilon$; decay sweep uses
an envelope $\varepsilon\,n^{-p}$ with fixed $\varepsilon=0.5$. Outputs land in
`checkpoints/corrupt_sweeps/`.

```bash
mkdir -p checkpoints/corrupt_sweeps
# noise: eps in {1e-3, 1e-2, 1e-1}
for eps in 1e-3 1e-2 1e-1; do
    .venv/bin/python diagnostic.py --mode corrupt \
        --corruption-mode noise --epsilon $eps \
        --out checkpoints/corrupt_sweeps/diag_noise_eps${eps}.pt \
        --num-rollouts 16 --seq-len 10001 --k-min 2 --k-max 10000 \
        --dtype float64 --seed 1
done
# decay: eps=0.5, p in {0.25, 0.5, 1.0, 1.5, 2.0}
for p in 0.25 0.5 1.0 1.5 2.0; do
    .venv/bin/python diagnostic.py --mode corrupt \
        --corruption-mode decay --epsilon 0.5 --corrupt-p $p \
        --out checkpoints/corrupt_sweeps/diag_decay_p${p}.pt \
        --num-rollouts 16 --seq-len 10001 --k-min 2 --k-max 10000 \
        --dtype float64 --seed 1
done
```

### 4. Plot the paper figures

The `plot.py` `--stem S` option writes `S-signed.pdf` and `S-qm.pdf`. The paper
uses the `bb-diag-*` and `bb-corrupt-*` stems. `--fit-n-min 10 --fit-n-max 2000`
matches `plot.py`'s defaults and the paper's power-law fit window.

```bash
# signed + QM condition panels (bb-diag-{600,50k,oracle}-{signed,qm}.pdf)
.venv/bin/python plot.py --diag checkpoints/diag_seqlen1024_training600.pt \
    --out-dir checkpoints --stem bb-diag-600 --fit-n-min 10 --fit-n-max 2000
.venv/bin/python plot.py --diag checkpoints/diag_seqlen1024_training50k.pt \
    --out-dir checkpoints --stem bb-diag-50k --fit-n-min 10 --fit-n-max 2000
.venv/bin/python plot.py --diag checkpoints/diag_oracle.pt \
    --out-dir checkpoints --stem bb-diag-oracle --fit-n-min 10 --fit-n-max 2000

# corrupted-oracle signed + QM panels (bb-corrupt-*-{signed,qm}.pdf)
.venv/bin/python plot.py --diag checkpoints/corrupt_sweeps/diag_noise_eps1e-3.pt \
    --out-dir checkpoints --stem bb-corrupt-noise1e-3 --fit-n-min 10 --fit-n-max 2000
.venv/bin/python plot.py --diag checkpoints/corrupt_sweeps/diag_noise_eps1e-2.pt \
    --out-dir checkpoints --stem bb-corrupt-noise1e-2 --fit-n-min 10 --fit-n-max 2000
.venv/bin/python plot.py --diag checkpoints/corrupt_sweeps/diag_noise_eps1e-1.pt \
    --out-dir checkpoints --stem bb-corrupt-noise1e-1 --fit-n-min 10 --fit-n-max 2000
.venv/bin/python plot.py --diag checkpoints/corrupt_sweeps/diag_decay_p0.25.pt \
    --out-dir checkpoints --stem bb-corrupt-decay-p025 --fit-n-min 10 --fit-n-max 2000
.venv/bin/python plot.py --diag checkpoints/corrupt_sweeps/diag_decay_p0.5.pt \
    --out-dir checkpoints --stem bb-corrupt-decay-p05 --fit-n-min 10 --fit-n-max 2000
.venv/bin/python plot.py --diag checkpoints/corrupt_sweeps/diag_decay_p1.0.pt \
    --out-dir checkpoints --stem bb-corrupt-decay-p10 --fit-n-min 10 --fit-n-max 2000
.venv/bin/python plot.py --diag checkpoints/corrupt_sweeps/diag_decay_p1.5.pt \
    --out-dir checkpoints --stem bb-corrupt-decay-p15 --fit-n-min 10 --fit-n-max 2000
.venv/bin/python plot.py --diag checkpoints/corrupt_sweeps/diag_decay_p2.0.pt \
    --out-dir checkpoints --stem bb-corrupt-decay-p20 --fit-n-min 10 --fit-n-max 2000

# variance-only panels: bb-variance-only-bfts.pdf and bb-variance-only-corrupt.pdf
.venv/bin/python plot_variance_only.py

# corrupted-oracle (C2) sweep panels:
#   diag_corrupt_{noise,decay}_{c2,bn}.pdf
.venv/bin/python plot_corrupt_sweep.py

# intro figure (panel a: CLT Gaussian vs Beta posterior; panel b: V_n schematic)
.venv/bin/python plot_intro_fig.py \
    --checkpoint checkpoints/seqlen1024_training50k.pt \
    --out checkpoints/beta-bernoulli-intro.pdf
```

All figures are written next to the checkpoints, under `checkpoints/`
(gitignored). `plot_variance_only.py` and `plot_corrupt_sweep.py` are run from
`experiments/beta_bernoulli/` and read/write `checkpoints/` by relative path.

If you have diagnostic data at a larger `k_max` (e.g., from a long run), use
`--plot-k-max 1024` on `plot.py` to truncate to the in-distribution range
before plotting.

## Environment

Python 3.14, torch 2.11, numpy 2.4, matplotlib 3.10. This is a SEPARATE
environment from the top-level repo (which uses TabPFN + JAX). Install:
```bash
python3 -m venv .venv && .venv/bin/python -m pip install torch numpy matplotlib
```
