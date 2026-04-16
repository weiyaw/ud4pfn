# Beta-Bernoulli predictive-CLT diagnostic testbed

A small meta-trained PFN on Beta-Bernoulli binary sequences, plus the exact
Beta-Bernoulli Bayes posterior predictive distribution (PPD) as a
precision-floor reference. The whole testbed exists to run the
predictive-CLT conditions from `sandra_new0804.tex` against a controlled,
analytically-tractable predictor.

## 1. Source files

| file | purpose |
|------|---------|
| `pfn.py` | minimal `TransformerEncoder`-based PFN (Mueller et al., 2022), no positional encoding |
| `data.py` | Beta-Bernoulli sampler: Gamma hyperprior on $(\alpha,\beta)$, iid Bernoulli$(\theta)$ with $\theta\sim\text{Beta}(\alpha,\beta)$ |
| `train.py` | meta-training loop (AdamW + cosine, BCE on query tokens) |
| `diagnostic.py` | computes the per-rollout quantities below. `--mode pfn` or `--mode oracle` |
| `plot.py` | two plot families from the same `b` tensor: `--conditions signed` and `--conditions qm` (defaults to both) |

## 2. Notation

For a rollout $Y_{1:N}\in\{0,1\}^N$ drawn from Beta-Bernoulli, let $g_k$
be the predictor's one-step-ahead estimate of $\Pr(Y_{k+1}{=}1\mid Y_{1:k})$.
For the Bayes PPD, $g_k = (\alpha + \sum_{i=1}^k Y_i)/(\alpha+\beta+k)$;
for the PFN, $g_k$ is read out from a forward pass.

Define

$$
\Delta_k := g_k - g_{k-1},
\qquad
b_k := \mathbb{E}[\Delta_k \mid Y_{1:k-1}].
$$

For binary $Y_k$, $b_k$ is a *two-point* conditional expectation, computable
exactly from three forward passes of the predictor:

$$
b_k \;=\; g_{k-1}\cdot g_k(Y_{1:k-1}, 1) \;+\; (1-g_{k-1})\cdot g_k(Y_{1:k-1}, 0) \;-\; g_{k-1}.
$$

We probe $k\in[k_{\text{min}}, k_{\text{max}}]$ on $R$ independent rollouts; per-rollout
quantities get an $(r)$ superscript.

## 3. The two condition families

### 3a. Signed-tail conditions (`--conditions signed`)

Theorem 1.2 of `sandra_new0804.tex` is stated for a vector of probe
points; here we specialise to a single probe, so each statement below
is a scalar statement in the free rate parameter $\gamma\in(0,1]$. All
four are path-wise $\mathbb{P}$-a.s.; the `signed` plot shows every
rollout individually with no cross-rollout averaging.

To make the conditions computable per-rollout, define the running signed
and squared partial sums

$$
P_n^{(r)} = \sum_{k=k_{\text{min}}}^n b_k^{(r)},
\qquad
Q_n^{(r)} = \sum_{k=k_{\text{min}}}^n (b_k^{(r)})^2,
$$

and their infinite-limit counterparts
$L^{(r)} = \lim_{n\to\infty}P_n^{(r)}$ and $Q_\infty^{(r)} = \lim_{n\to\infty}Q_n^{(r)}$
(in the finite run we substitute $P_{k_{\text{max}}}^{(r)}$ and
$Q_{k_{\text{max}}}^{(r)}$ for these, see the finite-$k_{\text{max}}$ caveat below). By telescoping,
$\sum_{k\ge n}b_k^{(r)} = L^{(r)} - P_{n-1}^{(r)}$ and
$\sum_{k\ge n}(b_k^{(r)})^2 = Q_\infty^{(r)} - Q_{n-1}^{(r)}$.

The four conditions, as $n\to\infty$:

- **Eq 3:** $\sum_{k\ge n}b_k\to 0$
- **(1A) first bullet:** $n^{\gamma/2}\sum_{k>n}b_k\to 0$
- **(1A) second bullet, $k{=}n$ version:** $n^{\gamma/2}\lvert b_n\rvert\to 0$
- **(1B):** $n^\gamma\sum_{k\ge n}b_k^2\to 0$

Sandra's original (1A)-second has $\sup_{k\ge n}\lvert\cdot\rvert$ in
place of $\lvert b_n\rvert$; we test the pointwise $k{=}n$ version.

#### Eq 3: $\sum_{k\ge n} b_k \to 0$

- *Plotted*: $P_n^{(r)}$ on linear $y$.
- *Visually passing*: each rollout's trajectory plateaus at some finite limit $L^{(r)}$.
- *Verdict*: **inconclusive for both PFNs** — sign-structure dependent (see note at the end of §3a).

| | observed |
|---|---|
| `oracle` | float64 random-walk floor, trajectories in $\pm 1.5\times 10^{-15}$ at $n=1500$ |
| `seqlen1024_training600` | trajectories drift visibly in $\pm 10^{-2}$ throughout the probed window — never plateau |
| `seqlen1024_training50k` | trajectories tighter ($\pm 5\times 10^{-3}$) and several curves bend visibly toward a plateau in the second half of the window, but still not clearly flat within $n\le 600$ |

#### (1A) first bullet: $n^{\gamma/2}\sum_{k>n}b_k \to 0$

- *Plotted*: $n^{\gamma/2}\,\lvert L^{(r)} - P_n^{(r)}\rvert$ on log–log (rate-weighted residual gap to the estimated limit).
- *Visually passing*: curve decays monotonically to zero with $n$ (ignore the right edge, see the finite-$k_{\text{max}}$ caveat below).
- *Verdict*: **inconclusive for both PFNs** — sign-structure dependent.

| | observed (within $n \ll k_{\text{max}}$) |
|---|---|
| `oracle` | noise floor $\sim\epsilon\, n^{(\gamma+1)/2}$ |
| `seqlen1024_training600` | roughly flat across all tested $\gamma$ — not cleanly vanishing, not cleanly growing |
| `seqlen1024_training50k` | visibly decaying across all tested $\gamma$, steeper than the 600-step PFN, but still not a clean power-law decay |

#### (1A) second bullet, $k{=}n$ version: $n^{\gamma/2}\lvert b_n\rvert \to 0$

- *Plotted*: $n^{\gamma/2}\,\lvert b_n^{(r)}\rvert$ on log–log (pointwise in $n$, no tail sum, not affected by finite-$k_{\text{max}}$).
- *Visually passing*: curve decays monotonically to zero with $n$.
- *Verdict*: analytic threshold $\gamma<2\beta$. At the 600-step $\widehat\beta\approx 0.971$ the threshold is $\gamma<1.94$; at the 50k-step local $\widehat\beta\in[1.03, 1.19]$ the threshold is $\gamma\in[2.06, 2.38]$. **All tested $\gamma\in\{0.25,0.5,0.75,1.0\}$ pass for every PFN.**

(Sandra's original (1A)-second has $\sup_{k\ge n}\lvert\cdot\rvert$ in place of $\lvert b_n\rvert$; we test the pointwise $k{=}n$ version.)

| | observed |
|---|---|
| `oracle` | noise floor $\sim\epsilon\, n^{\gamma/2}$ |
| `seqlen1024_training600` | clean monotone decay at every tested $\gamma$ |
| `seqlen1024_training50k` | clean monotone decay at every tested $\gamma$, steeper slopes than the 600-step PFN |

#### (1B): $n^\gamma \sum_{k\ge n} b_k^2 \to 0$

- *Plotted*: $n^\gamma\,(Q_\infty^{(r)} - Q_{n-1}^{(r)})$ on log–log (rate-weighted residual of the running sum of squares).
- *Visually passing*: curve decays monotonically to zero with $n$ (ignore right edge, finite-$k_{\text{max}}$ caveat).
- *Verdict*: analytic threshold $\gamma<2\beta-1$. At the 600-step $\widehat\beta\approx 0.971$ the threshold is $\gamma<0.94$, so $\gamma\in\{0.25,0.5,0.75\}$ pass and $\gamma{=}1$ is borderline (just $0.06$ above). At the 50k-step local $\widehat\beta\in[1.03, 1.19]$ the threshold is $\gamma\in[1.06, 1.38]$, so $\gamma{=}1$ passes (marginally at $\widehat\beta=1.03$, comfortably at $\widehat\beta=1.19$).

| | observed (within $n \ll k_{\text{max}}$) |
|---|---|
| `oracle` | noise floor $\sim\epsilon^2\, n^\gamma\, k_{\text{max}}$ |
| `seqlen1024_training600` | clean decay at $\gamma\in\{0.25,0.5,0.75\}$; roughly flat at $\gamma{=}1$ |
| `seqlen1024_training50k` | clean decay at every tested $\gamma$ including $\gamma{=}1$ |

**Sign-structure dependence (Eq 3 and (1A)-first).** The decay of the
*signed* tail sum $\sum_{k\ge n}b_k$ is not determined by the decay
exponent $\beta$ of $\lvert b_k\rvert$ alone: it also depends on how the
signs of the $b_k$ are arranged along the sequence. Under a pure
power-law $\lvert b_k\rvert\sim k^{-\beta}$ with $\beta < 1$, the
absolute sum $\sum_{k\ge n}\lvert b_k\rvert$ diverges; but the *signed*
sum can still converge rapidly through cancellation between positive
and negative $b_k$ — the textbook example is
$\sum_k (-1)^k/k=-\ln 2$, where $\lvert b_k\rvert = 1/k$ with $\beta=1$
is *not* absolutely summable, yet the signed sum converges. The signed
decay rate can be anywhere from the absolute upper bound $n^{1-\beta}$
down to 0 depending on the actual sign pattern, so from $\widehat\beta$
alone we cannot say whether Eq 3 and (1A)-first pass; we can only
observe the signed partial sum $P_n^{(r)}$ empirically.

**Finite-$k_{\text{max}}$ caveat.** In the finite run we substitute
$\widehat L^{(r)} := P_{k_{\text{max}}}^{(r)}$ and
$\widehat Q_\infty^{(r)} := Q_{k_{\text{max}}}^{(r)}$ for the true
infinite limits. As $n\to k_{\text{max}}$, the (1A)-first and (1B)
rate-weighted residual panels collapse to zero *by construction* — the
residual at the endpoint is $P_{k_{\text{max}}}^{(r)} - P_{k_{\text{max}}}^{(r)}=0$,
independent of whether the condition holds. Trust those two panels
only for $n\ll k_{\text{max}}$; the plot draws a red dashed line at
$n=k_{\text{max}}/2$ as a conservative cut-off. The Eq-3 panel
($P_n^{(r)}$ directly) and the (1A)-second panel ($n^{\gamma/2}\lvert b_n^{(r)}\rvert$)
are *not* affected. The QM conditions of §3b use no finite-limit
substitution and are unaffected.

### 3b. Strict quasi-martingale conditions (`--conditions qm`)

These are the ICML-era conditions — statements about the *expectation*
of the absolute drift rather than path-wise statements:

- QM: $\sum_{n\ge 1}\mathbb{E}\lvert b_n\rvert < \infty$
- $\sqrt n$-weighted variant: $\sum_{n\ge 1}\sqrt{n}\,\mathbb{E}\lvert b_n\rvert < \infty$

The expectation is estimated by the cross-rollout empirical mean
$\widehat{\mathbb{E}}\lvert b_n\rvert := \frac{1}{R}\sum_{r=1}^R \lvert b_n^{(r)}\rvert$.
The `qm` plot then shows the three quantities below, each with its own
verdict and observed behaviour.

#### (i) $\widehat{\mathbb{E}}\lvert b_n\rvert$ vs $n$, log–log, with OLS power-law fit $\widehat{\mathbb{E}}\lvert b_n\rvert\approx Cn^{-\widehat\beta}$

$\widehat\beta$ is the negative of the fitted log–log slope — equivalently,
the OLS estimate of the power-law decay exponent of
$\widehat{\mathbb{E}}\lvert b_n\rvert$.

| | observed |
|---|---|
| `oracle` | $\widehat\beta = -0.001\pm 0.007$ (flat floor at machine $\epsilon$, no structure to fit) |
| `seqlen1024_training600` | $\widehat\beta = 0.971\pm 0.004$ (see §7). **Caveat**: OLS CI is underestimated because log–log residuals are strongly autocorrelated (lag-1 $\approx 0.99$); treat the point estimate as accurate to ~$\pm 0.1$, not $\pm 0.004$. |
| `seqlen1024_training50k` | $\widehat\beta$ is fit-window-dependent, ranging from 1.03 to 1.19 across reasonable windows; see §7 for the full table. The local decay exponent is near 1, direction uncertain. |

#### (ii) $S(n) = \sum_{m\le n}\widehat{\mathbb{E}}\lvert b_m\rvert$ vs $n$

- *Visually passing* (QM holds): $S(n)$ eventually flattens.
- *Verdict*: 600-step PFN at $\widehat\beta\approx 0.971$ **fails** (threshold $\widehat\beta>1$). 50k-step PFN at local $\widehat\beta\in[1.03, 1.19]$ is **near-borderline**: strictly above the threshold in the window-$[10,300]$ fit, strictly on the threshold in tail-weighted windows; $S(n)$ growth has slowed substantially but has not visibly plateaued within the probed window.

| | observed |
|---|---|
| `oracle` | $S(n_{\text{max}}) = 4.1\times 10^{-13}$, growing linearly as $\epsilon n$ |
| `seqlen1024_training600` | monotonically growing throughout the window, no plateau ($S(600)=1.58\times 10^{-2}$) |
| `seqlen1024_training50k` | still monotonically growing, but the growth is much slower ($S(600)=5.07\times 10^{-3}$, about $3\times$ smaller than the 600-step counterpart at the same $n$) |

#### (iii) $T(n) = \sum_{m\le n}\sqrt{m}\,\widehat{\mathbb{E}}\lvert b_m\rvert$ vs $n$

- *Visually passing* ($\sqrt n$-weighted variant holds): $T(n)$ eventually flattens.
- *Verdict*: **fails for both PFNs** — the variant needs $\widehat\beta>3/2$, which is further from all observed local $\widehat\beta$ values than the QM threshold.

| | observed |
|---|---|
| `oracle` | $T(n_{\text{max}}) = 3.8\times 10^{-11}$, growing as $\epsilon n^{3/2}$ |
| `seqlen1024_training600` | monotonically growing throughout the window, no plateau ($T(600)=0.134$) |
| `seqlen1024_training50k` | still monotonically growing, $T(600)=2.51\times 10^{-2}$ (about $5\times$ smaller than the 600-step counterpart at the same $n$) |

## 4. Oracle precision floor

For the exact Bayes PPD, $b_k \equiv 0$ analytically at every $k$, so
*every* condition holds trivially. The oracle plots instead show
**floating-point roundoff amplified by rate weighting**: $\lvert b_k\rvert$ is at
$\epsilon_{f64}\approx 2.2\times 10^{-16}$, $\lvert P_n\rvert$ is a random walk at
$\epsilon\sqrt n$, $n^{\gamma/2}\lvert b_n\rvert$ has floor $\sim n^{\gamma/2}\epsilon$,
and $n^\gamma(Q_\infty - Q_{n-1})$ has floor $\sim n^\gamma(k_{\text{max}}-n)\epsilon^2$.
Any PFN curve many orders of magnitude above these is real signal; a
curve tracking them is at the numerical limit.

## 5. Checkpoints

Three predictors, each with a `.pt` tensor and two plots:

| stem | data | signed plot | qm plot |
|---|---|---|---|
| `oracle` (f64, $k\le 20000$, $R=32$) | `diag_oracle.pt` | `diag_oracle_signed.pdf` | `diag_oracle_qm.pdf` |
| `seqlen1024_training600` ($R=32$) | `diag_seqlen1024_training600.pt` | `diag_seqlen1024_training600_signed.pdf` | `diag_seqlen1024_training600_qm.pdf` |
| `seqlen1024_training50k` ($R=32$) | `diag_seqlen1024_training50k.pt` | `diag_seqlen1024_training50k_signed.pdf` | `diag_seqlen1024_training50k_qm.pdf` |

Trained weights: `seqlen1024_training600.pt`, `seqlen1024_training50k.pt`. Both PFNs share the same architecture: $d=64, L=2, H=4, \text{dim\_ff}=128$; they differ only in the number of meta-training steps (600 vs 50k).

## 6. Reproduce

```bash
# train
.venv/bin/python train.py --seq-len 1024 --steps 600 --batch-size 16 \
    --d-model 64 --nhead 4 --nlayers 2 --dim-feedforward 128 \
    --warmup-steps 50 --out checkpoints/seqlen1024_training600.pt
.venv/bin/python train.py --seq-len 1024 --steps 50000 --batch-size 16 \
    --d-model 64 --nhead 4 --nlayers 2 --dim-feedforward 128 \
    --warmup-steps 1000 --out checkpoints/seqlen1024_training50k.pt

# diagnose
.venv/bin/python diagnostic.py --mode pfn --checkpoint checkpoints/seqlen1024_training600.pt \
    --out checkpoints/diag_seqlen1024_training600.pt --num-rollouts 32 --seq-len 1024 \
    --k-min 2 --k-max 600 --dtype float32 --seed 1
.venv/bin/python diagnostic.py --mode pfn --checkpoint checkpoints/seqlen1024_training50k.pt \
    --out checkpoints/diag_seqlen1024_training50k.pt --num-rollouts 32 --seq-len 1024 \
    --k-min 2 --k-max 600 --dtype float64 --seed 1
.venv/bin/python diagnostic.py --mode oracle --out checkpoints/diag_oracle.pt \
    --num-rollouts 32 --seq-len 20001 --k-min 2 --k-max 20000 \
    --dtype float64 --seed 1

# plot (produces both signed and qm for each predictor)
.venv/bin/python plot.py --diag checkpoints/diag_seqlen1024_training600.pt --out-dir checkpoints \
    --stem diag_seqlen1024_training600 --fit-n-min 10 --fit-n-max 300
.venv/bin/python plot.py --diag checkpoints/diag_seqlen1024_training50k.pt --out-dir checkpoints \
    --stem diag_seqlen1024_training50k --fit-n-min 10 --fit-n-max 300
.venv/bin/python plot.py --diag checkpoints/diag_oracle.pt --out-dir checkpoints \
    --stem diag_oracle --fit-n-min 10 --fit-n-max 10000
```

## 7. Training-lever comparisons

The two PFNs share the architecture $(d=64, L=2, H=4, \text{dim\_ff}=128)$
and the training `seq_len=1024`, differing only in step count.

**Training step count** (`seqlen1024_training600` vs `seqlen1024_training50k`,
same `seq_len=1024`, 600 vs 50,000 steps, 9,600 vs 800,000 tasks seen).
Mean $\lvert b_k\rvert$ drops from $2.6\times 10^{-5}$ to $8.8\times 10^{-6}$
($\sim 3\times$). The fitted local $\widehat\beta$ on the $[10, 300]$
window moves from 0.971 to 1.187. **However, the power-law fit for `seqlen1024_training50k` is
fragile:** its log–log residuals are strongly autocorrelated (lag-1
correlation 0.995) and sweep coherently between $\pm 0.4$ in log units,
so the underlying $|b_k|$ curve has concavity on log–log and is not a
clean power law. $\widehat\beta$ depends on the fit window:

| fit window | $\widehat\beta$ |
|---|---|
| [10, 300] | 1.187 |
| [20, 300] | 1.114 |
| [10, 500] | 1.130 |
| [20, 500] | 1.080 |
| [30, 400] | 1.031 |

The tail-weighted windows give $\widehat\beta$ close to 1, not 1.19. The
honest statement is that training for $83\times$ more steps moves the
local $|b_k|$ decay exponent on this window *somewhere* in the range
$\widehat\beta\in[1.03, 1.19]$, with the precise value driven by
whichever slice of the window you weight most. The OLS $\pm 0.027$
confidence interval from any single window is wildly narrow because of
the residual autocorrelation. So "QM now passes at $\widehat\beta>1$"
should be read as "the local decay exponent is near 1, perhaps slightly
above, window-dependent," not as a clean passage.

## 8. Environment

Python 3.14, torch 2.11, numpy 2.4, matplotlib 3.10. Install:
```bash
python3 -m venv .venv && .venv/bin/python -m pip install torch numpy matplotlib
```
