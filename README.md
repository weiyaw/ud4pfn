# Uncertainty Decomposition for Bayes-Filtered Transformers via Bayesian Predictive Inference

## What this is

Code to reproduce the experiments in the paper *Uncertainty Decomposition for
Bayes-Filtered Transformers via Bayesian Predictive Inference*. The methods
build credible bands and an entropy-based uncertainty decomposition on top of
TabPFN v2.5 by treating its one-step-ahead predictive rule as a martingale
posterior and applying a predictive central-limit theorem.

Paper: [arXiv link forthcoming]

## Setup

Requires [uv](https://docs.astral.sh/uv/) and **Python >= 3.11**. From the
repository root:

```bash
uv sync
```

`pyproject.toml` and `uv.lock` are the authoritative environment definitions.
They pin `tabpfn==6.2.0` and `torch==2.9.0` (the versions used in the paper).
On Linux, uv installs JAX with its pip-bundled CUDA 12 runtime and installs the
CUDA 12.8 build of PyTorch. On other platforms, it installs the standard PyPI
builds of JAX and PyTorch. The Linux CUDA environment requires a compatible
NVIDIA GPU and driver, but does not require a separately installed CUDA
toolkit.

Run commands inside the managed environment with `uv run`, for example:

```bash
uv run python run-ghat.py --help
uv run pytest
```

`requirements.txt` is retained as a generated, runtime-only export for pip
compatibility; do not edit it manually. Regenerate it after dependency changes
with:

```bash
uv export --locked --no-dev --no-emit-project --no-hashes \
  --output-file requirements.txt
```

On Linux, pip users must also provide the PyTorch CUDA 12.8 index referenced by
the uv project:

```bash
python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r requirements.txt
```

On non-Linux platforms, `python -m pip install -r requirements.txt` is
sufficient.

All `run-*.py` and `visual-*.py` scripts must be executed **from the repository
root** (paths such as the Hydra output directory and `fibre_strength.csv` are
resolved relative to the working directory or to the script location).

The `beta_bernoulli/` diagnostic testbed uses a **separate, self-contained
environment** (PyTorch only, no TabPFN or JAX); see
[`beta_bernoulli/README.md`](beta_bernoulli/README.md).

## TabPFN checkpoints

The TabPFN v2.5 model weights are not bundled. Download the two checkpoints

- `tabpfn-v2.5-classifier-v2.5_default.ckpt`
- `tabpfn-v2.5-regressor-v2.5_default.ckpt`

from <https://huggingface.co/Prior-Labs/tabpfn_2_5/tree/main> and place them in
a `tabpfn-model/` directory at the repository root:

```
tabpfn-model/
├── tabpfn-v2.5-classifier-v2.5_default.ckpt
└── tabpfn-v2.5-regressor-v2.5_default.ckpt
```

`tabpfn-model/` is gitignored. The scripts load these paths directly (see
`run-ghat.py`, `run-real-analysis.py`, `run-bootstrap.py`, `run-copula.py`).

## Running the experiments

`run-experiments.sh` runs every `run-*.py` invocation needed to produce the
artifacts the figures and tables depend on. The `run-*.py` scripts compute and
cache intermediate tensors; the `visual-*.py` scripts read those caches and
render the figures/tables.

## Artifact map

Hydra writes each repetition to
`outputs/<id>/setup=... x_design=... shuffle=... n_est=... n=... m=... seed=.../`
containing `data.pickle`, `gn.pickle`, `g0_to_gn.pickle`, and (when
`mc_samples>0`) `gn_plus_1.pickle`. Bootstrap and copula runs add
`bootstrap-<B>.pickle` and `copula-<B>-<T>.pickle` alongside.

| Producer | Writes | Format |
|---|---|---|
| `run-ghat.py` | `outputs/<id>/setup=.../{data,gn,g0_to_gn,gn_plus_1,setup}.pickle` | pickle |
| `run-bootstrap.py` | `outputs/<id>/setup=.../bootstrap-200.pickle` | pickle |
| `run-copula.py` | `outputs/<id>/setup=.../copula-200-1000.pickle` | pickle |
| `run-real-analysis.py` | `outputs/real-analysis/setup=.../*.pickle` | pickle |
| `visual-*.py` | `figures/` (override with `$UD4PFN_FIGDIR`) | pdf |
| `beta_bernoulli/*` | `beta_bernoulli/checkpoints/` | pt (tensors/ckpts), pdf (figures) |

`outputs/` and `figures/` are gitignored. Set the environment variable
`UD4PFN_FIGDIR` to redirect figures elsewhere (e.g. into a paper repo) without
editing code:

```bash
UD4PFN_FIGDIR=/path/to/paper/images python visual-gap.py
```

## Reproducing each paper artifact

Run the relevant block of `run-experiments.sh` first, then the visual script.

| Paper artifact | Run (block in `run-experiments.sh`) | Visualize | Command |
|---|---|---|---|
| Coverage tables | coverage block: `run-ghat.py` (id=coverage) + `run-bootstrap.py` + `run-copula.py` | `visual-coverage.py` | `python visual-coverage.py` |
| Gap band figures `gap-<setup>` | gap block: `run-ghat.py` (id=gap) | `visual-gap.py` | `python visual-gap.py` |
| Real-data figures `labour-force-vn`, `fibre-strength-vn` | real-analysis block: `run-real-analysis.py` | `visual-real-analysis.py` | `python visual-real-analysis.py` |
| Entropy decomposition `ud-logreg-*`, `ud-two-moons*` | entropic-ud + entropic-ud-vary-n blocks: `run-ghat.py` | `visual-decompose.py` | `python visual-decompose.py` |
| Beta-Bernoulli diagnostics (`bb-diag-*`, `bb-corrupt-*`, intro fig) | — | see `beta_bernoulli/` | [`beta_bernoulli/README.md`](beta_bernoulli/README.md) |

## Hardware and runtime

- **Coverage sweep**: large. The coverage block runs `run-ghat.py` across 5
  setups × 3 dataset sizes × 50 seeds, followed by per-repetition bootstrap and
  copula runs; the paper reports roughly **700 GPU-hours** in total on NVIDIA
  L40S GPUs. Intended for a GPU cluster.
- **Beta-Bernoulli diagnostics**: each `diagnostic.py` run is about **25 min on
  an H100**.
- **Plotting from cached artifacts**: seconds on CPU, once the `outputs/` (or
  `beta_bernoulli/checkpoints/`) tensors exist.
- **Internet**: the labour-force setup downloads the Mroz dataset at runtime
  (via `statsmodels.datasets.get_rdataset("Mroz", "carData")`), so
  `run-real-analysis.py setup=labour-force` needs network access.

## File structure

```
+-- conf/                    (default configurations for the run-*.py scripts)

+-- run-experiments.sh       (bash script to compute artifacts for all plots in the paper;
|                             all outputs are saved in the outputs/ directory)
+-- run-ghat.py              (computes terms required for V_n and U_n)
+-- run-bootstrap.py         (computes bootstrap-based credible intervals)
+-- run-copula.py            (computes Nagler and Rügamer 2025, copula-based credible intervals)
+-- run-real-analysis.py     (computes V_n for the real-data analysis)

+-- visual-*.py              (scripts to generate and save figures used in the paper)

+-- constants.py             (shared constants used throughout the repository)
+-- data.py                  (data-generating process logic)
+-- metrics.py               (credible intervals, coverage, and entropy-based uncertainty decomposition)
+-- posterior.py             (predictive CLT logic, i.e., Gaussian approximation of the martingale posterior)
+-- pred_rule.py             (extensions of the vanilla TabPFN predictive rule with helper methods)
+-- pr_copula/               (copula-based martingale posterior, adapted from Fong et al. 2023)
+-- beta_bernoulli/          (self-contained predictive-CLT diagnostic testbed; own README + env)
+-- utils.py                 (miscellaneous utility functions)
```

## Data and licensing

- **TabPFN v2.5** model and weights: distributed by Prior Labs under the Prior
  Labs License (Apache-2.0 with an attribution requirement). Downloaded
  separately (see above); not redistributed here.
- **PSID labour-force data** (Mroz): derived from the University of Michigan
  Panel Study of Income Dynamics and used under the PSID Conditions of Use
  (<https://psidonline.isr.umich.edu/>) — academic use with attribution. Fetched
  at runtime via `statsmodels` (`carData::Mroz`), not redistributed here.
- **Fibre-strength data** (`fibre_strength.csv`): from Example 7.4 of Hamada,
  Wilson, Reese, and Martz, *Bayesian Reliability* (Springer, 2008). Included
  for academic reproduction under fair use.
- **`pr_copula/`**: adapted from the MP codebase by Edwin Fong et al.
  (<https://github.com/edfong/MP>), used under the MIT License. See
  [`pr_copula/NOTICE`](pr_copula/NOTICE) for the full attribution and license
  text.

## Acknowledgements

The `pr_copula/` module adapts the copula-based martingale posterior
implementation of Fong, Holmes, and Walker (2023) from
<https://github.com/edfong/MP> (MIT License).
