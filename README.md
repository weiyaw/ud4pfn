# Predictive CLT experiments

This repository contains the reusable predictive central-limit-theorem method
and the five experiment groups used in the paper.

## Environment

Python 3.11 or newer is required. `uv` is authoritative for the reusable
method and the coverage, gap, entropic-UD, and real-data experiments:

```bash
uv sync --frozen
uv sync --frozen --group dev
```

Run supported commands from the repository root with `uv run python -m ...`.
`requirements.txt` is a generated runtime-only compatibility export, not a
dependency source of truth.

Beta--Bernoulli is deliberately separate. It uses its own PyTorch-only
environment and commands run from `experiments/beta_bernoulli/`; see its
[experiment README](experiments/beta_bernoulli/README.md).

## Model checkpoints

Store PFN checkpoints in `pfn-model/`:

```text
pfn-model/
├── tabpfn-v3-classifier-v3_default.ckpt
├── tabpfn-v3-regressor-v3_default.ckpt
├── tabicl-classifier-v2-20260212.ckpt
└── tabicl-regressor-v2-20260212.ckpt
```

The TabPFN weights are downloaded separately from Prior Labs. TabICL downloads
its official checkpoint from Hugging Face when it is not already available.
Model weights are not redistributed with this repository.

## Experiments

Each experiment owns its data definitions, configuration, runner, plotter, and
reproduction notes.

| Group | Generate | Baselines / plot |
|---|---|---|
| Coverage | `uv run python -m experiments.coverage.run ...` | `experiments.coverage.run_bootstrap`, `experiments.coverage.run_copula`, `experiments.coverage.plot` |
| Gap | `uv run python -m experiments.gap.run ...` | `uv run python -m experiments.gap.plot` |
| Entropic UD | `uv run python -m experiments.entropic_ud.run` | `uv run python -m experiments.entropic_ud.plot` |
| Entropic varying \(n\) | `uv run python -m experiments.entropic_ud.run --config-name vary_n` | included by the entropic plotter |
| Real analysis | `uv run python -m experiments.real_analysis.run ...` | `uv run python -m experiments.real_analysis.plot` |
| Beta--Bernoulli | separate local workflow | [README](experiments/beta_bernoulli/README.md) |

`run-experiments.sh` contains the complete paper sweeps for the four root-`uv`
experiment groups. It does not invoke Beta--Bernoulli.

For a lightweight end-to-end check with the actual TabPFN checkpoints, run:

```bash
./run-smoke.sh
```

This runs one network-free case from each root-`uv` experiment group with one
estimator and one Monte Carlo draw. Smoke artifacts are retained under
`outputs/smoke/`.

## Artifact contract

Numerical outputs remain centrally located under:

```text
outputs/
├── coverage/
├── gap/
├── entropic-ud/
├── entropic-ud-vary-n/
└── real-analysis/
```

Synthetic `data.pickle` files contain `x_prev`, `y_prev`, `t`, `x_grid`,
`grid_shape`, and `true_prob`. Real-analysis data omits `true_prob`. Core
artifacts retain these shapes:

```text
gn.pickle          (p, m)
g0_to_gn.pickle    (n + 1, p, m)
gn_plus_1.pickle   (mc_samples, p, m), omitted when mc_samples == 0
```

Coverage baselines write `bootstrap-<B>.pickle` and
`copula-<B>-<T>.pickle`. `setup.pickle` is retired and unsupported.

Figures default to `figures/`. Set `UD4PFN_FIGDIR` to redirect all supported
plotters.

## Repository layout

```text
predictive_clt/                 reusable method and public API
experiments/_shared/            path/runtime and artifact infrastructure
experiments/coverage/           frequentist coverage and complete pr_copula tree
experiments/gap/                gap-in-observation experiments
experiments/entropic_ud/        entropy decomposition
experiments/real_analysis/      Mroz and fibre illustrations
experiments/beta_bernoulli/     independent PyTorch-only diagnostic
legacy/                         unsupported historical Gamma/CID/quasi material
tests/                          CPU/stub and cached-artifact tests
```

The dependency boundary is one-way: experiments may import `predictive_clt`;
the reusable package never imports experiments. Supported experiments do not
import one another except for `experiments._shared`.

## Testing and reproducibility

```bash
uv run pytest
```

The required suite uses CPU stubs and cached-artifact fixtures; it requires no
network, GPU, model checkpoint, or fresh paper sweep. To verify the generated
requirements export:

```bash
uv export --locked --no-dev --no-emit-project --no-hashes \
  --output-file requirements.txt
```

The optional `./run-smoke.sh` check is separate from pytest. It requires both
TabPFN v3 checkpoint files shown above and uses TabPFN's automatic device
selection.

## Data and licensing

- TabPFN model code and weights are distributed by Prior Labs under its
  applicable license and attribution terms.
- The Mroz labour-force dataset is fetched through `statsmodels` and requires
  network access when recomputing that setup.
- `experiments/real_analysis/fibre_strength.csv` is retained for academic
  reproduction of the reliability illustration.
- `experiments/coverage/pr_copula/` retains the complete attributed vendor
  source and its adjacent `NOTICE`.
- Repository-authored code is licensed under [LICENSE](LICENSE).
