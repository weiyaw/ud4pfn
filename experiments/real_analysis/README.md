# Real-data illustrations

This experiment produces the labour-force and fibre-strength predictive-CLT
figures. Commands run from the repository root in the shared `uv` environment.

```bash
uv run python -m experiments.real_analysis.run setup=labour-force
uv run python -m experiments.real_analysis.run setup=fibre-strength
uv run python -m experiments.real_analysis.plot
```

Run the lightweight, network-free actual-checkpoint smoke case with:

```bash
uv run python -m experiments.real_analysis.run --config-name smoke
```

It uses the full local fibre-strength data with two evaluation points, one
estimator, and one Monte Carlo draw. Artifacts are written under
`outputs/smoke/real-analysis/`.

The labour-force computation downloads the Mroz dataset and therefore requires
network access. Fibre data is stored beside this module. Plotting either setup
from cached artifacts requires no network access.

Artifacts remain under `outputs/real-analysis/`; figures are
`labour-force-vn.pdf` and `fibre-strength-vn.pdf`.
