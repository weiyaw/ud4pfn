# Frequentist coverage

Coverage reproduction has three stages:

1. Generate predictive-CLT artifacts with `experiments.coverage.run`.
2. Generate bootstrap artifacts and, for supported Gaussian/Poisson regression
   runs, copula artifacts.
3. Generate the paper tables with `experiments.coverage.plot`.

```bash
uv run python -m experiments.coverage.run setup=gaussian-linear-multivariate
uv run python -m experiments.coverage.run_bootstrap rep_dir=<run-directory>
uv run python -m experiments.coverage.run_copula rep_dir=<run-directory>
uv run python -m experiments.coverage.plot
```

The paper defaults are sample sizes 200, 500, and 1000, seeds 1000--1049,
`sobol-10d`, 16 estimators, and 1000 Monte Carlo draws. Bootstrap artifacts use
`B=200`; copula artifacts use 200 rollouts of length 1000. Tables report
pointwise and simultaneous coverage at alpha 0.05 and 0.20.

The complete attributed `pr_copula` source tree, including its `NOTICE` and
experimental modules, is retained beside the coverage baseline runner.
