# Gap-in-observation experiments

The seven one-dimensional DGPs in this directory use the fixed one-gap design:
half the covariates lie in `[-8,-2]` and half in `[2,8]`.

```bash
uv run python -m experiments.gap.run setup=gaussian-linear data_size=200
uv run python -m experiments.gap.plot
```

The paper sweep uses sample sizes 200, 500, and 1000, seed 1000,
`n_estimators=64`, a 100-point grid on `[-10,10]`, and 1000 Monte Carlo draws.
The plotter writes:

- `gap-gaussian-linear.pdf`
- `gap-gaussian-polynomial.pdf`
- `gap-gaussian-linear-dependent-error.pdf`
- `gap-gaussian-sine.pdf`
- `gap-poisson-linear.pdf`
- `gap-probit-mixture.pdf`
- `gap-categorical-linear.pdf`
