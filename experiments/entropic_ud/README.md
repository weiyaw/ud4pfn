# Entropic uncertainty decomposition

One runner supports both the fixed-data visual experiments and the varying
context-length sweep:

```bash
uv run python -m experiments.entropic_ud.run
uv run python -m experiments.entropic_ud.run --config-name vary_n
uv run python -m experiments.entropic_ud.plot
```

Supported setups are `logistic-linear`, `two-moons-1`, `two-moons-2`, and
`spiral`. Cached artifacts map to these paper figures:

- `ud-logreg-xstar.pdf`
- `ud-logreg-context-length.pdf`
- `ud-logreg-context-length-prop.pdf`
- `ud-two-moons.pdf`
- `ud-two-moons-spiral.pdf`

The varying-\(n\) configuration uses context sizes 75 through 200 in steps of
five, seeds 1000--1049, and writes no `gn_plus_1.pickle`.
