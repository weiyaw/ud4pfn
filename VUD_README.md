# VUD-on-TabPFN comparison (rebuttal experiments)

This branch adds a replication of Variational Uncertainty Decomposition
(VUD; Jayasekera et al., 2025, arXiv:2509.02327, code github.com/jacobyhsi/VUD)
applied to TabPFN, compared against the paper's predictive-CLT decomposition on
the same data, grids, and TabPFN checkpoints.

## Recipe (their code's defaults)

For each query point x*: K = 15 single-point auxiliary candidates
z ~ N(x*, (0.1 * per-feature std of the data)^2); exact enumeration over the
fantasy label at each candidate; forward-KL coherence score per candidate,
KL( p(y|x*, D) || sum_u p(u|z, D) p(y|x*, z, u, D) ); keep the 5 lowest-KL
candidates; report min Va among the kept and max_Ve = H_total - min Va.
Stated deviations from their LLM pipeline: no permutation ensembling over
context orderings (TabPFN is row invariant) and no prompt serialisation
(tables are TabPFN's native input). TabPFN runs with n_estimators=8.

## Scripts and outputs

| Setting | Script | Command | Output stem (in `vud_outputs/`) |
|---|---|---|---|
| Two Moons, n=100 | `vud_two_moons.py` | `python vud_two_moons.py` | `vud_two_moons_n100_eval144_est8` |
| Three-class spiral, n=200 | `vud_spiral_logreg.py` | `python vud_spiral_logreg.py --setup spiral` | `vud_spiral_n200_eval144_est8` |
| Logistic-linear, n=75 | `vud_spiral_logreg.py` | `python vud_spiral_logreg.py --setup logistic-linear` | `vud_logistic_n75_eval151_est8` |

The Two Moons experiment was run first; `vud_spiral_logreg.py` applies the
identical recipe to the paper's other two settings and computes the CLT side
in-script with the paper's pipeline. The two scripts will eventually be
merged. `vud_two_moons.py` reads its data, evaluation grid, and CLT-side
values from `vud_outputs/two_moons_inputs_n100_grid60_est8.npz` (committed;
arrays `x_prev`/`y_prev` are the training data, `x_grid` the 60x60 grid,
`gn`/`total_entropy`/`epis_clt` the paper pipeline's values at
n_estimators=8).

Each run writes a `.npz` (per-point arrays: `H_sub` total entropy,
`epis_clt_sub` CLT epistemic component, `Va` per-candidate aleatoric bounds,
`KLf` per-candidate coherence scores, `minVa`, `maxVe_raw`) and a `.png`
(side-by-side maps). The `vud_spiral_logreg.py` runs additionally write a
`_table.txt` (summary statistics and probe rows); the Two Moons statistics
are computed from the `.npz` arrays as described in the last section.

Naming note: `evalNNN` is the number of grid points VUD was evaluated at —
144 of the 3600-point (60x60) CLT grid for Two Moons and the spiral (VUD is
expensive per query point); all 151 points of the one-dimensional grid for
logistic-linear. The `_sub` suffix in the `.npz` array names means the same
restriction: those arrays hold the CLT values at the evaluated points.

## Reproducing the reported statistics

The violation and correlation statistics quoted in the author response are
computed from the `.npz` arrays: a candidate "violates" when
`Va > H_sub` at its point (exact fantasy enumeration, so no Monte Carlo
error); a point "reports a negative epistemic lower bound" when
`maxVe_raw < 0`; Spearman correlations are between `epis_clt_sub` and
`maxVe_raw`.
