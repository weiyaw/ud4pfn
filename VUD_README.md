# VUD-on-TabPFN comparison (rebuttal experiments)

This branch adds a faithful replication of Variational Uncertainty Decomposition
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

| Setting | Script | Command | Output stem (in `vud_pilot_outputs/`) |
|---|---|---|---|
| Two Moons, n=100 | `vud_pilot_faithful.py` | `python vud_pilot_faithful.py` | `vud_faithful_n100_sub12_est8` |
| Three-class spiral, n=200 | `vud_pilot_faithful2.py` | `python vud_pilot_faithful2.py --setup spiral` | `vud_faithful_spiral_n200_sub144_est8` |
| Logistic-linear, n=75 | `vud_pilot_faithful2.py` | `python vud_pilot_faithful2.py --setup logistic-linear` | `vud_faithful_logistic_n75_sub151_est8` |

Each run writes a `.npz` (per-point arrays: `H_sub` total entropy,
`epis_clt_sub` CLT epistemic component, `Va` per-candidate aleatoric bounds,
`KLf` per-candidate coherence scores, `minVa`, `maxVe_raw`), a `.png`
(side-by-side maps), and a `_table.txt` (summary statistics and probe rows).

Naming note: the Two Moons stem says `sub12` (subgrid stride, 12x12 = 144
evaluated points); the later script names by evaluated-point count
(`sub144` = 144 of the 3600-point CLT grid for the spiral; `sub151` = the
full 151-point grid for logistic-linear).

`vud_pilot_shared2.py` and the `vud_shared_*` outputs are a variant that
shares one candidate lattice across query points instead of perturbing about
each x*; it is not used in any reported number and is retained to document
sensitivity to the auxiliary-query strategy.

`vud_bb_pilot.py` / `vud_bb_timing.py` run the same comparison on the
Beta-Bernoulli model of Appendix H.

## Reproducing the reported statistics

The violation and correlation statistics quoted in the author response are
computed from the `.npz` arrays: a candidate "violates" when
`Va > H_sub` at its point (exact fantasy enumeration, so no Monte Carlo
error); a point "reports a negative epistemic lower bound" when
`maxVe_raw < 0`; Spearman correlations are between `epis_clt_sub` and
`maxVe_raw`.
