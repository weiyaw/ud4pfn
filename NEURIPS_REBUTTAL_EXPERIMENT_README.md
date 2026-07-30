# Archive: VUD-on-Beta-Bernoulli and early diagnostic material

Base: submission code plus the post-submission Beta-Bernoulli figure
amendments (branch point `2523e06`).

## 1. What we are trying to do

Archive branch. Snapshot (30 Jul 2026) of rebuttal material produced but not
cited in any posted reply, kept for reassessment at the discussion phase or
camera-ready:

- The VUD-vs-CLT comparison on the Beta-Bernoulli model of Appendix H, where
  the fantasy enumeration is exact — a sanity check of the same comparison
  the `vud-tabpfn` branch runs on TabPFN.
- The first (R=16) sup-statistic and T-split condition summaries — since
  superseded by the R=200 rerun on `bb-conditions-r200`; the R=16 numbers
  are never to be quoted.
- The original VUD scripts under their earlier names, superseded on
  `vud-tabpfn` (`vud_pilot_faithful.py` -> `vud_two_moons.py`,
  `vud_pilot_faithful2.py` -> `vud_spiral_logreg.py`, `vud_pilot_outputs/`
  -> `vud_outputs/` with `evalNNN` stems; the rebuttal-cited numbers are
  identical on both branches, and `vud-tabpfn` additionally restores the Two
  Moons input npz that `vud_pilot_faithful.py` needs to run).

## 2. Code to run

```bash
python vud_bb_pilot.py     # VUD-vs-CLT on the Beta-Bernoulli model
python vud_bb_timing.py    # wall-clock comparison on the same model
```

The superseded scripts (`vud_pilot_faithful*.py`,
`beta_bernoulli/supstat_tsplit.py`) run on their current branches under
their new names; run them there.

## 3. Expected figures/tables/artifacts

- `vud_pilot_outputs/vud_bb_pilot_m8.npz` / `_table.txt` (committed) — the
  Beta-Bernoulli VUD comparison arrays and summary table.
- `rebuttal/vud_bb_timing.md` (committed) — the timing comparison.
- `rebuttal/supstat_tsplit.md` (committed) — the superseded R=16 condition
  tables; historical record only.
