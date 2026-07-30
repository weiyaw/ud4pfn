# Archive branch (internal, not shared with reviewers)

Snapshot of the NeurIPS 2026 rebuttal work as of 30 Jul 2026, kept for
material we produced but chose not to put into the rebuttal for the VUD
reviewer (47GX). Reassess for the next iteration (discussion phase or
camera-ready).

Not cited in any posted reply:

- `vud_bb_pilot.py`, `vud_bb_timing.py`, `rebuttal/vud_bb_timing.md`,
  `vud_pilot_outputs/vud_bb_pilot_m8.*` — the VUD-vs-CLT comparison on the
  Beta-Bernoulli model of Appendix H, where the fantasy enumeration is
  exact.
- `beta_bernoulli/supstat_tsplit.py`, `rebuttal/supstat_tsplit.md` —
  finite-horizon sup statistics and T-split condition summaries from the
  saved diagnostics (also on the `h9jb` branch; sup-statistic numbers were
  deliberately left out of the posted rebuttal). Based on the 16 saved
  rollouts — h9jB has since asked for ~200, so regenerate before quoting.

Superseded by renamed versions on `vud-pilot` (shared with 47GX):

- `vud_pilot_faithful.py` -> `vud_two_moons.py`, `vud_pilot_faithful2.py`
  -> `vud_spiral_logreg.py`, `vud_pilot_outputs/` -> `vud_outputs/` with
  stems named by evaluated-point count (`evalNNN`). The rebuttal-cited
  numbers are identical on both branches; `vud-pilot` also restores the
  Two Moons input npz that `vud_pilot_faithful.py` needs to run.
