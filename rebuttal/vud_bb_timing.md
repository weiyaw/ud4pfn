| Model | n | CLT: prefix + v_n + entropies (s) | VUD exact tree m=4 (s) | VUD exact tree m=8 (s) | CLT forwards | VUD forwards (m=8) |
|---|---|---|---|---|---|---|
| pfn-600 | 10 | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.011 ± 0.001 | 10 (batch 1) | 9 (batch ≤ 2^8) |
| pfn-600 | 50 | 0.023 ± 0.001 | 0.005 ± 0.001 | 0.034 ± 0.004 | 50 (batch 1) | 9 (batch ≤ 2^8) |
| pfn-600 | 200 | 0.115 ± 0.002 | 0.011 ± 0.001 | 0.128 ± 0.009 | 200 (batch 1) | 9 (batch ≤ 2^8) |
| pfn-50k | 10 | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.012 ± 0.000 | 10 (batch 1) | 9 (batch ≤ 2^8) |
| pfn-50k | 50 | 0.022 ± 0.000 | 0.004 ± 0.000 | 0.026 ± 0.002 | 50 (batch 1) | 9 (batch ≤ 2^8) |
| pfn-50k | 200 | 0.115 ± 0.002 | 0.011 ± 0.000 | 0.126 ± 0.004 | 200 (batch 1) | 9 (batch ≤ 2^8) |

Wall-clock per context (mean +/- sd over 3 theta values x 3 repeats, single CPU) of the
predictive-CLT pipeline (prefix trajectory + v_n + closed-form entropies) vs VUD exact
fantasy-tree enumeration, both on the identical Beta-Bernoulli BFT and code paths as
vud_bb_pilot.py. At these context lengths the two are comparable at m = 8 and VUD is
cheaper at m = 4, because the 2^m branches batch into m+1 forwards while the prefix runs
n sequential batch-1 forwards. The relevant contrast is scaling: VUD-exact doubles its
branch count per unit of fantasy depth m (2^m branches), and its bound certifies only
~ m/(n+m) of the epistemic uncertainty (vud_bb_pilot_m8_table.txt, truncation-scaling
block: 3.8% at n = 200, m = 8), so matching the CLT decomposition at n = 200 would need
m ~ n, i.e. a 2^200-branch tree; the CLT pipeline delivers the full decomposition in n
forwards, independent of any probe depth.
