# A Principled Framework for Uncertainty Decomposition in TabPFN

## Reproducing Results from the Paper

### Environment setup

A requirements file, `requirements.txt`, is provided for convenience. We
recommend using a CPU version of `jax` to avoid potential GPU conflicts with
PyTorch.

### Experiment scripts

The experiments are organised into multiple `run-*.py` scripts for generating
artifacts on an HPC cluster or workstation, and corresponding `visual-*.py`
scripts for producing the plots and tables reported in the paper. The
`run-experiments.sh` bash script executes the relevant `run-*.py` scripts with
the appropriate configurations to generate all artifacts required for the
figures and tables in the paper.

| Experiment                              | Run Script             | Visualization Script      |
| --------------------------------------- | ---------------------- | ------------------------- |
| Coverage                                | `run-ghat.py`, `run-bootstrap.py`, `run-copula.py`          | `visual-coverage.py`      |
| Gap                                     | `run-ghat.py`          | `visual-gap.py`           |
| Real data analysis                      | `run-real-analysis.py` | `visual-real-analysis.py` |
| Entropy-based uncertainty decomposition | `run-ghat.py`          | `visual-decompose.py`     |


## File structure

```
+-- conf/                    (default configurations for the `run-*.py` scripts)

+-- run-experiments.sh       (bash script to compute artifacts for all plots in the paper;
|                             all outputs are saved in the `outputs/` directory)
+-- run-ghat.py              (computes terms required for V_n and U_n)
+-- run-bootstrap.py         (computes bootstrap-based credible intervals)
+-- run-copula.py            (computes Nagler and Rügamer 2025, copula-based credible intervals)
+-- run-real-analysis.py     (computes V_n for the real-data analysis)

+-- visual-*.py              (scripts to generate and save figures used in the paper)

+-- constants.py             (shared constants used throughout the repository)
+-- data.py                  (data-generating process logic)
+-- metrics.py               (credible intervals, coverage, and entropy-based uncertainty decomposition)
+-- posterior.py             (predictive CLT logic, i.e., Gaussian approximation of the martingale posterior)
+-- pred_rule.py             (extensions of the vanilla TabPFN predictive rule with helper methods)
+-- pr_copula/               (Copula-based martingale posterior adapted from Fong et al 2023 to implement Nagler and Rügamer 2025)
+-- utils.py                 (miscellaneous utility functions)
```
