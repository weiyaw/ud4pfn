import logging
from pathlib import Path
from timeit import default_timer as timer

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from experiments._shared.artifacts import load_run_metadata, read_pickle, write_pickle
from experiments._shared.runtime import (
    CLASSIFIER_CHECKPOINT_PATH,
    REGRESSOR_CHECKPOINT_PATH,
    register_githash_resolver,
)
from predictive_clt import (
    TabPFNClassifierPPD,
    TabPFNRegressorPPD,
    compute_gn,
)


REGRESSION_EXPERIMENTS = {
    "gaussian-linear-multivariate",
    "gaussian-linear-dependent-error-multivariate",
    "poisson-linear-multivariate",
}
CLASSIFICATION_EXPERIMENTS = {
    "probit-mixture-multivariate",
    "categorical-linear-multivariate",
}


def build_predictive_rule(experiment_name: str, n_estimators: int):
    if experiment_name in REGRESSION_EXPERIMENTS:
        return TabPFNRegressorPPD(
            n_estimators=n_estimators,
            softmax_temperature=1.0,
            fit_mode="low_memory",
            model_path=str(REGRESSOR_CHECKPOINT_PATH),
        )
    if experiment_name in CLASSIFICATION_EXPERIMENTS:
        return TabPFNClassifierPPD(
            n_estimators=n_estimators,
            softmax_temperature=1.0,
            fit_mode="low_memory",
            model_path=str(CLASSIFIER_CHECKPOINT_PATH),
        )
    raise ValueError(f"Unknown setup '{experiment_name}'")


def compute_bootstrap_predictions(
    predictive_rule,
    t,
    x_grid,
    x_prev,
    y_prev,
    n_bootstrap: int,
    seed: int,
):
    """Return bootstrap predictions with shape (B, p, m)."""
    n = x_prev.shape[0]
    rng = np.random.default_rng(seed)
    preds = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        g_boot = compute_gn(
            predictive_rule,
            t=t,
            x_grid=x_grid,
            x_prev=x_prev[idx],
            y_prev=y_prev[idx],
        )
        preds.append(g_boot)

    return np.stack(preds, axis=0)


def save_bootstrap_samples_for_rep(rep_dir: str, cfg: DictConfig) -> None:
    directory = Path(rep_dir)
    outpath = directory / f"bootstrap-{cfg.bootstrap_samples}.pickle"
    if outpath.exists() and not bool(cfg.overwrite):
        logging.info(f"Skipping existing {outpath}")
        return

    rep_data = read_pickle(directory / "data.pickle")
    metadata = load_run_metadata(directory, required_fields=("setup", "n_est", "seed"))
    experiment_name = metadata["setup"]
    n_estimators = metadata["n_est"]
    seed = metadata["seed"]

    predictive_rule = build_predictive_rule(experiment_name, n_estimators)

    logging.info(f"Computing {cfg.bootstrap_samples} bootstrap samples for {rep_dir}.")
    start = timer()
    bootstrap_preds = compute_bootstrap_predictions(
        predictive_rule=predictive_rule,
        t=rep_data["t"],
        x_grid=rep_data["x_grid"],
        x_prev=rep_data["x_prev"],
        y_prev=rep_data["y_prev"],
        n_bootstrap=int(cfg.bootstrap_samples),
        seed=seed + int(cfg.seed_offset),
    )
    # bootstrap_preds: (bootstrap_samples, num of t grid, num of x_grid)
    elapsed = timer() - start
    write_pickle(
        outpath,
        {
            "bootstrap_samples": int(cfg.bootstrap_samples),
            "seed_offset": int(cfg.seed_offset),
            "predictions": bootstrap_preds,
            "elapsed_seconds": elapsed,
        },
    )
    logging.info(f"Saved bootstrap samples for {rep_dir} in {elapsed:.2f}s")


@hydra.main(version_base=None, config_path="conf", config_name="bootstrap")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    logging.info(f"Hydra version: {hydra.__version__}")
    logging.info(OmegaConf.to_yaml(cfg))

    rep_dir = cfg.rep_dir
    if not Path(rep_dir).exists():
        raise FileNotFoundError(f"Repetition directory not found: {rep_dir}")

    if not (Path(rep_dir) / "data.pickle").exists():
        raise FileNotFoundError(f"data.pickle not found in {rep_dir}")

    save_bootstrap_samples_for_rep(rep_dir, cfg)


if __name__ == "__main__":
    register_githash_resolver()
    main()
