# %%
import os
from tabpfn import TabPFNClassifier, TabPFNRegressor
import warnings
from typing import Callable
import torch

import numpy as np
import utils
import logging
from timeit import default_timer as timer

from omegaconf import DictConfig, OmegaConf
import hydra

import credible_set
from credible_set import TabPFNClassifierPPD, TabPFNRegressorPPD
import data


REGRESSION = [
    "gaussian-linear",
    "gaussian-polynomial",
    "gaussian-linear-dependent-error",
    "gamma-linear",
    "gaussian-sine",
]

CLASSIFICATION = [
    "probit-mixture",
    "poisson-linear",
]

Y_STAR_MAP = {
    **{k: 0.0 for k in REGRESSION},
    **{k: 1 for k in CLASSIFICATION},
    "gamma-linear": 4.0,
}


@hydra.main(version_base=None, config_path="conf", config_name="ghat")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    logging.info(f"Hydra version: {hydra.__version__}")
    logging.info(OmegaConf.to_yaml(cfg))

    n = int(cfg.data_size)
    m = int(cfg.x_grid_size)
    x_design = cfg.x_design
    mc_samples = int(cfg.mc_samples)
    setup_name = cfg.setup
    seed = int(cfg.seed)
    tabpfn_n_estimators = int(cfg.n_estimators)
    tabpfn_average_before_softmax = cfg.average_before_softmax
    x_grid = np.linspace(-10, 10, m).reshape(-1, 1)

    torch.manual_seed(8655 + seed)
    rng = np.random.default_rng(1907 + seed)
    rng_others, rng_setup = rng.spawn(2)

    # convert setup (kebab-case) to Camalcase and ensure PPD suffix
    try:
        Setup = getattr(data, utils.kebab_to_camel(setup_name))
        setup = Setup(n, rng_setup, x_design)
    except AttributeError:
        raise ValueError(f"Data {setup_name} not found in data module")

    savedir = os.path.relpath(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )

    if setup_name in REGRESSION:
        clf = TabPFNRegressorPPD(
            y_star=Y_STAR_MAP[setup_name],
            n_estimators=tabpfn_n_estimators,
            average_before_softmax=tabpfn_average_before_softmax,
            softmax_temperature=1.0,
            fit_mode="low_memory",
            model_path="tabpfn-model/tabpfn-v2-regressor.ckpt",
        )
    elif setup_name in CLASSIFICATION:
        clf = TabPFNClassifierPPD(
            y_star=Y_STAR_MAP[setup_name],
            n_estimators=tabpfn_n_estimators,
            average_before_softmax=tabpfn_average_before_softmax,
            softmax_temperature=1.0,
            fit_mode="low_memory",
            model_path="tabpfn-model/tabpfn-v2-classifier.ckpt",
        )
    else:
        raise ValueError(f"Unknown data {data}")

    X = setup.X
    y = setup.y

    logging.info(f"Saving outputs to {savedir}")

    utils.write_to_local(
        f"{savedir}/data.pickle",
        {"setup": setup, "x_grid": x_grid},
    )

    start = timer()
    g0_to_gn = credible_set.compute_g0_to_gn(clf, x_grid, X, y)
    utils.write_to_local(f"{savedir}/g0_to_gn.pickle", g0_to_gn)
    logging.info(f"Built g0_to_gn in {timer() - start:.2f} seconds")

    start = timer()
    gn = credible_set.compute_gn(clf, x_grid, X, y)
    utils.write_to_local(f"{savedir}/gn.pickle", gn)
    logging.info(f"Built gn in {timer() - start:.2f} seconds")

    start = timer()
    gn_plus_1 = credible_set.sample_gn_plus_1(
        rng_others, clf, x_grid, X, y, size=mc_samples
    )
    utils.write_to_local(f"{savedir}/gn_plus_1.pickle", gn_plus_1)
    logging.info(f"Built gn_plus_1 in {timer() - start:.2f} seconds")


if __name__ == "__main__":
    OmegaConf.register_new_resolver("githash", utils.githash)
    main()
