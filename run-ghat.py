# %%
import os
import warnings
from typing import Callable
import torch

import numpy as np
import jax
import jax.random as jr
import utils
import logging
from timeit import default_timer as timer

from omegaconf import DictConfig, OmegaConf
import hydra

import credible_set
from pred_rule import TabPFNClassifierPPD, TabPFNRegressorPPD
import data

from constants import REGRESSION, CLASSIFICATION, Y_STAR_MAP

os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"

@hydra.main(version_base=None, config_path="conf", config_name="ghat")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    logging.info(f"Hydra version: {hydra.__version__}")
    logging.info(OmegaConf.to_yaml(cfg))

    n = int(cfg.data_size)
    m = int(cfg.x_grid_size)
    x_design = cfg.x_design
    shuffle_data = cfg.shuffle_data
    mc_samples = int(cfg.mc_samples)
    setup_name = cfg.setup
    seed = int(cfg.seed)
    tabpfn_n_estimators = int(cfg.n_estimators)
    tabpfn_average_before_softmax = cfg.average_before_softmax
    x_grid = np.linspace(-10, 10, m).reshape(-1, 1)

    torch.manual_seed(8655 + seed)
    key = jr.key(1907 + seed)
    key_others, key_setup = jr.split(key)

    # convert setup (kebab-case) to Camalcase and ensure PPD suffix
    try:
        Setup = getattr(data, utils.kebab_to_camel(setup_name))
        setup = Setup(key_setup, n, shuffle_data, x_design)
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
        key_others, clf, x_grid, X, y, size=mc_samples
    )
    utils.write_to_local(f"{savedir}/gn_plus_1.pickle", gn_plus_1)
    logging.info(f"Built gn_plus_1 in {timer() - start:.2f} seconds")


if __name__ == "__main__":
    OmegaConf.register_new_resolver("githash", utils.githash)
    main()
