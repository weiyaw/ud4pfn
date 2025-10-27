# %%
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
import os

import forward
import data


# # Original iid design
# X, y, x_grid, true_curve, title = data.load_syn_linear_gaussian_regression(
#     n=n, m=m, design="iid", rng=rng_data
# )

# X, y, x_grid, true_curve, title = data.load_syn_mixture_probit(n=n, m=m, design="iid")

# utils.write_to_local(
#     f"{savedir}/data.pickle",
#     {"X": X, "y": y, "x_grid": x_grid, "true_curve": true_curve, "title": title},
# )


@hydra.main(version_base=None, config_path="conf", config_name="forward")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    logging.info(f"Hydra version: {hydra.__version__}")
    logging.info(OmegaConf.to_yaml(cfg))

    n = cfg.data_size
    m = cfg.x_grid_size

    savedir = os.path.relpath(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )

    torch.manual_seed(cfg.seed * 71)
    rng = np.random.default_rng(1907 + cfg.seed)
    rng_others, rng_data = rng.spawn(2)

    # Original iid design
    X, y, x_grid, true_curve, title = data.load_syn_linear_gaussian_regression(
        n=n, m=m, design="iid", rng=rng_data
    )

    utils.write_to_local(
        f"{savedir}/data.pickle",
        {"X": X, "y": y, "x_grid": x_grid, "true_curve": true_curve, "title": title},
    )

    start = timer()
    if cfg.task == "regression":
        clf = TabPFNRegressor(
            n_estimators=64,
            average_before_softmax=True,
            softmax_temperature=1.0,
            fit_mode="low_memory",
        )
        g_hat = forward.build_g_hat_linreg(clf, X, y, x_grid, 0.5)
    elif cfg.task == "classification":
        clf = TabPFNClassifier(
            n_estimators=1,
            average_before_softmax=True,
            softmax_temperature=1.0,
            fit_mode="low_memory",
        )
        g_hat = forward.build_g_hat_logreg(clf, X, y, x_grid)
    else:
        raise ValueError(f"Unknown task: {cfg.task}")

    logging.info(f"Built g_hat in {timer() - start:.2f} seconds")
    utils.write_to_local(f"{savedir}/ghat.pickle", g_hat)


if __name__ == "__main__":
    OmegaConf.register_new_resolver("githash", utils.githash)
    main()
