# %%

import logging
import os
from timeit import default_timer as timer

import hydra
import jax.random as jr
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from scipy.stats import qmc

import data
import posterior
import utils
from constants import CLASSIFICATION, REGRESSION, T_MAP
from pred_rule import TabPFNClassifierPPD, TabPFNRegressorPPD

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
    n_estimators = int(cfg.n_estimators)

    torch.manual_seed(8655 + seed)
    key = jr.key(1907 + seed)
    key_others, key_setup = jr.split(key)

    if cfg.fix_data:
        # keep the random number generator fixed if required
        key_setup = jr.key(6683)

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
            n_estimators=n_estimators,
            softmax_temperature=1.0,
            fit_mode="low_memory",
            model_path="tabpfn-model/tabpfn-v2.5-regressor-v2.5_default.ckpt",
        )
    elif setup_name in CLASSIFICATION:
        clf = TabPFNClassifierPPD(
            n_estimators=n_estimators,
            softmax_temperature=1.0,
            fit_mode="low_memory",
            model_path="tabpfn-model/tabpfn-v2.5-classifier-v2.5_default.ckpt",
        )
    else:
        raise ValueError(f"Unknown data {data}")

    x_prev = setup.X
    y_prev = setup.y
    d = x_prev.shape[1]
    if d == 1:
        if isinstance(setup, data.LogisticLinear):
            # same as Jayasekera et al 2025
            x_grid = np.linspace(-15.0, 15.0, 151).reshape(-1, 1)
        else:
            x_grid = np.linspace(-10, 10, m).reshape(-1, 1)
        grid_shape = (x_grid.shape[0],)
    elif d > 2 or (d == 2 and isinstance(setup, data.DataMultivariate)):
        # Mirror DataMultivariate Sobol construction: scrambled Sobol in [-1, 1]^d.
        sampler = qmc.Sobol(d=d, scramble=True, rng=np.random.default_rng(50194))
        x01 = sampler.random(n=m)
        x_grid = (2.0 * x01 - 1.0).astype(np.float32)
        grid_shape = (x_grid.shape[0],)
    elif d == 2:
        if isinstance(setup, data.TwoMoons1):
            # same as Jayasekera et al 2025
            lin1 = np.linspace(-1.5, 2.6, m)
            lin2 = np.linspace(-1.5, 2.6, m)
        elif isinstance(setup, data.TwoMoons2):
            # same as Jayasekera et al 2025
            lin1 = np.linspace(-3.0, 3.6, m)
            lin2 = np.linspace(-2.5, 3.1, m)
        else:
            lin1 = np.linspace(-4.0, 4.0, m)
            lin2 = np.linspace(-4.0, 4.0, m)
        x1, x2 = np.meshgrid(lin1, lin2, indexing="ij")
        x_grid = np.stack([x1, x2], axis=-1).reshape(-1, 2)
        grid_shape = (len(lin1), len(lin2))

    t = np.array(T_MAP[setup_name])

    logging.info(f"Saving outputs to {savedir}")

    # Save this in case we need to inspect data generating distribution
    utils.write_to_local(f"{savedir}/setup.pickle", setup)

    # This is pure numpy array, so it's faster to load
    true_prob = np.stack([setup.get_true_event(x_grid, st) for st in t])
    utils.write_to_local(
        f"{savedir}/data.pickle",
        {
            "x_prev": x_prev,
            "y_prev": y_prev,
            "t": t,
            "x_grid": x_grid,
            "grid_shape": grid_shape,
            "true_prob": true_prob,
        },
    )

    start = timer()
    g0_to_gn = posterior.compute_g0_to_gn(clf, t, x_grid, x_prev, y_prev)
    utils.write_to_local(f"{savedir}/g0_to_gn.pickle", g0_to_gn)
    logging.info(f"Built g0_to_gn in {timer() - start:.2f} seconds")

    start = timer()
    gn = posterior.compute_gn(clf, t, x_grid, x_prev, y_prev)
    utils.write_to_local(f"{savedir}/gn.pickle", gn)
    logging.info(f"Built gn in {timer() - start:.2f} seconds")

    if mc_samples > 0:
        start = timer()
        gn_plus_1 = posterior.sample_gn_plus_1(
            key_others, clf, t, x_grid, x_prev, y_prev, size=mc_samples
        )
        utils.write_to_local(f"{savedir}/gn_plus_1.pickle", gn_plus_1)
        logging.info(f"Built gn_plus_1 in {timer() - start:.2f} seconds")
    else:
        logging.info(f"mc_samples={mc_samples}, skipping gn_plus_1")


if __name__ == "__main__":
    OmegaConf.register_new_resolver("githash", utils.githash)
    main()

# %%
