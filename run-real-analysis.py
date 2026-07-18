import logging
import os
from timeit import default_timer as timer

import hydra
import jax.random as jr
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

import data
import posterior
import utils
from constants import T_MAP
from pred_rule import TabPFNClassifierPPD

os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"


@hydra.main(version_base=None, config_path="conf", config_name="real-analysis")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    logging.info(f"Hydra version: {hydra.__version__}")
    logging.info(OmegaConf.to_yaml(cfg))

    m = int(cfg.x_grid_size)
    shuffle_data = cfg.shuffle_data
    mc_samples = int(cfg.mc_samples)
    setup_name = cfg.setup
    seed = int(cfg.seed)
    n_estimators = int(cfg.n_estimators)

    torch.manual_seed(8655 + seed)
    key = jr.key(1907 + seed)
    key_others, _ = jr.split(key)

    if setup_name == "labour-force":
        setup = data.LabourForce(shuffle_data)
    elif setup_name == "fibre-strength":
        setup = data.FibreStrength(shuffle_data)
    else:
        raise ValueError(f"Unknown data {setup_name} for real analysis.")

    savedir = os.path.relpath(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )

    clf = TabPFNClassifierPPD(
        n_estimators=n_estimators,
        softmax_temperature=1.0,
        fit_mode="low_memory",
        model_path="tabpfn-model/tabpfn-v2.5-classifier-v2.5_default.ckpt",
    )

    x_prev = setup.X
    y_prev = setup.y
    assert x_prev.shape[1] == 1, "Only 1D data is supported"
    x_grid = np.linspace(x_prev.min(), x_prev.max(), m).reshape(-1, 1)
    grid_shape = (m,)
    t = np.array(T_MAP[setup_name])

    logging.info(f"Saving outputs to {savedir}")

    # Save this in case we need to inspect data generating distribution
    utils.write_to_local(f"{savedir}/setup.pickle", setup)

    # This is pure numpy array, so it's faster to load
    utils.write_to_local(
        f"{savedir}/data.pickle",
        {
            "x_prev": x_prev,
            "y_prev": y_prev,
            "t": t,
            "x_grid": x_grid,
            "grid_shape": grid_shape,
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

    start = timer()
    gn_plus_1 = posterior.sample_gn_plus_1(
        key_others, clf, t, x_grid, x_prev, y_prev, size=mc_samples
    )
    utils.write_to_local(f"{savedir}/gn_plus_1.pickle", gn_plus_1)
    logging.info(f"Built gn_plus_1 in {timer() - start:.2f} seconds")


if __name__ == "__main__":
    OmegaConf.register_new_resolver("githash", utils.githash)
    main()

# %%
