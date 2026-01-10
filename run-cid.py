import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import torch
import joblib
import math
from pathlib import Path
from tqdm import trange
from tqdm.auto import tqdm
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from timeit import default_timer as timer
import utils
import os
import data

from functools import partial

from constants import REGRESSION, CLASSIFICATION, T_MAP
from pred_rule import TabPFNRegressorPPD, assert_ppd_args_shape

os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"


@jax.jit(static_argnames=["n", "get_x", "x_design"])
def sample_x_truth(key, n, x_prev, get_x, x_design):
    """get_x is a function that returns (n, d) array"""
    return get_x(key, n, x_design)


@jax.jit(static_argnames=["n"])
def sample_x_dirac(key, n, x_prev, x_mass):
    """x_mass is a (d, ) array"""
    return jnp.tile(x_mass, (n, 1))


def run_rollout(key, clf, x_new, sample_x, x_init, y_init, rollout_depth, save_path):
    """
    Run the rollout with some initial dataset x_init and y_init, then rollout up
    to rollout_depth.

    Parameters
    ----------
    key: jax.random.PRNGKey
        Random key for sampling.
    clf: TabPFNRegressorPPD
        PPD regressor.
    x_new : (m, d) array
        Query covariates.
    sample_x : Callable
        Function to sample the next covariate x. It will be called with
        arguments (key, n, x_prev).
    x_init : (n, d) array
        Initial dataset.
    y_init : (n,) array
        Initial dataset.
    rollout_depth : int
        Depth of the rollout.
    save_path : Path
        Path to save the computed bias.

    Returns
    -------
    x_rollout : (rollout_depth, d) array
        Rolled out covariates.
    y_rollout : (rollout_depth,) array
        Rolled out targets.
    Saves rollout in {save_path}/rollout.pickle and return the rollout. It skips
    the computation if the file already exists.
    """
    assert_ppd_args_shape(x_new, x_init, y_init)
    n0 = x_init.shape[0]
    assert rollout_depth >= n0
    rollout_path = save_path / "rollout.pickle"
    if os.path.exists(rollout_path):
        logging.info("rollout exists")
        rollout = utils.read_from(rollout_path)
        x_rollout = rollout["x"]
        y_rollout = rollout["y"]
    else:
        start = timer()
        x_rollout = np.vstack([x_init, np.zeros((rollout_depth, x_init.shape[1]))])
        y_rollout = np.append(y_init, np.zeros((rollout_depth,)))
        for i in trange(n0, rollout_depth + n0, desc="rollout", leave=False):
            loopkey = jr.fold_in(key, i)
            loopkey, subkey_x, subkey_y = jr.split(loopkey, 3)
            x_curr = sample_x(subkey_x, 1, x_rollout[:i])
            y_curr, _ = clf.sample(subkey_y, x_curr, x_rollout[:i], y_rollout[:i])
            x_rollout[i] = x_curr
            y_rollout[i] = y_curr.squeeze()
        assert x_rollout.shape[0] == y_rollout.shape[0] == rollout_depth + n0
        utils.write_to(rollout_path, {"x": x_rollout, "y": y_rollout})
        logging.info(f"rollout: {timer() - start:.2f} secs")
    return x_rollout, y_rollout


def compute_Fn_Qn(clf, x_rollout, y_rollout, x_new, t, u, n_points, save_path):
    """
    Compute F_n(x_new, t) and Q_n(x_new, u) along the rollout trajectory where
    F_n(x_new, t) = P( y <= t | x_new, x_{1:n}, y_{1:n}) and Q_n(x_new, u) is
    the quantile (inverse cdf) function of F_n.

    For each i point in n_points, it will evaluate F_n and Q_n on x_rollout[:i]
    and y_rollout[:i].

    Parameters
    ----------
    clf: TabPFNRegressorPPD
        PPD regressor.
    x_rollout : (N, d) array
        Rolled out covariates.
    y_rollout : (N,) array
        Rolled out targets.
    x_new : (m, d) array
        Query covariates.
    t: (p, ) array
        Event of the PPD.
    u: (q, ) array
        Quantile of the PPD.
    n_points : list of int of length K
        List of points to evaluate F_n and Q_n.
    save_path : Path
        Path to save the computed bias.

    Returns
    -------
    For each n, it saves F_n(x_new, t) and Q_n(x_new, u) along the rollout
    trajectory in {save_path}/FnQn-{n}.pickle as a dictionary with keys "F_n",
    "Q_n", "x_new", "t", "q", and "n". It skips the computation if the file
    already exists.
    """
    assert x_rollout.shape[0] == y_rollout.shape[0]
    assert x_rollout.shape[0] >= max(n_points)

    start = timer()
    for n in n_points:
        F_n_path = save_path / f"FnQn-{n}.pickle"
        if os.path.exists(F_n_path):
            logging.info(f"FnQn-{n} exists")
        else:
            x_prev, y_prev = x_rollout[:n], y_rollout[:n]
            F_n = clf.cdf(t, x_new, x_prev, y_prev)  # (p, m)
            Q_n = clf.icdf(u, x_new, x_prev, y_prev)  # (q, m)
            assert F_n.shape == (t.shape[0], x_new.shape[0])
            utils.write_to(
                F_n_path,
                {"F_n": F_n, "Q_n": Q_n, "x_new": x_new, "t": t, "u": u, "n": n},
            )
    logging.info(f"FnQn: {timer() - start:.2f} secs")


@hydra.main(version_base=None, config_path="conf", config_name="cid")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    logging.info(f"Hydra version: {hydra.__version__}")
    logging.info(OmegaConf.to_yaml(cfg))

    n_estimators = int(cfg.n_estimators)
    outer_idx = int(cfg.outer_idx)
    n_grid_size = int(cfg.n_grid_size)
    n_horizon = int(cfg.n_horizon)
    seed = int(cfg.seed)
    x_rollout = cfg.x_rollout  # "truth" or "dirac:xx"
    n0 = 25
    x_design = "uniform:0:1"
    shuffle_data = True

    torch.set_num_threads(1)

    # reproducibility
    key = jr.key(seed)
    key, key_outer, key_setup = jr.split(key, 3)
    torch.manual_seed(seed)

    setup = data.Gamma(key_setup, n=n0, shuffle=shuffle_data, x_design=x_design)

    x_prev = setup.X
    y_prev = setup.y

    x_new = np.linspace(0, 1, 3).reshape(-1, 1)
    t = np.linspace(0, 20, 101)
    u = np.linspace(0.01, 0.99, 99)
    assert x_new.ndim == 2 and t.ndim == 1

    savedir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logging.info(f"Experiment directory: {savedir}")

    if x_rollout == "truth":
        sample_x = partial(sample_x_truth, get_x=setup.get_x, x_design=x_design)
    elif x_rollout.startswith("dirac"):
        # extract dirac-xx
        mass = float(x_rollout.split(":")[1])
        x_mass = np.full((1, x_prev.shape[1]), mass)
        sample_x = partial(sample_x_dirac, x_mass=x_mass)
    else:
        raise ValueError(f"Unknown x_rollout: {x_rollout}")

    clf = TabPFNRegressorPPD(
        n_estimators=n_estimators,
        softmax_temperature=1.0,
        fit_mode="low_memory",
        model_path="tabpfn-model/tabpfn-v2.5-regressor-v2.5_default.ckpt",
    )

    # ------------------------------------------------------------
    # 3.  Run A Single Outer Path (Quasi-Martingale Check)
    # ------------------------------------------------------------
    # Run one outer path (indexed by outer_idx). We rollout until the largest
    # value of n_points and compute F_n(x_new, t) term along the way.

    n_points = np.rint(np.linspace(n0, n0 + n_horizon, n_grid_size)).astype(int)
    logging.info(f"Number of n_points: {len(n_points)}")

    key_outer = jr.fold_in(key_outer, outer_idx)
    save_path = savedir / f"outer-{outer_idx}"
    start = timer()

    key_path, key_eval = jr.split(key_outer)
    x_rollout, y_rollout = run_rollout(
        key_path, clf, x_new, sample_x, x_prev, y_prev, n_horizon, save_path
    )
    compute_Fn_Qn(clf, x_rollout, y_rollout, x_new, t, u, n_points, save_path)
    logging.info(f"outer-{outer_idx}: {timer() - start:.2f} secs")


if __name__ == "__main__":
    OmegaConf.register_new_resolver("githash", utils.githash)
    main()

# %%
