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
from pred_rule import TabPFNClassifierPPD, TabPFNRegressorPPD, assert_ppd_args_shape
from posterior import compute_gn, sample_gn_plus_1

os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"


def sample_delta(key, clf, sample_x, t, x_new, x_prev, y_prev, size):
    """
    Sample Δ_n(x_new, t) given x_{1:n-1}, y_{1:n-1} for the event A = {y=t} or A
    = {y <= t} which depends on clf. Here, Δ_n(x_new, t) = P_n(x_new, t) -
    P_{n-1}(x_new, t) with P_n(x_new, t) = P( A | x_new, x_{1:n}, y_{1:n}) and
    P_{n-1}(x_new, t) = P( A | x_new, x_{1:n-1}, y_{1:n-1}).

    This quantity is random because of the randomness in x_n and y_n. The
    distribution of x_n is defined in sample_x, and y_n | x_n the TabPFN given
    by clf.

    Parameters
    ----------
    key: jax.random.PRNGKey
        Random key for sampling.
    clf: TabPFNClassifierPPD or TabPFNRegressorPPD
        PPD classifier or regressor.
    sample_x : Callable
        Function to sample the next covariate x. It will be called with
        arguments (key, n, x_prev).
    t: (p, ) array
        Event of the PPD.
    x_new : (m, d) array
        Query covariates.
    x_prev : (n, d) array
        Historical covariates x_{1:n-1}
    y_prev : (n,) array
        Historical targets y_{1:n-1}.
    size : int
        Number of Monte Carlo samples for the inner Monte Carlo estimate.

    Return:
    -------
    (size, p, m) array
        Samples of Δ_n(x_new, t). Taking the mean along the 0th-axis will give a
        Monte Carlo estimate of E_n[Δ_n(x_new, t)].
    """
    P_n = clf.predict_event(t, x_new, x_prev, y_prev)  # (p, m)
    key, subkey = jr.split(key)
    x_curr = sample_x(subkey, size, x_prev)  # x_n: (size, d)
    key, subkey = jr.split(key)
    y_curr, _ = clf.sample(subkey, x_curr, x_prev, y_prev, size=1)  # y_n: (1, size)
    assert x_curr.shape[0] == y_curr.shape[1] == size

    deltas = []
    for i in trange(size, desc="inner MC", leave=False):
        x_plus_1 = np.vstack([x_prev, x_curr[i]])  # x_{1:n}
        y_plus_1 = np.append(y_prev, y_curr[0, i])  # y_{1:n}
        P_n_plus_1 = clf.predict_event(t, x_new, x_plus_1, y_plus_1)  # (p, m)
        deltas.append(P_n_plus_1 - P_n)

    return np.stack(deltas, axis=0)  # (size, p, m)


@jax.jit(static_argnames=["n"])
def sample_x_dirac(key, n, x_prev, x_mass):
    """x_mass is a (d, ) array"""
    return jnp.tile(x_mass, (n, 1))


@jax.jit(static_argnames=["n", "get_x", "x_design"])
def sample_x_truth(key, n, x_prev, get_x, x_design):
    """get_x is a function that returns (n, d) array"""
    return get_x(key, n, x_design)


def run_single_outer_path(
    key, clf, sample_x, t, x_new, x_init, y_init, n_points, mc_delta, save_path
):
    """
    Sample from Δ_n(x_new, t) along the rollout trajectory where Δ_n(x_new, t) =
    P_n(x_new, t) - P_{n-1}(x_new, t) with P_n(x_new, t) = P( A | x_new,
    x_{1:n}, y_{1:n}) and P_{n-1}(x_new, t) = P( A | x_new, x_{1:n-1},
    y_{1:n-1}).

    The function will start the rollout with some initial dataset x_init and
    y_init, then rollout up to max(n_points) - 1. After the rollout, the
    function will will sample from Δ_n(x_new, t) at all the n specified in
    n_points.

    Parameters
    ----------
    key: jax.random.PRNGKey
        Random key for sampling.
    clf: TabPFNClassifierPPD or TabPFNRegressorPPD
        PPD classifier or regressor.
    sample_x : Callable
        Function to sample the next covariate x. It will be called with
        arguments (key, n, x_prev).
    t: (p, ) array
        Event of the PPD.
    x_new : (m, d) array
        Query covariates.
    x_init : (n, d) array
        Initial dataset.
    y_init : (n,) array
        Initial dataset.
    n_points : list of int of length K
        List of rollout depth the outer path.
    mc_delta : int
        Number of samples for the inner Monte Carlo estimate.
    save_path : Path
        Path to save the computed bias.

    Returns
    -------
    For each n, it saves the samples of Δ_n | x_{1:n-1}, y_{1:n-1} in
    {save_path}/delta-{n}.pickle as a dictionary with keys "delta", "x_new",
    "t", and "n". It skips the computation if the file already exists.
    """

    key_path, key_eval = jr.split(key)
    assert_ppd_args_shape(x_new, x_init, y_init)

    n0 = x_init.shape[0]
    rollout_depth = max(n_points) - 1 - n0
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
            loopkey = jr.fold_in(key_path, i)
            loopkey, subkey_x, subkey_y = jr.split(loopkey, 3)
            x_curr = sample_x(subkey_x, 1, x_rollout[:i])
            y_curr, _ = clf.sample(subkey_y, x_curr, x_rollout[:i], y_rollout[:i])
            x_rollout[i] = x_curr
            y_rollout[i] = y_curr.squeeze()
        assert x_rollout.shape[0] == y_rollout.shape[0] == rollout_depth + n0
        utils.write_to(rollout_path, {"x": x_rollout, "y": y_rollout})
        logging.info(f"rollout: {timer() - start:.2f} secs")

    for n in n_points:
        delta_path = save_path / f"delta-{n}.pickle"
        if os.path.exists(delta_path):
            logging.info(f"delta-{n} exists")
        else:
            start = timer()
            subkey = jr.fold_in(key_eval, n)
            x_prev, y_prev = x_rollout[: n - 1], y_rollout[: n - 1]
            deltas_n = sample_delta(
                subkey, clf, sample_x, t, x_new, x_prev, y_prev, mc_delta
            )
            assert deltas_n.shape == (mc_delta, t.shape[0], x_new.shape[0])
            utils.write_to(
                delta_path, {"delta": deltas_n, "x_new": x_new, "t": t, "n": n}
            )
            logging.info(f"delta-{n}: {timer() - start:.2f} secs")


@hydra.main(version_base=None, config_path="conf", config_name="quasi")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    logging.info(f"Hydra version: {hydra.__version__}")
    logging.info(OmegaConf.to_yaml(cfg))

    n_estimators = int(cfg.n_estimators)
    n0 = int(cfg.n0)
    n_grid_size = int(cfg.n_grid_size)
    n_horizon = int(cfg.n_horizon)
    outer_idx = int(cfg.outer_idx)
    mc_delta = int(cfg.mc_delta)
    seed = int(cfg.seed)
    setup_name = cfg.setup
    x_design = cfg.x_design
    x_rollout = cfg.x_rollout
    shuffle_data = cfg.shuffle_data

    torch.set_num_threads(1)

    # reproducibility
    key = jr.key(seed)
    key, key_outer, key_setup = jr.split(key, 3)
    torch.manual_seed(seed)

    # ------------------------------------------------------------
    # 1. Load Data and Setup
    # ------------------------------------------------------------
    # convert setup (kebab-case) to Camalcase and ensure PPD suffix
    try:
        Setup = getattr(data, utils.kebab_to_camel(setup_name))
        setup = Setup(key_setup, n0, shuffle_data, x_design)
    except AttributeError:
        raise ValueError(f"Data {setup_name} not found in data module")

    x_prev = setup.X
    y_prev = setup.y

    # x_new is a grid
    if x_design.startswith("uniform:"):
        minval, maxval = x_design.split(":")[1:]
        x_new = np.linspace(float(minval), float(maxval), 11).reshape(-1, 1)
    else:
        x_new = np.linspace(-10, 10, 11).reshape(-1, 1)
    t = np.array(T_MAP[setup_name])
    assert x_new.ndim == 2 and t.ndim == 1

    savedir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logging.info(f"Experiment directory: {savedir}")

    if x_rollout == "truth":
        sample_x = partial(sample_x_truth, get_x=setup.get_x, x_design=x_design)
    elif x_rollout.startswith("dirac"):
        # extract dirac-xx
        mass = float(x_rollout.split("-")[1])
        x_mass = np.full((1, x_prev.shape[1]), mass)
        sample_x = partial(sample_x_dirac, x_mass=x_mass)
    else:
        raise ValueError(f"Unknown x_rollout: {x_rollout}")

    # ------------------------------------------------------------
    # 2.  Initialize Classifier/Regressor
    # ------------------------------------------------------------
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
        raise ValueError(f"Unknown setup {setup_name}")

    # ------------------------------------------------------------
    # 3.  Run A Single Outer Path (Quasi-Martingale Check)
    # ------------------------------------------------------------
    # Run one outer path (indexed by outer_idx). We rollout until the largest
    # value of k_points and compute the bias term along the way.

    # 30 points on the linear scale, 30 points on the geom scale
    n_start = n0 + 100
    n_end = n0 + n_horizon
    n_points_lin = np.rint(np.linspace(n_start, n_end, n_grid_size)).astype(int)
    n_points_geom = np.rint(np.geomspace(n_start, n_end, n_grid_size)).astype(int)
    n_points = np.unique(np.concatenate([n_points_lin, n_points_geom]))
    logging.info(f"n_points: {n_points.tolist()}")
    logging.info(f"Number of n_points: {len(n_points)}")

    key_outer = jr.fold_in(key_outer, outer_idx)
    save_path = savedir / f"outer-{outer_idx}"
    start = timer()
    run_single_outer_path(
        key_outer,
        clf,
        sample_x,
        t,
        x_new,
        x_prev,
        y_prev,
        n_points,
        mc_delta,
        save_path,
    )
    logging.info(f"outer-{outer_idx}: {timer() - start:.2f} secs")


if __name__ == "__main__":
    OmegaConf.register_new_resolver("githash", utils.githash)
    main()

# %%
