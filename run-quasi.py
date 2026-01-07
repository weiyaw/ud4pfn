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


def inner_mc_delta(key, clf, sample_x, t, x_new, x_prev, y_prev, mc_inner):
    """
    Compute | E_n[Δ_{n+1}] | for the event A = {y=t} or {y <= t}, depending on
    clf. Here Δ_{n+1} = P_{n+1} - P_n with P_n = P( A | x_new, x_prev, y_prev).

    This is similar to calling compute_gn and sample_gn_plus_1, except that
    x_new is always used for growing x.

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
        Historical covariates.
    y_prev : (n,) array
        Historical targets.
    mc_inner : int
        Number of Monte Carlo samples for the inner Monte Carlo estimate.

    Return:
    -------
    (p, m) array
        A Monte Carlo estimate of | E_n[Δ_{n+1}] | for the event A = {y=t} or {y
        <= t}, depending on clf.
    """
    P_n = clf.predict_event(t, x_new, x_prev, y_prev)  # (p, m)
    key, subkey = jr.split(key)
    x_curr = sample_x(subkey, mc_inner, x_prev)  # (mc_inner, d)
    y_curr, _ = clf.sample(subkey, x_curr, x_prev, y_prev, size=1)  # (1, mc_inner, )
    assert x_curr.shape[0] == y_curr.shape[1] == mc_inner

    deltas = []
    for i in trange(mc_inner, desc="inner MC", leave=False):
        x_plus_1 = np.vstack([x_prev, x_curr[i]])
        y_plus_1 = np.append(y_prev, y_curr[0, i])
        P_n_plus_1 = clf.predict_event(t, x_new, x_plus_1, y_plus_1)  # (p, m)
        deltas.append(P_n_plus_1 - P_n)

    return abs(np.mean(deltas, axis=0))  # (p, m)


def inner_mc_delta2(key, clf, t, x_new, x_prev, y_prev, m_inner):
    """
    Monte Carlo estimate of | E_n[Δ_{n+1}] | for the event A = {y=t} or {y <=
    t}, depending on clf. Here Δ_{n+1} = P_{n+1} - P_n with P_n = P( A | x_new,
    x_prev, y_prev).

    This version use Bayesian bootstrap to draw future x_plus_1.
    """
    P_n = compute_gn(clf, t, x_new, x_prev, y_prev)  # (p, m)
    P_n_plus_1 = sample_gn_plus_1(
        key, clf, t, x_new, x_prev, y_prev, m_inner
    )  # (mc_samples, p, m)
    return abs(np.mean(P_n_plus_1 - P_n, axis=0))


@jax.jit(static_argnames=["n"])
def sample_x_dirac(key, n, x_prev, x_mass):
    """x_mass is a (d, ) array"""
    return jnp.tile(x_mass, (n, 1))


@jax.jit(static_argnames=["n", "get_x", "x_design"])
def sample_x_truth(key, n, x_prev, get_x, x_design):
    """get_x is a function that returns (n, d) array"""
    return get_x(key, n, x_design)


def run_single_outer_path(
    key, clf, sample_x, t, x_new, x_init, y_init, k_points, mc_inner, save_path
):
    """
    Draw a single sample from the random variable | E[Δ_{k+1} | x_init, y_init,
    x_k, y_k]|, where the randomness of this random variable is coming from
    simulating x_k and y_k. Here Δ_{k+1} = P_{k+1} - P_k with P_k = P( A |
    x_init, y_init, x_k, y_k). The event A = {y=t} or {y <= t}, depends on clf.

    Currently, all x_k are set to x_new.

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
    k_points : list of int of length K
        List of rollout depth the outer path.
    mc_inner : int
        Number of samples for the inner Monte Carlo estimate.
    save_path : Path
        Path to save the computed bias.

    Return:
    -------
    For each k, it saves the computed bias = | E[Δ_{k+1} | x_init, y_init, x_k,
    y_k]| in {save_path}/bias-{k}.pickle as a dictionary with keys "bias", "t",
    and "k". It skips the computation if the file already exists.
    """
    key_path, key_eval = jr.split(key)
    assert_ppd_args_shape(x_new, x_init, y_init)

    prev_k = 0
    x_prev = x_init
    y_prev = y_init
    for k in k_points:
        # rollout to length k by repeatedly appending (x_new, y_new) to x_prev
        # and y_prev
        start = timer()
        for j in trange(prev_k, k, desc=f"Rollout {prev_k}-{k}", leave=False):
            loopkey = jr.fold_in(key_path, j)
            loopkey, subkey_x, subkey_y = jr.split(loopkey, 3)
            x_curr = sample_x(subkey_x, 1, x_prev)
            y_curr, _ = clf.sample(subkey_y, x_curr, x_prev, y_prev)
            assert y_curr.size == 1
            x_prev = np.vstack([x_prev, x_curr])
            y_prev = np.append(y_prev, y_curr)
        logging.info(f"Rollout {prev_k}-{k} takes {timer() - start:.2f} seconds")

        bias_save_path = save_path / f"bias-{k}.pickle"
        if os.path.exists(bias_save_path):
            logging.info(f"bias-{k} exists")
        else:
            # compute | E[Δ_{k+1} \mid x_init, y_init, x_k, y_k] |
            start = timer()
            subkey = jr.fold_in(key_eval, k)
            bias_k = inner_mc_delta(
                subkey, clf, sample_x, t, x_new, x_prev, y_prev, mc_inner
            )  # (p, m)
            assert bias_k.shape == (t.shape[0], x_new.shape[0])  # (p, m)
            utils.write_to(
                bias_save_path, {"bias": bias_k, "x_new": x_new, "t": t, "k": k}
            )
            logging.info(f"bias-{k} takes {timer() - start:.2f} seconds")
        prev_k = k


@hydra.main(version_base=None, config_path="conf", config_name="quasi")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    logging.info(f"Hydra version: {hydra.__version__}")
    logging.info(OmegaConf.to_yaml(cfg))

    n_estimators = int(cfg.n_estimators)
    n = int(cfg.n0)
    rollout_depth = int(cfg.rollout_depth)
    outer_idx = int(cfg.outer_idx)
    mc_inner = int(cfg.mc_inner)
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
        setup = Setup(key_setup, n, shuffle_data, x_design)
    except AttributeError:
        raise ValueError(f"Data {setup_name} not found in data module")

    x_prev = setup.X
    y_prev = setup.y

    # x_new is a grid
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
    # Run one outer path (indexed by outer_idx). We simulate adding data points
    # (up to rollout_depth) and compute the bias term.

    # 15 points on the linear scale, 15 points on the log10 scale
    k_points_lin = np.linspace(0, rollout_depth, 21, dtype=int)
    k_points_log = np.rint(np.geomspace(1, rollout_depth, 21)).astype(int)
    k_points = np.unique(np.concatenate([k_points_lin, k_points_log]))
    logging.info(f"k_points: {k_points.tolist()}")
    logging.info(f"Number of k_points: {len(k_points)}")

    loopkey_outer = jr.fold_in(key_outer, outer_idx)
    save_path = savedir / f"outer-{outer_idx}"
    run_single_outer_path(
        loopkey_outer,
        clf,
        sample_x,
        t,
        x_new,
        x_prev,
        y_prev,
        k_points,
        mc_inner,
        save_path,
    )


if __name__ == "__main__":
    OmegaConf.register_new_resolver("githash", utils.githash)
    main()

# %%
