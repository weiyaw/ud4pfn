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

from constants import REGRESSION, CLASSIFICATION, T_MAP
from pred_rule import TabPFNClassifierPPD, TabPFNRegressorPPD, assert_ppd_args_shape

os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"


def inner_mc_delta(key, clf, t, x_new, x_prev, y_prev, m_inner):
    """
    Monte Carlo estimate of | E_n[Δ_{n+1}] | for the event A = {y=t} or {y <= t}, depending on clf.
    Here Δ_{n+1} = P_{n+1} - P_n with P_n = P( A | x_new, x_prev, y_prev).

    This is similar to calling compute_gn and sample_gn_plus_1, except that x_new is always used for growing x.

    Parameters
    ----------
    key: jax.random.PRNGKey
        Random key for sampling.
    clf: TabPFNClassifierPPD or TabPFNRegressorPPD
        PPD classifier or regressor.
    t: (p, ) array
        Event of the PPD.
    x_new : (m, d) array
        Query covariates.
    x_prev : (n, d) array
        Historical covariates.
    y_prev : (n,) array
        Historical targets.

    Return:
    -------
    (p, m) array
        Monte Carlo estimate of | E_n[Δ_{n+1}] | for the event A = {y=t} or {y <= t}, depending on clf.
    """
    P_n = clf.predict_event(t, x_new, x_prev, y_prev)  # (p, m)
    key, subkey = jr.split(key)
    y_new, _ = clf.sample(subkey, x_new, x_prev, y_prev, size=m_inner)  # (m_inner, )
    assert y_new.shape[0] == m_inner

    deltas = []
    for i in trange(y_new.shape[0], desc="inner MC", leave=False):
        x_plus_1 = np.vstack([x_prev, x_new])
        y_plus_1 = np.append(y_prev, y_new[i])
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


def run_single_outer_path(key, clf, t, x_new, x_init, y_init, k_samp, m_inner):
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
    t: (p, ) array
        Event of the PPD.
    x_new : (1, d) array
        Query covariates.
    x_init : (n, d) array
        Initial dataset.
    y_init : (n,) array
        Initial dataset.
    k_samp : list of int of length K
        List of rollout depth the outer path.
    m_inner : int
        Number of samples for the inner Monte Carlo estimate.

    Return:
    -------
    (p, K) array
        Monte Carlo estimate of | E[Δ_{k+1} | x_init, y_init, x_k, y_k] | for
        the event A = {y=t} or {y <= t}, depending on clf.
    """
    delta_path = []
    prev_k = 0
    key_path, key_eval = jr.split(key)
    assert_ppd_args_shape(x_new, x_init, y_init)

    x_prev = x_init
    y_prev = y_init
    for k in tqdm(k_samp, desc="outer", position=0):
        # rollout to length k by repeatedly appending (x_new, y_new) to x_prev
        # and y_prev
        for j in range(prev_k, k):
            subkey = jr.fold_in(key_path, j)
            y_new, _ = clf.sample(subkey, x_new, x_prev, y_prev)
            x_prev = np.vstack([x_prev, x_new])
            y_prev = np.append(y_prev, y_new)

        # compute | E[Δ_{k+1} \mid x_init, y_init, x_k, y_k] |
        subkey = jr.fold_in(key_eval, k)
        delta_k = inner_mc_delta(
            subkey, clf, t, x_new, x_prev, y_prev, m_inner
        )  # (p, 1)
        delta_path.append(delta_k)
        prev_k = k

    delta_path = np.concatenate(delta_path, axis=1)  # (p, K)
    assert delta_path.shape == (t.shape[0], len(k_samp))
    return delta_path


@hydra.main(version_base=None, config_path="conf", config_name="quasi")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    logging.info(f"Hydra version: {hydra.__version__}")
    logging.info(OmegaConf.to_yaml(cfg))

    n_estimators = int(cfg.n_estimators)
    n = int(cfg.n0)
    n_horizon = int(cfg.n_horizon)
    outer_idx = int(cfg.outer_idx)
    m_inner = int(cfg.m_inner)
    seed = int(cfg.seed)
    setup_name = cfg.setup
    x_design = cfg.x_design
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

    # Query covariate: 0.0 for 1D/2D (center of domain)
    x_new = np.zeros((1, x_prev.shape[1]))
    t = np.array(T_MAP[setup_name])

    savedir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logging.info(f"Experiment directory: {savedir}")

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
    # Run one outer path (indexed by outer_idx). We simulate adding data points (up to n_horizon)
    # and compute the delta term.

    k_points = np.unique(np.linspace(0, n_horizon, 41, dtype=int))
    logging.info(f"k_points: {k_points.tolist()}")

    loopkey_outer = jr.fold_in(key_outer, outer_idx)
    tag = f"outer{outer_idx}"
    save_path = savedir / f"{tag}.pickle"
    if save_path.exists():
        logging.info(f"{tag} already exists.")
    else:
        logging.info(f"Fresh run for {tag}.")
        start = timer()
        delta_path = run_single_outer_path(
            loopkey_outer, clf, t, x_new, x_prev, y_prev, k_points, m_inner
        )

        assert delta_path.shape == (t.shape[0], k_points.size)
        utils.write_to(save_path, {"delta": delta_path, "k": k_points, "t": t})
        logging.info(f"Built {tag} in {timer() - start:.2f} seconds")


if __name__ == "__main__":
    OmegaConf.register_new_resolver("githash", utils.githash)
    main()
