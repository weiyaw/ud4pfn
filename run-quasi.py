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

from pred_rule import TabPFNClassifierPPD, TabPFNRegressorPPD
from constants import REGRESSION, CLASSIFICATION, Y_STAR_MAP

os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"


def inner_mc_delta(key, clf, t, x_new, x_prev, y_prev, m_inner):
    """
    Monte Carlo estimate of | E_n[Δ_{n+1}] | for the event A = {y=t} or {y <= t}, depending on clf.
    Here Δ_{n+1} = P_{n+1} - P_n with P_n = P( A | x_new, x_prev, y_prev).

    This is similar to calling compute_gn and sample_gn_plus_1, except that x_new is always used for growing x.
    """
    P_t = clf.predict_event(t, x_new, x_prev, y_prev)
    key, subkey = jr.split(key)
    y_new, _ = clf.sample(subkey, x_new, x_prev, y_prev, size=m_inner)

    deltas = []
    for i in trange(y_new.shape[0], desc="inner MC", leave=False):
        # Derive subkey from base key using loop index i
        # subkey = jr.fold_in(key, i)
        # y_new, _ = clf.sample(subkey, x_new, x_prev, y_prev)
        X_br = np.vstack([x_prev, x_new])
        y_br = np.append(y_prev, y_new[i])

        P_next_t = clf.predict_event(t, x_new, X_br, y_br)
        deltas.append(P_next_t - P_t)

    return abs(np.mean(deltas))

def inner_mc_delta2(key, clf, t, x_new, x_prev, y_prev, m_inner):
    """
    Monte Carlo estimate of | E_n[Δ_{n+1}] | for the event A = {y=t} or {y <= t}, depending on clf.
    Here Δ_{n+1} = P_{n+1} - P_n with P_n = P( A | x_new, x_prev, y_prev).

    This version use Bayesian bootstrap to draw future x_plus_1.
    """
    P_n = compute_gn(clf, t, x_new, x_prev, y_prev) # (p, m)
    P_n_plus_1 = sample_gn_plus_1(key, clf, t, x_new, x_prev, y_prev, m_inner) # (mc_samples, p, m)
    return abs(np.mean(P_n_plus_1 - P_n, axis=0))

def run_single_outer_path(key, clf, t, x_new, x_prev, y_prev, k_samp, m_inner):

    delta_vals = []
    prev_k = 0

    for k in tqdm(k_samp, desc="outer", position=0):
        key_outer = jr.fold_in(key, k)
        key_outer, key_growth = jr.split(key_outer)

        # rollout to length k by repeatedly appending (x_new, y_new)
        for j in range(prev_k, k):
            subkey = jr.fold_in(key_growth, j)
            y_new, _ = clf.sample(subkey, x_new, x_prev, y_prev)
            x_prev = np.vstack([x_prev, x_new])
            y_prev = np.append(y_prev, y_new)

        # conditional delta for event
        key_outer, subkey = jr.split(key_outer)
        delta_k = inner_mc_delta(subkey, clf, t, x_new, x_prev, y_prev, m_inner)
        delta_vals.append(delta_k)
        prev_k = k

    return np.array(delta_vals)


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
    x_new = np.zeros((1, x_prev.shape[1]), dtype=np.float32)
    t = np.array([Y_STAR_MAP[setup_name]])
    if setup_name == "gaussian-linear-susan":
        x_new = np.array([0.5, 0.5])
        t = np.array([2.5])

    savedir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logging.info(f"Experiment directory: {savedir}")

    # ------------------------------------------------------------
    # 2.  Initialize Classifier/Regressor
    # ------------------------------------------------------------

    if setup_name in REGRESSION:
        clf = TabPFNRegressorPPD(
            n_estimators=n_estimators,
            softmax_temperature=1.0,
            model_path="tabpfn-model/tabpfn-v2-regressor.ckpt",
        )
    elif setup_name in CLASSIFICATION:
        clf = TabPFNClassifierPPD(
            n_estimators=n_estimators,
            softmax_temperature=1.0,
            model_path="tabpfn-model/tabpfn-v2-classifier.ckpt",
        )
    else:
        raise ValueError(f"Unknown setup {setup_name}")

    # ------------------------------------------------------------
    # 3.  Run A Single Outer Path (Quasi-Martingale Check)
    # ------------------------------------------------------------
    # Run one outer path (indexed by outer_idx). We simulate adding data points (up to n_horizon)
    # and compute the delta term.

    k_samp = np.unique(np.logspace(np.log10(1), np.log10(n_horizon), 10).astype(int))
    np.save(savedir / "k_samp.npy", k_samp)
    logging.info(f"k_samp shape: {k_samp.shape}")

    loopkey_outer = jr.fold_in(key_outer, outer_idx)
    tag = f"outer{outer_idx}"
    chk_path = savedir / f"{tag}.npy"
    if chk_path.exists():
        logging.info(f"{tag} already exists.")
    else:
        logging.info(f"Fresh run for {tag}.")
        start = timer()
        delta_val = run_single_outer_path(
            loopkey_outer, clf, t, x_new, x_prev, y_prev, k_samp, m_inner
        )

        np.save(savedir / f"{tag}.npy", delta_val)
        logging.info(f"Built {tag} in {timer() - start:.2f} seconds")


if __name__ == "__main__":
    OmegaConf.register_new_resolver("githash", utils.githash)
    main()
