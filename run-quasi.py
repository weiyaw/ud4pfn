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


def inner_mc_bias(key, clf, X, y, x_new, m_inner):
    """
    Monte Carlo estimate of | E_n[Δ_{n+1}] | for the event {y=0}.
    Here Δ_{n+1} = P_{n+1} - P_n with P_n = P(y=0 | x_new, G_n).
    """
    P_t = clf.predict_event(x_new, X, y)

    diffs = []
    for i in trange(m_inner, desc="inner MC", leave=False):
        # Derive subkey from base key using loop index i
        subkey = jr.fold_in(key, i)
        y_branch, _ = clf.sample(subkey, x_new, X, y)
        X_br = np.vstack([X, x_new])
        y_br = np.append(y, y_branch)

        P_next_t = clf.predict_event(x_new, X_br, y_br)
        diffs.append(P_next_t - P_t)

    return abs(np.mean(diffs))


def run_single_outer_path(key, clf, X0, y0, k_samp, m_inner, x_new):

    bias_vals = []
    prev_k = 0
    X, y = X0.copy(), y0.copy()

    for k in tqdm(k_samp, desc="outer", position=0):
        key_outer = jr.fold_in(key, k)
        key_outer, key_growth = jr.split(key_outer)

        # grow history to length k by repeatedly appending (x_new, y_new)
        for j in range(prev_k, k):
            subkey = jr.fold_in(key_growth, j)
            y_new, _ = clf.sample(subkey, x_new, X, y)
            X = np.vstack([X, x_new])
            y = np.append(y, y_new)

        # conditional bias for event
        key_outer, subkey = jr.split(key_outer)
        b_k = inner_mc_bias(subkey, clf, X, y, x_new=x_new, m_inner=m_inner)
        bias_vals.append(b_k)
        prev_k = k

    return np.array(bias_vals)


@hydra.main(version_base=None, config_path="conf", config_name="quasi")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    logging.info(f"Hydra version: {hydra.__version__}")
    logging.info(OmegaConf.to_yaml(cfg))

    n_estimators = int(cfg.n_estimators)
    n = int(cfg.n0)
    n_horizon = int(cfg.n_horizon)
    r_outer = int(cfg.r_outer)
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

    k_samp = np.unique(
        np.concatenate(
            [np.arange(1, 21), np.arange(25, 101, 5), np.arange(120, n_horizon + 1, 20)]
        )
    ).astype(int)

    logging.info(f"k_samp shape: {k_samp.shape}")

    # ------------------------------------------------------------
    # 1. Load Data and Setup
    # ------------------------------------------------------------
    # convert setup (kebab-case) to Camalcase and ensure PPD suffix
    try:
        Setup = getattr(data, utils.kebab_to_camel(setup_name))
        setup = Setup(key_setup, n, shuffle_data, x_design)
    except AttributeError:
        raise ValueError(f"Data {setup_name} not found in data module")

    X = setup.X
    y = setup.y

    # Query covariate: 0.0 for 1D/2D (center of domain)
    x_new = np.zeros((1, X.shape[1]), dtype=np.float32)

    savedir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logging.info(f"Experiment directory: {savedir}")

    # ------------------------------------------------------------
    # 2.  Initialize Classifier/Regressor
    # ------------------------------------------------------------

    if setup_name in REGRESSION:
        clf = TabPFNRegressorPPD(
            y_star=Y_STAR_MAP[setup_name],
            n_estimators=n_estimators,
            model_path="tabpfn-model/tabpfn-v2-regressor.ckpt",
        )
    elif setup_name in CLASSIFICATION:
        clf = TabPFNClassifierPPD(
            y_star=Y_STAR_MAP[setup_name],
            n_estimators=n_estimators,
            model_path="tabpfn-model/tabpfn-v2-classifier.ckpt",
        )
    else:
        raise ValueError(f"Unknown setup {setup_name}")

    # ------------------------------------------------------------
    # 3.  Run Outer Paths (Quasi-Martingale Check)
    # ------------------------------------------------------------
    # We run multiple outer paths (r_outer). For each path, we simulate
    # adding data points (up to n_horizon) and compute the delta/bias term.
    bias_per_path = []
    for r in range(r_outer):
        loopkey_outer = jr.fold_in(key_outer, r)
        tag = f"outer{r}"
        chk_path = savedir / f"{tag}.npy"
        if chk_path.exists():
            logging.info(f"Loading {tag} from checkpoint.")
            bias_per_path.append(np.load(chk_path))
        else:
            logging.info(f"Fresh run for {tag}.")
            start = timer()
            bias_val = run_single_outer_path(
                loopkey_outer,
                clf,
                X,
                y,
                k_samp,
                m_inner=m_inner,
                x_new=x_new,
            )

            np.save(savedir / f"{tag}.npy", bias_val)
            logging.info(f"Built {tag} in {timer() - start:.2f} seconds")

    # bias_per_path = np.vstack(bias_per_path)  # (r_outer, len(k_samp))
    # bias_mean = bias_per_path.mean(axis=0)
    # partial       = np.cumsum(np.sqrt(n + k_samp + 1) * np.abs(bias_mean)) # This line was incomplete in original file


if __name__ == "__main__":
    OmegaConf.register_new_resolver("githash", utils.githash)
    main()
