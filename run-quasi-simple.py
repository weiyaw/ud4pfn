import logging
import os
from pathlib import Path
from timeit import default_timer as timer

import hydra
import jax.random as jr
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from scipy.special import expit
from tqdm import trange

import utils
from pred_rule import TabPFNClassifierPPD, assert_ppd_args_shape

os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"

SUPPORT_X = np.array([-1, 0, 1, 2]).reshape(-1, 1)
PMF_X = np.array([0.25, 0.25, 0.25, 0.25])


def compute_delta_and_weight(clf, t, x_new, x_prev, y_prev):
    """
    Return necessary stuff to compute E[Δ_n(x_new, t) given x_{1:n-1},
    y_{1:n-1}] exactly for the event A = {y=t}.  Here, Δ_n(x_new, t) = P_n(x_new, t)
    - P_{n-1}(x_new, t) with P_n(x_new, t) = P( A | x_new, x_{1:n}, y_{1:n}) and
    P_{n-1}(x_new, t) = P( A | x_new, x_{1:n-1}, y_{1:n-1}).

    Here, we assume P(X = x) is uniform over 4 values: -1, 0, 1, 2, Y can take 2
    values: 0, 1, and clf is a classifier.

    Parameters
    ----------
    clf: TabPFNClassifierPPD
        PPD classifier
    t: (p, ) array
        Event of the PPD.
    x_new : (m, d) array
        Query covariates.
    x_prev : (n, d) array
        Historical covariates x_{1:n-1}.
    y_prev : (n,) array
        Historical targets y_{1:n-1}.

    Return:
    -------
    (8, p, m) array Samples of Δ_n(x_new, t).  Taking the mean along the
    0th-axis will give a Monte Carlo estimate of E_n[Δ_n(x_new, t)].

    (8,) array Corresponding weights for the samples.
    """

    P_n = clf.predict_event(t, x_new, x_prev, y_prev)  # (p, m)
    x_curr = SUPPORT_X
    y_curr = np.array([0, 1])
    cond_y_x = clf.pmf(y_curr, x_curr, x_prev, y_prev)  # (2, 4)
    joint_y_x = cond_y_x * PMF_X

    assert x_curr.ndim == 2
    assert y_curr.ndim == 1

    deltas = []
    weights = []
    for i, y in enumerate(y_curr):
        for j, x in enumerate(x_curr):
            x_plus_1 = np.vstack([x_prev, x])
            y_plus_1 = np.append(y_prev, y)
            P_n_plus_1 = clf.predict_event(t, x_new, x_plus_1, y_plus_1)  # (p, m)
            deltas.append(P_n_plus_1 - P_n)
            weights.append(joint_y_x[i, j])

    return np.stack(deltas, axis=0), np.stack(weights, axis=0)  # (2*4, p, m), (2*4, )


def run_single_outer_path(key, clf, t, x_new, x_init, y_init, n_points, save_path):
    """
    Sample from Δ_n(x_new, t) along the rollout trajectory where Δ_n(x_new, t) =
    P_n(x_new, t) - P_{n-1}(x_new, t) with P_n(x_new, t) = P( A | x_new,
    x_{1:n}, y_{1:n}) and P_{n-1}(x_new, t) = P( A | x_new, x_{1:n-1},
    y_{1:n-1}).

    The function will start the rollout with some initial dataset x_init and
    y_init, then rollout up to max(n_points) - 1. After the rollout, the
    function will return objects for computing E[Δ_n(x_new, t) | x_{1:n-1},
    y_{1:n-1}] at all the n specified in n_points.

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
    For each n, it saves the computation of Δ_n | x_{1:n-1}, y_{1:n-1} and its
    corresponding weight in {save_path}/delta-{n}.pickle as a dictionary with
    keys "delta", "weight", "x_new", "t", and "n". It skips the computation if the file
    already exists.
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
            x_curr = SUPPORT_X[jr.choice(subkey_x, SUPPORT_X.shape[0], (1,), p=PMF_X)]
            y_curr, _ = clf.sample(subkey_y, x_curr, x_rollout[:i], y_rollout[:i])
            x_rollout[i] = x_curr
            y_rollout[i] = y_curr.squeeze()
        assert x_rollout.shape[0] == y_rollout.shape[0] == rollout_depth + n0
        utils.write_to(rollout_path, {"x": x_rollout, "y": y_rollout})
        logging.info(f"rollout: {timer() - start:.2f} secs")

    start = timer()
    for n in n_points:
        delta_path = save_path / f"delta-{n}.pickle"
        if os.path.exists(delta_path):
            logging.info(f"delta-{n} exists")
        else:
            x_prev, y_prev = x_rollout[: n - 1], y_rollout[: n - 1]
            deltas_n, weights = compute_delta_and_weight(clf, t, x_new, x_prev, y_prev)
            assert deltas_n.shape == (8, t.shape[0], x_new.shape[0])
            assert weights.shape == (8,)
            utils.write_to(
                delta_path,
                {"delta": deltas_n, "weight": weights, "x_new": x_new, "t": t, "n": n},
            )
    logging.info(f"delta: {timer() - start:.2f} secs")


def generate_initial_context(key_x, key_y, n0):
    # Simple 1D logistic regression
    x_init = SUPPORT_X[jr.choice(key_x, len(SUPPORT_X), (n0,), p=PMF_X)]
    y_init = jr.bernoulli(key_y, expit(-0.5 + 2 * x_init))
    y_init = y_init.squeeze().astype(int)
    return x_init, y_init


@hydra.main(version_base=None, config_path="conf", config_name="quasi-simple")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    logging.info(f"Hydra version: {hydra.__version__}")
    logging.info(OmegaConf.to_yaml(cfg))

    n_estimators = int(cfg.n_estimators)
    n0 = int(cfg.n0)
    n_grid_size = int(cfg.n_grid_size)
    n_horizon = int(cfg.n_horizon)
    outer_idx = int(cfg.outer_idx)
    seed = int(cfg.seed)

    torch.set_num_threads(1)

    # reproducibility
    key = jr.key(seed)
    key, key_outer, key_setup = jr.split(key, 3)
    torch.manual_seed(seed)

    if cfg.fix_data:
        # fix initial context across rollouts/outer paths
        _, subkey_x, subkey_y = jr.split(key_setup, 3)
        x_init, y_init = generate_initial_context(subkey_x, subkey_y, n0)

    # x_new is a grid
    x_new = np.array([-1.0, 0.0, 1.0]).reshape(-1, 1)
    t = np.array([0, 1])
    assert x_new.ndim == 2 and t.ndim == 1

    savedir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logging.info(f"Experiment directory: {savedir}")

    # initialize Classifier
    clf = TabPFNClassifierPPD(
        n_estimators=n_estimators,
        softmax_temperature=1.0,
        fit_mode="low_memory",
        model_path="tabpfn-model/tabpfn-v2.5-classifier-v2.5_default.ckpt",
    )

    # Run one outer path (indexed by outer_idx). We rollout until the largest
    # value of n_points and compute the bias term along the way.
    n_start = n0 + 100
    n_end = n0 + n_horizon
    n_head = np.arange(n0, n_start, 5).astype(int)
    n_tail = np.rint(np.geomspace(n_start, n_end, n_grid_size)).astype(int)
    n_points = np.unique(np.concatenate([n_head, n_tail]))
    logging.info(f"n_points: {n_points.tolist()}")
    logging.info(f"Number of n_points: {len(n_points)}")

    if not cfg.fix_data:
        # for each rollout, generate some fresh initial context
        key_x, key_y = jr.split(jr.fold_in(key_setup, outer_idx))
        x_init, y_init = generate_initial_context(key_x, key_y, n0)

    key_outer = jr.fold_in(key_outer, outer_idx)
    save_path = savedir / f"outer-{outer_idx}"
    start = timer()
    run_single_outer_path(key_outer, clf, t, x_new, x_init, y_init, n_points, save_path)
    logging.info(f"outer-{outer_idx}: {timer() - start:.2f} secs")


if __name__ == "__main__":
    OmegaConf.register_new_resolver("githash", utils.githash)
    main()

# %%
