import warnings
import os
from timeit import default_timer as timer
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import jax
import jax.numpy as jnp
import numpy as np
import logging
import time

from jaxtyping import ArrayLike

import utils

from pred_rule import TabPFNRegressorPPD
from pr_copula.main_copula_regression_conditional import (
    fit_copula_cregression,
    predict_copula_cregression,
)
from pr_copula import sample_copula_regression_functions as samp_mvcr

jax.config.update("jax_enable_x64", True)


def _prepare_copula_cregression_inputs(
    x_grid: ArrayLike,
    t_grid: ArrayLike,
):
    t_mesh_idx, x_mesh_idx = jnp.meshgrid(
        jnp.arange(jnp.shape(t_grid)[0]),
        jnp.arange(jnp.shape(x_grid)[0]),
        indexing="ij",
    )
    x_mesh_flat = x_grid[x_mesh_idx.reshape(-1)]
    t_mesh_flat = t_grid[t_mesh_idx.reshape(-1)]
    return x_mesh_flat, t_mesh_flat


def _predictive_resample_cregression_with_init(
    logcdf: ArrayLike,
    logpdf: ArrayLike,
    x: ArrayLike,
    x_test: ArrayLike,
    rho_opt: ArrayLike,
    rho_x_opt: ArrayLike,
    B_postsamples: int,
    T_fwdsamples: int = 5000,
    seed: int = 100,
):
    key = jax.random.PRNGKey(seed)
    key, *subkeys = jax.random.split(key, B_postsamples + 1)
    subkeys = jnp.asarray(subkeys)

    n = jnp.shape(x)[0]
    logging.info("Predictive resampling...")
    start = time.time()
    logcdf_conditionals_pr, logpdf_joints_pr = (
        samp_mvcr.predictive_resample_loop_cregression_B(
            subkeys,
            logcdf,
            logpdf,
            x,
            x_test,
            rho_opt,
            rho_x_opt,
            n,
            T_fwdsamples,
        )
    )
    logcdf_conditionals_pr = logcdf_conditionals_pr.block_until_ready()
    end = time.time()
    logging.info("Predictive resampling time: {}s".format(round(end - start, 3)))
    return logcdf_conditionals_pr, logpdf_joints_pr


def _tabpfn_init_cregression(
    x_train: ArrayLike,
    y_train: ArrayLike,
    x_test: ArrayLike,
    y_test: ArrayLike,
):

    regressor = TabPFNRegressorPPD(
        n_estimators=16,
        softmax_temperature=1.0,
        fit_mode="low_memory",
        model_path="tabpfn-model/tabpfn-v2.5-regressor-v2.5_default.ckpt",
        device="cpu",
    )
    regressor.fit(np.asarray(x_train), np.asarray(y_train))

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="overflow encountered in cast",
            category=RuntimeWarning,
        )
        pred_output = regressor.predict(np.asarray(x_test), output_type="full")

    bardist = pred_output["criterion"]
    logits = pred_output["logits"]
    bardist.borders = bardist.borders.cpu()

    y_test_torch = torch.tensor(np.asarray(y_test), dtype=torch.float32)[:, None]
    logcdf = np.log(bardist.cdf(logits, y_test_torch))
    logpdf = np.log(bardist.pdf(logits, y_test_torch))

    return (
        jnp.asarray(logcdf, dtype=jnp.float64),
        jnp.asarray(logpdf, dtype=jnp.float64)[:, None],
    )


def copula_regression(
    x_prev: ArrayLike,
    y_prev: ArrayLike,
    rollout_times: int,
    rollout_length: int,
    x_grid: ArrayLike,
    t_grid: ArrayLike,
) -> tuple[ArrayLike, Any]:

    # Return logcdf which is log P_N(y < t_grid | x_grid) with shape
    # (rollout_times, t_grid.shape[0], x_grid.shape[0])

    x_mesh_flat, t_mesh_flat = _prepare_copula_cregression_inputs(x_grid, t_grid)
    copula_cregression_obj = fit_copula_cregression(
        y_prev, x_prev, single_x_bandwidth=False, n_perm_optim=10, n_perm=10
    )
    n = len(y_prev)
    logging.info("Bandwidth is {}".format(copula_cregression_obj.rho_opt))
    logging.info("Bandwidth is {}".format(copula_cregression_obj.rho_x_opt))
    logging.info("Preq loglik is {}".format(copula_cregression_obj.preq_loglik / n))

    logcdf_init, logpdf_init = _tabpfn_init_cregression(
        x_prev, y_prev, x_mesh_flat, t_mesh_flat
    )

    logcdf, _ = _predictive_resample_cregression_with_init(
        logcdf_init,  # (rollout_times, t_grid.shape[0] * x_grid.shape[0])
        logpdf_init,  # (rollout_times, t_grid.shape[0] * x_grid.shape[0], 1)
        x_prev,
        x_mesh_flat,
        copula_cregression_obj.rho_opt,
        copula_cregression_obj.rho_x_opt,
        B_postsamples=rollout_times,
        T_fwdsamples=rollout_length,
    )

    logcdf = logcdf.reshape(rollout_times, jnp.shape(t_grid)[0], jnp.shape(x_grid)[0])

    # y_samp, x_samp = _draw_cregression_samples_from_logcdf(
    #     key, logcdf, y_pr, x_pr, rollout_times
    # )

    # y_samp = batched_inverse_transform(y_encoder, y_samp[..., np.newaxis])
    # x_samp = batched_inverse_transform(x_encoder, x_samp)
    # recursion_data = {"y": y_samp.squeeze(-1), "x": x_samp}
    return logcdf, copula_cregression_obj


def save_copula_samples_for_rep(rep_dir: str, cfg: DictConfig) -> None:
    """Compute and save copula regression logcdf for a given repetition directory."""
    outpath = f"{rep_dir}/copula-{cfg.rollout_times}-{cfg.rollout_length}.pickle"
    if os.path.exists(outpath) and not bool(cfg.overwrite):
        logging.info(f"Skipping existing {outpath}")
        return

    rep_data = utils.read_from(f"{rep_dir}/data.pickle")

    logging.info(f"Computing copula regression for {rep_dir}.")
    start = timer()
    logcdf, _ = copula_regression(
        x_prev=rep_data["x_prev"],
        y_prev=rep_data["y_prev"],
        rollout_times=int(cfg.rollout_times),
        rollout_length=int(cfg.rollout_length),
        x_grid=rep_data["x_grid"],
        t_grid=rep_data["t"],
    )
    # logcdf: (rollout_times, t_grid.shape[0], x_grid.shape[0])
    elapsed = timer() - start
    utils.write_to_local(
        outpath,
        {
            "logcdf": np.asarray(logcdf),
            "rollout_times": int(cfg.rollout_times),
            "rollout_length": int(cfg.rollout_length),
            "seed": int(cfg.seed),
            "elapsed_seconds": elapsed,
        },
    )
    logging.info(f"Saved copula samples for {rep_dir} in {elapsed:.2f}s")


@hydra.main(version_base=None, config_path="conf", config_name="copula")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    logging.info(f"Hydra version: {hydra.__version__}")
    logging.info(OmegaConf.to_yaml(cfg))

    rep_dir = cfg.rep_dir
    if not os.path.exists(rep_dir):
        raise FileNotFoundError(f"Repetition directory not found: {rep_dir}")

    if not os.path.exists(f"{rep_dir}/data.pickle"):
        raise FileNotFoundError(f"data.pickle not found in {rep_dir}")

    save_copula_samples_for_rep(rep_dir, cfg)


if __name__ == "__main__":
    OmegaConf.register_new_resolver("githash", utils.githash)
    main()
