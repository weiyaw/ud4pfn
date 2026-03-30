import logging
import os
import re
from timeit import default_timer as timer

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

import posterior
import utils
from constants import CLASSIFICATION, REGRESSION
from pred_rule import TabPFNClassifierPPD, TabPFNRegressorPPD

os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"


def parse_from_outdir(outdir: str, key: str) -> str:
    match = re.search(rf"{key}=([^\s]+)", outdir)
    if not match:
        raise ValueError(f"Could not parse '{key}' from outdir: {outdir}")
    return match.group(1)


def get_clf(setup_name: str, n_estimators: int):
    if setup_name in REGRESSION:
        return TabPFNRegressorPPD(
            n_estimators=n_estimators,
            softmax_temperature=1.0,
            fit_mode="low_memory",
            model_path="tabpfn-model/tabpfn-v2.5-regressor-v2.5_default.ckpt",
        )
    if setup_name in CLASSIFICATION:
        return TabPFNClassifierPPD(
            n_estimators=n_estimators,
            softmax_temperature=1.0,
            fit_mode="low_memory",
            model_path="tabpfn-model/tabpfn-v2.5-classifier-v2.5_default.ckpt",
        )
    raise ValueError(f"Unknown setup '{setup_name}'")


def compute_bootstrap_predictions(
    clf,
    t,
    x_grid,
    x_prev,
    y_prev,
    n_bootstrap: int,
    seed: int,
):
    """Return bootstrap predictions with shape (B, p, m)."""
    n = x_prev.shape[0]
    rng = np.random.default_rng(seed)
    preds = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        g_boot = posterior.compute_gn(
            clf=clf,
            t=t,
            x_grid=x_grid,
            x_prev=x_prev[idx],
            y_prev=y_prev[idx],
        )
        preds.append(g_boot)

    return np.stack(preds, axis=0)


def save_bootstrap_samples_for_rep(rep_dir: str, cfg: DictConfig) -> None:
    outpath = f"{rep_dir}/bootstrap_samples.pickle"
    if os.path.exists(outpath) and not bool(cfg.overwrite):
        logging.info(f"Skipping existing {outpath}")
        return

    rep_data = utils.read_from(f"{rep_dir}/data.pickle")

    setup_name = parse_from_outdir(rep_dir, "setup")
    n_estimators = int(parse_from_outdir(rep_dir, "n_est"))
    seed = int(parse_from_outdir(rep_dir, "seed"))

    clf = get_clf(setup_name, n_estimators)

    start = timer()
    bootstrap_preds = compute_bootstrap_predictions(
        clf=clf,
        t=rep_data["t"],
        x_grid=rep_data["x_grid"],
        x_prev=rep_data["x_prev"],
        y_prev=rep_data["y_prev"],
        n_bootstrap=int(cfg.bootstrap_samples),
        seed=seed + int(cfg.seed_offset),
    )

    elapsed = timer() - start
    utils.write_to_local(
        outpath,
        {
            "bootstrap_samples": int(cfg.bootstrap_samples),
            "seed_offset": int(cfg.seed_offset),
            "predictions": bootstrap_preds,
            "elapsed_seconds": elapsed,
        },
    )
    logging.info(f"Saved bootstrap samples for {rep_dir} in {elapsed:.2f}s")


@hydra.main(version_base=None, config_path="conf", config_name="bootstrap")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    logging.info(f"Hydra version: {hydra.__version__}")
    logging.info(OmegaConf.to_yaml(cfg))

    base_dir = os.path.join(cfg.outputs_root, cfg.id)
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Output id directory not found: {base_dir}")

    rep_dirs = utils.get_matching_dirs(base_dir, cfg.include_regex)
    rep_dirs = [
        d
        for d in rep_dirs
        if os.path.exists(f"{d}/data.pickle") and os.path.exists(f"{d}/gn.pickle")
    ]
    rep_dirs.sort()

    if not rep_dirs:
        raise RuntimeError(f"No repetition directories found in {base_dir}")

    logging.info(f"Computing bootstrap samples for {len(rep_dirs)} runs")
    for rep_dir in rep_dirs:
        save_bootstrap_samples_for_rep(rep_dir, cfg)


if __name__ == "__main__":
    OmegaConf.register_new_resolver("githash", utils.githash)
    main()
