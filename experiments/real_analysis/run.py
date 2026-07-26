"""Generate predictive-CLT artifacts for the two real-data illustrations."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer as timer
from typing import Callable, Literal

import hydra
import jax.random as jr
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from experiments._shared.artifacts import write_pickle
from experiments._shared.runtime import (
    CLASSIFIER_CHECKPOINT_PATH,
    register_githash_resolver,
)
from predictive_clt import (
    TabPFNClassifierPPD,
    compute_g0_to_gn,
    compute_gn,
    sample_gn_plus_1,
)

from .data import FibreStrength, LabourForce


@dataclass(frozen=True)
class ExperimentDefinition:
    data_loader: Callable[[bool], object]
    task: Literal["classification", "regression"]
    events: tuple[int | float, ...]
    default_event_index: int


EXPERIMENT_DEFINITIONS = {
    "labour-force": ExperimentDefinition(LabourForce, "classification", (0, 1), 1),
    "fibre-strength": ExperimentDefinition(FibreStrength, "classification", (0, 1), 1),
}


def run_experiment(cfg: DictConfig, output_dir: str | Path) -> Path:
    experiment_name = str(cfg.setup)
    if experiment_name not in EXPERIMENT_DEFINITIONS:
        supported = ", ".join(EXPERIMENT_DEFINITIONS)
        raise ValueError(
            f"Unknown setup '{experiment_name}'. Supported setups: {supported}"
        )

    seed = int(cfg.seed)
    torch.manual_seed(8655 + seed)
    key_others, _ = jr.split(jr.key(1907 + seed))

    definition = EXPERIMENT_DEFINITIONS[experiment_name]
    setup = definition.data_loader(bool(cfg.shuffle_data))
    x_prev = np.asarray(setup.X)
    y_prev = np.asarray(setup.y)
    if x_prev.ndim != 2 or x_prev.shape[1] != 1:
        raise ValueError("Real-analysis runners support one covariate")

    m = int(cfg.x_grid_size)
    x_grid = np.linspace(x_prev.min(), x_prev.max(), m).reshape(-1, 1)
    t = np.asarray(definition.events)
    output_dir = Path(output_dir)

    predictive_rule = TabPFNClassifierPPD(
        n_estimators=int(cfg.n_estimators),
        softmax_temperature=1.0,
        fit_mode="low_memory",
        model_path=str(CLASSIFIER_CHECKPOINT_PATH),
    )
    write_pickle(
        output_dir / "data.pickle",
        {
            "x_prev": x_prev,
            "y_prev": y_prev,
            "t": t,
            "x_grid": x_grid,
            "grid_shape": (m,),
        },
    )

    start = timer()
    gn = compute_gn(predictive_rule, t, x_grid, x_prev, y_prev)
    write_pickle(output_dir / "gn.pickle", gn)
    logging.info("Built gn in %.2f seconds", timer() - start)

    start = timer()
    g0_to_gn = compute_g0_to_gn(predictive_rule, t, x_grid, x_prev, y_prev)
    write_pickle(output_dir / "g0_to_gn.pickle", g0_to_gn)
    logging.info("Built g0_to_gn in %.2f seconds", timer() - start)

    mc_samples = int(cfg.mc_samples)
    if mc_samples > 0:
        gn_plus_1 = sample_gn_plus_1(
            key_others,
            predictive_rule,
            t,
            x_grid,
            x_prev,
            y_prev,
            size=mc_samples,
        )
        write_pickle(output_dir / "gn_plus_1.pickle", gn_plus_1)
    return output_dir


@hydra.main(version_base=None, config_path="conf", config_name="run")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    run_experiment(cfg, output_dir)


if __name__ == "__main__":
    register_githash_resolver()
    main()
