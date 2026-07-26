"""Generate predictive-CLT artifacts for entropic UD experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import hydra
import jax.random as jr
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from experiments._shared.artifacts import write_pickle
from experiments._shared.predictive_rule import build_predictive_rule
from experiments._shared.runtime import register_githash_resolver
from predictive_clt import (
    compute_g0_to_gn,
    compute_gn,
    sample_gn_plus_1,
)

from .data import LogisticLinear, Spiral, TwoMoons1, TwoMoons2


@dataclass(frozen=True)
class ExperimentDefinition:
    data_factory: Callable
    task: Literal["classification", "regression"]
    events: tuple[int | float, ...]
    default_event_index: int


EXPERIMENT_DEFINITIONS = {
    "logistic-linear": ExperimentDefinition(
        LogisticLinear, "classification", (0, 1), 1
    ),
    "two-moons-1": ExperimentDefinition(TwoMoons1, "classification", (0, 1), 1),
    "two-moons-2": ExperimentDefinition(TwoMoons2, "classification", (0, 1), 1),
    "spiral": ExperimentDefinition(Spiral, "classification", (0, 1, 2), 1),
}


def build_evaluation_grid(experiment_name: str, size: int):
    if experiment_name == "logistic-linear":
        grid = np.linspace(-15.0, 15.0, 151).reshape(-1, 1)
        return grid, (151,)
    if experiment_name == "two-moons-1":
        lower, upper = (-1.5, 2.6), (-1.5, 2.6)
    elif experiment_name == "two-moons-2":
        lower, upper = (-3.0, 3.6), (-2.5, 3.1)
    elif experiment_name == "spiral":
        lower, upper = (-4.0, 4.0), (-4.0, 4.0)
    else:
        raise ValueError(f"Unknown setup '{experiment_name}'")
    first = np.linspace(*lower, size)
    second = np.linspace(*upper, size)
    x1, x2 = np.meshgrid(first, second, indexing="ij")
    return np.stack([x1, x2], axis=-1).reshape(-1, 2), (size, size)


def run_experiment(cfg: DictConfig, output_dir: str | Path) -> Path:
    experiment_name = str(cfg.setup)
    if experiment_name not in EXPERIMENT_DEFINITIONS:
        supported = ", ".join(EXPERIMENT_DEFINITIONS)
        raise ValueError(
            f"Unknown setup '{experiment_name}'. Supported setups: {supported}"
        )
    seed = int(cfg.seed)
    torch.manual_seed(8655 + seed)
    key_others, key_setup = jr.split(jr.key(1907 + seed))
    if bool(cfg.fix_data):
        key_setup = jr.key(6683)

    definition = EXPERIMENT_DEFINITIONS[experiment_name]
    x_design = None if cfg.x_design is None else str(cfg.x_design)
    setup = definition.data_factory(
        key_setup, int(cfg.data_size), bool(cfg.shuffle_data), x_design
    )
    x_grid, grid_shape = build_evaluation_grid(experiment_name, int(cfg.x_grid_size))
    t = np.asarray(definition.events)
    x_prev, y_prev = setup.X, setup.y
    output_dir = Path(output_dir)
    predictive_rule = build_predictive_rule(
        str(cfg.pfn), definition.task, int(cfg.n_estimators)
    )
    write_pickle(
        output_dir / "data.pickle",
        {
            "x_prev": x_prev,
            "y_prev": y_prev,
            "t": t,
            "x_grid": x_grid,
            "grid_shape": grid_shape,
            "true_prob": np.stack(
                [setup.get_true_event(x_grid, event) for event in t]
            ),
        },
    )
    gn = compute_gn(predictive_rule, t, x_grid, x_prev, y_prev)
    write_pickle(output_dir / "gn.pickle", gn)
    g0_to_gn = compute_g0_to_gn(predictive_rule, t, x_grid, x_prev, y_prev)
    write_pickle(output_dir / "g0_to_gn.pickle", g0_to_gn)
    if int(cfg.mc_samples) > 0:
        gn_plus_1 = sample_gn_plus_1(
            key_others,
            predictive_rule,
            t,
            x_grid,
            x_prev,
            y_prev,
            size=int(cfg.mc_samples),
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
