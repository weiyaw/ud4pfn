"""Generate predictive-CLT artifacts for one-gap experiments."""

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
from experiments._shared.predictive_rule import build_predictive_rule
from experiments._shared.runtime import (
    register_githash_resolver,
)
from predictive_clt import (
    compute_g0_to_gn,
    compute_gn,
    sample_gn_plus_1,
)

from .data import (
    CategoricalLinear,
    GaussianLinear,
    GaussianLinearDependentError,
    GaussianPolynomial,
    GaussianSine,
    PoissonLinear,
    ProbitMixture,
)


@dataclass(frozen=True)
class ExperimentDefinition:
    data_factory: Callable
    task: Literal["classification", "regression"]
    events: tuple[int | float, ...]
    default_event_index: int


_REGRESSION_EVENTS = (-2.0, -1.0, 0.0, 1.0, 2.0)
EXPERIMENT_DEFINITIONS = {
    "gaussian-linear": ExperimentDefinition(
        GaussianLinear, "regression", _REGRESSION_EVENTS, 2
    ),
    "gaussian-polynomial": ExperimentDefinition(
        GaussianPolynomial, "regression", _REGRESSION_EVENTS, 2
    ),
    "gaussian-linear-dependent-error": ExperimentDefinition(
        GaussianLinearDependentError, "regression", _REGRESSION_EVENTS, 2
    ),
    "gaussian-sine": ExperimentDefinition(
        GaussianSine, "regression", _REGRESSION_EVENTS, 2
    ),
    "poisson-linear": ExperimentDefinition(
        PoissonLinear, "regression", (1.0, 2.0, 3.0), 1
    ),
    "probit-mixture": ExperimentDefinition(ProbitMixture, "classification", (0, 1), 1),
    "categorical-linear": ExperimentDefinition(
        CategoricalLinear, "classification", (0, 1), 1
    ),
}


def run_experiment(cfg: DictConfig, output_dir: str | Path) -> Path:
    experiment_name = str(cfg.setup)
    if experiment_name not in EXPERIMENT_DEFINITIONS:
        supported = ", ".join(EXPERIMENT_DEFINITIONS)
        raise ValueError(
            f"Unknown setup '{experiment_name}'. Supported setups: {supported}"
        )
    if str(cfg.x_design) != "one-gap":
        raise ValueError("Gap experiments require x_design='one-gap'")

    seed = int(cfg.seed)
    torch.manual_seed(8655 + seed)
    key_others, key_setup = jr.split(jr.key(1907 + seed))
    if bool(cfg.fix_data):
        key_setup = jr.key(6683)

    definition = EXPERIMENT_DEFINITIONS[experiment_name]
    setup = definition.data_factory(
        key_setup,
        int(cfg.data_size),
        bool(cfg.shuffle_data),
        str(cfg.x_design),
    )
    x_prev, y_prev = setup.X, setup.y
    x_grid = np.linspace(-10, 10, int(cfg.x_grid_size)).reshape(-1, 1)
    t = np.asarray(definition.events)
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
            "grid_shape": (x_grid.shape[0],),
            "true_prob": np.stack(
                [setup.get_true_event(x_grid, event) for event in t]
            ),
        },
    )

    start = timer()
    gn = compute_gn(predictive_rule, t, x_grid, x_prev, y_prev)
    write_pickle(output_dir / "gn.pickle", gn)
    logging.info("Built gn in %.2f seconds", timer() - start)
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
