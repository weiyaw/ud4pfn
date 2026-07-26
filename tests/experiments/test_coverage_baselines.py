import os

import numpy as np
from omegaconf import OmegaConf

from experiments._shared.artifacts import read_pickle, write_pickle
from experiments.coverage import run_bootstrap, run_copula


def make_run(tmp_path):
    run_dir = tmp_path / "setup=gaussian-linear-multivariate n_est=2 n=4 m=3 seed=1000"
    hydra = run_dir / ".hydra"
    hydra.mkdir(parents=True)
    (hydra / "config.yaml").write_text(
        "setup: gaussian-linear-multivariate\n"
        "n_estimators: 2\n"
        "data_size: 4\n"
        "x_grid_size: 3\n"
        "seed: 1000\n"
    )
    write_pickle(
        run_dir / "data.pickle",
        {
            "t": np.array([0.0]),
            "x_grid": np.zeros((3, 10)),
            "x_prev": np.zeros((4, 10)),
            "y_prev": np.arange(4.0),
        },
    )
    return run_dir


def test_bootstrap_artifact_schema_and_overwrite(monkeypatch, tmp_path):
    run_dir = make_run(tmp_path)
    monkeypatch.setattr(run_bootstrap, "build_predictive_rule", lambda *args: object())
    monkeypatch.setattr(
        run_bootstrap,
        "compute_gn",
        lambda predictive_rule, t, x_grid, x_prev, y_prev: np.full(
            (1, 3), y_prev.mean()
        ),
    )
    config = OmegaConf.create(
        {"bootstrap_samples": 5, "seed_offset": 7, "overwrite": True}
    )
    run_bootstrap.save_bootstrap_samples_for_rep(str(run_dir), config)
    artifact_path = run_dir / "bootstrap-5.pickle"
    artifact = read_pickle(artifact_path)

    assert set(artifact) == {
        "bootstrap_samples",
        "seed_offset",
        "predictions",
        "elapsed_seconds",
    }
    assert artifact["predictions"].shape == (5, 1, 3)
    modified = os.path.getmtime(artifact_path)
    config.overwrite = False
    run_bootstrap.save_bootstrap_samples_for_rep(str(run_dir), config)
    assert os.path.getmtime(artifact_path) == modified


def test_copula_artifact_schema(monkeypatch, tmp_path):
    run_dir = make_run(tmp_path)
    monkeypatch.setattr(
        run_copula,
        "copula_regression",
        lambda **kwargs: (np.zeros((2, 1, 3)), object()),
    )
    config = OmegaConf.create(
        {
            "rollout_times": 2,
            "rollout_length": 4,
            "seed": 9,
            "overwrite": True,
        }
    )
    run_copula.save_copula_samples_for_rep(str(run_dir), config)
    artifact = read_pickle(run_dir / "copula-2-4.pickle")

    assert set(artifact) == {
        "logcdf",
        "rollout_times",
        "rollout_length",
        "seed",
        "elapsed_seconds",
    }
    assert artifact["logcdf"].shape == (2, 1, 3)
