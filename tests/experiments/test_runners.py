import logging
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from experiments._shared.artifacts import read_pickle


class DummyPredictiveRule:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def install_stubs(monkeypatch, module):
    monkeypatch.setattr(
        module,
        "build_predictive_rule",
        lambda pfn, task, n_estimators: DummyPredictiveRule(
            pfn=pfn, task=task, n_estimators=n_estimators
        ),
    )

    def gn(predictive_rule, t, x_grid, x_prev, y_prev):
        return np.full((len(t), len(x_grid)), 0.5, dtype=np.float32)

    def g0_to_gn(predictive_rule, t, x_grid, x_prev, y_prev):
        value = np.full((len(y_prev) + 1, len(t), len(x_grid)), 0.5, dtype=np.float32)
        value[0] = np.nan
        return value

    def gn_plus_1(key, predictive_rule, t, x_grid, x_prev, y_prev, size):
        return np.full((size, len(t), len(x_grid)), 0.5, dtype=np.float32)

    monkeypatch.setattr(module, "compute_gn", gn)
    monkeypatch.setattr(module, "compute_g0_to_gn", g0_to_gn)
    monkeypatch.setattr(module, "sample_gn_plus_1", gn_plus_1)


@pytest.mark.parametrize(
    "module_name,setup,extra",
    [
        ("coverage", "gaussian-linear-multivariate", {"x_design": "sobol-10d"}),
        ("gap", "gaussian-linear", {"x_design": "one-gap"}),
        (
            "entropic_ud",
            "logistic-linear",
            {"x_design": "gaussian:1.5:3.0"},
        ),
    ],
)
@pytest.mark.parametrize("pfn", ["tabpfn", "tabicl"])
def test_synthetic_runners_write_preserved_schema(
    monkeypatch, tmp_path, caplog, module_name, setup, extra, pfn
):
    module = __import__(f"experiments.{module_name}.run", fromlist=["run"])
    install_stubs(monkeypatch, module)
    caplog.set_level(logging.INFO)
    data_size = 8 if module_name == "coverage" else 6
    config = OmegaConf.create(
        {
            "setup": setup,
            "pfn": pfn,
            "data_size": data_size,
            "x_grid_size": 4,
            "shuffle_data": True,
            "fix_data": False,
            "seed": 1000,
            "n_estimators": 2,
            "mc_samples": 3,
            **extra,
        }
    )

    module.run_experiment(config, tmp_path)

    assert {path.name for path in tmp_path.iterdir()} == {
        "data.pickle",
        "gn.pickle",
        "g0_to_gn.pickle",
        "gn_plus_1.pickle",
    }
    data = read_pickle(tmp_path / "data.pickle")
    p, m, n = len(data["t"]), len(data["x_grid"]), len(data["y_prev"])
    assert read_pickle(tmp_path / "gn.pickle").shape == (p, m)
    g0_to_gn = read_pickle(tmp_path / "g0_to_gn.pickle")
    assert g0_to_gn.shape == (n + 1, p, m)
    assert g0_to_gn.dtype == np.float32
    assert read_pickle(tmp_path / "gn_plus_1.pickle").shape == (3, p, m)
    assert not (tmp_path / "setup.pickle").exists()
    assert "Built gn in " in caplog.text
    assert "Built g0_to_gn in " in caplog.text
    assert "Built gn_plus_1 in " in caplog.text


def test_entropic_vary_n_omits_monte_carlo_artifact(monkeypatch, tmp_path, caplog):
    from experiments.entropic_ud import run

    install_stubs(monkeypatch, run)
    caplog.set_level(logging.INFO)
    config = OmegaConf.create(
        {
            "setup": "logistic-linear",
            "pfn": "tabpfn",
            "data_size": 6,
            "x_grid_size": 4,
            "x_design": "gaussian:1.5:3.0",
            "shuffle_data": True,
            "fix_data": False,
            "seed": 1000,
            "n_estimators": 2,
            "mc_samples": 0,
        }
    )
    run.run_experiment(config, tmp_path)
    assert not (tmp_path / "gn_plus_1.pickle").exists()
    assert "mc_samples=0, skipping gn_plus_1" in caplog.text
    assert "Built gn_plus_1 in " not in caplog.text


def test_real_runner_supports_both_setups_without_network(
    monkeypatch, tmp_path, caplog
):
    from experiments.real_analysis import run

    install_stubs(monkeypatch, run)
    caplog.set_level(logging.INFO)

    class LocalData:
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([0, 1, 1])

        def __init__(self, shuffle):
            pass

    original = dict(run.EXPERIMENT_DEFINITIONS)
    monkeypatch.setattr(
        run,
        "EXPERIMENT_DEFINITIONS",
        {
            name: replace(definition, data_loader=LocalData)
            for name, definition in original.items()
        },
    )
    for pfn in ("tabpfn", "tabicl"):
        for experiment_name in original:
            output_dir = tmp_path / pfn / experiment_name
            config = OmegaConf.create(
                {
                    "setup": experiment_name,
                    "pfn": pfn,
                    "x_grid_size": 4,
                    "shuffle_data": True,
                    "seed": 1000,
                    "n_estimators": 2,
                    "mc_samples": 3,
                }
            )
            run.run_experiment(config, output_dir)
            assert not (output_dir / "setup.pickle").exists()
            assert read_pickle(output_dir / "data.pickle")["y_prev"].shape == (3,)
            assert read_pickle(output_dir / "gn.pickle").shape == (2, 4)
    assert "Built gn in " in caplog.text
    assert "Built g0_to_gn in " in caplog.text
    assert "Built gn_plus_1 in " in caplog.text


@pytest.mark.parametrize(
    "module_name",
    ["coverage", "gap", "entropic_ud", "real_analysis"],
)
def test_runner_main_logs_resolved_config(
    monkeypatch, tmp_path, caplog, module_name
):
    module = __import__(f"experiments.{module_name}.run", fromlist=["run"])
    config = OmegaConf.create({"setup": "logged-setup"})
    monkeypatch.setattr(module, "run_experiment", lambda cfg, output_dir: None)
    monkeypatch.setattr(
        module.hydra.core.hydra_config.HydraConfig,
        "get",
        lambda: SimpleNamespace(
            runtime=SimpleNamespace(output_dir=str(tmp_path))
        ),
    )
    caplog.set_level(logging.INFO)

    module.main.__wrapped__(config)

    assert f"Hydra version: {module.hydra.__version__}" in caplog.text
    assert "setup: logged-setup" in caplog.text


@pytest.mark.parametrize(
    "module_name",
    ["coverage", "gap", "entropic_ud", "real_analysis"],
)
def test_runner_errors_list_supported_setups(tmp_path, module_name):
    module = __import__(f"experiments.{module_name}.run", fromlist=["run"])
    config = OmegaConf.create(
        {
            "setup": "unsupported",
            "pfn": "tabpfn",
            "x_design": "one-gap",
            "seed": 0,
        }
    )
    with pytest.raises(ValueError, match="Supported setups:"):
        module.run_experiment(config, tmp_path)
