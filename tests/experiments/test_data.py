from types import SimpleNamespace

import jax.random as jr
import numpy as np
import pandas as pd
import pytest

from experiments.coverage.run import EXPERIMENT_DEFINITIONS as COVERAGE
from experiments.entropic_ud.run import EXPERIMENT_DEFINITIONS as ENTROPIC
from experiments.gap.run import EXPERIMENT_DEFINITIONS as GAP
from experiments.real_analysis.data import FibreStrength, LabourForce


def test_supported_setup_names_are_exact():
    assert set(GAP) == {
        "gaussian-linear",
        "gaussian-polynomial",
        "gaussian-linear-dependent-error",
        "gaussian-sine",
        "poisson-linear",
        "probit-mixture",
        "categorical-linear",
    }
    assert set(COVERAGE) == {
        "gaussian-linear-multivariate",
        "gaussian-linear-dependent-error-multivariate",
        "poisson-linear-multivariate",
        "probit-mixture-multivariate",
        "categorical-linear-multivariate",
    }
    assert set(ENTROPIC) == {
        "logistic-linear",
        "two-moons-1",
        "two-moons-2",
        "spiral",
    }


@pytest.mark.parametrize("name,definition", list(GAP.items()))
def test_gap_data_shapes_events_and_reproducibility(name, definition):
    first = definition.data_factory(jr.key(1), 9, True, "one-gap")
    second = definition.data_factory(jr.key(1), 9, True, "one-gap")
    grid = np.linspace(-10, 10, 5)[:, None]

    assert first.X.shape == (9, 1)
    assert first.y.shape == (9,)
    np.testing.assert_array_equal(first.X, second.X)
    np.testing.assert_array_equal(first.y, second.y)
    assert np.all((first.X < -2) | (first.X > 2))
    for event in definition.events:
        assert first.get_true_event(grid, event).shape == (5,)
    if definition.task == "classification":
        assert np.issubdtype(first.y.dtype, np.integer)


@pytest.mark.parametrize("name,definition", list(COVERAGE.items()))
def test_coverage_data_shapes_events_and_reproducibility(name, definition):
    first = definition.data_factory(jr.key(2), 8, True, "sobol-10d")
    second = definition.data_factory(jr.key(2), 8, True, "sobol-10d")
    grid = np.zeros((4, 10), dtype=np.float32)

    assert first.X.shape == (8, 10)
    assert first.y.shape == (8,)
    np.testing.assert_array_equal(first.X, second.X)
    np.testing.assert_array_equal(first.y, second.y)
    for event in definition.events:
        assert first.get_true_event(grid, event).shape == (4,)
    if definition.task == "classification":
        assert np.issubdtype(first.y.dtype, np.integer)


@pytest.mark.parametrize("name,definition", list(ENTROPIC.items()))
def test_entropic_data_shapes_events_and_reproducibility(name, definition):
    design = "gaussian:1.5:3.0" if name == "logistic-linear" else None
    first = definition.data_factory(jr.key(3), 9, True, design)
    second = definition.data_factory(jr.key(3), 9, True, design)
    dimension = first.X.shape[1]
    grid = np.zeros((4, dimension))

    assert first.X.shape == (9, dimension)
    assert first.y.shape == (9,)
    assert np.issubdtype(first.y.dtype, np.integer)
    np.testing.assert_array_equal(first.X, second.X)
    np.testing.assert_array_equal(first.y, second.y)
    for event in definition.events:
        assert first.get_true_event(grid, event).shape == (4,)


def test_labour_force_loader_without_network(monkeypatch):
    frame = pd.DataFrame({"lfp": ["no", "yes"], "inc": [10.0, 20.0]})
    fake = SimpleNamespace(data=frame)
    import statsmodels.api as sm

    monkeypatch.setattr(sm.datasets, "get_rdataset", lambda *args: fake)
    data = LabourForce(shuffle=False)

    np.testing.assert_array_equal(data.y, [0, 1])
    np.testing.assert_array_equal(data.X[:, 0], [10.0, 20.0])


def test_fibre_loader_is_module_relative(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    data = FibreStrength(shuffle=False)

    assert data.X.ndim == 2 and data.X.shape[1] == 1
    assert data.y.shape == (data.X.shape[0],)
    assert set(np.unique(data.y)) <= {0, 1}
