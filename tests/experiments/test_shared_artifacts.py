import pickle
import importlib

import pytest

from experiments._shared.artifacts import (
    find_run_directories,
    load_run_metadata,
    read_pickle,
    write_pickle,
)


def test_pickle_round_trip_creates_parent(tmp_path):
    artifact = tmp_path / "nested" / "data.pickle"
    value = {"shape": (2, 3), "seed": 1000}

    write_pickle(artifact, value)

    assert read_pickle(artifact) == value
    with artifact.open("rb") as handle:
        assert pickle.load(handle) == value


def test_run_discovery_is_sorted_and_filtered(tmp_path):
    (tmp_path / "setup=z n=2").mkdir()
    (tmp_path / "setup=a n=1").mkdir()
    (tmp_path / "not-a-run").mkdir()
    (tmp_path / "file").write_text("ignored")

    paths = find_run_directories(tmp_path, r"^setup=")

    assert [path.name for path in paths] == ["setup=a n=1", "setup=z n=2"]


def test_hydra_metadata_precedes_legacy_directory_name(tmp_path):
    run_dir = tmp_path / "setup=legacy n_est=1 n=2 m=3 seed=4"
    hydra_dir = run_dir / ".hydra"
    hydra_dir.mkdir(parents=True)
    (hydra_dir / "config.yaml").write_text(
        "\n".join(
            [
                "setup: configured",
                "n_estimators: 16",
                "data_size: 200",
                "x_grid_size: 100",
                "seed: 1000",
            ]
        )
    )

    metadata = load_run_metadata(run_dir)

    assert metadata == {
        "setup": "configured",
        "n_est": 16,
        "n": 200,
        "m": 100,
        "seed": 1000,
    }


def test_legacy_metadata_fallback_and_clear_missing_error(tmp_path):
    run_dir = tmp_path / "setup=gaussian-linear n_est=16 n=200 m=100 seed=1000"
    run_dir.mkdir()

    assert load_run_metadata(run_dir) == {
        "setup": "gaussian-linear",
        "n_est": 16,
        "n": 200,
        "m": 100,
        "seed": 1000,
    }

    broken_run = tmp_path / "setup=broken-run"
    broken_run.mkdir()
    with pytest.raises(ValueError, match=r"Missing metadata field 'm'.*setup=broken-run"):
        load_run_metadata(broken_run, required_fields=("setup", "m"))


def test_figure_directory_environment_override(monkeypatch, tmp_path):
    import experiments._shared.runtime as runtime

    destination = tmp_path / "paper-figures"
    monkeypatch.setenv("UD4PFN_FIGDIR", str(destination))
    reloaded = importlib.reload(runtime)

    assert reloaded.FIGURES_ROOT == destination
    assert destination.is_dir()

    monkeypatch.delenv("UD4PFN_FIGDIR")
    importlib.reload(runtime)
