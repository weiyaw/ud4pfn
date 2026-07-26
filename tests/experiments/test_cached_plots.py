import runpy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments._shared.artifacts import write_pickle


def g0_to_gn(p, m):
    values = np.empty((4, p, m), dtype=np.float32)
    values[0] = np.nan
    values[1] = 0.40
    values[2] = np.linspace(0.42, 0.45, m)[None, :]
    values[3] = np.linspace(0.46, 0.50, m)[None, :]
    return values


def write_cached_run(
    directory,
    events,
    x_grid,
    y,
    *,
    true_probability=True,
    bootstrap=False,
    copula=False,
):
    directory.mkdir(parents=True)
    p, m = len(events), len(x_grid)
    if x_grid.shape[1] == 1:
        x_prev = np.linspace(
            float(x_grid.min()), float(x_grid.max()), len(y), dtype=np.float32
        )[:, None]
    else:
        x_prev = np.zeros((len(y), x_grid.shape[1]), dtype=np.float32)
    data = {
        "x_prev": x_prev,
        "y_prev": np.asarray(y),
        "t": np.asarray(events),
        "x_grid": np.asarray(x_grid),
        "grid_shape": (m,) if x_grid.shape[1] == 1 else (2, 2),
    }
    if true_probability:
        data["true_prob"] = np.full((p, m), 0.5, dtype=np.float32)
    write_pickle(directory / "data.pickle", data)
    write_pickle(directory / "gn.pickle", np.full((p, m), 0.5, dtype=np.float32))
    write_pickle(directory / "g0_to_gn.pickle", g0_to_gn(p, m))
    draws = np.stack(
        [
            np.full((p, m), 0.48, dtype=np.float32),
            np.full((p, m), 0.52, dtype=np.float32),
        ]
    )
    write_pickle(directory / "gn_plus_1.pickle", draws)
    if bootstrap:
        write_pickle(
            directory / "bootstrap-200.pickle",
            {
                "bootstrap_samples": 2,
                "seed_offset": 0,
                "predictions": draws,
                "elapsed_seconds": 0.0,
            },
        )
    if copula:
        write_pickle(
            directory / "copula-200-1000.pickle",
            {
                "logcdf": np.log(draws),
                "rollout_times": 2,
                "rollout_length": 2,
                "seed": 0,
                "elapsed_seconds": 0.0,
            },
        )
    return directory


def set_runtime_roots(monkeypatch, outputs, figures):
    from experiments._shared import runtime

    monkeypatch.setattr(runtime, "OUTPUTS_ROOT", outputs)
    monkeypatch.setattr(runtime, "FIGURES_ROOT", figures)
    figures.mkdir(parents=True)


def test_gap_cached_artifacts_produce_all_figures(monkeypatch, tmp_path):
    from experiments.gap.run import EXPERIMENT_DEFINITIONS

    outputs, figures = tmp_path / "outputs", tmp_path / "figures"
    set_runtime_roots(monkeypatch, outputs, figures)
    x_grid = np.array([[-1.0], [1.0]])
    for name, definition in EXPERIMENT_DEFINITIONS.items():
        for n in (200, 500, 1000):
            directory = (
                outputs
                / "gap"
                / (
                    f"setup={name} x_design=one-gap shuffle=True "
                    f"n_est=64 n={n} m=2 seed=1000"
                )
            )
            labels = (
                [0, 1, 0] if definition.task == "classification" else [0.0, 1.0, 2.0]
            )
            write_cached_run(directory, definition.events, x_grid, labels)

    runpy.run_module("experiments.gap.plot", run_name="__main__")

    assert {path.name for path in figures.glob("gap-*.pdf")} == {
        f"gap-{name}.pdf" for name in EXPERIMENT_DEFINITIONS
    }
    plt.close("all")


def test_real_cached_artifacts_produce_both_figures(monkeypatch, tmp_path):
    outputs, figures = tmp_path / "outputs", tmp_path / "figures"
    set_runtime_roots(monkeypatch, outputs, figures)
    x_grid = np.array([[1.0], [2.0]])
    for name in ("labour-force", "fibre-strength"):
        directory = (
            outputs
            / "real-analysis"
            / (f"setup={name} shuffle=True n_est=64 m=2 seed=1000")
        )
        write_cached_run(directory, (0, 1), x_grid, [0, 1, 1], true_probability=False)

    runpy.run_module("experiments.real_analysis.plot", run_name="__main__")

    assert (figures / "labour-force-vn.pdf").exists()
    assert (figures / "fibre-strength-vn.pdf").exists()
    plt.close("all")


def test_entropic_standard_and_vary_n_caches(monkeypatch, tmp_path):
    outputs, figures = tmp_path / "outputs", tmp_path / "figures"
    set_runtime_roots(monkeypatch, outputs, figures)
    line = np.linspace(-15, 15, 151)[:, None]
    for n in (15, 50, 75, 150):
        directory = (
            outputs
            / "entropic-ud"
            / (
                f"setup=logistic-linear x_design=gaussian n_est=64 "
                f"n={n} m=151 seed=1000"
            )
        )
        write_cached_run(directory, (0, 1), line, [0, 1, 1])

    grid_axis = np.array([-1.0, 1.0])
    x1, x2 = np.meshgrid(grid_axis, grid_axis, indexing="ij")
    grid = np.stack([x1, x2], axis=-1).reshape(-1, 2)
    for name, n in (
        ("two-moons-1", 30),
        ("two-moons-1", 100),
        ("two-moons-2", 30),
        ("two-moons-2", 100),
        ("spiral", 200),
    ):
        events = (0, 1, 2) if name == "spiral" else (0, 1)
        labels = [0, 1, 2] if name == "spiral" else [0, 1, 1]
        directory = (
            outputs
            / "entropic-ud"
            / (f"setup={name} x_design=None n_est=64 n={n} m=2 seed=1000")
        )
        write_cached_run(directory, events, grid, labels)

    vary_dir = outputs / "entropic-ud-vary-n"
    generic = write_cached_run(
        vary_dir / "setup=logistic-linear n=75 seed=1000",
        (0, 1),
        line,
        [0, 1, 1],
    )
    import experiments._shared.artifacts as artifacts

    original = artifacts.find_run_directories

    def matching(directory, regex):
        if Path(directory) == vary_dir:
            return [generic] * 50
        return original(directory, regex)

    monkeypatch.setattr(artifacts, "find_run_directories", matching)
    runpy.run_module("experiments.entropic_ud.plot", run_name="__main__")

    expected = {
        "ud-logreg-xstar.pdf",
        "ud-logreg-context-length.pdf",
        "ud-logreg-context-length-prop.pdf",
        "ud-two-moons.pdf",
        "ud-two-moons-spiral.pdf",
    }
    assert expected <= {path.name for path in figures.glob("*.pdf")}
    plt.close("all")


def test_coverage_tables_read_cached_artifacts(monkeypatch, tmp_path, capsys):
    from experiments.coverage.run import EXPERIMENT_DEFINITIONS

    outputs, figures = tmp_path / "outputs", tmp_path / "figures"
    set_runtime_roots(monkeypatch, outputs, figures)
    x_grid = np.zeros((2, 10))
    for name, definition in EXPERIMENT_DEFINITIONS.items():
        for n in (200, 500, 1000):
            directory = (
                outputs
                / "coverage"
                / (
                    f"setup={name} x_design=sobol-10d shuffle=True "
                    f"n_est=16 n={n} m=2 seed=1000"
                )
            )
            write_cached_run(
                directory,
                definition.events,
                x_grid,
                [0, 1, 1],
                bootstrap=True,
                copula=definition.task == "regression",
            )

    runpy.run_module("experiments.coverage.plot", run_name="__main__")

    captured = capsys.readouterr().out
    assert "gaussian-linear-multivariate" in captured
    assert "categorical-linear-multivariate" in captured
    plt.close("all")
