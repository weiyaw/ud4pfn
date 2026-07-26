from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir


EXPERIMENTS_ROOT = Path(__file__).resolve().parents[2] / "experiments"


@pytest.mark.parametrize(
    "group,expected",
    [
        (
            "coverage",
            {
                "setup": "gaussian-linear-multivariate",
                "data_size": 4,
                "x_grid_size": 2,
                "x_design": "sobol-10d",
                "id": "smoke/coverage",
            },
        ),
        (
            "gap",
            {
                "setup": "gaussian-linear",
                "data_size": 4,
                "x_grid_size": 2,
                "x_design": "one-gap",
                "id": "smoke/gap",
            },
        ),
        (
            "entropic_ud",
            {
                "setup": "two-moons-1",
                "data_size": 4,
                "x_grid_size": 2,
                "x_design": None,
                "id": "smoke/entropic-ud",
            },
        ),
        (
            "real_analysis",
            {
                "setup": "fibre-strength",
                "x_grid_size": 2,
                "id": "smoke/real-analysis",
            },
        ),
    ],
)
def test_smoke_config_is_lightweight_and_writes_to_smoke_outputs(group, expected):
    config_dir = EXPERIMENTS_ROOT / group / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(config_name="smoke", return_hydra_config=True)

    for field, value in expected.items():
        assert config[field] == value
    assert config.shuffle_data is True
    assert config.seed == 1000
    assert config.pfn == "tabpfn"
    assert config.n_estimators == 1
    assert config.mc_samples == 1
    assert "pfn=tabpfn" in config.hydra.run.dir
    assert config.hydra.run.dir.startswith(
        f"./outputs/smoke/{group.replace('_', '-')}/"
    )
