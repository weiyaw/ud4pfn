import ast
from pathlib import Path

import predictive_clt


EXPECTED_PUBLIC_API = {
    "TabPFNClassifierPPD",
    "TabPFNRegressorPPD",
    "compute_gn",
    "compute_g0_to_gn",
    "sample_gn_plus_1",
    "compute_un",
    "compute_vn",
    "build_pointwise_band",
    "build_simultaneous_band",
    "build_ellipsoid_band",
    "compute_ellipsoid_log_volume",
}


def test_public_api_is_exact():
    assert set(predictive_clt.__all__) == EXPECTED_PUBLIC_API
    assert all(hasattr(predictive_clt, name) for name in EXPECTED_PUBLIC_API)


def test_predictive_clt_does_not_import_experiments():
    package_root = Path(predictive_clt.__file__).resolve().parent
    for source_path in package_root.glob("*.py"):
        tree = ast.parse(source_path.read_text(), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported = [node.module]
            else:
                continue
            assert not any(
                name == "experiments" or name.startswith("experiments.")
                for name in imported
            ), f"{source_path} imports an experiment module"
