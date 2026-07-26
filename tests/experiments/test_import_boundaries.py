import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED = {"coverage", "gap", "entropic_ud", "real_analysis"}


def imported_modules(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module


def test_supported_experiments_do_not_cross_import():
    for experiment in SUPPORTED:
        for path in (REPO_ROOT / "experiments" / experiment).glob("*.py"):
            for imported in imported_modules(path):
                for other in SUPPORTED - {experiment}:
                    assert not (
                        imported == f"experiments.{other}"
                        or imported.startswith(f"experiments.{other}.")
                    ), f"{path} imports {other}"


def test_no_supported_code_imports_retired_root_modules():
    retired = {"data", "constants", "metrics", "posterior", "pred_rule", "utils"}
    for experiment in SUPPORTED:
        for path in (REPO_ROOT / "experiments" / experiment).glob("*.py"):
            assert retired.isdisjoint(imported_modules(path)), str(path)
