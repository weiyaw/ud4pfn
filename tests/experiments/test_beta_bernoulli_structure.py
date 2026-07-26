import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BETA_ROOT = REPO_ROOT / "experiments" / "beta_bernoulli"


def test_beta_bernoulli_is_relocated_and_self_contained():
    assert BETA_ROOT.is_dir()
    assert not (REPO_ROOT / "beta_bernoulli").exists()

    imports: set[str] = set()
    for source_path in BETA_ROOT.glob("*.py"):
        tree = ast.parse(source_path.read_text(), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)

    assert "predictive_clt" not in imports
    assert not any(name.startswith("predictive_clt.") for name in imports)
    assert "tabpfn" not in imports
    assert "jax" not in imports
    assert not any(name.startswith("jax.") for name in imports)


def test_beta_readme_documents_its_local_environment():
    readme = (BETA_ROOT / "README.md").read_text()

    assert "inside `experiments/beta_bernoulli/`" in readme
    assert ".venv/bin/python" in readme
    assert "SEPARATE" in readme
