"""Repository-relative runtime paths and Hydra setup."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_ROOT = REPO_ROOT / "outputs"
MODEL_ROOT = REPO_ROOT / "pfn-model"
CLASSIFIER_CHECKPOINT_PATH = (
    MODEL_ROOT / "tabpfn-v3-classifier-v3_default.ckpt"
)
REGRESSOR_CHECKPOINT_PATH = MODEL_ROOT / "tabpfn-v3-regressor-v3_default.ckpt"

_figure_override = os.environ.get("UD4PFN_FIGDIR")
if _figure_override:
    _selected_figure_root = Path(_figure_override).expanduser()
    FIGURES_ROOT = (
        _selected_figure_root
        if _selected_figure_root.is_absolute()
        else REPO_ROOT / _selected_figure_root
    )
else:
    FIGURES_ROOT = REPO_ROOT / "figures"
FIGURES_ROOT.mkdir(parents=True, exist_ok=True)

os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"


def githash() -> str:
    """Return the short Git commit hash used in Hydra run metadata."""

    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def register_githash_resolver() -> None:
    """Register Hydra's ``${githash:}`` resolver idempotently."""

    OmegaConf.register_new_resolver("githash", githash, replace=True)


__all__ = [
    "REPO_ROOT",
    "OUTPUTS_ROOT",
    "FIGURES_ROOT",
    "MODEL_ROOT",
    "CLASSIFIER_CHECKPOINT_PATH",
    "REGRESSOR_CHECKPOINT_PATH",
    "githash",
    "register_githash_resolver",
]
