"""Repository-relative runtime paths and Hydra setup."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_ROOT = REPO_ROOT / "outputs"
PFN_ROOT = REPO_ROOT / "pfn-model"
TABPFN_CLASSIFIER_CHECKPOINT_PATH = (
    PFN_ROOT / "tabpfn-v3-classifier-v3_default.ckpt"
)
TABPFN_REGRESSOR_CHECKPOINT_PATH = (
    PFN_ROOT / "tabpfn-v3-regressor-v3_default.ckpt"
)
TABICL_CLASSIFIER_CHECKPOINT_PATH = (
    PFN_ROOT / "tabicl-classifier-v2-20260212.ckpt"
)
TABICL_REGRESSOR_CHECKPOINT_PATH = (
    PFN_ROOT / "tabicl-regressor-v2-20260212.ckpt"
)

# Backwards-compatible aliases used by the TabPFN-only coverage baselines.
CLASSIFIER_CHECKPOINT_PATH = TABPFN_CLASSIFIER_CHECKPOINT_PATH
REGRESSOR_CHECKPOINT_PATH = TABPFN_REGRESSOR_CHECKPOINT_PATH

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
    "PFN_ROOT",
    "TABPFN_CLASSIFIER_CHECKPOINT_PATH",
    "TABPFN_REGRESSOR_CHECKPOINT_PATH",
    "TABICL_CLASSIFIER_CHECKPOINT_PATH",
    "TABICL_REGRESSOR_CHECKPOINT_PATH",
    "CLASSIFIER_CHECKPOINT_PATH",
    "REGRESSOR_CHECKPOINT_PATH",
    "githash",
    "register_githash_resolver",
]
