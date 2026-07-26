"""Filesystem helpers for experiment artifacts and legacy run metadata."""

from __future__ import annotations

import logging
import pickle
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


_METADATA_FIELDS = ("setup", "n_est", "n", "m", "seed")
_CONFIG_KEYS = {
    "setup": ("setup",),
    "n_est": ("n_est", "n_estimators"),
    "n": ("n", "data_size"),
    "m": ("m", "x_grid_size"),
    "seed": ("seed",),
}
_INTEGER_FIELDS = frozenset({"n_est", "n", "m", "seed"})


def write_pickle(path: str | Path, obj: Any, verbose: bool = False) -> None:
    """Write a highest-protocol pickle, creating its parent directory."""

    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if verbose:
        logging.info("Wrote artifact: %s", artifact_path)


def read_pickle(path: str | Path) -> Any:
    """Read and return a pickle artifact."""

    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def find_run_directories(
    directory: str | Path,
    pattern: str | re.Pattern[str] | None = None,
) -> list[Path]:
    """Return immediate child run directories in deterministic name order."""

    root = Path(directory)
    if not root.exists():
        return []
    regex = re.compile(pattern) if isinstance(pattern, str) else pattern
    return sorted(
        (
            entry
            for entry in root.iterdir()
            if entry.is_dir() and (regex is None or regex.search(entry.name))
        ),
        key=lambda entry: entry.name,
    )


def _config_value(config: Any, field: str) -> Any | None:
    for key in _CONFIG_KEYS[field]:
        value = OmegaConf.select(config, key, default=None)
        if value is not None:
            return value
    return None


def _legacy_value(run_dir: Path, field: str) -> str | None:
    match = re.search(
        rf"(?:^|\s){re.escape(field)}=([^\s]+)",
        run_dir.name,
    )
    return match.group(1) if match else None


def _normalise_metadata_value(field: str, value: Any) -> Any:
    if field in _INTEGER_FIELDS:
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid integer metadata field '{field}': {value!r}"
            ) from exc
    return str(value) if field == "setup" else value


def load_run_metadata(
    run_dir: str | Path,
    required_fields: Sequence[str] = _METADATA_FIELDS,
) -> dict[str, Any]:
    """Load canonical run metadata with Hydra-before-directory precedence.

    Canonical fields use artifact-directory names (``n_est``, ``n``, and
    ``m``). Hydra's configuration names (``n_estimators``, ``data_size``, and
    ``x_grid_size``) are accepted and normalised.
    """

    directory = Path(run_dir)
    if isinstance(required_fields, str):
        required_fields = (required_fields,)
    unknown = set(required_fields) - set(_METADATA_FIELDS)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown metadata field(s): {names}")

    config_path = directory / ".hydra" / "config.yaml"
    config = OmegaConf.load(config_path) if config_path.is_file() else None

    metadata: dict[str, Any] = {}
    for field in required_fields:
        value = _config_value(config, field) if config is not None else None
        if value is None:
            value = _legacy_value(directory, field)
        if value is None:
            raise ValueError(
                f"Missing metadata field '{field}' for run directory '{directory}'"
            )
        metadata[field] = _normalise_metadata_value(field, value)
    return metadata


__all__ = [
    "write_pickle",
    "read_pickle",
    "find_run_directories",
    "load_run_metadata",
]
