"""Data loaders owned by the real-analysis experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax.random as jr
import numpy as np
import pandas as pd


FIBRE_DATA_PATH = Path(__file__).resolve().with_name("fibre_strength.csv")


@dataclass
class RealData:
    X: np.ndarray
    y: np.ndarray


class LabourForce(RealData):
    """Mroz labour-force participation against non-labour income."""

    def __init__(self, shuffle: bool):
        import statsmodels.api as sm

        frame = sm.datasets.get_rdataset("Mroz", "carData").data
        self.y = frame.lfp.map({"no": 0, "yes": 1}).to_numpy(int)
        self.X = frame[["inc"]].to_numpy(float)
        if shuffle:
            permutation = jr.permutation(jr.key(7251), self.X.shape[0])
            self.X = self.X[permutation]
            self.y = self.y[permutation]


class FibreStrength(RealData):
    """Binary fibre reliability at the fixed 1.5 MPa threshold."""

    def __init__(self, shuffle: bool, csv_path: str | Path | None = None):
        frame = pd.read_csv(Path(csv_path) if csv_path is not None else FIBRE_DATA_PATH)
        self.X = frame[["length_mm"]].to_numpy(float)
        self.y = (frame["strength_mpa"] > 1.5).to_numpy(int)
        if shuffle:
            permutation = jr.permutation(jr.key(3753), self.X.shape[0])
            self.X = self.X[permutation]
            self.y = self.y[permutation]


__all__ = ["FIBRE_DATA_PATH", "RealData", "LabourForce", "FibreStrength"]
