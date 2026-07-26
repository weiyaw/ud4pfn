"""Public API for the reusable predictive central-limit-theorem method."""

from .bands import (
    build_ellipsoid_band,
    build_pointwise_band,
    build_simultaneous_band,
    compute_ellipsoid_log_volume,
)
from .posterior import (
    compute_g0_to_gn,
    compute_gn,
    compute_un,
    compute_vn,
    sample_gn_plus_1,
)
from .tabicl_adapter import TabICLClassifierPPD, TabICLRegressorPPD
from .tabpfn_adapter import TabPFNClassifierPPD, TabPFNRegressorPPD

__all__ = [
    "TabICLClassifierPPD",
    "TabICLRegressorPPD",
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
]
