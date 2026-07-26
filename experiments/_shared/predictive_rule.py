"""Construct predictive-distribution adapters for experiment runners."""

from __future__ import annotations

from typing import Literal

from predictive_clt import (
    TabICLClassifierPPD,
    TabICLRegressorPPD,
    TabPFNClassifierPPD,
    TabPFNRegressorPPD,
)

from .runtime import (
    TABICL_CLASSIFIER_CHECKPOINT_PATH,
    TABICL_REGRESSOR_CHECKPOINT_PATH,
    TABPFN_CLASSIFIER_CHECKPOINT_PATH,
    TABPFN_REGRESSOR_CHECKPOINT_PATH,
)


PFN = Literal["tabpfn", "tabicl"]
Task = Literal["classification", "regression"]


def build_predictive_rule(pfn: str, task: Task, n_estimators: int):
    """Build the selected PFN adapter for a classification or regression task."""

    if pfn not in ("tabpfn", "tabicl"):
        raise ValueError(
            f"Unknown pfn '{pfn}'. Supported PFNs: tabpfn, tabicl"
        )
    if task not in ("classification", "regression"):
        raise ValueError(
            f"Unknown task '{task}'. Supported tasks: classification, regression"
        )

    if pfn == "tabpfn":
        options = {
            "n_estimators": n_estimators,
            "softmax_temperature": 1.0,
            "fit_mode": "low_memory",
        }
        if task == "classification":
            return TabPFNClassifierPPD(
                **options,
                model_path=str(TABPFN_CLASSIFIER_CHECKPOINT_PATH),
            )
        return TabPFNRegressorPPD(
            **options,
            model_path=str(TABPFN_REGRESSOR_CHECKPOINT_PATH),
        )

    options = {
        "n_estimators": n_estimators,
        "allow_auto_download": True,
    }
    if task == "classification":
        return TabICLClassifierPPD(
            **options,
            softmax_temperature=1.0,
            model_path=str(TABICL_CLASSIFIER_CHECKPOINT_PATH),
        )
    return TabICLRegressorPPD(
        **options,
        model_path=str(TABICL_REGRESSOR_CHECKPOINT_PATH),
    )


__all__ = ["PFN", "Task", "build_predictive_rule"]
