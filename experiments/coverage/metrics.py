"""Coverage scores and bootstrap bands."""

import numpy as np


def compute_pointwise_coverage(true_curve, bands):
    intervals = [(band["lower"], band["upper"]) for band in bands]
    if not intervals:
        return np.nan
    for lower, upper in intervals:
        assert true_curve.shape == lower.shape == upper.shape
    if any(
        np.any(np.isnan(lower)) or np.any(np.isnan(upper))
        for lower, upper in intervals
    ):
        return np.nan
    return np.mean(
        np.asarray(
            [
                (true_curve >= lower) & (true_curve <= upper)
                for lower, upper in intervals
            ]
        )
    )


def compute_simultaneous_coverage(true_curve, bands):
    intervals = [(band["lower"], band["upper"]) for band in bands]
    if not intervals:
        return np.nan
    for lower, upper in intervals:
        assert true_curve.shape == lower.shape == upper.shape
    if any(
        np.any(np.isnan(lower)) or np.any(np.isnan(upper))
        for lower, upper in intervals
    ):
        return np.nan
    return np.mean(
        [
            np.all((true_curve >= lower) & (true_curve <= upper))
            for lower, upper in intervals
        ]
    )


def build_bootstrap_pointwise_band(mean, bootstrap_samples, alpha=0.05):
    mean, bootstrap_samples = np.asarray(mean), np.asarray(bootstrap_samples)
    assert mean.ndim == 1
    assert bootstrap_samples.ndim == 2
    assert bootstrap_samples.shape[1] == mean.shape[0]
    lower = np.clip(np.quantile(bootstrap_samples, alpha / 2, axis=0), 0, 1)
    upper = np.clip(np.quantile(bootstrap_samples, 1 - alpha / 2, axis=0), 0, 1)
    return {
        "mean": mean,
        "lower": lower,
        "upper": upper,
        "width": np.mean(upper - lower),
    }


def build_bootstrap_simultaneous_band(
    mean, bootstrap_samples, alpha=0.05, studentize=False, eps=1e-12
):
    mean, bootstrap_samples = np.asarray(mean), np.asarray(bootstrap_samples)
    assert mean.ndim == 1
    assert bootstrap_samples.ndim == 2
    assert bootstrap_samples.shape[1] == mean.shape[0]
    difference = bootstrap_samples - mean[None, :]
    if studentize:
        standard_error = np.maximum(
            np.std(bootstrap_samples, axis=0, ddof=1), eps
        )
        critical = np.quantile(
            np.max(np.abs(difference) / standard_error[None, :], axis=1),
            1 - alpha,
        )
        delta = critical * standard_error
    else:
        standard_error = None
        critical = np.quantile(np.max(np.abs(difference), axis=1), 1 - alpha)
        delta = critical
    lower = np.clip(mean - delta, 0, 1)
    upper = np.clip(mean + delta, 0, 1)
    return {
        "mean": mean,
        "lower": lower,
        "upper": upper,
        "c_alpha": critical,
        "se": standard_error,
        "width": np.mean(upper - lower),
    }


__all__ = [
    "compute_pointwise_coverage",
    "compute_simultaneous_coverage",
    "build_bootstrap_pointwise_band",
    "build_bootstrap_simultaneous_band",
]
