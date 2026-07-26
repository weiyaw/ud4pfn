import numpy as np

from predictive_clt.bands import (
    build_ellipsoid_band,
    build_pointwise_band,
    build_simultaneous_band,
    compute_ellipsoid_log_volume,
)


def test_pointwise_band_schema_and_width():
    mean = np.array([0.25, 0.75])
    covariance = np.array([0.01, 0.04])

    band = build_pointwise_band(mean, covariance)

    assert set(band) == {"mean", "lower", "upper", "se", "width"}
    np.testing.assert_allclose(np.asarray(band["se"]), [0.1, 0.2])
    np.testing.assert_allclose(
        np.asarray(band["upper"] - band["lower"]),
        np.asarray(band["width"]),
    )


def test_simultaneous_band_schema_and_determinism():
    mean = np.array([0.25, 0.5, 0.75])
    covariance = np.diag([0.01, 0.02, 0.03])

    first = build_simultaneous_band(mean, covariance)
    second = build_simultaneous_band(mean, covariance)

    assert set(first) == {
        "mean",
        "lower",
        "upper",
        "c_alpha",
        "se",
        "draws",
        "width",
    }
    np.testing.assert_array_equal(np.asarray(first["draws"]), np.asarray(second["draws"]))


def test_ellipsoid_band_schema_and_log_volume():
    mean = np.array([0.4, 0.6])
    covariance = np.diag([0.01, 0.04])

    band = build_ellipsoid_band(mean, covariance)

    assert set(band) == {
        "mean",
        "lower",
        "upper",
        "radius",
        "log_volume",
    }
    assert band["log_volume"] == compute_ellipsoid_log_volume(
        covariance, band["radius"]
    )
    assert compute_ellipsoid_log_volume(np.zeros((2, 2)), 1.0) == -np.inf
