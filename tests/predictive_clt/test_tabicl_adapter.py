import types

import jax.random as jr
import numpy as np
import pytest

tabicl = pytest.importorskip("tabicl")

from predictive_clt.tabicl_adapter import (  # noqa: E402
    TabICLClassifierPPD,
    TabICLRegressorPPD,
)


def _data():
    x_prev = np.arange(12, dtype=np.float32).reshape(6, 2)
    y_prev = np.linspace(-1.0, 1.0, 6, dtype=np.float32)
    x_new = np.arange(8, dtype=np.float32).reshape(4, 2)
    return x_new, x_prev, y_prev


def _stub_regressor(monkeypatch):
    regressor = TabICLRegressorPPD(
        n_estimators=1, allow_auto_download=False, device="cpu"
    )
    calls = {"fit": 0, "predict": 0}

    def fit(self, X, y):
        calls["fit"] += 1
        return self

    def predict(self, X, output_type="mean", alphas=None):
        calls["predict"] += 1
        assert output_type == "raw_quantiles"
        base = np.linspace(-2.0, 2.0, 9, dtype=np.float32)
        return np.stack([base + 0.25 * i for i in range(X.shape[0])])

    monkeypatch.setattr(regressor, "fit", types.MethodType(fit, regressor))
    monkeypatch.setattr(
        regressor, "predict", types.MethodType(predict, regressor)
    )
    return regressor, calls


def _stub_classifier(monkeypatch, classes=np.array([2, 5, 9])):
    classifier = TabICLClassifierPPD(
        n_estimators=1, allow_auto_download=False, device="cpu"
    )
    probabilities = np.array(
        [
            [0.1, 0.3, 0.6],
            [0.2, 0.5, 0.3],
            [0.7, 0.2, 0.1],
            [0.25, 0.25, 0.5],
        ],
        dtype=np.float32,
    )

    def fit(self, X, y):
        self.classes_ = np.asarray(classes)
        return self

    def predict_proba(self, X):
        return probabilities[: X.shape[0]]

    monkeypatch.setattr(classifier, "fit", types.MethodType(fit, classifier))
    monkeypatch.setattr(
        classifier,
        "predict_proba",
        types.MethodType(predict_proba, classifier),
    )
    return classifier, probabilities


def test_regressor_cdf_shape_bounds_and_monotonicity(monkeypatch):
    regressor, calls = _stub_regressor(monkeypatch)
    x_new, x_prev, y_prev = _data()

    probabilities = regressor.cdf(
        np.array([-1.0, 0.0, 1.0]),
        x_new=x_new,
        x_prev=x_prev,
        y_prev=y_prev,
    )

    assert probabilities.shape == (3, 4)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert np.all(np.diff(probabilities, axis=0) >= 0)
    assert calls == {"fit": 1, "predict": 1}


def test_regressor_icdf_shape_and_monotonicity(monkeypatch):
    regressor, _ = _stub_regressor(monkeypatch)
    x_new, x_prev, y_prev = _data()

    quantiles = regressor.icdf(
        np.array([0.1, 0.5, 0.9]),
        x_new=x_new,
        x_prev=x_prev,
        y_prev=y_prev,
    )

    assert quantiles.shape == (3, 4)
    assert np.all(np.isfinite(quantiles))
    assert np.all(np.diff(quantiles, axis=0) >= 0)


def test_regressor_sample_is_reproducible_and_returns_metadata(monkeypatch):
    regressor, _ = _stub_regressor(monkeypatch)
    x_new, x_prev, y_prev = _data()
    key = jr.key(12)

    first, first_meta = regressor.sample(
        key, x_new=x_new, x_prev=x_prev, y_prev=y_prev, size=5
    )
    second, _ = regressor.sample(
        key, x_new=x_new, x_prev=x_prev, y_prev=y_prev, size=5
    )

    assert first.shape == (5, 4)
    np.testing.assert_array_equal(first, second)
    assert first_meta["raw_quantiles"].shape == (4, 9)
    assert "quantile_distribution" in first_meta


def test_regressor_predict_event_delegates_to_cdf(monkeypatch):
    regressor, _ = _stub_regressor(monkeypatch)
    x_new, x_prev, y_prev = _data()
    threshold = np.array([0.0])

    actual = regressor.predict_event(threshold, x_new, x_prev, y_prev)
    expected = regressor.cdf(threshold, x_new, x_prev, y_prev)

    np.testing.assert_allclose(actual, expected)


def test_regressor_rejects_invalid_shapes(monkeypatch):
    regressor, _ = _stub_regressor(monkeypatch)
    x_new, x_prev, y_prev = _data()

    with pytest.raises(AssertionError):
        regressor.cdf(
            np.array([0.0]), x_new[:, :1], x_prev=x_prev, y_prev=y_prev
        )


def test_classifier_pmf_multiclass_and_absent_class(monkeypatch):
    classifier, probabilities = _stub_classifier(monkeypatch)
    x_new, x_prev, _ = _data()
    y_prev = np.array([2, 5, 9, 2, 5, 9])

    result = classifier.pmf(
        np.array([9, 2, 100]),
        x_new=x_new,
        x_prev=x_prev,
        y_prev=y_prev,
    )

    assert result.shape == (3, 4)
    np.testing.assert_allclose(result[0], probabilities[:, 2])
    np.testing.assert_allclose(result[1], probabilities[:, 0])
    np.testing.assert_array_equal(result[2], 0.0)


def test_classifier_sample_preserves_labels_and_is_reproducible(monkeypatch):
    classes = np.array(["low", "medium", "high"])
    classifier, probabilities = _stub_classifier(monkeypatch, classes=classes)
    x_new, x_prev, _ = _data()
    y_prev = np.array(["low", "medium", "high", "low", "medium", "high"])
    key = jr.key(7)

    first, metadata = classifier.sample(
        key, x_new=x_new, x_prev=x_prev, y_prev=y_prev, size=6
    )
    second, _ = classifier.sample(
        key, x_new=x_new, x_prev=x_prev, y_prev=y_prev, size=6
    )

    assert first.shape == (6, 4)
    assert first.dtype == classes.dtype
    assert set(np.unique(first)) <= set(classes)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(metadata["probs"], probabilities)


def test_real_quantile_distribution_checkpoint_free():
    from tabicl._model.quantile_dist import QuantileDistribution
    import torch

    raw_quantiles = torch.tensor(
        [
            [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
            [-1.0, -0.7, -0.4, -0.2, 0.0, 0.3, 0.7, 1.2, 1.8],
        ],
        dtype=torch.float32,
    )
    distribution = QuantileDistribution(raw_quantiles)
    thresholds = torch.tensor(
        [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]], dtype=torch.float32
    )

    probabilities = distribution.cdf(thresholds)
    recovered = distribution.icdf(probabilities)

    assert probabilities.shape == thresholds.shape
    assert torch.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    torch.testing.assert_close(recovered, thresholds)
