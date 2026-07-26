import importlib
import sys
import types
import pathlib

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="TabPFN adapters depend on torch")


@pytest.fixture()
def pred_rule_module(monkeypatch):
    """Import the adapter module with lightweight TabPFN stubs."""

    repo_root = pathlib.Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(repo_root))

    class DummyBarDistribution:
        def __init__(self, loc: float = 0.5):
            self.loc = float(loc)
            self.borders = torch.zeros(1)

        def icdf(self, logits_row, p):
            # mimic torch return type used by bardist_sample
            return torch.tensor(self.loc + p, dtype=torch.float32)

        def cdf(self, logits, ys):
            val = torch.clamp(torch.tensor(self.loc, dtype=torch.float32), 0.0, 1.0)
            return torch.full_like(ys, val)

    class DummyRegressor:
        def __init__(self, *args, **kwargs):
            self._loc = 0.5

        def fit(self, X, y):
            self._loc = float(np.mean(y))
            return self

        def predict(self, X, output_type="full"):
            logits = torch.zeros((X.shape[0], 4), dtype=torch.float32)
            return {"logits": logits, "criterion": DummyBarDistribution(self._loc)}

    class DummyClassifier:
        def __init__(self, *args, **kwargs):
            self.classes_ = np.array([0, 1])

        def fit(self, X, y):
            return self

        def predict_proba(self, X):
            base = np.array([0.25, 0.75], dtype=np.float32)
            return np.tile(base, (X.shape[0], 1))

        def predict_logits(self, X):
            probabilities = self.predict_proba(X).astype(np.float64)
            return np.log(probabilities)

    stub = types.ModuleType("tabpfn")
    stub.TabPFNClassifier = DummyClassifier
    stub.TabPFNRegressor = DummyRegressor
    monkeypatch.setitem(sys.modules, "tabpfn", stub)
    for name in (
        "predictive_clt",
        "predictive_clt.posterior",
        "predictive_clt.tabpfn_adapter",
    ):
        sys.modules.pop(name, None)

    pred_rule = importlib.import_module("predictive_clt.tabpfn_adapter")
    pred_rule = importlib.reload(pred_rule)
    return pred_rule


def test_assert_ppd_args_shape_valid(pred_rule_module):
    x_prev = np.ones((4, 2), dtype=np.float32)
    y_prev = np.arange(4, dtype=np.float32)
    x_new = np.zeros((3, 2), dtype=np.float32)
    pred_rule_module.assert_ppd_args_shape(x_new, x_prev, y_prev)


def test_assert_ppd_args_shape_invalid_mismatch(pred_rule_module):
    x_prev = np.ones((4, 3), dtype=np.float32)
    y_prev = np.arange(5, dtype=np.float32)
    x_new = np.zeros((3, 3), dtype=np.float32)
    with pytest.raises(AssertionError):
        pred_rule_module.assert_ppd_args_shape(x_new, x_prev, y_prev)


def test_tabpfn_regressor_sample(pred_rule_module):
    import jax.random as jr

    key = jr.PRNGKey(0)
    x_prev = np.random.randn(5, 2).astype(np.float32)
    y_prev = np.linspace(0.1, 0.9, 5, dtype=np.float32)
    x_new = np.random.randn(3, 2).astype(np.float32)

    reg = pred_rule_module.TabPFNRegressorPPD(y_star=0.5)

    samples, meta = reg.sample(
        key=key, x_new=x_new, x_prev=x_prev, y_prev=y_prev, size=4
    )

    assert samples.shape == (4, x_new.shape[0])
    assert "bardist" in meta
    assert "logits" in meta


def test_tabpfn_regressor_predict_event(pred_rule_module):
    x_prev = np.random.randn(5, 2).astype(np.float32)
    y_prev = np.linspace(0.0, 1.0, 5, dtype=np.float32)
    x_new = np.random.randn(3, 2).astype(np.float32)

    reg = pred_rule_module.TabPFNRegressorPPD(y_star=0.5)

    # In our dummy, cdf always returns self.loc (mean of y_prev).
    # mean(0.0..1.0) = 0.5. So cdf values should be 0.5.
    probs = reg.predict_event(
        t=np.array([0.5]), x_new=x_new, x_prev=x_prev, y_prev=y_prev
    )

    assert probs.shape == (1, x_new.shape[0])
    np.testing.assert_allclose(probs, 0.5)


def test_tabpfn_regressor_predict_event_vector(pred_rule_module):
    x_prev = np.random.randn(5, 2).astype(np.float32)
    y_prev = np.linspace(0.0, 1.0, 5, dtype=np.float32)
    x_new = np.random.randn(3, 2).astype(np.float32)

    reg = pred_rule_module.TabPFNRegressorPPD(
        y_star=0.5
    )  # y_star arg is still there for now

    # Pass vector t
    t_vals = np.array([0.2, 0.8])
    probs = reg.predict_event(t=t_vals, x_new=x_new, x_prev=x_prev, y_prev=y_prev)

    assert probs.shape == (2, x_new.shape[0])
    # Dummy CDF always returns 0.5 regardless of t
    np.testing.assert_allclose(probs, 0.5)


def test_tabpfn_classifier_sample(pred_rule_module):
    import jax.random as jr

    key = jr.PRNGKey(1)

    x_prev = np.random.randn(5, 2).astype(np.float32)
    y_prev = np.array([0, 1, 0, 1, 0])
    x_new = np.random.randn(3, 2).astype(np.float32)

    classifier = pred_rule_module.TabPFNClassifierPPD(y_star=1)

    samples, meta = classifier.sample(
        key=key, x_new=x_new, x_prev=x_prev, y_prev=y_prev, size=4
    )

    assert samples.shape == (4, x_new.shape[0])
    assert "probs" in meta
    assert meta["probs"].shape == (x_new.shape[0], 2)


def test_tabpfn_classifier_predict_event(pred_rule_module):
    x_prev = np.random.randn(5, 2).astype(np.float32)
    y_prev = np.array([0, 1, 0, 1, 0])
    x_new = np.random.randn(3, 2).astype(np.float32)

    # y_star = 1. The DummyClassifier always returns [0.25, 0.75].
    # So prob of class 1 is 0.75.
    classifier = pred_rule_module.TabPFNClassifierPPD(y_star=1)

    probs = classifier.predict_event(
        t=np.array([1]), x_new=x_new, x_prev=x_prev, y_prev=y_prev
    )

    assert probs.shape == (1, x_new.shape[0])
    np.testing.assert_allclose(probs, 0.75)


def test_tabpfn_classifier_predict_event_unsupported_class(pred_rule_module):
    x_prev = np.random.randn(5, 2).astype(np.float32)
    y_prev = np.array([0, 1, 0, 1, 0])
    x_new = np.random.randn(3, 2).astype(np.float32)

    # y_star = 99 (not in [0, 1]). Should return 0.
    classifier = pred_rule_module.TabPFNClassifierPPD(y_star=99)

    probs = classifier.predict_event(
        t=np.array([99]), x_new=x_new, x_prev=x_prev, y_prev=y_prev
    )

    assert probs.shape == (1, x_new.shape[0])
    np.testing.assert_allclose(probs, 0.0)


def test_tabpfn_classifier_predict_event_multiclass(pred_rule_module):
    x_prev = np.random.randn(5, 2).astype(np.float32)
    y_prev = np.array([0, 1, 0, 1, 0])
    x_new = np.random.randn(3, 2).astype(np.float32)

    # Classes: [0, 1]. Base probs: [0.25, 0.75].
    # Query t = [0, 1, 99].
    # Expected:
    # t=0 -> 0.25
    # t=1 -> 0.75
    # t=99 -> 0.0

    classifier = pred_rule_module.TabPFNClassifierPPD(y_star=1)
    t_vals = np.array([0, 1, 99])

    probs = classifier.predict_event(
        t=t_vals, x_new=x_new, x_prev=x_prev, y_prev=y_prev
    )

    assert probs.shape == (3, x_new.shape[0])
    np.testing.assert_allclose(probs[0], 0.25)
    np.testing.assert_allclose(probs[1], 0.75)
    np.testing.assert_allclose(probs[2], 0.0)
