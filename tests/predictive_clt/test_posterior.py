import importlib
import pathlib
import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="TabPFN adapters depend on torch")


@pytest.fixture()
def posterior_module(monkeypatch):
    """Import predictive_clt.posterior with lightweight TabPFN stubs."""

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

        # sample method is needed for sample_gn_plus_1
        def sample(
            self, rng=None, x_new=None, x_prev=None, y_prev=None, size=1, key=None
        ):
            # dummy implementation
            draws = np.tile(
                np.linspace(0.1, 0.9, x_new.shape[0], dtype=np.float32), (size, 1)
            )
            return draws, {}

        # predict_event method is needed for compute_gn
        def predict_event(self, t, x_new, x_prev, y_prev):
            # dummy implementation
            t = np.atleast_1d(t)
            return np.full((t.shape[0], x_new.shape[0]), 0.7, dtype=np.float32)

    class DummyClassifier:
        def __init__(self, *args, **kwargs):
            self.classes_ = np.array([0, 1])

        def fit(self, X, y):
            return self

        def predict_proba(self, X):
            base = np.array([0.25, 0.75], dtype=np.float32)
            return np.tile(base, (X.shape[0], 1))

        # sample method is needed for sample_gn_plus_1
        def sample(
            self, rng=None, x_new=None, x_prev=None, y_prev=None, size=1, key=None
        ):
            # dummy implementation
            # return shape (size, m)
            return np.zeros((size, x_new.shape[0]), dtype=np.float32), {}

        # predict_event method is needed for compute_gn
        def predict_event(self, t, x_new, x_prev, y_prev):
            t = np.atleast_1d(t)
            return np.full((t.shape[0], x_new.shape[0]), 0.42, dtype=np.float32)

    stub_tabpfn = types.ModuleType("tabpfn")
    stub_tabpfn.TabPFNClassifier = DummyClassifier
    stub_tabpfn.TabPFNRegressor = DummyRegressor
    monkeypatch.setitem(sys.modules, "tabpfn", stub_tabpfn)
    for name in (
        "predictive_clt",
        "predictive_clt.posterior",
        "predictive_clt.tabpfn_adapter",
    ):
        sys.modules.pop(name, None)

    posterior = importlib.import_module("predictive_clt.posterior")
    posterior = importlib.reload(posterior)
    return posterior


def test_compute_gn_uses_predict_event(posterior_module):
    calls = {"fit": 0, "predict_event": 0}

    class Stub:
        def fit(self, X, y):
            calls["fit"] += 1
            return self

        def predict_event(self, t, x_new, x_prev, y_prev):
            calls["predict_event"] += 1
            t = np.atleast_1d(t)
            return np.full((t.shape[0], x_new.shape[0]), 0.42, dtype=np.float32)

    x_prev = np.ones((4, 2), dtype=np.float32)
    y_prev = np.arange(4, dtype=np.float32)
    x_grid = np.zeros((3, 2), dtype=np.float32)
    t = np.array([0])

    result = posterior_module.compute_gn(Stub(), t, x_grid, x_prev, y_prev)

    assert result.shape == (1, x_grid.shape[0])
    np.testing.assert_allclose(result, 0.42)
    assert calls == {"fit": 0, "predict_event": 1}
    # Actually compute_gn implementation:
    # return predictive_rule.predict_event(
    #     x_new=x_grid, x_prev=x_prev, y_prev=y_prev
    # )
    # It does NOT call fit explicitly. predict_event inside the predictive rule
    # might call fit.
    # But here Stub.predict_event does not call fit.
    # So calls should be {"fit": 0, "predict_event": 1} if compute_gn doesn't call fit.
    # Looking at posterior.py: compute_gn simply calls
    # predictive_rule.predict_event.
    # So fit is not called by compute_gn itself.


def test_compute_gn_classifier_single_class_prefix(posterior_module):
    predictive_rule = posterior_module.TabPFNClassifierPPD()
    predictive_rule.predict_event = lambda **kwargs: pytest.fail(
        "predictive rule should not be called"
    )
    x_prev = np.array([[0.0], [1.0]], dtype=np.float32)
    y_prev = np.array([1, 1], dtype=np.int32)
    x_grid = np.array([[-1.0], [0.0], [1.0]], dtype=np.float32)

    result = posterior_module.compute_gn(
        predictive_rule, np.array([0, 1, 2]), x_grid, x_prev, y_prev
    )

    assert result.dtype == np.float32
    np.testing.assert_array_equal(
        result,
        np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.float32),
    )


def test_compute_gn_regressor_low_diversity_prefix(posterior_module):
    predictive_rule = posterior_module.TabPFNRegressorPPD()
    predictive_rule.predict_event = lambda **kwargs: pytest.fail(
        "predictive rule should not be called"
    )
    x_prev = np.array([[0.0], [1.0]], dtype=np.float32)
    y_prev = np.array([0.5, 0.5], dtype=np.float32)
    x_grid = np.array([[-1.0], [1.0]], dtype=np.float32)

    result = posterior_module.compute_gn(
        predictive_rule,
        np.array([0.25, 0.5, 0.75]),
        x_grid,
        x_prev,
        y_prev,
    )

    assert result.dtype == np.float32
    np.testing.assert_array_equal(
        result,
        np.array([[0, 0], [1, 1], [1, 1]], dtype=np.float32),
    )


def test_compute_gn_tabicl_classifier_single_class_prefix(posterior_module):
    predictive_rule = posterior_module.TabICLClassifierPPD(
        n_estimators=1, allow_auto_download=False
    )
    predictive_rule.predict_event = lambda **kwargs: pytest.fail(
        "predictive rule should not be called"
    )
    x_prev = np.array([[0.0], [1.0]], dtype=np.float32)
    y_prev = np.array([5, 5], dtype=np.int32)
    x_grid = np.array([[-1.0], [0.0], [1.0]], dtype=np.float32)

    result = posterior_module.compute_gn(
        predictive_rule, np.array([4, 5]), x_grid, x_prev, y_prev
    )

    np.testing.assert_array_equal(
        result, np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
    )


def test_compute_gn_tabicl_regressor_low_diversity_prefix(posterior_module):
    predictive_rule = posterior_module.TabICLRegressorPPD(
        n_estimators=1, allow_auto_download=False
    )
    predictive_rule.predict_event = lambda **kwargs: pytest.fail(
        "predictive rule should not be called"
    )
    x_prev = np.array([[0.0], [1.0]], dtype=np.float32)
    y_prev = np.array([0.5, 0.5], dtype=np.float32)
    x_grid = np.array([[-1.0], [1.0]], dtype=np.float32)

    result = posterior_module.compute_gn(
        predictive_rule,
        np.array([0.25, 0.5]),
        x_grid,
        x_prev,
        y_prev,
    )

    np.testing.assert_array_equal(
        result, np.array([[0, 0], [1, 1]], dtype=np.float32)
    )


def test_sample_gn_plus_1_supports_tabicl_adapter(posterior_module):
    import jax.random as jr

    predictive_rule = posterior_module.TabICLRegressorPPD(
        n_estimators=1, allow_auto_download=False
    )

    def sample(key, x_new, x_prev, y_prev, size=1):
        return np.zeros((size, x_new.shape[0]), dtype=np.float32), {}

    def predict_event(t, x_new, x_prev, y_prev):
        return np.full(
            (np.atleast_1d(t).size, x_new.shape[0]), 0.5, dtype=np.float32
        )

    predictive_rule.sample = sample
    predictive_rule.predict_event = predict_event
    draws = posterior_module.sample_gn_plus_1(
        key=jr.key(0),
        predictive_rule=predictive_rule,
        t=np.array([0.5]),
        x_grid=np.zeros((2, 1)),
        x_prev=np.ones((3, 1)),
        y_prev=np.arange(3),
        size=3,
    )

    assert draws.shape == (3, 1, 2)
    np.testing.assert_array_equal(draws, 0.5)


def test_sample_gn_plus_1_returns_expected_shape(posterior_module):
    import jax.random as jr

    class Stub:
        def fit(self, X, y):
            return self

        def sample(self, key, x_new, x_prev, y_prev, size=1):
            draws = np.tile(
                np.linspace(0.1, 0.9, x_new.shape[0], dtype=np.float32), (size, 1)
            )
            return draws, {}

        def predict_event(self, t, x_new, x_prev, y_prev):
            t = np.atleast_1d(t)
            return np.linspace(0.2, 0.8, x_new.shape[0], dtype=np.float32)[None, :]

    key = jr.key(0)
    x_prev = np.ones((5, 2), dtype=np.float32)
    y_prev = np.linspace(0.0, 1.0, 5, dtype=np.float32)
    x_grid = np.zeros((4, 2), dtype=np.float32)
    t = np.array([0.5])

    draws = posterior_module.sample_gn_plus_1(
        key=key,
        predictive_rule=Stub(),
        t=t,
        x_grid=x_grid,
        x_prev=x_prev,
        y_prev=y_prev,
        size=6,
    )

    assert draws.shape == (6, 1, x_grid.shape[0])


def test_sample_gn_plus_1_supports_one_draw(posterior_module):
    import jax.random as jr

    class Stub:
        def sample(self, key, x_new, x_prev, y_prev, size=1):
            return np.zeros((size, x_new.shape[0]), dtype=np.float32), {}

        def predict_event(self, t, x_new, x_prev, y_prev):
            return np.zeros((np.atleast_1d(t).size, x_new.shape[0]))

    draws = posterior_module.sample_gn_plus_1(
        key=jr.key(0),
        predictive_rule=Stub(),
        t=np.array([0.5]),
        x_grid=np.zeros((2, 1)),
        x_prev=np.ones((3, 1)),
        y_prev=np.arange(3),
        size=1,
    )

    assert draws.shape == (1, 1, 2)


def test_compute_g0_to_gn_calls_compute_gn_all_prefixes(monkeypatch, posterior_module):
    call_counts = []

    def fake_compute_gn(predictive_rule, t, x_grid, x_prev, y_prev):
        call_counts.append(len(y_prev))
        t = np.atleast_1d(t)
        return np.full((t.shape[0], x_grid.shape[0]), 0.5, dtype=np.float32)

    monkeypatch.setattr(posterior_module, "compute_gn", fake_compute_gn)

    x_prev = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    y_prev = np.array([0, 0, 1], dtype=np.int32)
    x_grid = np.linspace(0.0, 1.0, 4, dtype=np.float32)[:, None]
    t = np.array([0])

    # We need an instance of TabPFNClassifierPPD (or something that passes isinstance check if checked)
    # posterior.py checks isinstance(predictive_rule, TabPFNClassifierPPD)
    # Our fixture mocks pred_rule.TabPFNClassifierPPD, so we should instantiate that.

    predictive_rule = posterior_module.TabPFNClassifierPPD(y_star=1)

    vn = posterior_module.compute_g0_to_gn(predictive_rule, t, x_grid, x_prev, y_prev)

    assert vn.shape == (y_prev.shape[0] + 1, 1, x_grid.shape[0])
    assert np.all(np.isnan(vn[0]))
    # All steps call compute_gn which returns 0.5
    np.testing.assert_allclose(vn[1], 0.5)
    np.testing.assert_allclose(vn[2], 0.5)
    np.testing.assert_allclose(vn[3], 0.5)
    assert call_counts == [1, 2, 3]


def test_compute_g0_to_gn_regressor_calls_compute_gn_each_prefix(
    monkeypatch, posterior_module
):
    call_counts = []

    def fake_compute_gn(predictive_rule, t, x_grid, x_prev, y_prev):
        call_counts.append(len(y_prev))
        t = np.atleast_1d(t)
        return np.full((t.shape[0], x_grid.shape[0]), 0.7, dtype=np.float32)

    monkeypatch.setattr(posterior_module, "compute_gn", fake_compute_gn)

    x_prev = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    y_prev = np.array([1.5, 1.5, 1.8], dtype=np.float32)
    x_grid = np.linspace(0.0, 1.0, 5, dtype=np.float32)[:, None]
    t = np.array([0.5])

    predictive_rule = posterior_module.TabPFNRegressorPPD(y_star=0.5)

    vn = posterior_module.compute_g0_to_gn(predictive_rule, t, x_grid, x_prev, y_prev)

    assert vn.shape == (y_prev.shape[0] + 1, 1, x_grid.shape[0])
    assert np.all(np.isnan(vn[0]))
    np.testing.assert_allclose(vn[1:], 0.7)
    # k=1: len=1 < 2 short circuit -> empirical cdf?
    # posterior.py:
    # if isinstance(predictive_rule, TabPFNRegressorPPD):
    #    if y_prev.shape[0] < 2 or np.unique(y_prev).size < 2:
    #        return float(np.mean(y_prev <= predictive_rule.y_star))

    # Wait, compute_gn has guard clauses!
    # But we PATCHED compute_gn.
    # The loop in compute_g0_to_gn calls compute_gn.
    # So if compute_gn logic is patched, it should just be called.
    # UNLESS compute_g0_to_gn ALSO has guard clauses?
    # posterior.py:
    # for k in trange(1, n + 1):
    #     if isinstance(predictive_rule, TabPFNClassifierPPD): ...
    #     g0_to_gn[k, :] = compute_gn(...)

    # It seems compute_g0_to_gn for regressor does NOT have inner loop guards, it delegates to compute_gn.
    # So call_counts should be [1, 2, 3].

    assert call_counts == [1, 2, 3]


def test_compute_un_pointwise_shape(posterior_module):
    rng = np.random.default_rng(1)
    m = 3
    size = 5
    n = 4
    gn = np.linspace(0.2, 0.6, m, dtype=np.float32)
    draws = np.stack([rng.uniform(0.1, 0.9, m) for _ in range(size)], axis=0).astype(
        np.float32
    )

    pointwise = posterior_module.compute_un(
        gn=gn, gn_plus_1=draws, n=n, type="pointwise"
    )

    assert pointwise.shape == (m,)


def test_compute_un_simultaneous_shape(posterior_module):
    rng = np.random.default_rng(2)
    m = 4
    size = 6
    n = 5
    gn = np.linspace(0.1, 0.4, m, dtype=np.float32)
    draws = np.stack([rng.uniform(0.0, 1.0, m) for _ in range(size)], axis=0).astype(
        np.float32
    )

    simultaneous = posterior_module.compute_un(
        gn=gn, gn_plus_1=draws, n=n, type="simultaneous"
    )

    assert simultaneous.shape == (m, m)


def test_compute_vn_uses_increments_from_two_through_n(posterior_module):
    trajectory = np.array(
        [
            [np.nan, np.nan],
            [0.0, 0.0],
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )
    weighted = np.array([[2.0, 4.0], [6.0, 6.0]])

    pointwise = posterior_module.compute_vn(trajectory, type="pointwise")
    simultaneous = posterior_module.compute_vn(trajectory, type="simultaneous")

    np.testing.assert_allclose(pointwise, np.mean(weighted**2, axis=0))
    np.testing.assert_allclose(
        simultaneous,
        np.mean(np.einsum("ij,ik->ijk", weighted, weighted), axis=0),
    )
