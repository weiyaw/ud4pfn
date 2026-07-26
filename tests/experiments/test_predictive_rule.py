import pytest

from experiments._shared import predictive_rule


class DummyAdapter:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.mark.parametrize(
    "pfn,task,constructor_name,checkpoint_name",
    [
        (
            "tabpfn",
            "classification",
            "TabPFNClassifierPPD",
            "tabpfn-v3-classifier-v3_default.ckpt",
        ),
        (
            "tabpfn",
            "regression",
            "TabPFNRegressorPPD",
            "tabpfn-v3-regressor-v3_default.ckpt",
        ),
        (
            "tabicl",
            "classification",
            "TabICLClassifierPPD",
            "tabicl-classifier-v2-20260212.ckpt",
        ),
        (
            "tabicl",
            "regression",
            "TabICLRegressorPPD",
            "tabicl-regressor-v2-20260212.ckpt",
        ),
    ],
)
def test_build_predictive_rule_selects_adapter(
    monkeypatch, pfn, task, constructor_name, checkpoint_name
):
    monkeypatch.setattr(predictive_rule, constructor_name, DummyAdapter)

    adapter = predictive_rule.build_predictive_rule(pfn, task, n_estimators=7)

    assert adapter.kwargs["n_estimators"] == 7
    assert adapter.kwargs["model_path"].endswith(checkpoint_name)
    if pfn == "tabpfn":
        assert adapter.kwargs["fit_mode"] == "low_memory"
        assert adapter.kwargs["softmax_temperature"] == 1.0
        assert "allow_auto_download" not in adapter.kwargs
    else:
        assert adapter.kwargs["allow_auto_download"] is True
        assert "fit_mode" not in adapter.kwargs
        assert ("softmax_temperature" in adapter.kwargs) == (
            task == "classification"
        )


def test_build_predictive_rule_rejects_unknown_pfn():
    with pytest.raises(
        ValueError, match=r"Unknown pfn 'other'.*tabpfn, tabicl"
    ):
        predictive_rule.build_predictive_rule(
            "other", "classification", n_estimators=1
        )


def test_build_predictive_rule_rejects_unknown_task():
    with pytest.raises(ValueError, match="Unknown task 'other'"):
        predictive_rule.build_predictive_rule("tabpfn", "other", n_estimators=1)
