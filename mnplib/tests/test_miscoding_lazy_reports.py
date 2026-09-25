"""Metrics-only reports do not request pairwise redundancy."""

from inspect import signature

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from mnplib.miscoding import Miscoding, rank_features, select_features, subset_analysis


SUBSET_FIELDS = {
    "deficiency", "surplus", "miscoding", "is_reliable", "failure_reason",
    "resolved_n_bins", "n_samples", "n_observed_joint_states",
    "mean_joint_occupancy", "n_singleton_joint_states", "singleton_fraction",
    "mask", "n_selected_features", "selected_features", "selected_feature_names",
}


@pytest.fixture
def data():
    rng = np.random.default_rng(17)
    X = pd.DataFrame(rng.integers(0, 2, size=(300, 5)), columns=list("abcde"))
    return X, X.a.to_numpy()


@pytest.mark.parametrize("subset", [[], [0], [0, 1, 2], [0, 1, 2, 3, 4]])
@pytest.mark.parametrize("functional", [False, True])
def test_subset_reports_do_not_request_pairwise_redundancy(data, subset, functional, monkeypatch):
    reference_metric = Miscoding().fit(*data)
    reference_metric.redundancy_matrix()
    reference = reference_metric.subset_analysis(subset)
    metric = Miscoding().fit(*data)
    def unexpected(*args):
        raise AssertionError("Pairwise redundancy must stay lazy.")
    monkeypatch.setattr(Miscoding, "_feature_pair_redundancy", unexpected)
    actual = (subset_analysis(subset, X=data[0], y=data[1]) if functional
              else metric.subset_analysis(subset))
    assert set(actual) == SUBSET_FIELDS
    for key, value in actual.items():
        np.testing.assert_equal(value, reference[key])
    for method in (metric.miscoding_subset, metric.deficiency_subset, metric.surplus_subset):
        method(subset)
    assert metric._redundancy_matrix_ is None


def test_subset_analysis_signatures():
    assert list(signature(Miscoding.subset_analysis).parameters) == ["self", "subset"]
    assert list(signature(subset_analysis).parameters) == ["subset", "X", "y", "X_type", "y_type"]


def test_model_analysis_uses_subset_report_without_pair_queries(data, monkeypatch):
    X, y = data
    model = DecisionTreeClassifier(random_state=0).fit(X, y)
    metric = Miscoding().fit(X, y)

    def unexpected(*args):
        raise AssertionError("Model subset diagnostics must not request pairwise redundancy.")

    monkeypatch.setattr(metric, "_feature_pair_redundancy", unexpected)
    report = metric.model_analysis(model)
    assert set(report) == SUBSET_FIELDS
    assert report["is_reliable"]
    assert report["miscoding"] == pytest.approx(0.0)
    assert metric._redundancy_matrix_ is None


@pytest.mark.parametrize("method", ["rank_features", "select_features"])
def test_detailed_search_matches_full_report_without_redundancy(data, method, monkeypatch):
    options = {"criterion": "miscoding"} if method == "rank_features" else {}
    expected = getattr(Miscoding().fit(*data), method)(return_details=True, **options)
    metric = Miscoding().fit(*data)
    def unexpected(*args):
        raise AssertionError("Search must not compute unused redundancy.")
    monkeypatch.setattr(metric, "_feature_pair_redundancy", unexpected)
    actual = getattr(metric, method)(return_details=True, include_redundancy=False, **options)
    assert "redundancy" not in actual
    pd.testing.assert_frame_equal(actual["path"], expected["path"])
    key = "feature_order" if method == "rank_features" else "selected_features"
    assert actual[key] == expected[key]
    if method == "select_features":
        assert set(actual["subset"]) == set(expected["subset"]) == SUBSET_FIELDS
        for name, value in actual["subset"].items():
            np.testing.assert_equal(value, expected["subset"][name])
    assert metric._redundancy_matrix_ is None


def test_functional_options_and_unreliable_subset(monkeypatch):
    X = np.arange(60).reshape(20, 3).astype(str)
    y = np.arange(20).astype(str)
    def unexpected(*args):
        raise AssertionError("Pairwise redundancy must stay lazy.")
    monkeypatch.setattr(Miscoding, "_feature_pair_redundancy", unexpected)
    result = subset_analysis([0, 1, 2], X=X, y=y)
    assert set(result) == SUBSET_FIELDS
    assert not result["is_reliable"] and np.isnan(result["miscoding"])
    for function in (select_features, rank_features):
        details = function(X=X, y=y, return_details=True, include_redundancy=False)
        assert "redundancy" not in details


def test_redundancy_can_be_requested_after_metrics_only_search(data):
    metric = Miscoding().fit(*data)
    metric.rank_features(criterion="miscoding", return_details=True, include_redundancy=False)
    assert metric._redundancy_matrix_ is None
    pd.testing.assert_frame_equal(metric.redundancy_matrix(), Miscoding().fit(*data).redundancy_matrix())
