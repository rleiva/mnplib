"""Metrics-only reports do not request pairwise redundancy."""

import numpy as np
import pandas as pd
import pytest

from mnplib.miscoding import Miscoding, rank_features, select_features, subset_analysis


@pytest.fixture
def data():
    rng = np.random.default_rng(17)
    X = pd.DataFrame(rng.integers(0, 2, size=(300, 5)), columns=list("abcde"))
    return X, X.a.to_numpy()


@pytest.mark.parametrize("subset", [[], [0], [0, 1, 2], [0, 1, 2, 3, 4]])
def test_metrics_only_subset_matches_full_report_without_pair_queries(data, subset, monkeypatch):
    reference = Miscoding().fit(*data).subset_analysis(subset)
    metric = Miscoding().fit(*data)
    def unexpected(*args):
        raise AssertionError("Pairwise redundancy must stay lazy.")
    monkeypatch.setattr(metric, "_feature_pair_redundancy", unexpected)
    actual = metric.subset_analysis(subset, include_weights=False)
    assert "redundancy_weights" not in actual and "feature_weights" not in actual
    for key, value in actual.items():
        np.testing.assert_equal(value, reference[key])
    for method in (metric.miscoding_subset, metric.deficiency_subset, metric.surplus_subset):
        method(subset)


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
    assert metric._pair_redundancy_cache_ == {}


def test_functional_options_and_unreliable_subset(monkeypatch):
    X = np.arange(60).reshape(20, 3).astype(str)
    y = np.arange(20).astype(str)
    def unexpected(*args):
        raise AssertionError("Pairwise redundancy must stay lazy.")
    monkeypatch.setattr(Miscoding, "_feature_pair_redundancy", unexpected)
    result = subset_analysis([0, 1, 2], X=X, y=y, include_weights=False)
    assert not result["is_reliable"] and np.isnan(result["miscoding"])
    for function in (select_features, rank_features):
        details = function(X=X, y=y, return_details=True, include_redundancy=False)
        assert "redundancy" not in details


def test_redundancy_can_be_requested_after_metrics_only_search(data):
    metric = Miscoding().fit(*data)
    metric.rank_features(criterion="miscoding", return_details=True, include_redundancy=False)
    assert metric._pair_redundancy_cache_ == {}
    pd.testing.assert_frame_equal(metric.redundancy_matrix(), Miscoding().fit(*data).redundancy_matrix())
