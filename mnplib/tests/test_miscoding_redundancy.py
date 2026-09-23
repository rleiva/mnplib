"""On-demand redundancy, fitted snapshots, and cache lifecycle."""

import pickle
from itertools import combinations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from mnplib import Miscoding, Nescience
from mnplib.miscoding import miscoding_subset
from mnplib.utils import (
    _resolve_bins,
    empirical_distribution_array,
    empirical_distribution_vector,
)


@pytest.fixture
def data():
    X = np.random.default_rng(7).integers(0, 2, size=(240, 5))
    return X, X[:, 0].copy()


def test_fit_and_feature_queries_do_not_compute_pairwise_redundancy(data, monkeypatch):
    def unexpected_pair(self, i, j):
        raise AssertionError("Pairwise redundancy was not requested.")

    monkeypatch.setattr(Miscoding, "_feature_pair_redundancy", unexpected_pair)
    metric = Miscoding().fit(*data)
    for method in (metric.deficiency_feature, metric.surplus_feature, metric.miscoding_feature):
        assert np.isfinite(method()).all()
    assert len(metric.feature_analysis()) == data[0].shape[1]
    assert metric.subset_analysis([])["is_reliable"]
    assert metric.subset_analysis([0])["miscoding"] == 0.0
    assert metric._pair_redundancy_cache_ == {}
    assert metric._redundancy_matrix_ is None
    assert all(len(features) <= 1 for features, _, _ in metric._code_length_cache_)
    Nescience().fit(*data)


def test_subset_requests_only_selected_pairs_and_reuses_symmetric_values(data, monkeypatch):
    metric = Miscoding().fit(*data)
    metric.subset_analysis([3, 1, 4])
    assert set(metric._pair_redundancy_cache_) == {(1, 3), (1, 4), (3, 4)}
    assert metric._redundancy_matrix_ is None

    def unexpected_length(*args, **kwargs):
        raise AssertionError("A cached pair must not recompute code lengths.")

    monkeypatch.setattr(metric, "_code_length_for_indices", unexpected_length)
    for i, j in [(1, 3), (1, 4), (3, 4)]:
        assert metric._feature_pair_redundancy(i, j) == metric._feature_pair_redundancy(j, i)
    assert metric._feature_pair_redundancy(2, 2) == 1.0


def test_full_matrix_completes_missing_pairs_once_and_returns_copies(data, monkeypatch):
    metric = Miscoding().fit(*data)
    metric.subset_analysis([0, 1])
    calls = []
    original_length = metric._code_length

    def record_length(columns, numeric, *, n_bins):
        calls.append(len(columns))
        return original_length(columns, numeric, n_bins=n_bins)

    monkeypatch.setattr(metric, "_code_length", record_length)
    expected = metric.redundancy_
    assert calls.count(2) == 9
    assert len(metric._pair_redundancy_cache_) == 10
    assert metric._redundancy_matrix_ is not None

    def unexpected_matrix(*args, **kwargs):
        raise AssertionError("The full matrix is already cached.")

    monkeypatch.setattr(metric, "_feature_redundancy_matrix", unexpected_matrix)
    returned = metric.redundancy_
    returned[:] = -1
    frame = metric.redundancy_matrix()
    frame.iloc[:, :] = -1
    np.testing.assert_array_equal(metric.redundancy_, expected)
    assert metric._pair_redundancy_cache_[(0, 1)] == expected[0, 1]
    report = metric.subset_analysis([0, 1])
    np.testing.assert_allclose(
        report["redundancy_weights"], 1.0 / expected[:2, :2].sum(axis=1),
    )


@pytest.mark.parametrize("n_bins", [3, "auto", "adaptive"])
@pytest.mark.parametrize("kind", ["numeric", "categorical", "mixed"])
def test_pairwise_values_and_subset_weights_match_empirical_formulas(n_bins, kind):
    rng = np.random.default_rng(5)
    X = pd.DataFrame(rng.normal(size=(200, 3)), columns=["a", "b", "c"])
    if kind != "numeric":
        for column in (X.columns if kind == "categorical" else ["b"]):
            X[column] = np.where(X[column] > 0, "high", "low")
    X["constant_1"] = 1.0
    X["constant_2"] = 2.0
    y = rng.integers(0, 2, size=len(X))
    metric = Miscoding(n_bins=n_bins).fit(X, y)
    bins = _resolve_bins(n_bins, len(X), subset_size=2)
    expected = np.eye(X.shape[1])
    lengths = [empirical_distribution_vector(
        X.iloc[:, j], numeric=metric.X_isnumeric_[j], n_bins=bins,
    ).code_length for j in range(X.shape[1])]
    for i, j in combinations(range(X.shape[1]), 2):
        joint = empirical_distribution_array(
            X.iloc[:, [i, j]], numeric=[metric.X_isnumeric_[i], metric.X_isnumeric_[j]],
            n_bins=bins,
        ).code_length
        denominator = max(lengths[i], lengths[j])
        value = (1.0 if denominator == 0 else
                 np.clip(1.0 - (joint - min(lengths[i], lengths[j])) / denominator, 0, 1))
        expected[i, j] = expected[j, i] = value

    selected = [0, 2, 4]
    report = metric.subset_analysis(selected)
    weights = 1.0 / expected[np.ix_(selected, selected)].sum(axis=1)
    np.testing.assert_allclose(report["redundancy_weights"], weights)
    np.testing.assert_allclose(report["feature_weights"], weights * metric.feature_code_lengths_[selected])
    assert metric._redundancy_matrix_ is None
    np.testing.assert_allclose(metric.redundancy_, expected)


@pytest.mark.parametrize("method", ["select_features", "rank_features"])
@pytest.mark.parametrize("return_details", [False, True])
def test_feature_search_materializes_full_matrix_only_for_details(data, method, return_details):
    metric = Miscoding().fit(*data)
    result = getattr(metric, method)(max_features=1, return_details=return_details)
    if return_details:
        assert result["redundancy"].shape == (5, 5)
        assert len(metric._pair_redundancy_cache_) == 10
        assert metric._redundancy_matrix_ is not None
    else:
        assert metric._pair_redundancy_cache_ == {}
        assert metric._redundancy_matrix_ is None


def test_functional_subset_score_does_not_compute_pairs(data, monkeypatch):
    calls = []
    original = Miscoding._feature_pair_redundancy

    def record_pair(self, i, j):
        calls.append((i, j))
        return original(self, i, j)

    monkeypatch.setattr(Miscoding, "_feature_pair_redundancy", record_pair)
    value = miscoding_subset([1, 3], X=data[0], y=data[1])
    assert np.isfinite(value)
    assert calls == []


@pytest.mark.parametrize("full_matrix", [False, True])
def test_refit_resets_all_redundancy_caches(data, full_matrix):
    metric = Miscoding().fit(*data)
    metric.subset_analysis([0, 1])
    if full_matrix:
        metric.redundancy_matrix()
    X = pd.DataFrame(data[0][:, [2, 3]], columns=["first", "second"])
    y = data[0][:, 4]
    metric.set_params(n_bins=3).fit(X, y)
    expected = Miscoding(n_bins=3).fit(X, y)
    assert metric._pair_redundancy_cache_ == {}
    assert metric._redundancy_matrix_ is None
    assert metric._empirical_summary_cache_ == {}
    assert all(len(features) <= 1 for features, _, _ in metric._code_length_cache_)
    pd.testing.assert_frame_equal(metric.feature_analysis(), expected.feature_analysis())
    pd.testing.assert_frame_equal(metric.redundancy_matrix(), expected.redundancy_matrix())


@pytest.mark.parametrize("failure", ["bins", "target", "type"])
def test_failed_refit_does_not_expose_cached_diagnostics(data, failure):
    metric = Miscoding().fit(*data)
    metric.redundancy_matrix()
    X, y = data
    if failure == "bins":
        metric.set_params(n_bins=1)
    elif failure == "type":
        metric.set_params(X_type="invalid")
    else:
        y = None
    with pytest.raises(ValueError):
        metric.fit(X, y)
    with pytest.raises(NotFittedError):
        check_is_fitted(metric)
    with pytest.raises(NotFittedError):
        metric.redundancy_
    with pytest.raises(NotFittedError):
        metric.subset_analysis([0, 1])
    assert metric._pair_redundancy_cache_ == {}
    assert metric._redundancy_matrix_ is None


@pytest.mark.parametrize("as_dataframe", [False, True])
def test_external_data_mutation_does_not_change_fitted_diagnostics(data, as_dataframe):
    X, y = data
    if as_dataframe:
        X = pd.DataFrame(X, columns=list("abcde"))
        y = pd.Series(y)
    metric = Miscoding().fit(X, y)
    reference = Miscoding().fit(X, y)
    metric.subset_analysis([0, 1])
    if as_dataframe:
        X.iloc[:, :] = 0
        X.columns = list("vwxyz")
        y.iloc[:] = 1
        pd.testing.assert_frame_equal(metric._model_X_, reference._model_X_)
    else:
        X[:] = 0
        y[:] = 1
        np.testing.assert_array_equal(metric._model_X_, reference._model_X_)
    np.testing.assert_array_equal(metric.X_, reference.X_)
    np.testing.assert_array_equal(metric.y_, reference.y_)
    pd.testing.assert_frame_equal(metric.feature_analysis(), reference.feature_analysis())
    assert metric.subset_analysis([1, 2])["miscoding"] == reference.subset_analysis([1, 2])["miscoding"]
    pd.testing.assert_frame_equal(metric.redundancy_matrix(), reference.redundancy_matrix())


def test_parameter_changes_apply_on_refit_without_mixing_cached_quantities():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(500, 3))
    y = X[:, 0] + X[:, 1]
    metric = Miscoding(n_bins=3).fit(X, y)
    reference = Miscoding(n_bins=3).fit(X, y)
    metric.subset_analysis([0, 1])
    metric.set_params(n_bins=5, X_type="categorical", y_type="categorical")
    assert metric.get_params()["n_bins"] == 5
    assert metric.subset_analysis([1, 2])["resolved_n_bins"] == 3
    assert metric.subset_analysis([1, 2])["miscoding"] == reference.subset_analysis([1, 2])["miscoding"]
    np.testing.assert_allclose(metric.redundancy_, reference.redundancy_)
    metric.fit(X, y)
    reference = Miscoding(n_bins=5, X_type="categorical", y_type="categorical").fit(X, y)
    assert metric.subset_analysis([1, 2])["resolved_n_bins"] is None
    np.testing.assert_allclose(metric.redundancy_, reference.redundancy_)
    pd.testing.assert_frame_equal(metric.feature_analysis(), reference.feature_analysis())


@pytest.mark.parametrize("cache_state", ["empty", "partial", "full"])
def test_fitted_redundancy_caches_support_serialization_and_cloning(data, cache_state):
    metric = Miscoding().fit(*data)
    if cache_state == "partial":
        metric.subset_analysis([0, 1])
    elif cache_state == "full":
        metric.redundancy_matrix()
    restored = pickle.loads(pickle.dumps(metric))
    assert restored._pair_redundancy_cache_ == metric._pair_redundancy_cache_
    assert (restored._redundancy_matrix_ is None) == (metric._redundancy_matrix_ is None)
    np.testing.assert_allclose(restored.redundancy_, metric.redundancy_)
    with pytest.raises(NotFittedError):
        clone(metric).redundancy_


def test_redundancy_attribute_requires_successful_fit():
    with pytest.raises(NotFittedError):
        Miscoding().redundancy_
