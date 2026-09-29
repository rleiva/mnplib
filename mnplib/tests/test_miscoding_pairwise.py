"""On-demand pairwise miscoding, fitted snapshots, and cache lifecycle."""

import pickle
from itertools import combinations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from mnplib import Miscoding, Nescience
import mnplib.miscoding as miscoding_module
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


def test_fit_and_feature_queries_do_not_compute_pairwise_miscoding(data, monkeypatch):
    def unexpected_pair(self, i, j):
        raise AssertionError("Pairwise miscoding was not requested.")

    monkeypatch.setattr(Miscoding, "_feature_pair_miscoding", unexpected_pair)
    metric = Miscoding().fit(*data)
    for method in (metric.deficiency_feature, metric.surplus_feature, metric.miscoding_feature):
        assert np.isfinite(method()).all()
    assert len(metric.feature_analysis()) == data[0].shape[1]
    assert metric.subset_analysis([])["is_reliable"]
    assert metric.subset_analysis([0])["miscoding"] == 0.0
    assert metric._pairwise_miscoding_matrix_ is None
    assert all(len(features) <= 1 for features, _, _ in metric._empirical_cache_)
    Nescience().fit(*data)


def test_pair_queries_reuse_empirical_statistics_and_are_symmetric(data, monkeypatch):
    metric = Miscoding().fit(*data)
    for i, j in [(3, 1), (1, 4), (3, 4)]:
        metric._feature_pair_miscoding(i, j)
    assert metric._pairwise_miscoding_matrix_ is None

    def unexpected_distribution(*args, **kwargs):
        raise AssertionError("Cached empirical statistics must not be recomputed.")

    monkeypatch.setattr(miscoding_module, "empirical_distribution_array", unexpected_distribution)
    for i, j in [(1, 3), (1, 4), (3, 4)]:
        assert metric._feature_pair_miscoding(i, j) == metric._feature_pair_miscoding(j, i)
    assert metric._feature_pair_miscoding(2, 2) == 0.0


def test_full_matrix_completes_missing_pairs_once_and_returns_copies(data, monkeypatch):
    metric = Miscoding().fit(*data)
    metric._feature_pair_miscoding(0, 1)
    calls = []
    original_distribution = miscoding_module.empirical_distribution_array

    def record_distribution(X, **kwargs):
        calls.append(X.shape[1])
        return original_distribution(X, **kwargs)

    monkeypatch.setattr(miscoding_module, "empirical_distribution_array", record_distribution)
    expected = metric.pairwise_miscoding_
    assert calls.count(2) == 9
    assert metric._pairwise_miscoding_matrix_ is not None

    def unexpected_matrix(*args, **kwargs):
        raise AssertionError("The full matrix is already cached.")

    monkeypatch.setattr(metric, "_compute_pairwise_miscoding_matrix", unexpected_matrix)
    returned = metric.pairwise_miscoding_
    returned[:] = -1
    frame = metric.pairwise_miscoding_matrix()
    frame.iloc[:, :] = -1
    np.testing.assert_array_equal(metric.pairwise_miscoding_, expected)


@pytest.mark.parametrize("kind", ["numeric", "categorical", "mixed"])
def test_pairwise_values_match_empirical_formulas(kind):
    rng = np.random.default_rng(5)
    X = pd.DataFrame(rng.normal(size=(200, 3)), columns=["a", "b", "c"])
    if kind != "numeric":
        for column in (X.columns if kind == "categorical" else ["b"]):
            X[column] = np.where(X[column] > 0, "high", "low")
    X["constant_1"] = 1.0
    X["constant_2"] = 2.0
    y = rng.integers(0, 2, size=len(X))
    metric = Miscoding().fit(X, y)
    bins = _resolve_bins("adaptive", len(X), subset_size=2)
    expected = np.zeros((X.shape[1], X.shape[1]))
    lengths = [empirical_distribution_vector(
        X.iloc[:, j], numeric=metric.X_isnumeric_[j], n_bins=bins,
    ).code_length for j in range(X.shape[1])]
    for i, j in combinations(range(X.shape[1]), 2):
        joint = empirical_distribution_array(
            X.iloc[:, [i, j]], numeric=[metric.X_isnumeric_[i], metric.X_isnumeric_[j]],
            n_bins=bins,
        ).code_length
        denominator = max(lengths[i], lengths[j])
        value = (0.0 if denominator == 0 else
                 np.clip((joint - min(lengths[i], lengths[j])) / denominator, 0, 1))
        expected[i, j] = expected[j, i] = value

    assert metric._pairwise_miscoding_matrix_ is None
    np.testing.assert_allclose(metric.pairwise_miscoding_, expected)


@pytest.mark.parametrize("X_type", ["numeric", "categorical"])
def test_pairwise_miscoding_for_independent_duplicate_and_constant_features(X_type):
    independent = np.tile([[0, 0], [0, 1], [1, 0], [1, 1]], (30, 1))
    X = pd.DataFrame(independent, columns=["a", "b"])
    X["copy_a"] = X.a
    X["inverse_a"] = 1 - X.a
    X["constant_1"] = 1
    X["constant_2"] = 2
    metric = Miscoding(X_type=X_type).fit(X, X.a.to_numpy())

    matrix = metric.pairwise_miscoding_matrix()

    assert list(matrix.index) == list(matrix.columns) == list(X.columns)
    np.testing.assert_array_equal(np.diag(matrix), np.zeros(X.shape[1]))
    np.testing.assert_allclose(matrix, matrix.T)
    assert matrix.loc["a", "b"] == pytest.approx(1.0)
    assert matrix.loc["a", "copy_a"] == pytest.approx(0.0)
    assert matrix.loc["a", "inverse_a"] == pytest.approx(0.0)
    assert matrix.loc["constant_1", "constant_2"] == pytest.approx(0.0)
    assert matrix.loc["a", "constant_1"] == pytest.approx(1.0)


@pytest.mark.parametrize("constant", [False, True])
def test_pairwise_miscoding_of_a_single_feature_is_zero(data, constant):
    X, y = data
    X = np.ones((len(X), 1)) if constant else X[:, :1]
    metric = Miscoding().fit(X, y)

    pd.testing.assert_frame_equal(
        metric.pairwise_miscoding_matrix(),
        pd.DataFrame([[0.0]], index=["x0"], columns=["x0"]),
    )


def test_pairwise_miscoding_is_independent_of_target(data):
    X, y = data
    expected = Miscoding().fit(X, y).pairwise_miscoding_matrix()
    actual = Miscoding(y_type="numeric").fit(
        X, np.arange(len(X), dtype=float),
    ).pairwise_miscoding_matrix()

    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize("method", ["select_features", "rank_features"])
@pytest.mark.parametrize("return_details", [False, True])
def test_feature_search_materializes_full_matrix_only_for_details(data, method, return_details):
    metric = Miscoding().fit(*data)
    result = getattr(metric, method)(max_features=1, return_details=return_details)
    if return_details:
        pd.testing.assert_frame_equal(
            result["pairwise_miscoding"], metric.pairwise_miscoding_matrix(),
        )
        assert metric._pairwise_miscoding_matrix_ is not None
    else:
        assert metric._pairwise_miscoding_matrix_ is None


def test_functional_subset_score_does_not_compute_pairs(data, monkeypatch):
    calls = []
    original = Miscoding._feature_pair_miscoding

    def record_pair(self, i, j):
        calls.append((i, j))
        return original(self, i, j)

    monkeypatch.setattr(Miscoding, "_feature_pair_miscoding", record_pair)
    value = miscoding_subset([1, 3], X=data[0], y=data[1])
    assert np.isfinite(value)
    assert calls == []


@pytest.mark.parametrize("full_matrix", [False, True])
def test_refit_resets_all_pairwise_miscoding_caches(data, full_matrix):
    metric = Miscoding().fit(*data)
    metric.subset_analysis([0, 1])
    metric._feature_pair_miscoding(0, 1)
    if full_matrix:
        metric.pairwise_miscoding_matrix()
    X = pd.DataFrame(data[0][:, [2, 3]], columns=["first", "second"])
    y = data[0][:, 4]
    metric.fit(X, y)
    expected = Miscoding().fit(X, y)
    assert metric._pairwise_miscoding_matrix_ is None
    assert metric._empirical_cache_ == expected._empirical_cache_
    assert all(len(features) <= 1 for features, _, _ in metric._empirical_cache_)
    pd.testing.assert_frame_equal(metric.feature_analysis(), expected.feature_analysis())
    pd.testing.assert_frame_equal(metric.pairwise_miscoding_matrix(), expected.pairwise_miscoding_matrix())


@pytest.mark.parametrize("failure", ["target", "type"])
def test_failed_refit_does_not_expose_cached_diagnostics(data, failure):
    metric = Miscoding().fit(*data)
    metric.pairwise_miscoding_matrix()
    X, y = data
    if failure == "type":
        metric.set_params(X_type="invalid")
    else:
        y = None
    with pytest.raises(ValueError):
        metric.fit(X, y)
    with pytest.raises(NotFittedError):
        check_is_fitted(metric)
    with pytest.raises(NotFittedError):
        metric.pairwise_miscoding_
    with pytest.raises(NotFittedError):
        metric.subset_analysis([0, 1])
    assert metric._empirical_cache_ == {}
    assert metric._pairwise_miscoding_matrix_ is None


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
    pd.testing.assert_frame_equal(metric.pairwise_miscoding_matrix(), reference.pairwise_miscoding_matrix())


def test_parameter_changes_apply_on_refit_without_mixing_cached_quantities():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(500, 3))
    y = X[:, 0] + X[:, 1]
    metric = Miscoding().fit(X, y)
    reference = Miscoding().fit(X, y)
    metric.subset_analysis([0, 1])
    metric.set_params(X_type="categorical", y_type="categorical")
    assert metric.get_params()["X_type"] == "categorical"
    assert metric.subset_analysis([1, 2])["resolved_n_bins"] == _resolve_bins("adaptive", len(X), subset_size=2)
    assert metric.subset_analysis([1, 2])["miscoding"] == reference.subset_analysis([1, 2])["miscoding"]
    np.testing.assert_allclose(metric.pairwise_miscoding_, reference.pairwise_miscoding_)
    metric.fit(X, y)
    reference = Miscoding(X_type="categorical", y_type="categorical").fit(X, y)
    assert metric.subset_analysis([1, 2])["resolved_n_bins"] is None
    np.testing.assert_allclose(metric.pairwise_miscoding_, reference.pairwise_miscoding_)
    pd.testing.assert_frame_equal(metric.feature_analysis(), reference.feature_analysis())


@pytest.mark.parametrize("cache_state", ["features", "subset", "matrix"])
def test_fitted_pairwise_miscoding_caches_support_serialization_and_cloning(data, cache_state):
    metric = Miscoding().fit(*data)
    if cache_state == "subset":
        metric.subset_analysis([0, 1])
        metric._feature_pair_miscoding(0, 1)
    elif cache_state == "matrix":
        metric.pairwise_miscoding_matrix()
    restored = pickle.loads(pickle.dumps(metric))
    assert restored._empirical_cache_ == metric._empirical_cache_
    assert (restored._pairwise_miscoding_matrix_ is None) == (metric._pairwise_miscoding_matrix_ is None)
    np.testing.assert_allclose(restored.pairwise_miscoding_, metric.pairwise_miscoding_)
    with pytest.raises(NotFittedError):
        clone(metric).pairwise_miscoding_


def test_pairwise_miscoding_attribute_requires_successful_fit():
    with pytest.raises(NotFittedError):
        Miscoding().pairwise_miscoding_
