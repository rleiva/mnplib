"""Consistent empirical contexts, compact caching, and subset diagnostics."""

from dataclasses import FrozenInstanceError, fields
import gc
import weakref

import numpy as np
import pandas as pd
import pytest

import mnplib.miscoding as miscoding_module
from mnplib.miscoding import Miscoding
from mnplib.utils import _resolve_bins, empirical_distribution_array, empirical_distribution_vector


def record_distributions(monkeypatch):
    calls = []
    original = miscoding_module.empirical_distribution_array

    def record(X, *, numeric, n_bins):
        calls.append((X.shape[1], tuple(numeric), n_bins))
        return original(X, numeric=numeric, n_bins=n_bins)

    monkeypatch.setattr(miscoding_module, "empirical_distribution_array", record)
    return calls


def test_fit_resolves_one_feature_context_and_caches_compact_statistics(monkeypatch):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(120, 4))
    y = rng.normal(size=len(X))
    resolutions = []
    original = miscoding_module._resolve_bins

    def resolve(policy, n_samples, *, subset_size):
        resolutions.append((policy, n_samples, subset_size))
        return original(policy, n_samples, subset_size=subset_size)

    monkeypatch.setattr(miscoding_module, "_resolve_bins", resolve)
    calls = record_distributions(monkeypatch)
    metric = Miscoding().fit(X, y)

    bins = _resolve_bins("adaptive", len(X), subset_size=1)
    assert resolutions == [("adaptive", len(X), 1)]
    assert calls == [(1, (True,), bins)] * 5 + [(2, (True, True), bins)] * 4
    assert len(metric._empirical_cache_) == 9
    for statistics in metric._empirical_cache_.values():
        assert {field.name for field in fields(statistics)} == {
            "code_length", "n_samples", "n_states", "n_singletons",
        }
        assert isinstance(statistics.code_length, float)
        assert isinstance(statistics.n_samples, int)
        assert isinstance(statistics.n_states, int)
        assert isinstance(statistics.n_singletons, int)
        assert statistics.n_samples == len(X)
        assert 0 <= statistics.n_singletons <= statistics.n_states <= len(X)
        with pytest.raises(FrozenInstanceError):
            statistics.code_length = 0.0
    assert metric._redundancy_matrix_ is None


@pytest.mark.parametrize("kind", ["numeric", "categorical", "mixed"])
@pytest.mark.parametrize("target", ["numeric", "categorical", "constant"])
def test_feature_components_share_joint_and_marginal_code_lengths(kind, target):
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(120, 3)), columns=list("abc"))
    if kind != "numeric":
        for column in (X.columns if kind == "categorical" else ["b"]):
            X[column] = np.where(X[column] > 0, "high", "low")
    X["constant"] = 1.0
    y = (rng.normal(size=len(X)) if target == "numeric" else
         rng.integers(0, 3, size=len(X)) if target == "categorical" else np.zeros(len(X)))
    metric = Miscoding().fit(X, y)
    bins = _resolve_bins("auto", len(X))
    k_y = empirical_distribution_vector(y, numeric=metric.y_isnumeric_, n_bins=bins).code_length
    assert metric.target_code_length_ == k_y

    for j in range(X.shape[1]):
        k_x = empirical_distribution_vector(
            X.iloc[:, j], numeric=metric.X_isnumeric_[j], n_bins=bins,
        ).code_length
        k_xy = empirical_distribution_array(
            np.column_stack([X.iloc[:, j], y]),
            numeric=[metric.X_isnumeric_[j], metric.y_isnumeric_], n_bins=bins,
        ).code_length
        deficiency = 0.0 if k_y == 0 else np.clip((k_xy - k_x) / k_y, 0, 1)
        surplus = 0.0 if k_x == 0 else np.clip((k_xy - k_y) / k_x, 0, 1)
        assert metric.feature_code_lengths_[j] == k_x
        assert metric.deficiency_feature(j) == pytest.approx(deficiency)
        assert metric.surplus_feature(j) == pytest.approx(surplus)
        assert metric.miscoding_feature(j) == pytest.approx(max(deficiency, surplus))


def test_feature_scores_remain_finite_when_single_feature_subset_is_sparse():
    X = np.arange(12).reshape(-1, 1).astype(str)
    y = np.arange(12).astype(str)
    metric = Miscoding().fit(X, y)
    assert np.isfinite(metric.feature_analysis()[["deficiency", "surplus", "miscoding"]]).all().all()
    report = metric.subset_analysis([0])
    assert not report["is_reliable"]
    assert np.isnan(report["miscoding"])


def test_pair_and_subset_calculations_each_resolve_one_context(monkeypatch):
    X = np.random.default_rng(2).integers(0, 2, size=(200, 3))
    metric = Miscoding().fit(X, X[:, 0])
    resolutions = []
    original = metric._resolve_n_bins_for_subset

    def resolve(subset_size):
        resolutions.append(subset_size)
        return original(subset_size)

    monkeypatch.setattr(metric, "_resolve_n_bins_for_subset", resolve)
    metric._feature_pair_redundancy(0, 1)
    assert resolutions == [2]
    report = metric.subset_analysis([0, 1, 2])
    assert report["is_reliable"]
    assert resolutions == [2, 3]


def test_subset_cache_distinguishes_bins_and_reuses_feature_permutations(monkeypatch):
    x = np.linspace(0, 1, 120)
    X = np.column_stack([x, x, 2 * x])
    metric = Miscoding(y_type="numeric").fit(X, x)
    calls = record_distributions(monkeypatch)
    cached = set(metric._empirical_cache_)
    bins = _resolve_bins("adaptive", len(X), subset_size=2)
    report = metric.subset_analysis([2, 0])

    assert report["is_reliable"]
    assert calls == [(3, (True, True, True), bins), (2, (True, True), bins), (1, (True,), bins)]
    assert set(metric._empirical_cache_) - cached == {
        ((0, 2), True, bins), ((0, 2), False, bins), ((), True, bins),
    }
    subset_target = metric._empirical_cache_[((), True, bins)].code_length
    assert subset_target == empirical_distribution_vector(x, n_bins=bins).code_length
    assert subset_target != metric.target_code_length_

    calls.clear()
    repeated = metric.subset_analysis([0, 2])
    for key in ("deficiency", "surplus", "miscoding", "resolved_n_bins", "is_reliable"):
        assert repeated[key] == report[key]
    joint = metric._empirical_statistics_for_indices([2, 0], y_included=True, n_bins=bins)
    assert joint is metric._empirical_cache_[((0, 2), True, bins)]
    assert joint.n_states == report["n_observed_joint_states"]
    assert joint.n_singletons == report["n_singleton_joint_states"]
    assert calls == []


def test_unreliable_joint_does_not_compute_or_retain_marginals(monkeypatch):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 20))
    y = rng.normal(size=len(X))
    metric = Miscoding().fit(X, y)
    cached = metric._empirical_cache_.copy()
    calls = record_distributions(monkeypatch)
    report = metric.subset_analysis(list(range(20)))

    assert not report["is_reliable"]
    assert calls == [(21, (True,) * 21, 2)]
    assert set(metric._empirical_cache_) - cached.keys() == {(tuple(range(20)), True, 2)}
    assert all(metric._empirical_cache_[key] is value for key, value in cached.items())
    assert metric._redundancy_matrix_ is None

    calls.clear()
    repeated = metric.subset_analysis(list(reversed(range(20))))
    assert not repeated["is_reliable"]
    assert calls == []


@pytest.mark.parametrize("reliable", [False, True])
def test_single_feature_reliability_reuses_statistics_from_fit(monkeypatch, reliable):
    x = np.repeat(np.arange(8), 3) if reliable else np.arange(8)
    metric = Miscoding(X_type="categorical", y_type="categorical").fit(x[:, None], x)
    calls = record_distributions(monkeypatch)

    report = metric.subset_analysis([0])

    assert calls == []
    assert report["is_reliable"] is reliable
    assert report["n_observed_joint_states"] == 8
    assert report["n_singleton_joint_states"] == (0 if reliable else 8)
    assert report["singleton_fraction"] == (0.0 if reliable else 1.0)
    assert report["mean_joint_occupancy"] == (3.0 if reliable else 1.0)
    assert report["failure_reason"] == (None if reliable else "joint_distribution_too_sparse")
    for field in ("deficiency", "surplus", "miscoding"):
        assert report[field] == 0.0 if reliable else np.isnan(report[field])


def test_cached_statistics_match_empirical_distributions():
    x = np.repeat(np.arange(5), [1, 2, 3, 5, 8])
    X = pd.DataFrame({"a": x.astype(float), "b": np.where(x % 2, "odd", "even")})
    y = x % 3
    metric = Miscoding().fit(X, y)
    metric.subset_analysis([0, 1])

    for (features, y_included, bins), statistics in metric._empirical_cache_.items():
        columns = [metric.X_[:, j] for j in features]
        numeric = [metric.X_isnumeric_[j] for j in features]
        if y_included:
            columns.append(y)
            numeric.append(metric.y_isnumeric_)
        summary = empirical_distribution_array(
            np.asarray(columns, dtype=object).T, numeric=numeric, n_bins=bins,
        )
        assert statistics.code_length == summary.code_length
        assert statistics.n_samples == summary.n_samples
        assert statistics.n_states == summary.n_states
        assert statistics.n_singletons == np.count_nonzero(summary.counts == 1)


def test_distribution_arrays_are_released_after_extracting_statistics(monkeypatch):
    references = []
    original = miscoding_module.empirical_distribution_array

    def record(X, **kwargs):
        summary = original(X, **kwargs)
        references.extend(weakref.ref(value) for value in (
            summary, summary.states, summary.counts, summary.probabilities,
        ))
        return summary

    monkeypatch.setattr(miscoding_module, "empirical_distribution_array", record)
    X = np.random.default_rng(2).integers(0, 2, size=(200, 3))
    metric = Miscoding().fit(X, X[:, 0])
    metric.subset_analysis([0, 1])
    metric.redundancy_matrix()
    gc.collect()

    assert metric._empirical_cache_
    assert references
    assert all(reference() is None for reference in references)


@pytest.mark.parametrize("method", ["rank_features", "select_features"])
def test_search_path_diagnostics_match_subset_reports(method):
    rng = np.random.default_rng(4)
    X = rng.integers(0, 2, size=(400, 3))
    y = 4 * X[:, 0] + 2 * X[:, 1] + X[:, 2]
    metric = Miscoding().fit(X, y)
    path = getattr(metric, method)(return_details=True, include_redundancy=False)["path"]
    assert not path.empty
    for _, row in path.iterrows():
        report = metric.subset_analysis(list(row["selected_features"]))
        fields = (report.keys() & set(path.columns)) - {"selected_features", "selected_feature_names"}
        for key in fields:
            np.testing.assert_equal(row[key], report[key])
