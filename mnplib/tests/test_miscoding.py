"""
Tests for the Miscoding class.

These tests target empirical miscoding diagnostics:

    - Miscoding(X_type="auto", y_type="auto", n_bins="auto", min_improvement=0.0)
    - fit(X, y)
    - feature_deficiency()
    - feature_surplus()
    - feature_miscoding()
    - feature_redundancy()
    - feature_analysis()
    - subset_analysis(subset)
    - miscoding_subset(subset, mode=...)
    - select_features(...)
    - rank_features(...)
    - feature_analysis(X, y, **kwargs)
    - feature_redundancy(X, y, **kwargs)
    - miscoding_subset(X, y, subset, **kwargs)
    - select_features(X, y, **kwargs)
    - rank_features(X, y, **kwargs)
"""

import numpy as np
import pandas as pd
import pytest

from types import SimpleNamespace

from sklearn.exceptions import NotFittedError

import mnplib.miscoding as miscoding_module
from mnplib.miscoding import (
    Miscoding,
    _adaptive_n_bins,
    _auto_n_bins,
    feature_analysis,
    feature_redundancy,
    miscoding_subset,
    rank_features,
    select_features,
)
from mnplib.utils import empirical_distribution


RELIABILITY_FIELDS = {
    "is_reliable",
    "failure_reason",
    "n_samples",
    "n_observed_joint_states",
    "mean_joint_occupancy",
    "n_singleton_joint_states",
    "singleton_fraction",
}


def make_simple_classification_data():
    """Return a small dataset with two perfect features and one weak feature."""
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    X = np.column_stack(
        [
            y,
            np.array([0, 1, 0, 1, 1, 0, 1, 0]),
            np.array([5, 5, 7, 7, 5, 7, 5, 7]),
        ]
    )
    return X, y


def make_redundant_noisy_data():
    """Return a dataset with two identical imperfect features."""
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    noisy = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    X = np.column_stack([noisy, noisy])
    return X, y


def make_distributed_signal_data():
    """Return a dataset whose target is represented across several features."""
    rng = np.random.default_rng(1)
    n_samples = 2000

    x0 = rng.integers(0, 2, size=n_samples)
    x1 = rng.integers(0, 2, size=n_samples)
    x2 = rng.integers(0, 2, size=n_samples)
    x3 = rng.integers(0, 2, size=n_samples)
    y = 8 * x0 + 4 * x1 + 2 * x2 + x3
    noise = rng.integers(0, 16, size=(n_samples, 6))

    return np.column_stack([x0, x1, x2, x3, noise]), y


def make_sparse_subset_data():
    """Return a small high-dimensional dataset with sparse joint states."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 20))
    y = rng.integers(0, 3, size=30)
    return X, y


def make_unreliable_extension_data():
    """Return a dataset whose one-feature extensions are too sparse."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(8, 5))
    y = np.arange(8) % 3
    return X, y


def make_reliable_low_dimensional_data():
    """Return a low-dimensional categorical dataset with reliable diagnostics."""
    rng = np.random.default_rng(2)
    n_samples = 500
    x0 = rng.integers(0, 2, size=n_samples)
    x1 = rng.integers(0, 2, size=n_samples)
    y = 2 * x0 + x1
    return np.column_stack([x0, x1]), y


def _summary(code_length, counts=(5.0, 5.0)):
    counts = np.asarray(counts, dtype=float)
    return SimpleNamespace(
        counts=counts,
        code_length=float(code_length),
        n_samples=int(np.sum(counts)),
        n_states=int(counts.size),
    )


def test_constructor_defaults():
    metric = Miscoding()

    assert metric.X_type == "auto"
    assert metric.y_type == "auto"
    assert metric.n_bins == "auto"
    assert metric.min_improvement == pytest.approx(0.0)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"X_type": "invalid"}, "X_type"),
        ({"y_type": "invalid"}, "y_type"),
        ({"n_bins": "invalid"}, "n_bins"),
        ({"n_bins": 1}, "n_bins"),
        ({"min_improvement": -0.1}, "min_improvement"),
    ],
)
def test_constructor_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Miscoding(**kwargs)


def test_adaptive_n_bins_matches_auto_for_one_feature():
    assert _adaptive_n_bins(1000, 1) == _auto_n_bins(1000)


def test_adaptive_n_bins_is_at_least_two():
    assert all(_adaptive_n_bins(1, size) >= 2 for size in range(1, 20))


def test_adaptive_n_bins_is_non_increasing_with_subset_size():
    values = [_adaptive_n_bins(1000, size) for size in range(1, 20)]

    assert values == sorted(values, reverse=True)


def test_adaptive_n_bins_is_non_decreasing_with_sample_count():
    values = [_adaptive_n_bins(n_samples, 3) for n_samples in [10, 50, 100, 500]]

    assert values == sorted(values)


@pytest.mark.parametrize(
    "n_samples, subset_size, message",
    [
        (0, 1, "n_samples"),
        (-1, 1, "n_samples"),
        (10, 0, "subset_size"),
        (10, -1, "subset_size"),
    ],
)
def test_adaptive_n_bins_rejects_invalid_inputs(n_samples, subset_size, message):
    with pytest.raises(ValueError, match=message):
        _adaptive_n_bins(n_samples, subset_size)


def test_fit_sets_fitted_attributes_for_numpy_array():
    X, y = make_simple_classification_data()

    metric = Miscoding(X_type="categorical", y_type="categorical").fit(X, y)

    assert metric.is_fitted_ is True
    assert metric.n_samples_in_ == X.shape[0]
    assert metric.n_features_in_ == X.shape[1]
    assert metric.target_code_length_ >= 0.0
    assert metric.feature_code_lengths_.shape == (X.shape[1],)
    assert metric.redundancy_.shape == (X.shape[1], X.shape[1])
    assert list(metric.feature_names_in_) == ["x0", "x1", "x2"]
    assert metric.X_isnumeric_ == [False, False, False]
    assert metric.y_isnumeric_ is False


def test_fit_preserves_dataframe_feature_names_and_infers_mixed_types():
    X = pd.DataFrame(
        {
            "perfect": [0, 0, 1, 1, 0, 1],
            "category": ["a", "b", "a", "b", "a", "b"],
        }
    )
    y = np.array([0, 0, 1, 1, 0, 1])

    metric = Miscoding(X_type="auto", y_type="categorical").fit(X, y)

    assert list(metric.feature_names_in_) == ["perfect", "category"]
    assert metric.X_isnumeric_ == [True, False]


def test_perfect_feature_has_zero_deficiency_and_zero_surplus():
    X, y = make_simple_classification_data()

    metric = Miscoding(X_type="categorical", y_type="categorical").fit(X, y)

    assert metric.feature_deficiency()[0] == pytest.approx(0.0)
    assert metric.feature_surplus()[0] == pytest.approx(0.0)
    assert metric.feature_miscoding()[0] == pytest.approx(0.0)


def test_feature_diagnostics_have_expected_shape_and_range():
    X, y = make_simple_classification_data()

    metric = Miscoding(X_type="categorical", y_type="categorical").fit(X, y)

    deficiency = metric.feature_deficiency()
    surplus = metric.feature_surplus()
    miscoding = metric.feature_miscoding()

    assert deficiency.shape == (X.shape[1],)
    assert surplus.shape == (X.shape[1],)
    assert miscoding.shape == (X.shape[1],)

    assert np.all((0.0 <= deficiency) & (deficiency <= 1.0))
    assert np.all((0.0 <= surplus) & (surplus <= 1.0))
    assert np.all((0.0 <= miscoding) & (miscoding <= 1.0))
    assert np.allclose(miscoding, np.maximum(deficiency, surplus))


def test_feature_diagnostic_methods_return_copies():
    X, y = make_simple_classification_data()

    metric = Miscoding(X_type="categorical", y_type="categorical").fit(X, y)

    deficiency = metric.feature_deficiency()
    deficiency[:] = 999.0

    assert not np.all(metric.feature_deficiency() == 999.0)


def test_feature_analysis_returns_expected_columns_and_sorted_rows():
    X, y = make_simple_classification_data()

    metric = Miscoding(X_type="categorical", y_type="categorical").fit(X, y)
    table = metric.feature_analysis()

    assert list(table.columns) == [
        "feature_index",
        "feature_name",
        "is_numeric",
        "code_length",
        "deficiency",
        "surplus",
        "miscoding",
    ]
    assert len(table) == X.shape[1]
    assert table["miscoding"].is_monotonic_increasing
    assert table.iloc[0]["miscoding"] == pytest.approx(0.0)


def test_feature_redundancy_returns_symmetric_dataframe():
    X, y = make_simple_classification_data()

    metric = Miscoding(X_type="categorical", y_type="categorical").fit(X, y)
    redundancy = metric.feature_redundancy()

    assert isinstance(redundancy, pd.DataFrame)
    assert list(redundancy.index) == ["x0", "x1", "x2"]
    assert list(redundancy.columns) == ["x0", "x1", "x2"]
    assert np.allclose(redundancy.values, redundancy.values.T)
    assert np.allclose(np.diag(redundancy), 1.0)
    assert np.all((0.0 <= redundancy.values) & (redundancy.values <= 1.0))


def test_identical_features_have_high_pairwise_redundancy():
    X, y = make_redundant_noisy_data()

    metric = Miscoding(X_type="categorical", y_type="categorical").fit(X, y)
    redundancy = metric.feature_redundancy()

    assert redundancy.loc["x0", "x1"] == pytest.approx(1.0)


@pytest.mark.parametrize("mode", ["deficiency", "surplus", "miscoding"])
def test_miscoding_subset_accepts_valid_modes(mode):
    X, y = make_simple_classification_data()

    metric = Miscoding(X_type="categorical", y_type="categorical").fit(X, y)
    value = metric.miscoding_subset([0], mode=mode)

    assert isinstance(value, float)
    assert 0.0 <= value <= 1.0


def test_miscoding_subset_for_perfect_feature_is_zero():
    X, y = make_simple_classification_data()

    metric = Miscoding(X_type="categorical", y_type="categorical").fit(X, y)

    assert metric.miscoding_subset([0], mode="deficiency") == pytest.approx(0.0)
    assert metric.miscoding_subset([0], mode="surplus") == pytest.approx(0.0)
    assert metric.miscoding_subset([0], mode="miscoding") == pytest.approx(0.0)


def test_miscoding_subset_accepts_binary_mask():
    X, y = make_simple_classification_data()

    metric = Miscoding(X_type="categorical", y_type="categorical").fit(X, y)

    assert metric.miscoding_subset([1, 0, 0]) == pytest.approx(
        metric.miscoding_subset([0])
    )


def test_empty_subset_has_full_deficiency_and_zero_surplus():
    X, y = make_simple_classification_data()

    metric = Miscoding(X_type="categorical", y_type="categorical").fit(X, y)
    details = metric.subset_analysis([])

    assert details["deficiency"] == pytest.approx(1.0)
    assert details["surplus"] == pytest.approx(0.0)
    assert details["miscoding"] == pytest.approx(1.0)
    assert details["is_reliable"] is True
    assert details["failure_reason"] is None
    assert details["n_samples"] == X.shape[0]
    assert details["n_observed_joint_states"] is None
    assert details["mean_joint_occupancy"] is None
    assert details["n_singleton_joint_states"] is None
    assert details["singleton_fraction"] is None
    assert details["n_selected_features"] == 0


def test_empty_subset_has_zero_deficiency_for_constant_target():
    X = np.array([[0], [1], [2], [3]])
    y = np.zeros(4, dtype=int)

    metric = Miscoding(X_type="categorical", y_type="categorical").fit(X, y)
    details = metric.subset_analysis([])

    assert details["deficiency"] == pytest.approx(0.0)
    assert details["surplus"] == pytest.approx(0.0)
    assert details["miscoding"] == pytest.approx(0.0)
    assert details["is_reliable"] is True
    assert details["failure_reason"] is None


def test_empirical_subset_deficiency_is_clipped_to_unit_interval(monkeypatch):
    X, y = make_simple_classification_data()
    metric = Miscoding(X_type="categorical", y_type="categorical").fit(X, y)

    def fake_summary(*args, **kwargs):
        features = list(kwargs.get("features") or [])
        y_included = kwargs.get("y_included", False)
        if features and y_included:
            return _summary(100.0)
        if features:
            return _summary(0.0)
        if y_included:
            return _summary(10.0)
        return _summary(0.0)

    monkeypatch.setattr(metric, "_empirical_summary_for_indices", fake_summary)

    assert metric.miscoding_subset([0], mode="deficiency") == pytest.approx(1.0)


def test_empirical_subset_surplus_is_clipped_to_unit_interval(monkeypatch):
    X, y = make_simple_classification_data()
    metric = Miscoding(X_type="categorical", y_type="categorical").fit(X, y)

    def fake_summary(*args, **kwargs):
        features = list(kwargs.get("features") or [])
        y_included = kwargs.get("y_included", False)
        if features and y_included:
            return _summary(100.0)
        if features:
            return _summary(5.0)
        if y_included:
            return _summary(10.0)
        return _summary(0.0)

    monkeypatch.setattr(metric, "_empirical_summary_for_indices", fake_summary)

    assert metric.miscoding_subset([0], mode="surplus") == pytest.approx(1.0)


@pytest.mark.parametrize("mode", ["deficiency", "surplus", "miscoding"])
def test_miscoding_subset_matches_subset_analysis_with_adaptive_bins(mode):
    X, y = make_simple_classification_data()

    metric = Miscoding(
        X_type="categorical",
        y_type="categorical",
        n_bins="adaptive",
    ).fit(X, y)
    details = metric.subset_analysis([0, 2])

    assert metric.miscoding_subset([0, 2], mode=mode) == pytest.approx(
        details[mode]
    )
    assert details["miscoding"] == pytest.approx(
        max(details["deficiency"], details["surplus"])
    )


def test_empirical_subset_formulas_match_manual_code_lengths():
    X, y = make_simple_classification_data()

    metric = Miscoding(
        X_type="categorical",
        y_type="categorical",
        n_bins=2,
    ).fit(X, y)
    details = metric.subset_analysis([0, 2])

    k_xy = empirical_distribution(
        [X[:, 0], X[:, 2], y],
        [False, False, False],
        n_bins=2,
    ).code_length
    k_x = empirical_distribution(
        [X[:, 0], X[:, 2]],
        [False, False],
        n_bins=2,
    ).code_length
    k_y = empirical_distribution([y], [False], n_bins=2).code_length

    deficiency = (k_xy - k_x) / k_y
    surplus = (k_xy - k_y) / k_x

    assert details["deficiency"] == pytest.approx(deficiency)
    assert details["surplus"] == pytest.approx(surplus)
    assert details["miscoding"] == pytest.approx(max(deficiency, surplus))


def test_subset_analysis_includes_reliability_diagnostics():
    X, y = make_reliable_low_dimensional_data()

    metric = Miscoding(X_type="categorical", y_type="categorical").fit(X, y)
    details = metric.subset_analysis([0, 1])

    assert RELIABILITY_FIELDS.issubset(details)
    assert details["n_samples"] == X.shape[0]
    assert details["n_observed_joint_states"] >= 1
    assert details["mean_joint_occupancy"] >= 2.0
    assert details["n_singleton_joint_states"] >= 0
    assert 0.0 <= details["singleton_fraction"] <= 0.5


def test_reliable_small_subset_returns_finite_values():
    X, y = make_reliable_low_dimensional_data()

    metric = Miscoding(X_type="categorical", y_type="categorical").fit(X, y)
    details = metric.subset_analysis([0, 1])

    assert details["is_reliable"] is True
    assert details["failure_reason"] is None
    assert np.isfinite(details["deficiency"])
    assert np.isfinite(details["surplus"])
    assert np.isfinite(details["miscoding"])


def test_unreliable_high_dimensional_subset_returns_nan_values():
    X, y = make_sparse_subset_data()

    metric = Miscoding(X_type="numeric", y_type="categorical", n_bins=4).fit(X, y)
    details = metric.subset_analysis(list(range(8)))

    assert np.isnan(details["deficiency"])
    assert np.isnan(details["surplus"])
    assert np.isnan(details["miscoding"])
    assert details["is_reliable"] is False
    assert details["failure_reason"] == "joint_distribution_too_sparse"
    assert details["n_samples"] == X.shape[0]
    assert details["n_observed_joint_states"] == X.shape[0]
    assert details["mean_joint_occupancy"] == pytest.approx(1.0)
    assert details["n_singleton_joint_states"] == X.shape[0]
    assert details["singleton_fraction"] == pytest.approx(1.0)


@pytest.mark.parametrize("mode", ["deficiency", "surplus", "miscoding"])
def test_miscoding_subset_returns_nan_for_unreliable_subset(mode):
    X, y = make_sparse_subset_data()

    value = miscoding_subset(
        X,
        y,
        list(range(8)),
        mode=mode,
        X_type="numeric",
        y_type="categorical",
        n_bins=4,
    )

    assert np.isnan(value)


def test_subset_analysis_returns_expected_keys_and_shapes():
    X, y = make_simple_classification_data()

    metric = Miscoding(X_type="categorical", y_type="categorical").fit(X, y)
    details = metric.subset_analysis([0, 2])

    assert set(details.keys()) == {
        "deficiency",
        "surplus",
        "miscoding",
        *RELIABILITY_FIELDS,
        "features_in_use",
        "n_selected_features",
        "selected_feature_indices",
        "selected_feature_names",
        "redundancy_weights",
        "feature_weights",
    }
    assert details["features_in_use"].shape == (X.shape[1],)
    assert details["n_selected_features"] == 2
    assert details["selected_feature_indices"] == [0, 2]
    assert details["selected_feature_names"] == ["x0", "x2"]
    assert details["redundancy_weights"].shape == (2,)
    assert details["feature_weights"].shape == (2,)
    assert 0.0 <= details["miscoding"] <= 1.0


def test_empirical_subset_counts_duplicate_information_once():
    X, y = make_redundant_noisy_data()

    metric = Miscoding(X_type="categorical", y_type="categorical").fit(X, y)

    single = metric.subset_analysis([0])
    duplicated = metric.subset_analysis([1, 1])

    assert duplicated["redundancy_weights"] == pytest.approx([0.5, 0.5])
    assert duplicated["deficiency"] == pytest.approx(single["deficiency"])
    assert duplicated["surplus"] == pytest.approx(single["surplus"])
    assert duplicated["miscoding"] == pytest.approx(single["miscoding"])


def test_adaptive_subset_uses_consistent_bins_for_numerical_target(monkeypatch):
    x0 = np.tile([0.0, 1.0], 12)
    x1 = np.tile([0.0, 0.0, 1.0, 1.0], 6)
    x2 = np.tile([0.0, 1.0, 1.0, 0.0], 6)
    X = np.column_stack([x0, x1, x2])
    y = X[:, 0] + X[:, 1]
    calls = []
    original = miscoding_module.empirical_distribution

    def spy_distribution(columns, numeric, n_bins="auto"):
        calls.append((len(columns), tuple(numeric), n_bins))
        return original(columns=columns, numeric=numeric, n_bins=n_bins)

    monkeypatch.setattr(
        miscoding_module,
        "empirical_distribution",
        spy_distribution,
    )

    metric = Miscoding(
        X_type="numeric",
        y_type="numeric",
        n_bins="adaptive",
    ).fit(X, y)
    metric._code_length_cache_.clear()
    calls.clear()

    metric.subset_analysis([0, 1, 2])

    expected_bins = _adaptive_n_bins(X.shape[0], 3)
    assert calls == [
        (4, (True, True, True, True), expected_bins),
        (3, (True, True, True), expected_bins),
        (1, (True,), expected_bins),
    ]


def test_adaptive_subset_keeps_categorical_target_encoding(monkeypatch):
    x0 = np.tile([0.0, 1.0], 12)
    x1 = np.tile([0.0, 0.0, 1.0, 1.0], 6)
    x2 = np.tile([0.0, 1.0, 1.0, 0.0], 6)
    X = np.column_stack([x0, x1, x2])
    y = np.array(["low", "high"] * 12)
    calls = []
    original = miscoding_module.empirical_distribution

    def spy_distribution(columns, numeric, n_bins="auto"):
        calls.append((len(columns), tuple(numeric), n_bins))
        return original(columns=columns, numeric=numeric, n_bins=n_bins)

    monkeypatch.setattr(
        miscoding_module,
        "empirical_distribution",
        spy_distribution,
    )

    metric = Miscoding(
        X_type="numeric",
        y_type="categorical",
        n_bins="adaptive",
    ).fit(X, y)
    metric._code_length_cache_.clear()
    calls.clear()

    metric.subset_analysis([0, 2])

    expected_bins = _adaptive_n_bins(X.shape[0], 2)
    assert calls == [
        (3, (True, True, False), expected_bins),
        (2, (True, True), expected_bins),
        (1, (False,), expected_bins),
    ]


@pytest.mark.parametrize("mode", ["deficiency", "surplus", "miscoding"])
def test_one_feature_subset_adaptive_matches_auto(mode):
    X = np.linspace(0.0, 1.0, 40).reshape(-1, 1)
    y = np.sin(X[:, 0])

    auto = Miscoding(X_type="numeric", y_type="numeric", n_bins="auto").fit(X, y)
    adaptive = Miscoding(
        X_type="numeric",
        y_type="numeric",
        n_bins="adaptive",
    ).fit(X, y)

    assert adaptive.miscoding_subset([0], mode=mode) == pytest.approx(
        auto.miscoding_subset([0], mode=mode)
    )


def test_reliability_diagnostics_work_with_adaptive_categorical_target():
    X, y = make_sparse_subset_data()

    metric = Miscoding(
        X_type="numeric",
        y_type="categorical",
        n_bins="adaptive",
    ).fit(X, y)
    details = metric.subset_analysis(list(range(8)))

    assert details["is_reliable"] is False
    assert details["failure_reason"] == "joint_distribution_too_sparse"
    assert np.isnan(details["miscoding"])


def test_reliability_diagnostics_work_with_adaptive_numerical_target():
    X, _ = make_sparse_subset_data()
    y = np.random.default_rng(4).normal(size=X.shape[0])

    metric = Miscoding(
        X_type="numeric",
        y_type="numeric",
        n_bins="adaptive",
    ).fit(X, y)
    details = metric.subset_analysis(list(range(8)))

    assert details["is_reliable"] is False
    assert details["failure_reason"] == "joint_distribution_too_sparse"
    assert np.isnan(details["miscoding"])


def test_miscoding_subset_rejects_invalid_mode():
    X, y = make_simple_classification_data()

    metric = Miscoding(X_type="categorical", y_type="categorical").fit(X, y)

    with pytest.raises(ValueError, match="mode"):
        metric.miscoding_subset([0], mode="invalid")


def test_selected_indices_validation_errors():
    X, y = make_simple_classification_data()

    metric = Miscoding(X_type="categorical", y_type="categorical").fit(X, y)

    with pytest.raises(ValueError, match="one-dimensional"):
        metric.miscoding_subset([[0, 1]])

    with pytest.raises(ValueError, match="duplicate"):
        metric.miscoding_subset([0, 0])

    with pytest.raises(ValueError, match="selected indices"):
        metric.miscoding_subset([X.shape[1]])


def test_select_features_returns_binary_mask():
    X, y = make_simple_classification_data()

    metric = Miscoding(X_type="categorical", y_type="categorical").fit(X, y)
    mask = metric.select_features(max_features=1)

    assert mask.shape == (X.shape[1],)
    assert set(mask.tolist()) <= {0, 1}
    assert int(mask.sum()) <= 1


def test_select_features_selects_a_perfect_feature_first():
    X, y = make_simple_classification_data()

    metric = Miscoding(X_type="categorical", y_type="categorical").fit(X, y)
    details = metric.select_features(max_features=2, return_details=True)

    assert details["selected_feature_indices"][0] in {0, 2}
    assert details["path"].iloc[0]["miscoding"] == pytest.approx(0.0)


def test_select_features_stops_when_subset_miscoding_does_not_improve():
    X, y = make_simple_classification_data()

    metric = Miscoding(X_type="categorical", y_type="categorical").fit(X, y)
    details = metric.select_features(max_features=3, return_details=True)

    assert len(details["selected_feature_indices"]) == 1
    assert int(details["selected_features"].sum()) == 1
    assert details["subset"]["miscoding"] == pytest.approx(0.0)


def test_rank_features_continues_after_miscoding_stops_improving():
    X, y = make_simple_classification_data()

    metric = Miscoding(X_type="categorical", y_type="categorical").fit(X, y)
    strict = metric.select_features(max_features=3, return_details=True)
    details = metric.rank_features(return_details=True)

    assert len(strict["selected_feature_indices"]) == 1
    assert details["feature_order"] == [0, 2, 1]
    assert len(details["path"]) == X.shape[1]
    assert (
        details["path"].iloc[1:]["miscoding_improvement"] <= 0.0
    ).any()


def test_rank_features_returns_full_or_bounded_unique_order():
    X, y = make_simple_classification_data()

    metric = Miscoding(X_type="categorical", y_type="categorical").fit(X, y)
    full_order = metric.rank_features()
    bounded_order = metric.rank_features(max_features=2)

    assert len(full_order) == X.shape[1]
    assert len(set(full_order)) == X.shape[1]
    assert set(full_order) == set(range(X.shape[1]))
    assert len(bounded_order) == 2
    assert len(set(bounded_order)) == 2


def test_rank_features_return_details_has_expected_shape():
    X, y = make_simple_classification_data()

    metric = Miscoding(X_type="categorical", y_type="categorical").fit(X, y)
    details = metric.rank_features(max_features=2, return_details=True)

    assert set(details) == {
        "feature_order",
        "feature_names",
        "path",
        "features",
        "redundancy",
    }
    assert len(details["feature_order"]) == 2
    assert len(details["feature_names"]) == 2
    assert len(details["path"]) == 2
    assert {
        "step",
        "feature_index",
        "feature_name",
        "deficiency",
        "surplus",
        "miscoding",
        "is_reliable",
        "failure_reason",
        "n_samples",
        "n_observed_joint_states",
        "mean_joint_occupancy",
        "n_singleton_joint_states",
        "singleton_fraction",
        "deficiency_improvement",
        "surplus_change",
        "miscoding_improvement",
        "selected_feature_indices",
        "selected_feature_names",
    }.issubset(details["path"].columns)


def test_rank_features_deficiency_prioritizes_target_relevant_features():
    X, y = make_distributed_signal_data()

    metric = Miscoding(X_type="categorical", y_type="categorical").fit(X, y)
    order = metric.rank_features(max_features=6, criterion="deficiency")

    assert 1 <= len(order) <= 6
    assert len(set(order)) == len(order)
    assert set(order[:4]).issubset(set(range(10)))
    assert len(set(order[:4]) & {0, 1, 2, 3}) >= 3


def test_relevant_prefixes_lower_adaptive_deficiency_than_empty_subset():
    X, y = make_distributed_signal_data()

    metric = Miscoding(
        X_type="numeric",
        y_type="categorical",
        n_bins="adaptive",
    ).fit(X, y)
    empty = metric.miscoding_subset([], mode="deficiency")
    deficiencies = [
        metric.miscoding_subset(list(range(size)), mode="deficiency")
        for size in range(1, 5)
    ]

    assert all(value < empty for value in deficiencies)


def test_adaptive_deficiency_decreases_as_relevant_features_are_added():
    X, y = make_distributed_signal_data()

    metric = Miscoding(
        X_type="numeric",
        y_type="categorical",
        n_bins="adaptive",
    ).fit(X, y)
    deficiencies = [
        metric.miscoding_subset(list(range(size)), mode="deficiency")
        for size in range(1, 5)
    ]

    assert deficiencies[-1] < deficiencies[0]
    assert sum(
        later <= earlier
        for earlier, later in zip(deficiencies, deficiencies[1:])
    ) >= 2


def test_shuffled_target_does_not_improve_relevant_adaptive_prefixes():
    X, y = make_distributed_signal_data()
    shuffled = np.random.default_rng(42).permutation(y)

    real = Miscoding(
        X_type="numeric",
        y_type="categorical",
        n_bins="adaptive",
    ).fit(X, y)
    baseline = Miscoding(
        X_type="numeric",
        y_type="categorical",
        n_bins="adaptive",
    ).fit(X, shuffled)

    real_deficiency = real.miscoding_subset([0, 1, 2, 3], mode="deficiency")
    baseline_deficiency = baseline.miscoding_subset(
        [0, 1, 2, 3],
        mode="deficiency",
    )

    assert real_deficiency <= baseline_deficiency


def test_noise_features_do_not_collapse_adaptive_deficiency():
    X, y = make_distributed_signal_data()

    metric = Miscoding(
        X_type="numeric",
        y_type="categorical",
        n_bins="adaptive",
    ).fit(X, y)
    noise_deficiency = metric.miscoding_subset(
        [4, 5],
        mode="deficiency",
    )
    details = metric.subset_analysis([4, 5])

    assert np.isnan(noise_deficiency)
    assert details["is_reliable"] is False
    assert details["failure_reason"] == "joint_distribution_too_sparse"


def test_rank_features_miscoding_criterion_and_invalid_criterion():
    X, y = make_distributed_signal_data()

    metric = Miscoding(X_type="categorical", y_type="categorical").fit(X, y)
    order = metric.rank_features(criterion="miscoding")

    assert 1 <= len(order) <= X.shape[1]
    assert len(set(order)) == len(order)

    with pytest.raises(ValueError, match="criterion"):
        metric.rank_features(criterion="invalid")


def test_rank_features_with_adaptive_bins_returns_valid_unique_indices():
    X, y = make_distributed_signal_data()

    metric = Miscoding(
        X_type="numeric",
        y_type="categorical",
        n_bins="adaptive",
    ).fit(X, y)
    order = metric.rank_features()

    assert 1 <= len(order) <= X.shape[1]
    assert len(set(order)) == len(order)
    assert set(order).issubset(set(range(X.shape[1])))


def test_rank_features_with_adaptive_bins_respects_max_features():
    X, y = make_distributed_signal_data()

    metric = Miscoding(
        X_type="numeric",
        y_type="categorical",
        n_bins="adaptive",
    ).fit(X, y)

    order = metric.rank_features(max_features=3)

    assert len(order) <= 3
    assert len(set(order)) == len(order)


def test_rank_features_with_adaptive_bins_returns_full_path_details():
    X, y = make_distributed_signal_data()

    metric = Miscoding(
        X_type="numeric",
        y_type="categorical",
        n_bins="adaptive",
    ).fit(X, y)
    details = metric.rank_features(return_details=True)

    assert len(details["path"]) == len(details["feature_order"])
    assert len(details["path"]) <= X.shape[1]
    assert {
        "step",
        "feature_index",
        "feature_name",
        "deficiency",
        "surplus",
        "miscoding",
        "is_reliable",
        "failure_reason",
        "n_samples",
        "n_observed_joint_states",
        "mean_joint_occupancy",
        "n_singleton_joint_states",
        "singleton_fraction",
        "deficiency_improvement",
        "surplus_change",
        "miscoding_improvement",
        "selected_feature_indices",
        "selected_feature_names",
    }.issubset(details["path"].columns)


def test_candidate_extensions_include_reliability_fields():
    X, y = make_sparse_subset_data()

    metric = Miscoding(X_type="numeric", y_type="categorical", n_bins=4).fit(X, y)
    current = metric.subset_analysis([11])
    candidates = metric._candidate_extensions([11], current)

    assert RELIABILITY_FIELDS.issubset(candidates.columns)
    assert not candidates["is_reliable"].any()
    assert candidates["failure_reason"].eq(
        "joint_distribution_too_sparse"
    ).all()


def test_reliable_candidates_sort_before_unreliable_candidates():
    X, y = make_sparse_subset_data()

    metric = Miscoding(X_type="numeric", y_type="categorical", n_bins=4).fit(X, y)
    current = metric.subset_analysis([])
    reliable = metric._candidate_extensions([], current).iloc[[0]].copy()
    unreliable = metric._candidate_extensions([11], metric.subset_analysis([11])).iloc[
        [0]
    ].copy()
    candidates = pd.concat([unreliable, reliable], ignore_index=True)

    sorted_candidates = metric._sort_candidates(candidates, criterion="miscoding")

    assert bool(sorted_candidates.iloc[0]["is_reliable"]) is True
    assert bool(sorted_candidates.iloc[-1]["is_reliable"]) is False


def test_rank_features_stops_when_remaining_candidates_are_unreliable():
    X, y = make_sparse_subset_data()

    metric = Miscoding(X_type="numeric", y_type="categorical", n_bins=4).fit(X, y)
    details = metric.rank_features(max_features=20, return_details=True)

    assert len(details["feature_order"]) < 20
    assert details["path"]["is_reliable"].all()
    assert all(
        metric.subset_analysis(indices)["is_reliable"]
        for indices in details["path"]["selected_feature_indices"]
    )


def test_select_features_stops_when_remaining_candidates_are_unreliable():
    X, y = make_sparse_subset_data()

    metric = Miscoding(X_type="numeric", y_type="categorical", n_bins=4).fit(X, y)
    details = metric.select_features(return_details=True)

    assert len(details["selected_feature_indices"]) == 1
    assert details["subset"]["is_reliable"] is True
    assert details["path"]["is_reliable"].all()


def test_rank_and_select_stop_when_no_reliable_candidate_exists():
    X, y = make_unreliable_extension_data()

    metric = Miscoding(X_type="numeric", y_type="categorical", n_bins=8).fit(X, y)
    rank_details = metric.rank_features(return_details=True)
    select_details = metric.select_features(return_details=True)

    assert rank_details["feature_order"] == []
    assert rank_details["path"].empty
    assert select_details["selected_feature_indices"] == []
    assert int(select_details["selected_features"].sum()) == 0
    assert select_details["subset"]["is_reliable"] is True


def test_select_features_return_details():
    X, y = make_simple_classification_data()

    metric = Miscoding(X_type="categorical", y_type="categorical").fit(X, y)
    details = metric.select_features(max_features=2, return_details=True)

    assert set(details.keys()) == {
        "selected_features",
        "selected_feature_indices",
        "selected_feature_names",
        "min_improvement",
        "path",
        "subset",
        "features",
        "redundancy",
    }
    assert details["selected_features"].shape == (X.shape[1],)
    assert isinstance(details["path"], pd.DataFrame)
    assert isinstance(details["subset"], dict)
    assert isinstance(details["features"], pd.DataFrame)
    assert isinstance(details["redundancy"], pd.DataFrame)


def test_select_features_with_adaptive_bins_returns_valid_mask():
    X, y = make_distributed_signal_data()

    metric = Miscoding(
        X_type="numeric",
        y_type="categorical",
        n_bins="adaptive",
    ).fit(X, y)
    mask = metric.select_features(max_features=4)

    assert mask.shape == (X.shape[1],)
    assert set(mask.tolist()) <= {0, 1}


def test_select_features_with_adaptive_bins_returns_details():
    X, y = make_distributed_signal_data()

    metric = Miscoding(
        X_type="numeric",
        y_type="categorical",
        n_bins="adaptive",
    ).fit(X, y)
    details = metric.select_features(max_features=4, return_details=True)

    assert set(details.keys()) == {
        "selected_features",
        "selected_feature_indices",
        "selected_feature_names",
        "min_improvement",
        "path",
        "subset",
        "features",
        "redundancy",
    }
    assert details["subset"]["miscoding"] == pytest.approx(
        metric.miscoding_subset(
            details["selected_feature_indices"],
            mode="miscoding",
        )
    )
    assert details["subset"]["is_reliable"] is True
    assert np.isfinite(details["subset"]["deficiency"])
    assert np.isfinite(details["subset"]["surplus"])
    assert np.isfinite(details["subset"]["miscoding"])


def test_select_features_respects_min_improvement():
    X, y = make_simple_classification_data()

    metric = Miscoding(
        X_type="categorical",
        y_type="categorical",
        min_improvement=2.0,
    ).fit(X, y)

    mask = metric.select_features()

    assert np.array_equal(mask, np.zeros(X.shape[1], dtype=int))


def test_select_features_rejects_negative_arguments():
    X, y = make_simple_classification_data()

    metric = Miscoding(X_type="categorical", y_type="categorical").fit(X, y)

    with pytest.raises(ValueError, match="min_improvement"):
        metric.select_features(min_improvement=-1.0)

    with pytest.raises(ValueError, match="max_features"):
        metric.select_features(max_features=-1)


def test_fit_rejects_missing_target():
    X, _ = make_simple_classification_data()

    with pytest.raises(ValueError, match="requires a target"):
        Miscoding().fit(X, None)


def test_fit_rejects_empty_target():
    X = np.empty((0, 2))
    y = np.array([])

    with pytest.raises(ValueError, match="must not be empty"):
        Miscoding().fit(X, y)


def test_fit_rejects_inconsistent_lengths():
    X = np.array([[0], [1], [2]])
    y = np.array([0, 1])

    with pytest.raises(ValueError):
        Miscoding().fit(X, y)


def test_methods_requiring_fit_raise_not_fitted_error():
    metric = Miscoding()

    with pytest.raises(NotFittedError):
        metric.feature_deficiency()

    with pytest.raises(NotFittedError):
        metric.feature_surplus()

    with pytest.raises(NotFittedError):
        metric.feature_miscoding()

    with pytest.raises(NotFittedError):
        metric.feature_redundancy()

    with pytest.raises(NotFittedError):
        metric.feature_analysis()

    with pytest.raises(NotFittedError):
        metric.subset_analysis([])

    with pytest.raises(NotFittedError):
        metric.select_features()

    with pytest.raises(NotFittedError):
        metric.rank_features()

    with pytest.raises(NotFittedError):
        metric.miscoding_subset([])


def test_numeric_regression_target_is_supported():
    X = np.array(
        [
            [0.0, 10.0],
            [0.1, 11.0],
            [1.0, 20.0],
            [1.1, 21.0],
            [0.2, 12.0],
            [1.2, 22.0],
        ]
    )
    y = np.array([0.0, 0.1, 1.0, 1.1, 0.2, 1.2])

    metric = Miscoding(X_type="numeric", y_type="numeric", n_bins=2).fit(X, y)

    assert metric.y_isnumeric_ is True
    assert metric.X_isnumeric_ == [True, True]
    assert metric.feature_miscoding().shape == (2,)
    assert metric.feature_redundancy().shape == (2, 2)


def test_categorical_dataframe_values_are_supported():
    X = pd.DataFrame(
        {
            "letter": ["a", "a", "b", "b", "a", "b"],
            "flag": ["yes", "yes", "no", "no", "yes", "no"],
        }
    )
    y = np.array(["left", "left", "right", "right", "left", "right"])

    metric = Miscoding(X_type="auto", y_type="categorical").fit(X, y)

    assert metric.X_isnumeric_ == [False, False]
    assert metric.y_isnumeric_ is False
    assert metric.feature_miscoding().shape == (2,)


def test_functional_feature_analysis_matches_estimator():
    X, y = make_simple_classification_data()

    direct = feature_analysis(X, y, X_type="categorical", y_type="categorical")
    estimator = Miscoding(X_type="categorical", y_type="categorical").fit(X, y)

    pd.testing.assert_frame_equal(direct, estimator.feature_analysis())


def test_functional_feature_redundancy_matches_estimator():
    X, y = make_simple_classification_data()

    direct = feature_redundancy(X, y, X_type="categorical", y_type="categorical")
    estimator = Miscoding(X_type="categorical", y_type="categorical").fit(X, y)

    pd.testing.assert_frame_equal(direct, estimator.feature_redundancy())


def test_functional_miscoding_subset_matches_estimator():
    X, y = make_simple_classification_data()

    direct = miscoding_subset(
        X,
        y,
        [0, 2],
        X_type="categorical",
        y_type="categorical",
    )
    estimator = Miscoding(X_type="categorical", y_type="categorical").fit(X, y)

    assert direct == pytest.approx(estimator.miscoding_subset([0, 2]))


def test_functional_select_features_matches_estimator():
    X, y = make_simple_classification_data()

    direct = select_features(
        X,
        y,
        X_type="categorical",
        y_type="categorical",
        max_features=1,
    )

    estimator = Miscoding(X_type="categorical", y_type="categorical").fit(X, y)

    assert np.array_equal(direct, estimator.select_features(max_features=1))


def test_functional_rank_features_matches_estimator():
    X, y = make_simple_classification_data()

    direct = rank_features(
        X,
        y,
        X_type="categorical",
        y_type="categorical",
        max_features=2,
    )

    estimator = Miscoding(X_type="categorical", y_type="categorical").fit(X, y)

    assert direct == estimator.rank_features(max_features=2)


@pytest.mark.parametrize("mode", ["deficiency", "surplus", "miscoding"])
def test_functional_miscoding_subset_accepts_adaptive_bins(mode):
    X, y = make_distributed_signal_data()

    value = miscoding_subset(
        X,
        y,
        [0, 1],
        mode=mode,
        X_type="numeric",
        y_type="categorical",
        n_bins="adaptive",
    )

    assert isinstance(value, float)
    assert 0.0 <= value <= 1.0


def test_functional_miscoding_subset_returns_nan_for_unreliable_subset():
    X, y = make_sparse_subset_data()

    value = miscoding_subset(
        X,
        y,
        list(range(8)),
        X_type="numeric",
        y_type="categorical",
        n_bins=4,
    )

    assert np.isnan(value)


def test_functional_rank_features_accepts_adaptive_bins():
    X, y = make_distributed_signal_data()

    order = rank_features(
        X,
        y,
        X_type="numeric",
        y_type="categorical",
        n_bins="adaptive",
        max_features=3,
    )

    assert len(order) == 3
    assert len(set(order)) == 3


def test_functional_rank_features_details_include_reliability_metadata():
    X, y = make_sparse_subset_data()

    details = rank_features(
        X,
        y,
        X_type="numeric",
        y_type="categorical",
        n_bins=4,
        return_details=True,
        max_features=20,
    )

    assert len(details["feature_order"]) < 20
    assert RELIABILITY_FIELDS.issubset(details["path"].columns)
    assert details["path"]["is_reliable"].all()


def test_functional_select_features_accepts_adaptive_bins():
    X, y = make_distributed_signal_data()

    mask = select_features(
        X,
        y,
        X_type="numeric",
        y_type="categorical",
        n_bins="adaptive",
        max_features=4,
    )

    assert mask.shape == (X.shape[1],)
    assert set(mask.tolist()) <= {0, 1}


def test_functional_select_features_details_include_reliable_subset():
    X, y = make_sparse_subset_data()

    details = select_features(
        X,
        y,
        X_type="numeric",
        y_type="categorical",
        n_bins=4,
        return_details=True,
    )

    assert details["subset"]["is_reliable"] is True
    assert np.isfinite(details["subset"]["deficiency"])
    assert RELIABILITY_FIELDS.issubset(details["path"].columns)


def test_code_length_cache_is_populated_after_fit():
    X, y = make_simple_classification_data()

    metric = Miscoding(X_type="categorical", y_type="categorical").fit(X, y)

    assert isinstance(metric._code_length_cache_, dict)
    assert len(metric._code_length_cache_) > 0
