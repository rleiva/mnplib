"""
Tests for the Miscoding class.

These tests target the redundancy-discounted implementation:

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
    - feature_analysis(X, y, **kwargs)
    - feature_redundancy(X, y, **kwargs)
    - miscoding_subset(X, y, subset, **kwargs)
    - select_features(X, y, **kwargs)
"""

import numpy as np
import pandas as pd
import pytest

from sklearn.exceptions import NotFittedError

from mnplib.miscoding import (
    Miscoding,
    feature_analysis,
    feature_redundancy,
    miscoding_subset,
    select_features,
)


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
        ({"min_improvement": -0.1}, "min_improvement"),
    ],
)
def test_constructor_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Miscoding(**kwargs)


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
    assert details["n_features_in_use"] == 0


def test_subset_analysis_returns_expected_keys_and_shapes():
    X, y = make_simple_classification_data()

    metric = Miscoding(X_type="categorical", y_type="categorical").fit(X, y)
    details = metric.subset_analysis([0, 2])

    assert set(details.keys()) == {
        "deficiency",
        "surplus",
        "miscoding",
        "features_in_use",
        "n_features_in_use",
        "selected_feature_indices",
        "selected_feature_names",
        "redundancy_weights",
        "feature_weights",
    }
    assert details["features_in_use"].shape == (X.shape[1],)
    assert details["n_features_in_use"] == 2
    assert details["selected_feature_indices"] == [0, 2]
    assert details["selected_feature_names"] == ["x0", "x2"]
    assert details["redundancy_weights"].shape == (2,)
    assert details["feature_weights"].shape == (2,)
    assert 0.0 <= details["miscoding"] <= 1.0


def test_redundancy_discounted_subset_counts_duplicate_information_once():
    X, y = make_redundant_noisy_data()

    metric = Miscoding(X_type="categorical", y_type="categorical").fit(X, y)

    single = metric.subset_analysis([0])
    duplicated = metric.subset_analysis([1, 1])

    assert duplicated["redundancy_weights"] == pytest.approx([0.5, 0.5])
    assert duplicated["deficiency"] == pytest.approx(single["deficiency"])
    assert duplicated["surplus"] == pytest.approx(single["surplus"])
    assert duplicated["miscoding"] == pytest.approx(single["miscoding"])


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


def test_code_length_cache_is_populated_after_fit():
    X, y = make_simple_classification_data()

    metric = Miscoding(X_type="categorical", y_type="categorical").fit(X, y)

    assert isinstance(metric._code_length_cache_, dict)
    assert len(metric._code_length_cache_) > 0
