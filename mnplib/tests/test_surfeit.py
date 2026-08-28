"""
Tests for the simplified Surfeit class.

These tests cover explicit model strings and fitted estimators supported by the
canonical serializer layer.
"""

import zlib

import numpy as np
import pandas as pd
import pytest

from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression

from mnplib.regressor import NescienceRegressor
from mnplib.surfeit import (
    Surfeit,
    model_description,
    surfeit_model_score,
    surfeit_score,
)


def _linear_regression_problem():
    X, y = make_regression(
        n_samples=80,
        n_features=3,
        noise=0.1,
        random_state=42,
    )
    model = LinearRegression().fit(X, y)
    return X, y, model


def test_constructor_defaults():
    metric = Surfeit()

    assert metric.y_type == "auto"
    assert metric.n_bins == "auto"
    assert metric.zlib_level == 9
    assert metric.zlib_overhead == 6


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"y_type": "invalid"}, "y_type"),
        ({"zlib_level": -1}, "zlib_level"),
        ({"zlib_level": 10}, "zlib_level"),
        ({"zlib_overhead": -1}, "zlib_overhead"),
    ],
)
def test_constructor_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Surfeit(**kwargs)


def test_fit_sets_fitted_attributes_for_classification_target():
    X = np.array([[0.0], [0.1], [1.0], [1.1], [0.2], [1.2]])
    y = np.array([0, 0, 1, 1, 0, 1])

    metric = Surfeit(y_type="categorical").fit(X, y)

    assert metric.is_fitted_ is True
    assert metric.n_samples_in_ == len(y)
    assert metric.n_features_in_ == 1
    assert metric.feature_names_in_.tolist() == ["x0"]
    assert metric.y_isnumeric_ is False
    assert metric.len_y_ >= 0.0


def test_fit_sets_fitted_attributes_for_numeric_target():
    X = np.array([[0.0], [0.1], [1.0], [1.1], [0.2], [1.2]])
    y = np.array([1.0, 1.1, 2.0, 2.1, 1.2, 2.2])

    metric = Surfeit(y_type="numeric", n_bins=2).fit(X, y)

    assert metric.is_fitted_ is True
    assert metric.y_isnumeric_ is True
    assert metric.len_y_ >= 0.0


def test_fit_y_supports_string_only_usage():
    y = np.array([0, 0, 1, 1, 0, 1])
    model_string = "def model(x):\n    return int(x > 0)\n"

    metric = Surfeit(y_type="categorical").fit_y(y)
    value = metric.surfeit_string(model_string)

    assert metric.X_ is None
    assert metric.is_fitted_ is True
    assert isinstance(value, float)
    assert 0.0 <= value <= 1.0


def test_surfeit_string_returns_float_in_unit_interval():
    y = np.array([0, 0, 1, 1, 0, 1])
    model_string = "def model(x):\n    return int(x > 0)\n"

    metric = Surfeit(y_type="categorical").fit_y(y)
    value = metric.surfeit_string(model_string)

    assert isinstance(value, float)
    assert 0.0 <= value <= 1.0


def test_surfeit_score_matches_estimator_usage():
    y = np.array([0, 0, 1, 1, 0, 1])
    model_string = "def model(x):\n    return int(x > 0)\n"

    direct = surfeit_score(model_string, y, y_type="categorical")

    metric = Surfeit(y_type="categorical").fit_y(y)
    via_estimator = metric.surfeit_string(model_string)

    assert direct == pytest.approx(via_estimator)


def test_surfeit_model_works_for_fitted_linear_regression():
    X, y, model = _linear_regression_problem()

    metric = Surfeit(y_type="numeric").fit(X, y)
    value = metric.surfeit_model(model)

    assert isinstance(value, float)
    assert 0.0 <= value <= 1.0


def test_model_description_returns_lengths_and_surfeit():
    X, y, model = _linear_regression_problem()

    metric = Surfeit(y_type="numeric").fit(X, y)
    description = metric.model_description(model)

    assert {
        "model_string",
        "model_length",
        "model_compressed_length",
        "surfeit",
    }.issubset(description)
    assert description["model_type"] == "LinearRegression"
    assert description["selected_features"] == [0, 1, 2]
    assert description["n_selected_features"] == 3
    assert description["model_length"] == len(
        description["model_string"].encode("utf-8")
    )
    assert description["surfeit"] == pytest.approx(metric.surfeit_model(model))


def test_surfeit_model_matches_serializer_string_result():
    X, y, model = _linear_regression_problem()

    metric = Surfeit(y_type="numeric").fit(X, y)
    model_string = metric.model_description(model)["model_string"]

    assert metric.surfeit_model(model) == pytest.approx(
        metric.surfeit_string(model_string)
    )


def test_fit_preserves_dataframe_feature_names_for_model_api():
    X, y, _ = _linear_regression_problem()
    X_df = pd.DataFrame(X, columns=["a", "b", "c"])
    model = LinearRegression().fit(X_df, y)

    metric = Surfeit(y_type="numeric").fit(X_df, y)
    description = metric.model_description(model)

    assert metric.feature_names_in_.tolist() == ["a", "b", "c"]
    assert "model_string" in description
    assert 0.0 <= description["surfeit"] <= 1.0


def test_surfeit_model_accepts_explicit_feature_names():
    X, y, model = _linear_regression_problem()

    metric = Surfeit(y_type="numeric").fit(X, y)
    value = metric.surfeit_model(model, feature_names=["a", "b", "c"])

    assert 0.0 <= value <= 1.0


def test_surfeit_model_after_fit_y_uses_supplied_X_or_feature_names():
    X, y, model = _linear_regression_problem()

    metric = Surfeit(y_type="numeric").fit_y(y)
    from_X = metric.surfeit_model(model, X=X)
    from_names = metric.surfeit_model(model, feature_names=["a", "b", "c"])

    assert 0.0 <= from_X <= 1.0
    assert from_names == pytest.approx(from_X)


def test_surfeit_model_rejects_unsupported_estimator_type():
    X, y, _ = _linear_regression_problem()
    model = RandomForestRegressor(n_estimators=3, random_state=42).fit(X, y)
    metric = Surfeit(y_type="numeric").fit(X, y)

    with pytest.raises(ValueError, match="Unsupported scikit-learn model type"):
        metric.surfeit_model(model)


def test_surfeit_model_rejects_unfitted_supported_estimator():
    X, y, _ = _linear_regression_problem()
    metric = Surfeit(y_type="numeric").fit(X, y)

    with pytest.raises(NotFittedError):
        metric.surfeit_model(LinearRegression())


def test_surfeit_model_score_matches_estimator_usage():
    X, y, model = _linear_regression_problem()

    functional = surfeit_model_score(model, X, y, y_type="numeric")
    metric = Surfeit(y_type="numeric").fit(X, y)

    assert functional == pytest.approx(metric.surfeit_model(model))


def test_surfeit_model_score_slices_full_X_with_feature_indices():
    X, y, _ = _linear_regression_problem()
    selected = [0, 2]
    model = LinearRegression().fit(X[:, selected], y)

    functional = surfeit_model_score(
        model,
        X,
        y,
        feature_indices=selected,
        y_type="numeric",
    )
    metric = Surfeit(y_type="numeric").fit(X, y)
    direct = metric.surfeit_model(
        model,
        X=X[:, selected],
        feature_indices=selected,
    )

    assert functional == pytest.approx(direct)


def test_functional_model_description_matches_estimator_usage():
    X, y, model = _linear_regression_problem()

    direct = model_description(model, X, y, y_type="numeric")
    metric = Surfeit(y_type="numeric").fit(X, y)
    via_estimator = metric.model_description(model)

    assert direct["model_string"] == via_estimator["model_string"]
    assert direct["surfeit"] == pytest.approx(via_estimator["surfeit"])


def test_surfeit_model_matches_automl_candidate_artifact_surfeit():
    X, y = make_regression(
        n_samples=120,
        n_features=3,
        n_informative=2,
        noise=0.1,
        random_state=7,
    )
    automl = NescienceRegressor(
        models=["linear_regression"],
        n_bins=2,
        max_feature_prefixes=2,
    ).fit(X, y)
    result = automl.best_result_
    selected = list(result.artifacts.subset)

    value = automl.nescience_.surfeit_.surfeit_model(
        result.model.estimator,
        X=X[:, selected],
        feature_indices=selected,
    )

    assert value == pytest.approx(result.components["surfeit"])
    assert value == pytest.approx(
        automl.nescience_.surfeit_.surfeit_string(result.artifacts.model_string)
    )


def test_surfeit_string_requires_fitted_estimator():
    metric = Surfeit()

    with pytest.raises(NotFittedError):
        metric.surfeit_string("def model(x):\n    return x\n")


def test_surfeit_string_rejects_non_string_model():
    y = np.array([0, 0, 1, 1])
    metric = Surfeit(y_type="categorical").fit_y(y)

    with pytest.raises(TypeError, match="model_string"):
        metric.surfeit_string(123)


def test_surfeit_string_rejects_empty_string():
    y = np.array([0, 0, 1, 1])
    metric = Surfeit(y_type="categorical").fit_y(y)

    with pytest.raises(ValueError, match="must not be empty"):
        metric.surfeit_string("")


def test_validate_model_string_returns_utf8_bytes():
    model_string = "def modelo(x):\n    return 'á'\n"

    model_bytes = Surfeit._validate_model_string(model_string)

    assert isinstance(model_bytes, bytes)
    assert model_bytes == model_string.encode("utf-8")


def test_description_lengths_use_configured_compression():
    model_string = "def predict(x):\n    return 0\n"
    model_bytes = model_string.encode("utf-8")

    metric = Surfeit(zlib_level=1)
    lengths = metric.description_lengths(model_string)

    assert lengths == {
        "model_length": len(model_bytes),
        "model_compressed_length": len(zlib.compress(model_bytes, level=1)),
    }


def test_description_lengths_reject_invalid_model_string():
    metric = Surfeit()

    with pytest.raises(TypeError, match="model_string"):
        metric.description_lengths(123)

    with pytest.raises(ValueError, match="must not be empty"):
        metric.description_lengths("")


def test_effective_compressed_length_subtracts_overhead_and_clips():
    metric = Surfeit(zlib_overhead=6)

    assert metric._effective_compressed_length(
        compressed_length=20,
        model_length=100,
    ) == 14

    assert metric._effective_compressed_length(
        compressed_length=3,
        model_length=100,
    ) == 0

    assert metric._effective_compressed_length(
        compressed_length=200,
        model_length=100,
    ) == 100


def test_compress_bytes_returns_bytes():
    metric = Surfeit(zlib_level=9)
    data = b"abcabcabcabcabcabc"

    compressed = metric._compress_bytes(data)

    assert isinstance(compressed, bytes)
    assert len(compressed) > 0


def test_surfeit_from_lengths_is_clipped_to_unit_interval():
    y = np.array([0, 0, 1, 1])
    metric = Surfeit(y_type="categorical").fit_y(y)

    value = metric._surfeit_from_lengths(
        model_length=10,
        compressed_length=1000,
    )

    assert isinstance(value, float)
    assert 0.0 <= value <= 1.0


def test_target_code_length_constant_target_is_zero():
    y = np.array([1, 1, 1, 1])

    metric = Surfeit(y_type="categorical").fit_y(y)

    assert metric.len_y_ == pytest.approx(0.0)


def test_constant_target_yields_unit_surfeit_for_nonempty_description():
    y = np.array([1, 1, 1, 1])
    model_string = "def model(x):\n    return 1\n"

    metric = Surfeit(y_type="categorical").fit_y(y)

    assert metric.surfeit_string(model_string) == pytest.approx(1.0)


def test_manual_y_type_numeric_overrides_auto_detection():
    y = np.array([0, 1, 2, 3])

    metric = Surfeit(y_type="numeric", n_bins=2).fit_y(y)

    assert metric.y_isnumeric_ is True


def test_manual_y_type_categorical_overrides_auto_detection():
    y = np.array([0.0, 1.0, 2.0, 3.0])

    metric = Surfeit(y_type="categorical", n_bins=2).fit_y(y)

    assert metric.y_isnumeric_ is False


def test_auto_target_type_detects_binary_as_categorical():
    y = np.array([0, 1, 0, 1])

    metric = Surfeit(y_type="auto").fit_y(y)

    assert metric.y_isnumeric_ is False


def test_auto_target_type_detects_continuous_as_numeric():
    y = np.array([0.1, 0.2, 0.3, 0.4])

    metric = Surfeit(y_type="auto", n_bins=2).fit_y(y)

    assert metric.y_isnumeric_ is True


def test_fit_y_rejects_empty_target():
    metric = Surfeit()

    with pytest.raises(ValueError, match="must not be empty"):
        metric.fit_y([])


def test_fit_y_rejects_2d_target():
    metric = Surfeit()

    with pytest.raises(ValueError, match="one-dimensional"):
        metric.fit_y([[0], [1], [0], [1]])


def test_fit_rejects_inconsistent_lengths():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0, 1])

    with pytest.raises(ValueError):
        Surfeit().fit(X, y)


def test_surfeit_score_rejects_invalid_model_string():
    y = np.array([0, 0, 1, 1])

    with pytest.raises(ValueError, match="must not be empty"):
        surfeit_score("", y, y_type="categorical")
