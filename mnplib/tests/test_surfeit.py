"""
Tests for the simplified Surfeit class.

These tests target the string-based API:

    - Surfeit(y_type="auto", n_bins="auto", zlib_level=9, zlib_overhead=6)
    - fit(X, y)
    - fit_y(y)
    - surfeit_string(model_string)
    - surfeit_score(model_string, y, ...)

Model-specific serialization is intentionally outside the Surfeit class.
"""

import zlib

import numpy as np
import pytest

from sklearn.exceptions import NotFittedError

from mnplib.surfeit import Surfeit, surfeit_score


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
