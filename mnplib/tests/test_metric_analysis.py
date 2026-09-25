"""Analysis reports, conventional error context, and dimensionally valid surfeit."""

import zlib

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression

from mnplib import Inaccuracy, Nescience, Surfeit
from mnplib.inaccuracy import model_analysis as inaccuracy_model_analysis
from mnplib.inaccuracy import prediction_analysis
from mnplib.surfeit import description_analysis
from mnplib.surfeit import model_analysis as surfeit_model_analysis
from mnplib.utils import empirical_distribution_array


def test_categorical_prediction_analysis_reports_code_lengths_and_counts():
    y = np.array([0, 0, 1, 1])
    predictions = np.array([0, 1, 1, 1])
    metric = Inaccuracy().fit_y(y)
    report = metric.prediction_analysis(predictions)

    assert report["y_type"] == "categorical"
    assert report["resolved_n_bins"] is None
    assert report["n_samples"] == 4
    assert report["target_code_length_bits"] == 4
    assert report["prediction_code_length_bits"] == pytest.approx(3.2451124978365313)
    assert report["joint_code_length_bits"] == 6
    assert report["n_observed_joint_states"] == 3
    assert report["mean_joint_occupancy"] == pytest.approx(4 / 3)
    assert report["n_singleton_joint_states"] == 2
    assert report["singleton_fraction"] == pytest.approx(2 / 3)
    assert report["accuracy"] == 0.75
    assert report["inaccuracy"] == pytest.approx(metric.inaccuracy_predictions(predictions))
    assert "mae" not in report and "rmse" not in report
    assert report == prediction_analysis(predictions, y=y)


def test_numeric_prediction_analysis_reports_errors_and_resolved_bins():
    expected_bins = 3
    y = np.array([0., 1., 2., 3.])
    predictions = np.array([0., 2., 1., 5.])
    metric = Inaccuracy(y_type="numeric").fit_y(y)
    report = metric.prediction_analysis(predictions)

    assert report["y_type"] == "numeric"
    assert report["resolved_n_bins"] == expected_bins
    assert report["mae"] == pytest.approx(1.)
    assert report["rmse"] == pytest.approx(np.sqrt(1.5))
    assert "accuracy" not in report
    for key, columns in [("target_code_length_bits", [y]),
                         ("prediction_code_length_bits", [predictions]),
                         ("joint_code_length_bits", [predictions, y])]:
        summary = empirical_distribution_array(np.column_stack(columns), n_bins=expected_bins)
        assert report[key] == pytest.approx(summary.code_length)
    assert report == prediction_analysis(predictions, y=y, y_type="numeric")


def test_information_inaccuracy_and_classification_error_are_distinct():
    y = np.array(["a", "a", "b", "b"])
    predictions = np.array(["b", "b", "a", "a"])
    report = Inaccuracy().fit_y(y).prediction_analysis(predictions)

    assert report["inaccuracy"] == 0.
    assert report["accuracy"] == 0.


def test_joint_sparsity_is_descriptive_for_prediction_analysis():
    y = np.arange(30)
    report = Inaccuracy(y_type="categorical").fit_y(y).prediction_analysis(y)

    assert report["n_observed_joint_states"] == 30
    assert report["mean_joint_occupancy"] == 1.
    assert report["singleton_fraction"] == 1.
    assert report["inaccuracy"] == 0.
    assert "is_reliable" not in report
    assert "failure_reason" not in report


@pytest.mark.parametrize("predictions,inaccuracy,accuracy", [([1] * 4, 0., 1.), ([0] * 4, 1., 0.)])
def test_constant_prediction_analysis(predictions, inaccuracy, accuracy):
    report = Inaccuracy().fit_y([1] * 4).prediction_analysis(predictions)
    assert report["inaccuracy"] == inaccuracy
    assert report["accuracy"] == accuracy
    assert report["target_code_length_bits"] == 0.
    assert report["joint_code_length_bits"] == 0.


def test_model_analysis_uses_fitted_feature_coordinates_and_labels():
    rng = np.random.default_rng(4)
    X = pd.DataFrame(rng.normal(size=(80, 3)), columns=["a", "b", "c"])
    y = 2 * X["c"].to_numpy() + rng.normal(scale=0.01, size=len(X))
    selected = [2, 0]
    local_X = X.iloc[:, selected]
    model = LinearRegression().fit(local_X, y)

    for cls, functional, key in [(Inaccuracy, inaccuracy_model_analysis, "inaccuracy"),
                                  (Surfeit, surfeit_model_analysis, "surfeit")]:
        metric = cls().fit(X, y)
        implicit = metric.model_analysis(model, feature_indices=selected)
        explicit = metric.model_analysis(model, X=local_X, feature_indices=selected,
                                         feature_names=list(local_X.columns))
        assert implicit == explicit
        assert implicit == functional(model, X=X, y=y, feature_indices=selected)
        assert implicit[key] == pytest.approx(
            getattr(metric, f"{key}_model")(model, feature_indices=selected))


def test_inaccuracy_model_analysis_accepts_predictors_without_serializers():
    class Predictor:
        def predict(self, X):
            return np.asarray(X)[:, 0]

    y = np.array([0, 1, 0, 1])
    X = y.reshape(-1, 1)
    metric = Inaccuracy().fit_y(y)
    assert metric.model_analysis(Predictor(), X=X) == metric.prediction_analysis(y)


@pytest.mark.parametrize("cls", [Inaccuracy, Surfeit])
def test_model_analysis_requires_fitted_metric_and_estimator(cls):
    X = np.arange(12).reshape(6, 2)
    y = np.arange(6)
    with pytest.raises(NotFittedError):
        cls().model_analysis(LinearRegression())
    with pytest.raises(NotFittedError):
        cls().fit(X, y).model_analysis(LinearRegression())


def test_analysis_validates_predictions_and_models():
    with pytest.raises(NotFittedError):
        Inaccuracy().prediction_analysis([0, 1])
    metric = Inaccuracy().fit_y([0, 1])
    with pytest.raises(ValueError, match="same number of samples"):
        metric.prediction_analysis([0])
    with pytest.raises(ValueError, match="one-dimensional"):
        metric.prediction_analysis([[0], [1]])
    with pytest.raises(TypeError, match="predict"):
        metric.model_analysis(object(), X=np.zeros((2, 1)))
    model = DummyRegressor().fit(np.zeros((2, 1)), [0, 1])
    with pytest.raises(ValueError, match="Unsupported"):
        Surfeit().fit_y([0, 1]).model_analysis(model)


@pytest.mark.parametrize("n_samples,compressed_bytes,overhead,reference_source,reference_bits", [
    (4, 12, 2, "target", 4),
    (200, 12, 2, "compression", 80),
    (80, 12, 2, "both", 80),
    (200, 100, 2, "compression", 160),
    (4, 12, 20, "compression", 0),
])
def test_surfeit_reference_limits_use_bits(n_samples, compressed_bytes, overhead,
                                          reference_source, reference_bits, monkeypatch):
    y = np.tile([0, 1], n_samples // 2)
    metric = Surfeit(zlib_overhead=overhead).fit_y(y)
    monkeypatch.setattr(metric, "_compress_bytes", lambda data: b"x" * compressed_bytes)
    report = metric.description_analysis("x" * 20)

    assert report["model_code_length_bits"] == 160
    assert report["compressed_code_length_bits"] == 8 * compressed_bytes
    assert report["target_code_length_bits"] == n_samples
    assert report["reference_code_length_bits"] == reference_bits
    assert report["reference_source"] == reference_source
    assert report["compression_ratio"] == compressed_bytes / 20
    assert report["surfeit"] == pytest.approx(1. - reference_bits / 160)
    assert metric.surfeit_string("x" * 20) == report["surfeit"]


@pytest.mark.parametrize("level,overhead", [(0, 0), (1, 6), (9, 6), (9, 1000)])
def test_description_analysis_matches_utf8_zlib_and_functional_api(level, overhead):
    text = "predict: \u00e1\n" * 50
    y = np.tile([0, 1], 500)
    raw = text.encode("utf-8")
    compressed = zlib.compress(raw, level=level)
    effective_bits = 8 * min(len(raw), max(0, len(compressed) - overhead))
    metric = Surfeit(zlib_level=level, zlib_overhead=overhead).fit_y(y)
    report = metric.description_analysis(text)

    assert report["model_code_length_bits"] == 8 * len(raw)
    assert report["compressed_code_length_bits"] == 8 * len(compressed)
    assert report["effective_compressed_code_length_bits"] == effective_bits
    assert report["reference_code_length_bits"] == min(1000, effective_bits)
    assert report["surfeit"] == pytest.approx(1 - min(1000, effective_bits) / (8 * len(raw)))
    assert report == description_analysis(text, y=y, zlib_level=level, zlib_overhead=overhead)


def test_description_analysis_constant_target_and_compression_expansion():
    report = Surfeit().fit_y([1] * 4).description_analysis("a")
    assert report["target_code_length_bits"] == 0.
    assert report["reference_source"] == "target"
    assert report["surfeit"] == 1.
    assert report["compression_ratio"] > 1.


def test_description_analysis_requires_fit_and_valid_string():
    with pytest.raises(NotFittedError):
        Surfeit().description_analysis("model")
    metric = Surfeit().fit_y([0, 1])
    with pytest.raises(TypeError, match="model_string"):
        metric.description_analysis(object())
    with pytest.raises(ValueError, match="must not be empty"):
        metric.description_analysis("")


def test_model_analysis_and_nescience_share_corrected_surfeit():
    X = np.tile([[0., 0.], [1., 0.], [0., 1.], [1., 1.]], (50, 1))
    y = X[:, 0] + 2 * X[:, 1]
    model = LinearRegression().fit(X, y)
    report = Surfeit().fit(X, y).model_analysis(model)
    expected = 1 - report["reference_code_length_bits"] / report["model_code_length_bits"]
    explanation = Nescience().fit(X, y).model_analysis(model)

    assert report["model_string"] == explanation["model_string"]
    assert report["surfeit"] == pytest.approx(expected)
    assert explanation["surfeit"] == pytest.approx(expected)
