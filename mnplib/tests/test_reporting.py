"""Plain-text presentation of metric, model-search, and anomaly analyses."""

from copy import deepcopy
import pickle
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from mnplib import (
    ResidualAnalysis, Inaccuracy, Miscoding, Nescience, NescienceClassifier,
    NescienceRegressor, Surfeit, TimeSeries,
)
from mnplib.inaccuracy import prediction_analysis
from mnplib.miscoding import subset_analysis
from mnplib.reporting import format_analysis
from mnplib.surfeit import description_analysis


@pytest.fixture(scope="module")
def classification_data():
    X = pd.DataFrame(np.random.default_rng(4).integers(0, 2, size=(400, 3)),
                     columns=["signal", "context", "noise"])
    y = 2 * X.signal.to_numpy() + X.context.to_numpy()
    return X, y, DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)


@pytest.mark.parametrize("cls,key", [
    (Miscoding, "miscoding"), (Inaccuracy, "inaccuracy"),
    (Surfeit, "surfeit"), (Nescience, "nescience"),
])
def test_all_metric_model_reports_are_readable_and_unchanged(classification_data, cls, key, capsys):
    X, y, model = classification_data
    report = cls().fit(X, y).model_analysis(model)
    before = pickle.dumps(report)
    text = format_analysis(report)

    assert text.startswith(f"{key.title()} Analysis\n")
    assert f"{report[key]:.4f}" in text
    assert "model_string" not in text
    assert "array(" not in text
    assert max(map(len, text.splitlines())) <= 78
    assert pickle.dumps(report) == before
    assert capsys.readouterr().out == ""
    assert format_analysis(MappingProxyType(report)) == text


def test_subset_report_includes_names_reliability_and_joint_counts(classification_data):
    X, y, _ = classification_data
    report = Miscoding().fit(X, y).subset_analysis([0, 1])
    text = format_analysis(report)
    assert "signal, context" in text
    assert "Subset reliability" in text and "Reliable" in text
    assert "Joint distribution" in text
    assert "Observed states" in text
    assert "Mean occupancy" in text
    assert "Singleton fraction" in text
    assert format_analysis(subset_analysis([0, 1], X=X, y=y)) == text


def test_sparse_and_empty_subset_reports():
    X = np.arange(60).reshape(20, 3).astype(str)
    y = np.arange(20).astype(str)
    metric = Miscoding().fit(X, y)
    text = format_analysis(metric.subset_analysis([0, 1]))
    assert "Unreliable" in text
    assert "joint_distribution_too_sparse" in text
    assert "NaN" not in text
    empty = format_analysis(metric.subset_analysis([]))
    assert "Reliable" in empty and "(none)" in empty
    assert "Joint distribution" not in empty


def test_prediction_and_description_reports_include_information_scores_and_units():
    numeric = Inaccuracy(y_type="numeric").fit_y([0., 1., 2., 3.])
    report = numeric.prediction_analysis([0., 2., 1., 5.])
    text = format_analysis(report)
    assert "Inaccuracy Analysis" in text
    assert "MAE" not in text and "RMSE" not in text and "Accuracy" not in text
    assert "Code lengths (bits)" in text
    assert "Subset reliability" not in text
    categorical = format_analysis(prediction_analysis([0, 1, 1, 1], y=[0, 0, 1, 1]))
    assert "Inaccuracy Analysis" in categorical
    assert "Code lengths (bits)" in categorical
    assert "MAE" not in categorical and "RMSE" not in categorical and "Accuracy" not in categorical
    surfeit = format_analysis(description_analysis("predict: x0\n" * 30, y=[0, 0, 1, 1]))
    assert "Surfeit Analysis" in surfeit
    assert "Code lengths (bits)" in surfeit
    assert "Effective compressed" in surfeit
    assert "Reference source" in surfeit
    assert "predict:" not in surfeit


@pytest.mark.parametrize("cls,family,task,label", [
    (NescienceClassifier, "decision_tree", "Auto-Classification", "Accuracy"),
    (NescienceRegressor, "linear_regression", "Auto-Regression", "R-squared"),
])
def test_automl_reports_label_training_scores_and_only_recorded_parameters(
    classification_data, cls, family, task, label, monkeypatch,
):
    X, y, _ = classification_data
    model = cls(models=[family], random_state=0).fit(X, y)
    report = model.analysis()

    def unexpected(*args, **kwargs):
        raise AssertionError("Formatting must not evaluate or fit a model.")

    monkeypatch.setattr(model, "fit", unexpected)
    monkeypatch.setattr(model, "score", unexpected)
    monkeypatch.setattr(model, "predict", unexpected)
    before = pickle.dumps(report)
    text = format_analysis(report)
    assert text.startswith(f"{task} Analysis\n")
    assert report["candidate"] in text
    assert "Training data" in text and label in text
    assert "RMS weighting" in text and "Miscoding weight" in text
    for key in report["hyperparameters"]:
        assert key in text
    assert "warm_start" not in text
    assert pickle.dumps(report) == before


def test_time_series_and_lag_reports():
    y = np.tile([0., 1., 2., 1.], 60)
    model = TimeSeries(window_size=2, models=["autoregressive"]).fit(y)
    text = format_analysis(model.analysis())
    assert text.startswith("Auto-Time-Series Analysis\n")
    assert "Lagged training data" in text
    assert "R-squared" in text and "Lag window" in text
    assert "Selected lags" in text
    assert "y_lag_" in text
    table = model.lag_analysis()
    before = table.copy(deep=True)
    text = format_analysis(table)
    assert text.startswith("Lag Analysis\n")
    assert "Deficiency" in text and "Miscoding" in text
    pd.testing.assert_frame_equal(table, before)


@pytest.mark.parametrize("patterns,status", [
    ("none", "insufficient_anomalies"), ("one", "single_correction_pattern"),
    ("multiple", "ok"),
])
def test_anomaly_reports_cover_explanation_states(classification_data, patterns, status):
    X, y, _ = classification_data
    predictions = y.copy()
    if patterns == "one":
        predictions[y == 0] = 1
    elif patterns == "multiple":
        predictions = (y + 1) % 4
    model = ResidualAnalysis(task="classification").fit(X, y, predictions=predictions)
    report = model.analysis()
    before = pickle.dumps(report)
    text = format_analysis(report)
    assert text.startswith("Residual Analysis\n")
    assert model.feature_analysis()["status"] == status
    assert "Anomalies" in text and "Anomaly rate" in text
    assert "Empirical common-state code lengths (bits)" in text
    assert pickle.dumps(report) == before


def test_regression_anomaly_report_labels_kind_and_overall_detection(classification_data):
    X, y, _ = classification_data
    predictions = y.astype(float) + np.where(np.arange(len(y)) % 2, 5., -5.)
    model = ResidualAnalysis(task="regression").fit(X, y, predictions=predictions)
    text = format_analysis(model.analysis())
    assert "common_target_bins" in text
    assert "all_samples" in text
    assert "independent_numeric_bins" in text


def test_feature_table_and_empty_table(classification_data):
    X, y, _ = classification_data
    table = Miscoding().fit(X, y).feature_analysis()
    before = table.copy(deep=True)
    text = format_analysis(table)
    assert text.startswith("Feature Analysis\n")
    assert all(name in text for name in X.columns)
    assert "Code (bits)" in text
    assert f"{table.iloc[0]['code_length_bits']:.3f}" in text
    assert "Miscoding" in format_analysis(table.iloc[:0])
    pd.testing.assert_frame_equal(table, before)


def test_long_text_and_feature_lists_are_bounded(classification_data):
    X, y, model = classification_data
    report = Nescience().fit(X, y).model_analysis(model)
    report["selected_feature_names"] = [f"feature_{index}" for index in range(30)]
    report["model_type"] = "long_model_" * 30
    report["metadata"] = {"large": np.zeros(10000)}
    text = format_analysis(report)
    assert "+22 more" in text
    assert "feature_8" not in text
    assert max(map(len, text.splitlines())) <= 78
    table = pd.concat([Miscoding().fit(X, y).feature_analysis()] * 20, ignore_index=True)
    table["feature_name"] = "feature_name_" * 40
    assert len(format_analysis(table).splitlines()) < 80


@pytest.mark.parametrize("value,formatted", [(np.nan, "NaN"), (np.inf, "inf"), (-np.inf, "-inf")])
def test_nonfinite_scores_remain_explicit(value, formatted):
    text = format_analysis({"surfeit": value})
    assert formatted in text


@pytest.mark.parametrize("report", [None, [], 0.4, "analysis"])
def test_unsupported_input_types_are_rejected(report):
    with pytest.raises(TypeError, match="mapping or DataFrame"):
        format_analysis(report)


@pytest.mark.parametrize("report", [
    {}, {"score": 0.1}, {"surfeit": None}, {"miscoding": [0.1, 0.2]},
    {"inaccuracy": "0.1"}, {"nescience": True}, pd.DataFrame({"score": [0.1]}),
    {"n_anomalies": None, "anomaly_rate": 0.1},
    {"n_anomalies": 1.5, "anomaly_rate": 0.1},
])
def test_unrecognized_or_invalid_reports_are_rejected(report):
    with pytest.raises(ValueError):
        format_analysis(report)


def test_missing_optional_fields_and_supplied_failure_reason():
    report = {"miscoding": np.float64(0.25), "is_reliable": np.bool_(False),
              "n_samples": np.int64(20), "selected_features": np.array([2, 4])}
    text = format_analysis(report)
    assert "Not provided" in text and "2, 4" in text
    assert "0.2500" in text
    assert "None" not in text
    unchanged = deepcopy(report)
    format_analysis(report)
    np.testing.assert_equal(report, unchanged)
