"""Model-relative anomaly detection, explanation, and input validation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier

from mnplib import AnomalyDetector
from mnplib.anomalies import results_dataframe
from mnplib.utils import _resolve_bins


@pytest.fixture
def data():
    X = pd.DataFrame({"signal": np.tile([0., 1., 2., 3.], 40),
                      "context": np.repeat([0., 1.], 80)})
    return X, X["signal"].to_numpy()


def test_regression_bin_mismatches_and_direction(data):
    X, y = data
    predictions = y.copy()
    predictions[0] = 10
    predictions[3] = -10
    metric = AnomalyDetector(task="regression").fit(X, y, predictions=predictions)
    assert metric.anomalies().tolist() == [0, 3]
    assert metric.anomalies("over_predicted").tolist() == [0]
    assert metric.anomalies("under_predicted").tolist() == [3]
    table = metric.results_dataframe()
    assert table["sample_index"].tolist() == [0, 3]
    assert table["is_anomaly"].all()
    assert len(metric.results_dataframe(only_anomalies=False)) == len(y)
    assert table.loc[0, "y_pred_bin"] == metric.n_bins_
    assert table.loc[1, "y_pred_bin"] == -1


def test_classification_mismatches_with_string_labels(data):
    X, y = data
    labels = np.where(y > 1, "high", "low")
    predictions = labels.copy()
    predictions[:2] = "high"
    metric = AnomalyDetector().fit(X, labels, predictions=predictions)
    assert metric.task_ == "classification"
    assert metric.anomalies("misclassified").tolist() == [0, 1]
    assert metric.results_dataframe()["correct"].tolist() == [False, False]


@pytest.mark.parametrize("task,model", [("regression", LinearRegression()),
                                        ("classification", DecisionTreeClassifier(max_depth=1))])
def test_supplied_models_are_evaluated_without_retraining(data, task, model):
    X, y = data
    model.fit(X, y)
    predictions = model.predict(X)
    metric = AnomalyDetector(task=task).fit(X, y, model=model)
    assert metric.model_ is model
    np.testing.assert_array_equal(metric.y_pred_, predictions)


@pytest.mark.parametrize("task,family", [("regression", "linear_regression"),
                                         ("classification", "decision_tree")])
def test_automatic_model_workflow(data, task, family):
    X, y = data
    metric = AnomalyDetector(task=task,
                             auto_model_kwargs={"models": [family]}).fit(X, y)
    assert metric.model_.best_result_.is_reliable
    assert metric.analysis()["model_nescience"] == metric.model_.nescience()


def test_discretization_and_consolidated_explanation(data):
    X, y = data
    predictions = np.roll(y, 1)
    metric = AnomalyDetector(task="regression").fit(X, y, predictions=predictions)
    assert metric.n_bins_ == _resolve_bins("auto", len(y))
    report = metric.analysis()
    assert report["n_anomalies"] == len(metric.anomalies())
    assert "compressibility" in report
    assert "feature_analysis" in report
    assert "selection_path" in report
    assert all(isinstance(index, int) for index in report["selected_features"])
    assert set(report["selected_feature_names"]).issubset(X.columns)
    if report["selection_path"].shape[0]:
        assert report["selection_path"]["is_reliable"].all()


def test_empty_anomaly_set_has_a_clear_explanation(data):
    X, y = data
    metric = AnomalyDetector(task="regression").fit(X, y, predictions=y)
    report = metric.analysis()
    assert metric.results_dataframe().empty
    assert report["n_anomalies"] == 0
    assert report["status"] == "insufficient_anomalies"


def test_single_correction_pattern_is_reported(data):
    X, y = data
    labels = np.zeros(len(y), dtype=int)
    metric = AnomalyDetector(task="classification").fit(X, labels, predictions=np.ones(len(y)))
    assert metric.analysis()["status"] == "single_correction_pattern"


def test_functional_report_matches_estimator(data):
    X, y = data
    predictions = np.roll(y, 1)
    expected = AnomalyDetector(task="regression").fit(X, y, predictions=predictions)
    pd.testing.assert_frame_equal(results_dataframe(X, y, predictions, task="regression"),
                                  expected.results_dataframe())


@pytest.mark.parametrize("method", ["anomalies", "results_dataframe", "analysis"])
def test_fitted_state_is_required(method):
    with pytest.raises(NotFittedError):
        getattr(AnomalyDetector(), method)()


@pytest.mark.parametrize("options", [{"task": "other"}, {"X_type": "other"}])
def test_invalid_configuration_is_rejected(data, options):
    X, y = data
    with pytest.raises(ValueError):
        AnomalyDetector(**options).fit(X, y, predictions=y)


def test_constant_regression_target_uses_one_observed_bin(data):
    X, y = data
    y = np.ones_like(y)
    predictions = y.copy()
    predictions[:2] = [0, 2]
    metric = AnomalyDetector(task="regression").fit(X, y, predictions=predictions)
    assert metric.n_bins_ == 1
    assert metric.anomalies().tolist() == [0, 1]
    np.testing.assert_array_equal(metric.y_true_bin_, np.zeros(len(y)))
    np.testing.assert_array_equal(metric.y_pred_bin_[:2], [-1, 1])


def test_prediction_input_validation(data):
    X, y = data
    with pytest.raises(ValueError, match="either model or predictions"):
        AnomalyDetector().fit(X, y, model=LinearRegression().fit(X, y), predictions=y)
    with pytest.raises(ValueError, match="same number of samples"):
        AnomalyDetector().fit(X, y, predictions=y[:-1])
    with pytest.raises(NotFittedError):
        AnomalyDetector().fit(X, y, model=LinearRegression())


def test_anomaly_kinds_are_task_specific(data):
    X, y = data
    metric = AnomalyDetector(task="regression").fit(X, y, predictions=y)
    with pytest.raises(ValueError, match="classification"):
        metric.anomalies("misclassified")
    with pytest.raises(ValueError, match="kind"):
        metric.anomalies("other")
