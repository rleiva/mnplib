"""Model-relative anomaly detection, explanation, and input validation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from mnplib import ResidualAnalysis
from mnplib.utils import _resolve_bins
from mnplib.inaccuracy import Inaccuracy


@pytest.fixture
def data():
    X = pd.DataFrame({"signal": np.tile([0., 1., 2., 3.], 40),
                      "context": np.repeat([0., 1.], 80)})
    return X, X["signal"].to_numpy()


@pytest.mark.parametrize("values,task,expected", [
    ([0, 1, 2, 3], "auto", "classification"),
    ([0., 1., 2., 3.], "auto", "classification"),
    ([False, True, False, True], "auto", "classification"),
    (["a", "b", "c", "d"], "auto", "classification"),
    ([0.1, 0.2, 0.3, 0.4], "auto", "regression"),
    ([0, 1, 2, 3], "regression", "regression"),
    ([0.1, 0.2, 0.3, 0.4], "classification", "classification"),
])
def test_target_inference_and_explicit_task_selection(data, values, task, expected):
    X, _ = data
    y = np.tile(values, len(X) // len(values))
    metric = ResidualAnalysis(task=task).fit(X, y, predictions=y)
    assert metric.task_ == expected
    assert metric.anomalies().size == 0


def test_automatic_task_rejects_unsupported_target_types(data):
    X, y = data
    with pytest.raises(ValueError, match="Unsupported target type 'unknown'"):
        ResidualAnalysis().fit(X, y.astype(object), predictions=y)


def test_regression_bin_mismatches_and_direction(data):
    X, y = data
    predictions = y.copy()
    predictions[0] = 10
    predictions[3] = -10
    metric = ResidualAnalysis(task="regression").fit(X, y, predictions=predictions)
    assert metric.anomalies().tolist() == [0, 3]
    assert metric.anomalies("over_predicted").tolist() == [0]
    assert metric.anomalies("under_predicted").tolist() == [3]
    table = metric.results_dataframe(only_anomalies=True)
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
    metric = ResidualAnalysis().fit(X, labels, predictions=predictions)
    assert metric.task_ == "classification"
    assert metric.anomalies("misclassified").tolist() == [0, 1]
    assert metric.results_dataframe(only_anomalies=True)["correct"].tolist() == [False, False]


def test_discretization_and_consolidated_explanation(data):
    X, y = data
    predictions = np.roll(y, 1)
    metric = ResidualAnalysis(task="regression").fit(X, y, predictions=predictions)
    assert metric.n_bins_ == _resolve_bins("auto", len(y))
    report = metric.feature_analysis()
    assert report["n_anomalies"] == len(metric.anomalies())
    assert metric.compressibility()["scope"] == "anomalous_predicted_states"
    assert "feature_analysis" in report
    assert "selection_path" in report
    assert all(isinstance(index, int) for index in report["selected_features"])
    assert set(report["selected_feature_names"]).issubset(X.columns)
    if report["selection_path"].shape[0]:
        assert report["selection_path"]["is_reliable"].all()


def test_empty_anomaly_set_has_a_clear_explanation(data):
    X, y = data
    metric = ResidualAnalysis(task="regression").fit(X, y, predictions=y)
    report = metric.feature_analysis()
    assert metric.results_dataframe(only_anomalies=True).empty
    assert report["n_anomalies"] == 0
    assert report["status"] == "insufficient_anomalies"


def test_single_correction_pattern_is_reported(data):
    X, y = data
    labels = np.zeros(len(y), dtype=int)
    metric = ResidualAnalysis(task="classification").fit(X, labels, predictions=np.ones(len(y)))
    assert metric.feature_analysis()["status"] == "single_correction_pattern"


@pytest.mark.parametrize("method", ["anomalies", "results_dataframe", "analysis"])
def test_fitted_state_is_required(method):
    with pytest.raises(NotFittedError):
        getattr(ResidualAnalysis(), method)()


@pytest.mark.parametrize("options", [{"task": "other"}, {"X_type": "other"}])
def test_invalid_configuration_is_rejected(data, options):
    X, y = data
    with pytest.raises(ValueError):
        ResidualAnalysis(**options).fit(X, y, predictions=y)


def test_constant_regression_target_uses_one_observed_bin(data):
    X, y = data
    y = np.ones_like(y)
    predictions = y.copy()
    predictions[:2] = [0, 2]
    metric = ResidualAnalysis(task="regression").fit(X, y, predictions=predictions)
    assert metric.n_bins_ == 1
    assert metric.anomalies().tolist() == [0, 1]
    np.testing.assert_array_equal(metric.y_true_bin_, np.zeros(len(y)))
    np.testing.assert_array_equal(metric.y_pred_bin_[:2], [-1, 1])


def test_prediction_input_validation(data):
    X, y = data
    with pytest.raises(ValueError, match="same number of samples"):
        ResidualAnalysis().fit(X, y, predictions=y[:-1])
    with pytest.raises(TypeError, match="predictions"):
        ResidualAnalysis().fit(X, y)


def test_anomaly_kinds_are_task_specific(data):
    X, y = data
    metric = ResidualAnalysis(task="regression").fit(X, y, predictions=y)
    with pytest.raises(ValueError, match="classification"):
        metric.anomalies("misclassified")
    with pytest.raises(ValueError, match="kind"):
        metric.anomalies("other")


def test_conditional_lengths_and_anomalies_are_distinct():
    y = np.tile([0, 1], 100)
    metric = ResidualAnalysis().fit_y(y, predictions=1-y)
    report = metric.analysis()
    assert report["n_anomalies"] == len(y)
    assert report["target_conditional_code_length_bits"] == 0
    assert report["prediction_conditional_code_length_bits"] == 0
    assert metric.results_dataframe()["local_correction_information"].eq(0).all()
    assert report["inaccuracy"] == Inaccuracy(y_type="categorical").fit_y(y).inaccuracy_predictions(1-y)
    assert metric.feature_analysis()["status"] == "no_attributes"


def test_local_information_sums_to_conditional_length():
    y = np.tile([0, 0, 1, 1], 50)
    pred = np.tile([0, 1, 0, 1], 50)
    metric = ResidualAnalysis().fit_y(y, predictions=pred)
    assert metric.analysis()["target_conditional_code_length_bits"] == pytest.approx(200)
    assert metric.results_dataframe()["local_correction_information"].sum() == pytest.approx(200)
    assert metric.patterns_dataframe()["count"].sum() == 200
    assert metric.patterns_dataframe(only_anomalies=True)["count"].sum() == 100
    assert metric.analysis()["scope"] == "all_samples"


def test_regression_keeps_inaccuracy_encoding():
    y = np.linspace(0, 1, 200)
    pred = y + 20
    metric = ResidualAnalysis(task="regression").fit_y(y, predictions=pred)
    report = metric.analysis()
    expected = Inaccuracy(y_type="numeric").fit_y(y).prediction_analysis(pred)
    assert report["inaccuracy"] == expected["inaccuracy"]
    assert report["inaccuracy_encoding"] != report["correction_encoding"]
    assert report["n_anomalies"] == len(y)
    assert report["target_conditional_code_length_bits"] > 0


def test_features_are_lazy_cached_defensive_and_refit_clears(data, monkeypatch):
    from mnplib.miscoding import Miscoding
    X, y = data
    calls = []
    original = Miscoding.fit
    def counted(self, *args, **kwargs):
        calls.append(True)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Miscoding, "fit", counted)
    metric = ResidualAnalysis(task="regression").fit(X, y, predictions=np.roll(y, 1))
    metric.analysis()
    metric.results_dataframe()
    assert not calls
    report = metric.feature_analysis()
    report["feature_analysis"].drop(report["feature_analysis"].index, inplace=True)
    assert not metric.feature_analysis()["feature_analysis"].empty
    assert len(calls) == 1
    metric.fit_y(y, predictions=y)
    assert metric.feature_analysis()["status"] == "no_attributes"
    assert metric.compressibility()["status"] == "no_anomalies"


@pytest.mark.parametrize("values", [[], [[0], [1]], [0, np.nan], [0, np.inf]])
def test_invalid_vectors(values):
    with pytest.raises(ValueError):
        ResidualAnalysis().fit_y([0, 1], predictions=values)


def test_sample_identity_and_output_copies():
    metric = ResidualAnalysis().fit_y([0, 1], predictions=[1, 0], sample_ids=[5, 9])
    assert metric.results_dataframe()["sample_id"].tolist() == [5, 9]
    table = metric.results_dataframe()
    table.loc[0, "y_true"] = 99
    assert metric.y_[0] == 0
    with pytest.raises(ValueError, match="unique"):
        metric.fit_y([0, 1], predictions=[0, 1], sample_ids=[5, 5])


def test_attribute_distribution_uses_shared_edges(data):
    X, y = data
    metric = ResidualAnalysis(task="regression").fit(X, y, predictions=np.roll(y, 1))
    histogram = metric.attribute_distribution("signal", kind="under_predicted")
    assert sum(b["total_count"] for b in histogram["bins"]) == len(y)
    assert sum(b["anomaly_count"] for b in histogram["bins"]) == len(metric.anomalies("under_predicted"))
    with pytest.raises(ValueError, match="Unknown"):
        metric.attribute_distribution("missing")


def test_failed_refit_is_not_a_fitted_analysis():
    metric = ResidualAnalysis().fit_y([0, 1], predictions=[0, 1])
    with pytest.raises(ValueError):
        metric.fit([[1]], [0, 1], predictions=[0, 1])
    with pytest.raises(NotFittedError):
        metric.analysis()
