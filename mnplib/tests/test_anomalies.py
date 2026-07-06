"""
Tests for the redesigned anomaly-detection module.

These tests target the new estimator-style implementation expected at:

    mnplib.anomalies.AnomalyDetector

The suite covers regression and classification anomalies, anomaly tables,
supplied models, auto-estimator delegation, anomaly grouping, DataFrame feature
names, and validation errors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.datasets import make_classification, make_regression
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import mnplib.anomalies as anomalies_module
from mnplib.anomalies import AnomalyDetector, anomaly_table


class DummyAutoRegressor(BaseEstimator, RegressorMixin):
    """Deterministic regressor used to test auto-model delegation."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict(self, X):
        return np.repeat(self.mean_, np.asarray(X).shape[0])

    def score(self, X, y):
        y = np.asarray(y, dtype=float)
        denominator = np.sum((y - np.mean(y)) ** 2)
        if denominator == 0.0:
            return 0.0
        return float(1.0 - np.sum((y - self.predict(X)) ** 2) / denominator)

    def nescience_score(self):
        return 0.123


class DummyAutoClassifier(BaseEstimator, ClassifierMixin):
    """Deterministic classifier used to test auto-model delegation."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X, y):
        values, counts = np.unique(y, return_counts=True)
        self.classes_ = values
        self.majority_ = values[int(np.argmax(counts))]
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict(self, X):
        return np.repeat(self.majority_, np.asarray(X).shape[0])

    def predict_proba(self, X):
        X = np.asarray(X)
        probabilities = np.zeros((X.shape[0], len(self.classes_)), dtype=float)
        majority_position = int(np.where(self.classes_ == self.majority_)[0][0])
        probabilities[:, majority_position] = 1.0
        return probabilities

    def score(self, X, y):
        return float(np.mean(self.predict(X) == y))

    def nescience_score(self):
        return 0.456


@pytest.fixture()
def regression_dataframe():
    """Return a deterministic regression dataset with named features."""
    rng = np.random.default_rng(42)
    n_samples = 60
    signal = np.linspace(-3.0, 3.0, n_samples)
    auxiliary = rng.normal(size=n_samples)
    noise = rng.normal(scale=0.05, size=n_samples)

    X = pd.DataFrame(
        {
            "signal": signal,
            "auxiliary": auxiliary,
            "duplicate_signal": signal + rng.normal(scale=0.01, size=n_samples),
        }
    )
    y = 2.5 * signal - 0.4 * auxiliary + noise
    return X, y


@pytest.fixture()
def classification_data():
    """Return a deterministic binary classification dataset."""
    X, y = make_classification(
        n_samples=90,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        n_classes=2,
        random_state=42,
    )
    return X, y


@pytest.fixture()
def grouped_anomaly_data():
    """Return data where anomalous rows have a visible one-dimensional split."""
    rng = np.random.default_rng(42)
    n_samples = 80

    X = pd.DataFrame(
        {
            "group_axis": np.r_[
                rng.normal(-4.0, 0.20, 12),
                rng.normal(4.0, 0.20, 12),
                rng.normal(0.0, 1.00, n_samples - 24),
            ],
            "noise_axis": rng.normal(0.0, 1.0, n_samples),
            "redundant_axis": np.r_[
                rng.normal(-4.0, 0.20, 12),
                rng.normal(4.0, 0.20, 12),
                rng.normal(0.0, 1.00, n_samples - 24),
            ],
        }
    )

    y = rng.normal(0.0, 1.0, n_samples)
    predictions = y.copy()
    predictions[:24] += 20.0
    return X, y, predictions


def test_regression_residual_quantile_detects_injected_anomalies(regression_dataframe):
    X, y = regression_dataframe
    predictions = np.asarray(y).copy()
    predictions[[4, 17, 51]] += np.array([25.0, -30.0, 40.0])

    detector = AnomalyDetector(
        task="regression",
        anomaly_rule="residual_quantile",
        anomaly_quantile=0.90,
        random_state=42,
    )
    detector.fit_predictions(X, y, predictions)

    detected = set(detector.anomalies())

    assert {4, 17, 51}.issubset(detected)
    assert detector.task_ == "regression"
    assert detector.anomaly_rule_ == "residual_quantile"
    assert detector.anomaly_scores().shape == (len(y),)
    assert list(detector.feature_names_in_) == list(X.columns)

    summary = detector.summary()
    assert summary["task"] == "regression"
    assert summary["n_anomalies"] == int(np.sum(detector.anomaly_mask_))
    assert summary["anomaly_rate"] == pytest.approx(np.mean(detector.anomaly_mask_))
    assert "mean_absolute_residual" in summary

    table = detector.anomaly_table()
    assert set(table["sample_index"]).issubset(detected)
    assert {"residual", "absolute_residual", "standardized_residual", "direction"}.issubset(
        table.columns
    )


def test_regression_under_and_over_predicted_subsets():
    X = np.arange(20).reshape(-1, 1)
    y = np.arange(20, dtype=float)
    predictions = y.copy()
    predictions[3] = y[3] - 100.0
    predictions[14] = y[14] + 100.0

    detector = AnomalyDetector(
        task="regression",
        anomaly_rule="residual_quantile",
        anomaly_quantile=0.85,
    ).fit_predictions(X, y, predictions)

    assert 3 in detector.anomalies(kind="under_predicted")
    assert 14 in detector.anomalies(kind="over_predicted")


def test_standardized_residual_rule_flags_large_residuals(regression_dataframe):
    X, y = regression_dataframe
    predictions = np.asarray(y).copy()
    predictions[7] += 100.0

    detector = AnomalyDetector(
        task="regression",
        anomaly_rule="standardized_residual",
        z_score_threshold=3.0,
    ).fit_predictions(X, y, predictions)

    assert 7 in detector.anomalies()
    assert detector.anomaly_threshold_ == pytest.approx(3.0)
    assert detector.standardized_residual_[7] >= 3.0


def test_bin_mismatch_rule_for_regression():
    X = np.arange(40).reshape(-1, 1)
    y = np.linspace(0.0, 10.0, 40)
    predictions = y.copy()
    predictions[-5:] = 0.0

    detector = AnomalyDetector(
        task="regression",
        anomaly_rule="bin_mismatch",
        n_bins=4,
    ).fit_predictions(X, y, predictions)

    assert set(range(35, 40)).intersection(set(detector.anomalies()))
    assert set(np.unique(detector.anomaly_score_)).issubset({0.0, 1.0})


def test_auto_task_infers_regression(regression_dataframe):
    X, y = regression_dataframe

    detector = AnomalyDetector(
        task="auto",
        anomaly_rule="residual_quantile",
        anomaly_quantile=0.90,
    ).fit_predictions(X, y, y)

    assert detector.task_ == "regression"


def test_classification_misclassification_detects_changed_labels(classification_data):
    X, y = classification_data
    predictions = y.copy()
    predictions[[2, 9, 31]] = 1 - predictions[[2, 9, 31]]

    detector = AnomalyDetector(task="classification").fit_predictions(X, y, predictions)

    assert set(detector.anomalies()) == {2, 9, 31}
    assert detector.task_ == "classification"
    assert detector.anomaly_rule_ == "misclassification"
    assert np.array_equal(detector.anomaly_score_, detector.anomaly_mask_.astype(float))

    table = detector.anomaly_table()
    assert table["correct"].eq(False).all()
    assert table["anomaly_kind"].eq("misclassified").all()
    assert detector.summary()["n_misclassified"] == 3


def test_classification_probability_quantile_with_supplied_model(classification_data):
    X, y = classification_data

    detector = AnomalyDetector(
        task="classification",
        anomaly_rule="probability_quantile",
        anomaly_quantile=0.80,
        fit_model=True,
        random_state=42,
    )
    detector.fit(X, y, model=LogisticRegression(max_iter=1000, random_state=42))

    assert detector.true_class_probability_ is not None
    assert np.all(detector.true_class_probability_ >= 0.0)
    assert np.all(detector.true_class_probability_ <= 1.0)
    assert detector.anomaly_threshold_ is not None
    assert detector.anomaly_scores().shape == y.shape
    assert detector.anomaly_table(only_anomalies=False).shape[0] == len(y)


def test_auto_task_infers_classification(classification_data):
    X, y = classification_data
    detector = AnomalyDetector(task="auto").fit_predictions(X, y, y)
    assert detector.task_ == "classification"


def test_string_class_labels_are_supported(classification_data):
    X, y_numeric = classification_data
    y = np.array([f"class_{label}" for label in y_numeric])
    predictions = y.copy()
    predictions[[1, 4]] = np.where(predictions[[1, 4]] == "class_0", "class_1", "class_0")

    detector = AnomalyDetector(task="classification").fit_predictions(X, y, predictions)

    assert set(detector.anomalies()) == {1, 4}
    assert set(detector.anomaly_table()["y_true"]).issubset(set(y))


def test_supplied_regressor_is_cloned_and_fitted_when_requested():
    X, y = make_regression(n_samples=70, n_features=4, noise=0.2, random_state=42)
    model = DecisionTreeRegressor(max_depth=2, random_state=42)

    detector = AnomalyDetector(
        task="regression",
        fit_model=True,
        anomaly_quantile=0.90,
        random_state=42,
    )
    detector.fit(X, y, model=model)

    assert detector.model_ is not model
    assert detector.y_pred_.shape == y.shape
    assert detector.anomaly_table().shape[0] > 0


def test_supplied_fitted_classifier_is_used_directly_when_fit_model_false(classification_data):
    X, y = classification_data
    model = DecisionTreeClassifier(max_depth=1, random_state=42).fit(X, y)

    detector = AnomalyDetector(task="classification", fit_model=False)
    detector.fit(X, y, model=model)

    assert detector.model_ is model
    assert detector.y_pred_.shape == y.shape


def test_auto_regressor_delegation(monkeypatch, regression_dataframe):
    X, y = regression_dataframe

    monkeypatch.setattr(anomalies_module, "NescienceRegressor", DummyAutoRegressor)

    detector = AnomalyDetector(
        task="regression",
        anomaly_rule="residual_quantile",
        anomaly_quantile=0.90,
        auto_model_kwargs={"aggregation": "arithmetic"},
        random_state=42,
    )
    detector.fit(X, y)

    assert isinstance(detector.model_, DummyAutoRegressor)
    assert detector.model_.kwargs["aggregation"] == "arithmetic"
    assert detector.model_.kwargs["random_state"] == 42
    assert detector.summary()["model_nescience"] == pytest.approx(0.123)


def test_auto_classifier_delegation(monkeypatch, classification_data):
    X, y = classification_data

    monkeypatch.setattr(anomalies_module, "NescienceClassifier", DummyAutoClassifier)

    detector = AnomalyDetector(
        task="classification",
        auto_model_kwargs={"aggregation": "maximum"},
        random_state=42,
    )
    detector.fit(X, y)

    assert isinstance(detector.model_, DummyAutoClassifier)
    assert detector.model_.kwargs["aggregation"] == "maximum"
    assert detector.summary()["model_nescience"] == pytest.approx(0.456)


def test_auto_regressor_missing_class_raises_import_error(monkeypatch, regression_dataframe):
    X, y = regression_dataframe

    monkeypatch.setattr(anomalies_module, "NescienceRegressor", None)
    detector = AnomalyDetector(task="regression")

    with pytest.raises(ImportError):
        detector.fit(X, y)


def test_group_anomalies_one_dimension(grouped_anomaly_data):
    X, y, predictions = grouped_anomaly_data

    detector = AnomalyDetector(
        task="regression",
        anomaly_rule="residual_quantile",
        anomaly_quantile=0.70,
        random_state=42,
    ).fit_predictions(X, y, predictions)

    groups = detector.group_anomalies(
        dimensions=1,
        max_groups=3,
        filter_redundant=False,
    )

    assert not groups.empty
    assert len(groups) <= 3
    assert {
        "attribute_1",
        "attribute_1_name",
        "inertia",
        "cluster_0_size",
        "cluster_1_size",
        "balance",
        "n_anomalies",
    }.issubset(groups.columns)
    assert groups["inertia"].is_monotonic_increasing


def test_group_anomalies_two_dimensions_filters_repeated_attributes(grouped_anomaly_data):
    X, y, predictions = grouped_anomaly_data

    detector = AnomalyDetector(
        task="regression",
        anomaly_rule="residual_quantile",
        anomaly_quantile=0.70,
        random_state=42,
    ).fit_predictions(X, y, predictions)

    groups = detector.group_anomalies(
        dimensions=2,
        filter_redundant=False,
        filter_repeated_attributes=True,
        max_groups=2,
    )

    assert "attribute_2" in groups.columns
    assert "attribute_2_name" in groups.columns
    assert len(groups) <= 2


def test_group_points_by_name_and_index(grouped_anomaly_data):
    X, y, predictions = grouped_anomaly_data

    detector = AnomalyDetector(
        task="regression",
        anomaly_rule="residual_quantile",
        anomaly_quantile=0.70,
        random_state=42,
    ).fit_predictions(X, y, predictions)

    by_name = detector.group_points("group_axis")
    by_index = detector.group_points(0)

    assert {"sample_index", "cluster", "y_true", "y_pred", "anomaly_score", "group_axis"}.issubset(
        by_name.columns
    )
    assert len(by_name) == len(by_index)
    assert set(by_name["cluster"]).issubset({0, 1})


def test_group_anomalies_returns_empty_table_when_too_few_anomalies():
    X = np.arange(10).reshape(-1, 1)
    y = np.arange(10, dtype=float)
    predictions = y.copy()

    detector = AnomalyDetector(
        task="regression",
        anomaly_rule="residual_quantile",
        anomaly_quantile=0.99,
    ).fit_predictions(X, y, predictions)

    detector.anomaly_mask_[:] = False
    detector.anomaly_mask_[0] = True

    groups = detector.group_anomalies(dimensions=1)

    assert groups.empty
    assert "attribute_1_name" in groups.columns


def test_grouping_invalid_dimension_raises(grouped_anomaly_data):
    X, y, predictions = grouped_anomaly_data
    detector = AnomalyDetector(task="regression").fit_predictions(X, y, predictions)

    with pytest.raises(ValueError, match="dimensions"):
        detector.group_anomalies(dimensions=3)


def test_functional_anomaly_table_returns_dataframe():
    X = np.arange(12).reshape(-1, 1)
    y = np.arange(12, dtype=float)
    predictions = y.copy()
    predictions[-1] += 100.0

    table = anomaly_table(
        X,
        y,
        predictions,
        task="regression",
        anomaly_rule="residual_quantile",
        anomaly_quantile=0.90,
    )

    assert isinstance(table, pd.DataFrame)
    assert "anomaly_score" in table.columns
    assert len(table) >= 1


@pytest.mark.parametrize(
    "method_name,args",
    [
        ("anomalies", ()),
        ("anomaly_scores", ()),
        ("anomaly_table", ()),
        ("summary", ()),
        ("explain", ()),
        ("group_anomalies", ()),
        ("group_points", (0,)),
    ],
)
def test_unfitted_methods_raise_not_fitted_error(method_name, args):
    detector = AnomalyDetector()

    with pytest.raises(NotFittedError):
        getattr(detector, method_name)(*args)


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"task": "bad"}, "task"),
        ({"X_type": "mixed"}, "X_type"),
        ({"y_type": "mixed"}, "y_type"),
        ({"anomaly_rule": "bad"}, "anomaly_rule"),
        ({"anomaly_quantile": 1.0}, "anomaly_quantile"),
        ({"anomaly_quantile": 0.0}, "anomaly_quantile"),
        ({"z_score_threshold": 0.0}, "z_score_threshold"),
        ({"min_cluster_fraction": 0.6}, "min_cluster_fraction"),
        ({"redundancy_threshold": 1.1}, "redundancy_threshold"),
    ],
)
def test_invalid_configuration_raises(kwargs, error):
    X = np.arange(10).reshape(-1, 1)
    y = np.arange(10, dtype=float)
    detector = AnomalyDetector(**kwargs)

    with pytest.raises(ValueError, match=error):
        detector.fit_predictions(X, y, y)


def test_fit_rejects_model_and_predictions_together(regression_dataframe):
    X, y = regression_dataframe
    model = DecisionTreeRegressor(max_depth=2, random_state=42).fit(X, y)
    detector = AnomalyDetector(task="regression")

    with pytest.raises(ValueError, match="either model or predictions"):
        detector.fit(X, y, model=model, predictions=y)


def test_fit_rejects_wrong_prediction_length(regression_dataframe):
    X, y = regression_dataframe
    detector = AnomalyDetector(task="regression")

    with pytest.raises(ValueError, match="same number of samples"):
        detector.fit_predictions(X, y, np.asarray(y)[:-1])


def test_invalid_kind_for_task_raises(classification_data):
    X, y = classification_data
    detector = AnomalyDetector(task="classification").fit_predictions(X, y, y)

    with pytest.raises(ValueError, match="under_predicted"):
        detector.anomalies(kind="under_predicted")


def test_probability_quantile_without_predict_proba_raises(classification_data):
    X, y = classification_data
    predictions = y.copy()
    detector = AnomalyDetector(
        task="classification",
        anomaly_rule="probability_quantile",
    )

    with pytest.raises(ValueError, match="predict_proba"):
        detector.fit_predictions(X, y, predictions)


def test_unknown_attribute_name_raises(grouped_anomaly_data):
    X, y, predictions = grouped_anomaly_data
    detector = AnomalyDetector(task="regression").fit_predictions(X, y, predictions)

    with pytest.raises(ValueError, match="Unknown attribute"):
        detector.group_points("does_not_exist")


def test_invalid_attribute_index_raises(grouped_anomaly_data):
    X, y, predictions = grouped_anomaly_data
    detector = AnomalyDetector(task="regression").fit_predictions(X, y, predictions)

    with pytest.raises(ValueError, match="outside the valid range"):
        detector.group_points(99)


def test_fit_validates_dataframe_length(regression_dataframe):
    X, y = regression_dataframe
    detector = AnomalyDetector(task="regression")

    with pytest.raises(ValueError, match="inconsistent lengths"):
        detector.fit_predictions(X.iloc[:-1], y, y)
