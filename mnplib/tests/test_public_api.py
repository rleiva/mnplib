"""Consistent model metrics, feature coordinates, and search configuration."""

import copy

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from mnplib import (AnomalyDetector, Inaccuracy, Miscoding, Nescience,
                    NescienceClassifier, NescienceRegressor, Surfeit, TimeSeries)
from mnplib.automl import CandidateEvaluator
from mnplib.inaccuracy import inaccuracy_model, inaccuracy_predictions
from mnplib.miscoding import miscoding_model, miscoding_feature, miscoding_subset
from mnplib.nescience import nescience_model, model_analysis
from mnplib.surfeit import surfeit_model, surfeit_string


@pytest.fixture
def data():
    rng = np.random.default_rng(9)
    X = pd.DataFrame(rng.integers(0, 2, size=(400, 4)), columns=list("abcd"))
    y = 3 * X["d"].to_numpy() + rng.normal(scale=0.01, size=len(X))
    return X, y


def test_comparable_class_and_functional_metrics(data):
    X, y = data
    model = LinearRegression().fit(X, y)
    for cls, function, method in [
        (Miscoding, miscoding_model, "miscoding_model"),
        (Inaccuracy, inaccuracy_model, "inaccuracy_model"),
        (Surfeit, surfeit_model, "surfeit_model"),
        (Nescience, nescience_model, "nescience_model"),
    ]:
        metric = cls(n_bins=2).fit(X, y)
        value = getattr(metric, method)(model)
        assert np.isfinite(value)
        assert value == pytest.approx(function(model, X=X, y=y, n_bins=2))
        assert list(metric.feature_names_in_) == list(X.columns)
    report = model_analysis(model, X=X, y=y, n_bins=2)
    assert report["is_reliable"]
    assert report["nescience"] == pytest.approx(nescience_model(model, X=X, y=y, n_bins=2))


def test_explicit_primitives_and_feature_access(data):
    X, y = data
    model = LinearRegression().fit(X, y)
    metric = Miscoding(n_bins=2).fit(X, y)
    assert metric.miscoding_feature("d") == metric.miscoding_feature(3)
    assert metric.miscoding_feature().shape == (4,)
    assert metric.miscoding_feature(3) == miscoding_feature(3, X=X, y=y, n_bins=2)
    assert metric.miscoding_subset([0, 1]) == miscoding_subset([0, 1], X=X, y=y, n_bins=2)
    description = Surfeit(n_bins=2).fit(X, y).model_analysis(model)
    assert description["surfeit"] == surfeit_string(description["model_string"], y=y, n_bins=2)
    assert inaccuracy_predictions(model.predict(X), y=y, n_bins=2) == inaccuracy_model(model, X=X, y=y, n_bins=2)


def test_indices_and_boolean_masks_have_distinct_meanings(data):
    X, y = data
    metric = Miscoding(n_bins=2).fit(X.iloc[:, :2], y)
    assert metric.subset_analysis([0, 1])["selected_features"] == [0, 1]
    assert metric.subset_analysis([False, True])["selected_features"] == [1]
    with pytest.raises(ValueError, match="duplicate"):
        metric.miscoding_subset([0, 0])
    with pytest.raises(ValueError, match="dimension"):
        metric.miscoding_subset([True])
    assert metric.select_features().dtype == np.dtype(bool)


def test_subset_model_coordinates_match_candidate_evaluation(data):
    X, y = data
    indices = [3, 1]
    local_X = X.iloc[:, indices]
    model = DecisionTreeRegressor(max_depth=1, random_state=0).fit(local_X, y)
    metric = Nescience(n_bins=2).fit(X, y)
    direct = metric.model_analysis(model, feature_indices=indices)
    explicit = metric.model_analysis(model, X=local_X, feature_indices=indices)
    result = CandidateEvaluator(X=X.to_numpy(), y=y, nescience=metric,
                                feature_names=X.columns).evaluate(
        name="tree", family="decision_tree", model=model, feature_indices=indices)
    assert direct["selected_features"] == [3]
    assert direct["nescience"] == pytest.approx(result.nescience)
    assert direct["model_string"] == result.artifacts.model_string
    assert direct["model_string"] == explicit["model_string"]
    description = Surfeit(n_bins=2).fit(X, y).model_analysis(model, feature_indices=indices)
    assert description["selected_features"] == [3]
    assert description["surfeit"] == pytest.approx(result.components["surfeit"])


def test_explicit_subset_inputs_require_feature_coordinates(data):
    X, y = data
    local_X = X.iloc[:, [3, 1]]
    model = LinearRegression().fit(local_X, y)
    for cls, method in [(Miscoding, "miscoding_model"), (Nescience, "nescience_model")]:
        metric = cls(n_bins=2).fit(X, y)
        with pytest.raises(ValueError, match="feature_indices is required"):
            getattr(metric, method)(model, X=local_X)


@pytest.mark.parametrize("aggregation", ["euclidean", "arithmetic", "harmonic", "geometric", "maximum", "addition", "product"])
def test_unreliable_model_metrics_stay_nan(aggregation):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 10))
    y = rng.normal(size=20)
    model = LinearRegression().fit(X, y)
    metric = Nescience(n_bins=4, aggregation=aggregation).fit(X, y)
    report = metric.model_analysis(model)
    assert report["is_reliable"] is False
    assert report["failure_reason"] == "joint_distribution_too_sparse"
    with pytest.warns(RuntimeWarning, match="joint_distribution_too_sparse"):
        assert np.isnan(metric.nescience_model(model))
    assert np.isnan(report["nescience"])


@pytest.mark.parametrize("cls,method", [(Miscoding, "miscoding_model"), (Surfeit, "surfeit_model"),
                                        (Nescience, "nescience_model")])
def test_model_errors_are_clear(data, cls, method):
    X, y = data
    with pytest.raises(NotFittedError):
        getattr(cls(), method)(LinearRegression().fit(X, y))
    metric = cls(n_bins=2).fit(X, y)
    with pytest.raises(NotFittedError):
        getattr(metric, method)(LinearRegression())
    with pytest.raises(ValueError, match="Unsupported.*DummyRegressor"):
        getattr(metric, method)(DummyRegressor().fit(X, y))


def test_fit_y_allows_explicit_evaluation_inputs(data):
    X, y = data
    model = LinearRegression().fit(X, y)
    metric = Inaccuracy(n_bins=2).fit_y(y)
    assert metric.inaccuracy_model(model, X=X) == metric.inaccuracy_predictions(model.predict(X))
    metric = Surfeit(n_bins=2).fit_y(y)
    assert np.isfinite(metric.surfeit_model(model))


@pytest.mark.parametrize("cls,family", [(NescienceClassifier, "decision_tree"),
                                        (NescienceRegressor, "linear_regression")])
def test_weights_and_search_options_survive_clone_and_reach_evaluation(data, cls, family):
    X, y = data
    target = (y > 1).astype(int) if cls is NescienceClassifier else y
    weights = {"deficiency": 2., "surplus": 0.5, "inaccuracy": 3., "surfeit": 1.}
    options = {"decision_tree": {"n_jobs": 1}} if family == "decision_tree" else {family: {"patience": 2}}
    original = copy.deepcopy(options)
    estimator = clone(cls(models=[family], n_bins=2, weights=weights, search_options=options)).fit(X, target)
    assert estimator.weights == weights
    assert is_classifier(estimator) if cls is NescienceClassifier else is_regressor(estimator)
    assert options == original
    np.testing.assert_allclose(estimator.nescience_.weights_, [2., 0.5, 3., 1.])
    metric = Nescience(n_bins=2, weights=weights,
                       y_type="categorical" if cls is NescienceClassifier else "numeric").fit(X, target)
    assert metric.nescience_model(estimator) == pytest.approx(estimator.nescience())
    row = estimator.results_dataframe().iloc[0]
    assert row["candidate"] == estimator.best_candidate_name_
    assert row["n_selected_features"] == len(row["selected_features"])
    assert row["is_reliable"]


@pytest.mark.parametrize("cls", [NescienceClassifier, NescienceRegressor, TimeSeries])
def test_search_configuration_rejects_unknown_keys(data, cls):
    X, y = data
    metric = cls(search_options={"not_a_family": {}})
    with pytest.raises(ValueError, match="search_options"):
        metric.fit(y) if cls is TimeSeries else metric.fit(X, (y > 1).astype(int))


def test_forecast_score_and_input_validation():
    series = np.sin(np.arange(200) / 12.)
    metric = TimeSeries(models=["moving_average"], window_size=2, n_bins=2).fit(series[:160])
    assert np.isfinite(metric.score(series[160:]))
    for steps in (0, 1.5, True):
        with pytest.raises(ValueError, match="steps"):
            metric.forecast(steps)
    with pytest.raises(ValueError, match="y_future"):
        metric.score([1.])


@pytest.mark.parametrize("cls", [Miscoding, Nescience, NescienceClassifier,
                                 NescienceRegressor, TimeSeries, AnomalyDetector])
def test_subset_workflows_default_to_adaptive_bins(cls):
    assert clone(cls()).n_bins == "adaptive"
