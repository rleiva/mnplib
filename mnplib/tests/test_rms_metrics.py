"""RMS scalar definitions, weight semantics, and cross-interface consistency."""

import inspect
import math
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from mnplib import Miscoding, Mismodel, Nescience, NescienceClassifier, NescienceRegressor, TimeSeries
from mnplib.miscoding import (
    feature_analysis, miscoding, miscoding_feature, miscoding_model,
    miscoding_subset, model_analysis as miscoding_model_analysis,
    pairwise_miscoding_matrix, subset_analysis,
)
from mnplib.mismodel import mismodel
from mnplib.nescience import (
    model_analysis, nescience, nescience_components, nescience_model,
)
from mnplib.utils import _resolve_bins, empirical_distribution_array, empirical_distribution_vector


@pytest.mark.parametrize("a,b", [(0, 0), (1, 1), (0, 1), (0.2, 0.4), (0.6, 0.8),
                                 *[(t, t) for t in (0.1, 0.5, 0.9)]])
def test_canonical_pairs_are_normalized_symmetric_rms(a, b):
    expected = math.sqrt((a*a + b*b) / 2)
    for left, right in ((a, b), (b, a)):
        assert miscoding(deficiency=left, surplus=right) == pytest.approx(expected)
        assert Miscoding.aggregate_components(deficiency=left, surplus=right) == pytest.approx(expected)
        assert mismodel(inaccuracy=left, surfeit=right) == pytest.approx(expected)
        assert Mismodel.aggregate_components(inaccuracy=left, surfeit=right) == pytest.approx(expected)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_primitive_estimates_propagate(value):
    for a, b in ((value, 0), (0, value)):
        assert np.isnan(miscoding(deficiency=a, surplus=b))
        assert np.isnan(mismodel(inaccuracy=a, surfeit=b))


def test_finite_negative_primitive_estimates_are_invalid():
    with pytest.raises(ValueError, match="nonnegative"):
        miscoding(deficiency=-0.1, surplus=0.5)


@pytest.fixture
def categorical_data():
    bits = np.tile([[0, 0], [0, 1], [1, 0], [1, 1]], (30, 1))
    X = pd.DataFrame({"a": bits[:, 0], "b": bits[:, 1],
                      "joint": 2*bits[:, 0] + bits[:, 1], "constant": 1})
    return X, bits[:, 0]


def test_feature_and_dataset_miscoding_interfaces_agree(categorical_data):
    X, y = categorical_data
    metric = Miscoding(X_type="categorical").fit(X, y)
    d = metric.deficiency_feature()
    s = metric.surplus_feature()
    np.testing.assert_allclose(d, [0, 1, 0, 1], atol=1e-14)
    np.testing.assert_allclose(s, [0, 1, 0.5, 0], atol=1e-14)
    expected = np.sqrt((d*d + s*s) / 2)
    np.testing.assert_allclose(metric.miscoding_feature(), expected)
    np.testing.assert_allclose(miscoding_feature(X=X, y=y, X_type="categorical"), expected)
    for index, name in enumerate(X.columns):
        assert metric.miscoding_feature(name) == pytest.approx(expected[index])
        assert miscoding_feature(index, X=X, y=y, X_type="categorical") == pytest.approx(expected[index])
    pd.testing.assert_frame_equal(metric.feature_analysis(), feature_analysis(X=X, y=y, X_type="categorical"))
    for subset in ([], [0], [2], [3], [0, 1], list(range(X.shape[1])), [True, False, True, False]):
        report = metric.subset_analysis(subset)
        expected_value = math.sqrt((report["deficiency"]**2 + report["surplus"]**2) / 2)
        assert metric.miscoding_subset(subset) == pytest.approx(expected_value)
        assert miscoding_subset(subset, X=X, y=y, X_type="categorical") == pytest.approx(expected_value)
        functional = subset_analysis(subset, X=X, y=y, X_type="categorical")
        assert functional["miscoding"] == pytest.approx(expected_value)
    assert metric.miscoding_subset([]) == pytest.approx(1 / math.sqrt(2))


@pytest.mark.parametrize("kind", ["numeric", "categorical"])
def test_pairwise_miscoding_uses_both_directional_components(categorical_data, kind):
    X, y = categorical_data
    metric = Miscoding(X_type=kind).fit(X, y)
    bins = _resolve_bins("adaptive", len(X), subset_size=2)
    matrix = metric.pairwise_miscoding_matrix().to_numpy()
    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            left = empirical_distribution_vector(X.iloc[:, i], numeric=kind == "numeric", n_bins=bins).code_length
            right = empirical_distribution_vector(X.iloc[:, j], numeric=kind == "numeric", n_bins=bins).code_length
            joint = empirical_distribution_array(X.iloc[:, [i, j]].to_numpy(), numeric=kind == "numeric", n_bins=bins).code_length
            d = 0 if right == 0 else np.clip((joint - left) / right, 0, 1)
            s = 0 if left == 0 else np.clip((joint - right) / left, 0, 1)
            assert matrix[i, j] == pytest.approx(math.sqrt((d*d + s*s) / 2))
    np.testing.assert_allclose(matrix, matrix.T)
    np.testing.assert_allclose(metric.pairwise_miscoding_, matrix)
    np.testing.assert_allclose(pairwise_miscoding_matrix(X=X, y=y, X_type=kind), matrix)


def test_model_and_search_miscoding_reports_use_rms(categorical_data):
    X, y = categorical_data
    model = LinearRegression().fit(X, y)
    metric = Miscoding(X_type="categorical").fit(X, y)
    reports = [metric.model_analysis(model), miscoding_model_analysis(model, X=X, y=y, X_type="categorical")]
    for method in ("select_features", "rank_features"):
        details = getattr(metric, method)(return_details=True)
        if method == "select_features":
            reports.append(details["subset"])
        reports.extend(details["path"].to_dict("records"))
    for report in reports:
        assert report["miscoding"] == pytest.approx(math.sqrt(
            (report["deficiency"]**2 + report["surplus"]**2) / 2))
    assert miscoding_model(model, X=X, y=y, X_type="categorical") == pytest.approx(reports[0]["miscoding"])


@pytest.mark.parametrize("weights,expected", [(None, math.sqrt(0.3)),
                                               ({"miscoding": 1, "mismodel": 3}, math.sqrt(0.4))])
def test_hierarchical_nescience_numerical_example(weights, expected):
    m = Miscoding.aggregate_components(deficiency=0.2, surplus=0.4)
    p = Mismodel.aggregate_components(inaccuracy=0.6, surfeit=0.8)
    assert m*m == pytest.approx(0.1)
    assert p*p == pytest.approx(0.5)
    assert Nescience(weights=weights).aggregate_components(miscoding=m, mismodel=p) == pytest.approx(expected)


def test_default_hierarchy_matches_four_primitive_rms():
    for d, s, i, u in np.random.default_rng(42).random((100, 4)):
        actual = Nescience().aggregate_components(
            miscoding=miscoding(deficiency=d, surplus=s), mismodel=mismodel(inaccuracy=i, surfeit=u))
        assert actual == pytest.approx(math.sqrt((d*d + s*s + i*i + u*u) / 4))


@pytest.mark.parametrize("scale", [1e-300, 1, 1e300, np.finfo(float).max / 4])
def test_weight_scaling_is_invariant_and_finite(scale):
    weights = np.array([scale, 3*scale])
    original = weights.copy()
    metric = Nescience(weights=weights)
    with np.errstate(all="raise"):
        assert metric.aggregate_components(miscoding=0.2, mismodel=0.8) == pytest.approx(math.sqrt(0.49))
    np.testing.assert_array_equal(weights, original)


def test_maximum_finite_weights_do_not_overflow():
    metric = Nescience(weights=[np.finfo(float).max, np.finfo(float).max])
    with np.errstate(all="raise"):
        assert metric.aggregate_components(miscoding=0.2, mismodel=0.8) == pytest.approx(math.sqrt(0.34))


@pytest.mark.parametrize("weights,expected", [([1, 0], 0.2), ([0, 1], 0.8),
                                             ({"mismodel": 3}, math.sqrt(0.49))])
def test_top_level_weights_control_only_the_final_score(weights, expected):
    assert Nescience(weights=weights).aggregate_components(miscoding=0.2, mismodel=0.8) == pytest.approx(expected)
    assert Nescience(weights=[1, 0]).aggregate_components(miscoding=0, mismodel=1) == 0


@pytest.mark.parametrize("weights", [None, [0, 1], [1, 0]])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_dimensions_remain_visible_with_zero_weights(weights, value):
    for a, b in ((value, 0), (0, value)):
        assert np.isnan(Nescience(weights=weights).aggregate_components(miscoding=a, mismodel=b))


@pytest.mark.parametrize("weights", [
    [], [1], [1, 1, 1], [1, 1, 1, 1], [0, 0], [-1, 2], [np.nan, 1], [np.inf, 1],
    "12", b"12", 3, {1, 2}, [None, 1], ["1", 1], [1+2j, 1], [[1], [2]], [True, 1],
    {"deficiency": 1}, {"surplus": 1}, {"inaccuracy": 1}, {"surfeit": 1}, {"unknown": 1},
])
def test_invalid_weight_configurations_are_rejected(weights, categorical_data):
    metric = Nescience(weights=weights)
    with pytest.raises(ValueError):
        metric.aggregate_components(miscoding=0.2, mismodel=0.3)
    with pytest.raises(ValueError):
        metric.fit(*categorical_data)


def test_resolved_ratios_parameter_updates_and_cloning(categorical_data):
    X, y = categorical_data
    weights = {"mismodel": 3.0}
    metric = Nescience(X_type="categorical", weights=weights).fit(X, y)
    assert metric.weights is weights
    assert weights == {"mismodel": 3.0}
    np.testing.assert_array_equal(metric.weights_, [1, 3])
    assert clone(metric).get_params() == metric.get_params()
    args = dict(subset=[0], predictions=y, model_string="def predict(x):\n return x[0]\n")
    assert metric.analysis(**args)["weights"] == {"miscoding": 1, "mismodel": 3}
    metric.set_params(weights=[0, 2])
    report = metric.analysis(**args)
    assert report["nescience"] == pytest.approx(report["mismodel"])
    np.testing.assert_array_equal(metric.weights_, [0, 2])
    metric.fit(X, y)
    assert metric.analysis(**args)["weights"] == {"miscoding": 0, "mismodel": 2}


def test_all_nescience_interfaces_reconstruct_the_same_score(categorical_data, monkeypatch):
    X, y = categorical_data
    model = LinearRegression().fit(X, y)
    weights = {"miscoding": 1, "mismodel": 3}
    metric = Nescience(X_type="categorical", weights=weights).fit(X, y)
    subset_analysis_spy = Mock(wraps=metric.miscoding_.subset_analysis)
    monkeypatch.setattr(metric.miscoding_, "subset_analysis", subset_analysis_spy)
    report = metric.model_analysis(model)
    subset_analysis_spy.assert_called_once()
    expected = math.sqrt((report["miscoding"]**2 + 3*report["mismodel"]**2) / 4)
    assert report["nescience"] == pytest.approx(expected)
    assert "aggregation" not in report
    args = dict(subset=report["selected_features"], predictions=model.predict(X), model_string=report["model_string"])
    assert set(metric.components(**args)) == {"deficiency", "surplus", "inaccuracy", "surfeit"}
    assert metric.nescience(**args) == pytest.approx(expected)
    assert nescience(X=X, y=y, X_type="categorical", weights=weights, **args) == pytest.approx(expected)
    assert nescience_components(X=X, y=y, X_type="categorical", **args) == metric.components(**args)
    assert nescience_model(model, X=X, y=y, X_type="categorical", weights=weights) == pytest.approx(expected)
    functional = model_analysis(model, X=X, y=y, X_type="categorical", weights=weights)
    assert functional["nescience"] == pytest.approx(expected)
    assert report["mismodel"] == pytest.approx(Mismodel().fit(X, y).mismodel_model(model))


@pytest.mark.parametrize("cls", [Nescience, NescienceClassifier, NescienceRegressor, TimeSeries])
def test_metric_estimators_expose_only_top_level_weight_configuration(cls):
    assert "aggregation" not in inspect.signature(cls).parameters
    with pytest.raises(TypeError):
        cls(aggregation="euclidean")
    estimator = cls(weights=[1, 3])
    assert clone(estimator).weights == [1, 3]


@pytest.mark.parametrize("function", [nescience, nescience_components, nescience_model, model_analysis])
def test_functional_interfaces_have_no_aggregation_option(function):
    assert "aggregation" not in inspect.signature(function).parameters
    with pytest.raises(TypeError, match="aggregation"):
        function(aggregation="euclidean")


def test_direct_aggregation_requires_two_derived_metrics():
    assert Nescience.component_names_ == ("deficiency", "surplus", "inaccuracy", "surfeit")
    assert Nescience.weight_names_ == ("miscoding", "mismodel")
    with pytest.raises(TypeError):
        Nescience().aggregate_components(deficiency=0.2, surplus=0.4, inaccuracy=0.6, surfeit=0.8)


def test_nonfinite_mismodel_preserves_valid_miscoding(categorical_data, monkeypatch):
    metric = Nescience(weights=[1, 0]).fit(*categorical_data)
    monkeypatch.setattr(metric.mismodel_.inaccuracy_, "inaccuracy_predictions", lambda predictions: np.nan)
    report = metric.analysis(subset=[0], predictions=categorical_data[1], model_string="return x[0]")
    assert np.isfinite(report["miscoding"])
    assert np.isfinite(report["surfeit"])
    assert np.isnan(report["mismodel"])
    assert np.isnan(report["nescience"])


def test_time_series_tables_and_lag_reports_use_canonical_rms():
    y = np.sin(np.arange(160) / 8) + np.random.default_rng(1).normal(0, 0.1, 160)
    weights = {"miscoding": 1, "mismodel": 3}
    estimator = TimeSeries(window_size=3, models=["moving_average"], weights=weights).fit(y)
    table = estimator.results_dataframe()
    np.testing.assert_allclose(table["miscoding"], np.sqrt((table["deficiency"]**2 + table["surplus"]**2) / 2))
    np.testing.assert_allclose(table["mismodel"], np.sqrt((table["inaccuracy"]**2 + table["surfeit"]**2) / 2))
    np.testing.assert_allclose(table["nescience"], np.sqrt((table["miscoding"]**2 + 3*table["mismodel"]**2) / 4))
    report = estimator.analysis()
    assert report["weights"] == weights
    assert report["nescience"] == pytest.approx(table.iloc[0]["nescience"])
    lags = estimator.lag_analysis(max_lag=3)
    np.testing.assert_allclose(lags["miscoding"], np.sqrt((lags["deficiency"]**2 + lags["surplus"]**2) / 2))
    estimator.set_params(weights=[3, 1]).fit(y)
    report = estimator.analysis()
    assert report["nescience"] == pytest.approx(math.sqrt((3*report["miscoding"]**2 + report["mismodel"]**2) / 4))
