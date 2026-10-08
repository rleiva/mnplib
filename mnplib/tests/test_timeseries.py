"""Tests for the TimeSeries package."""

import numpy as np
import pandas as pd
import pytest

from sklearn.exceptions import NotFittedError
from sklearn.metrics import r2_score

from mnplib.timeseries import TimeSeries, FixedLinearForecaster
from mnplib.models import describe_model
from mnplib.models.language import parse, render, execute
from mnplib.timeseries.models import (
    exponential_smoothing_weights,
    moving_average_weights,
)


def make_series(n=80):
    rng = np.random.default_rng(42)
    y = np.zeros(n, dtype=float)
    for t in range(2, n):
        y[t] = 0.65 * y[t - 1] - 0.2 * y[t - 2] + rng.normal(scale=0.2)
    return y


def make_exogenous_series(n=90):
    rng = np.random.default_rng(123)
    x1 = rng.normal(size=n)
    x2 = np.sin(np.linspace(0.0, 8.0, n))
    y = np.zeros(n, dtype=float)
    for t in range(2, n):
        y[t] = 0.5 * y[t - 1] + 0.25 * x1[t - 1] - 0.1 * x2[t - 2] + rng.normal(scale=0.1)
    X = pd.DataFrame({"temperature": x1, "demand": x2})
    return y, X


def test_fit_selects_candidate_and_sets_public_attributes():
    y = make_series()
    ts = TimeSeries(window_size=5, random_state=42).fit(y)

    assert ts.is_fitted_
    assert ts.window_size_ == 5
    assert ts.X_supervised_.shape == (len(y) - 5, 5)
    assert ts.y_supervised_.shape == (len(y) - 5,)
    assert ts.best_result_.nescience == pytest.approx(ts.best_nescience_ if hasattr(ts, "best_nescience_") else ts.nescience())
    assert ts.model_ is ts.best_result_.model
    assert len(ts.candidate_results_) > 1
    assert len(ts.selected_feature_indices_) >= 1


def test_forecast_returns_requested_number_of_steps():
    y = make_series()
    ts = TimeSeries(window_size=4, models=["autoregressive"]).fit(y)

    forecast = ts.forecast(steps=6)

    assert isinstance(forecast, np.ndarray)
    assert forecast.shape == (6,)
    assert np.all(np.isfinite(forecast))


def test_intercept_only_forecast_retains_estimator_input_coordinates():
    ts = TimeSeries(window_size=3, models=["autoregressive"]).fit(np.ones(40))
    assert ts.best_artifacts_.subset == []
    assert ts.best_artifacts_.description.feature_indices
    np.testing.assert_array_equal(ts.forecast(3), np.ones(3))


def test_fitted_values_align_with_training_observations():
    y = make_series()
    ts = TimeSeries(window_size=4, models=["autoregressive"]).fit(y)

    assert ts.fitted_values_.shape == y.shape
    assert np.isnan(ts.fitted_values_[:ts.window_size_]).all()
    np.testing.assert_allclose(ts.fitted_values_[ts.window_size_:], ts.best_artifacts_.predictions)


def test_score_evaluates_observations_after_training():
    y = make_series(100)
    ts = TimeSeries(window_size=4, models=["autoregressive"]).fit(y[:80])
    expected = r2_score(y[80:], ts.forecast(20))
    assert ts.score(y[80:]) == pytest.approx(expected)


def test_results_dataframe_is_sorted_and_has_expected_columns():
    y = make_series()
    ts = TimeSeries(window_size=4).fit(y)

    df = ts.results_dataframe()

    expected = {
        "candidate",
        "family",
        "nescience",
        "native_estimator_score",
        "n_selected_features",
        "description_length",
        "selected_features",
        "selected_feature_names",
        "deficiency",
        "surplus",
        "inaccuracy",
        "surfeit",
        "is_reliable",
        "failure_reason",
        "n_samples",
        "n_observed_joint_states",
        "mean_joint_occupancy",
        "n_singleton_joint_states",
        "singleton_fraction",
    }
    assert expected.issubset(df.columns)
    values = df["nescience"].to_numpy(dtype=float)
    finite = np.isfinite(values)
    assert list(finite) == sorted(finite, reverse=True)
    assert np.all(np.diff(values[finite]) >= 0.0)
    assert bool(df.iloc[0]["is_reliable"])
    assert np.isfinite(float(df.iloc[0]["nescience"]))
    assert df.iloc[0]["candidate"] == ts.model_name_
    assert {"autoregressive", "moving_average", "exponential_smoothing"}.issubset(set(df["family"]))


def test_components_nescience_and_model_string():
    y = make_series()
    ts = TimeSeries(window_size=4).fit(y)

    components = ts.components()
    model_string = ts.model_description()["model_string"]
    description = ts.model_description()

    assert set(components) == {"deficiency", "surplus", "inaccuracy", "surfeit"}
    assert ts.nescience() == pytest.approx(ts.best_result_.nescience)
    assert render(parse(model_string)) == model_string
    np.testing.assert_allclose(execute(model_string, ts.X_supervised_),
                               ts.best_artifacts_.predictions, atol=0.002, rtol=0.01)
    assert description["candidate"] == ts.best_result_.name
    assert description["model_string"] == model_string
    assert description["surfeit"] == pytest.approx(components["surfeit"])


def test_analysis_contains_time_series_details():
    y = make_series()
    ts = TimeSeries(window_size=4).fit(y)

    explanation = ts.analysis()

    assert explanation["candidate"] == ts.model_name_
    assert explanation["task"] == "forecasting"
    assert explanation["evaluation_context"] == "lagged_training"
    assert explanation["native_estimator_score"] == pytest.approx(ts.best_result_.estimator_score)
    assert explanation["n_samples"] == len(y) - ts.window_size_
    assert explanation["window_size"] == ts.window_size_
    assert explanation["selected_feature_names"] == ts.selected_feature_names_
    assert {"deficiency", "surplus", "inaccuracy", "surfeit"}.issubset(explanation)
    assert "mismodel" in explanation


@pytest.mark.parametrize("family,options,parameters", [
    ("autoregressive", {}, {}),
    ("moving_average", {"windows": [2]}, {"window": 2}),
    ("exponential_smoothing", {"windows": [2], "alphas": [0.4]}, {"window": 2, "alpha": 0.4}),
])
def test_candidate_hyperparameters_are_shared_by_reports(family, options, parameters):
    ts = TimeSeries(window_size=4, models=[family], search_options={family: options})
    ts.fit(make_series(120))
    report = ts.analysis()
    assert ts.best_result_.hyperparameters == parameters
    assert report["hyperparameters"] == parameters
    assert ts.results_dataframe().iloc[0]["hyperparameters"] == parameters
    assert not parameters.keys() & ts.best_result_.metadata.keys()


def test_lag_analysis_methods_without_exogenous_data():
    y = make_series()
    ts = TimeSeries(window_size=5).fit(y)

    auto = ts.lag_analysis(max_lag=4)
    all_lags = ts.lag_analysis(max_lag=4)

    assert list(auto["lag"]) == [1, 2, 3, 4]
    assert {"lag", "feature_name", "deficiency", "surplus", "miscoding"}.issubset(auto.columns)
    assert len(all_lags) == len(auto)


def test_exogenous_data_feature_names_forecast_and_cross_lag_analysis():
    y, X = make_exogenous_series()
    ts = TimeSeries(window_size=4, models=["autoregressive"]).fit(y, X)

    assert list(ts.exogenous_feature_names_) == ["temperature", "demand"]
    assert "temperature_lag_1" in list(ts.feature_names_in_)

    future = X.tail(3).to_numpy()
    forecast = ts.forecast(steps=3, X_future=future)
    cross = ts.lag_analysis(max_lag=3).query("attribute == 'temperature'")
    all_lags = ts.lag_analysis(max_lag=2)

    assert forecast.shape == (3,)
    assert list(cross["lag"]) == [1, 2, 3]
    assert "attribute" in cross.columns
    assert set(all_lags.get("attribute", pd.Series(dtype=object)).dropna()).issubset({"temperature", "demand"})


def test_model_family_filtering():
    y = make_series()
    ts = TimeSeries(window_size=4, models=["moving_average"]).fit(y)

    assert set(result.family for result in ts.candidate_results_) == {"moving_average"}
    assert ts.model_name_.startswith("moving_average")


def test_moving_average_and_smoothing_configuration():
    y = make_series()
    ts = TimeSeries(window_size=5, models=['moving_average', 'exponential_smoothing'], search_options={'moving_average': {'windows': [2, 5]}, 'exponential_smoothing': {'alphas': [0.2, 0.8]}}).fit(y)

    names = {result.name for result in ts.candidate_results_}

    assert "moving_average_2" in names
    assert "moving_average_5" in names
    assert "exponential_smoothing_w2_a0.2" in names
    assert "exponential_smoothing_w5_a0.8" in names


def test_dataframe_exogenous_must_have_same_length():
    y = make_series()
    X = pd.DataFrame({"x": np.arange(len(y) - 1)})

    with pytest.raises(ValueError, match="inconsistent lengths"):
        TimeSeries(window_size=4).fit(y, X)


def test_invalid_configuration_errors():
    y = make_series()

    with pytest.raises(ValueError, match="Unknown model"):
        TimeSeries(models=["unknown"]).fit(y)

    with pytest.raises(ValueError, match="min_improvement"):
        TimeSeries(search_options={'autoregressive': {'min_improvement': -1}}).fit(y)

    with pytest.raises(ValueError, match="window_size"):
        TimeSeries(window_size=0).fit(y)

    with pytest.raises(ValueError, match="alphas"):
        TimeSeries(models=['exponential_smoothing'], search_options={'exponential_smoothing': {'alphas': [1.5]}}).fit(y)

    with pytest.raises(ValueError, match="ARIMA order"):
        TimeSeries(models=['arima'], search_options={'arima': {'orders': [(1, 0)]}}).fit(y)

    with pytest.raises(ValueError, match="state-space"):
        TimeSeries(models=['state_space'], search_options={'state_space': {'models': ['bad']}}).fit(y)

    with pytest.raises(ValueError, match="max_iter"):
        TimeSeries(models=['arima'], search_options={'arima': {'max_iter': 0}, 'state_space': {'max_iter': 0}}).fit(y)


def test_unfitted_methods_raise_not_fitted_error():
    ts = TimeSeries(window_size=4)

    with pytest.raises(NotFittedError):
        ts.forecast()
    with pytest.raises(NotFittedError):
        ts.results_dataframe()
    with pytest.raises(NotFittedError):
        ts.components()
    with pytest.raises(NotFittedError):
        ts.model_description()["model_string"]


def test_fixed_linear_forecaster_and_weight_helpers():
    X = np.arange(20, dtype=float).reshape(10, 2)
    y = X @ np.array([0.25, 0.75])
    model = FixedLinearForecaster(weights=[0.25, 0.75]).fit(X, y)

    assert np.allclose(model.predict(X), y)
    assert model.score(X, y) == pytest.approx(1.0)
    assert np.allclose(moving_average_weights(3), [1 / 3, 1 / 3, 1 / 3])
    assert np.allclose(exponential_smoothing_weights(3, 0.5).sum(), 1.0)


def test_fixed_model_description_is_stable():
    X = np.arange(20).reshape(10, 2)
    model = FixedLinearForecaster(weights=[0.5, 0.5]).fit(X, X.mean(axis=1))
    description = describe_model(model, feature_names=["y_lag_1", "y_lag_2"])
    assert description.canonical == "5.00e-01*x0+5.00e-01*x1"
    assert description.feature_names == ("y_lag_1", "y_lag_2")
    np.testing.assert_allclose(execute(description.ast, X), model.predict(X))


def test_autoregressive_search_uses_selection_options():
    y = make_series()
    ts = TimeSeries(window_size=4, models=['autoregressive'], search_options={'autoregressive': {'min_improvement': 0.0}}).fit(y)

    assert ts.search_options["autoregressive"]["min_improvement"] == pytest.approx(0.0)
    assert ts.best_result_.is_reliable


def test_candidate_components_are_computed_from_explicit_artifacts():
    y = make_series()
    ts = TimeSeries(window_size=4, models=["autoregressive"]).fit(y)
    result = ts.best_result_

    direct_components = {
        "deficiency": ts.miscoding_.deficiency_subset(result.artifacts.subset),
        "surplus": ts.miscoding_.surplus_subset(result.artifacts.subset),
        "inaccuracy": ts.mismodel_.inaccuracy_.inaccuracy_predictions(result.artifacts.predictions),
        "surfeit": ts.mismodel_.surfeit_.surfeit_string(result.artifacts.model_string),
    }

    assert result.components == pytest.approx(direct_components)
    assert result.nescience == pytest.approx(ts.nescience_.aggregate_components(
        miscoding=np.sqrt((direct_components["deficiency"]**2 + direct_components["surplus"]**2) / 2),
        mismodel=np.sqrt((direct_components["inaccuracy"]**2 + direct_components["surfeit"]**2) / 2)))


def test_candidate_results_include_subset_reliability_diagnostics():
    y = make_series()
    ts = TimeSeries(window_size=4, models=["autoregressive"]).fit(y)
    result = ts.best_result_
    diagnostics = ts.miscoding_.subset_analysis(result.artifacts.subset)

    assert result.is_reliable is True
    assert result.subset_diagnostics["is_reliable"] is True
    assert result.subset_diagnostics["failure_reason"] is None
    assert result.subset_diagnostics["n_samples"] == diagnostics["n_samples"]
    assert result.subset_diagnostics["resolved_n_bins"] == diagnostics["resolved_n_bins"]
    for row in ts.results_dataframe().itertuples():
        expected = ts.miscoding_.subset_analysis(row.selected_features)["resolved_n_bins"]
        assert row.resolved_n_bins == expected
    assert ts.analysis()["resolved_n_bins"] == diagnostics["resolved_n_bins"]


@pytest.mark.parametrize("family", ["arima", "state_space"])
def test_stateful_descriptions_raise_explicit_unsupported_error(family):
    model = TimeSeries(window_size=5, models=[family])
    with pytest.raises(ValueError, match="not supported.*schema version 1"):
        model.fit(make_series(90))
    assert model.diagnostics_[0]["reason"] == "unsupported_model_description"


def test_unsupported_descriptions_are_reported_and_other_families_remain_available():
    model = TimeSeries(window_size=4, models=["arima", "state_space", "moving_average"]).fit(make_series())
    assert set(model.results_dataframe()["family"]) == {"moving_average"}
    assert {item["family"] for item in model.diagnostics_} == {"arima", "state_space"}


def test_fit_warns_and_retains_finite_unreliable_candidate():
    rng = np.random.default_rng(123)
    y = rng.normal(size=10)

    with pytest.warns(RuntimeWarning, match="sparsely populated") as caught:
        model = TimeSeries(
            window_size=5,
            models=["moving_average"],
        ).fit(y)
    assert len(caught) == 1
    assert not model.best_result_.is_reliable
    assert np.isfinite(model.best_result_.nescience)
