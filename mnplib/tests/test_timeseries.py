"""Tests for the redesigned TimeSeries package."""

import numpy as np
import pandas as pd
import pytest

from sklearn.exceptions import NotFittedError

from mnplib.timeseries import TimeSeries, FixedLinearForecaster
from mnplib.timeseries.models import (
    canonical_fixed_model_string,
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
    ts = TimeSeries(window_size=5, n_bins=3, random_state=42).fit(y)

    assert ts.is_fitted_
    assert ts.window_size_ == 5
    assert ts.X_supervised_.shape == (len(y) - 5, 5)
    assert ts.y_supervised_.shape == (len(y) - 5,)
    assert ts.best_result_.nescience == pytest.approx(ts.best_nescience_ if hasattr(ts, "best_nescience_") else ts.nescience_score())
    assert ts.model_ is ts.best_result_.model
    assert len(ts.candidate_results_) > 1
    assert len(ts.selected_feature_indices_) >= 1


def test_forecast_returns_requested_number_of_steps():
    y = make_series()
    ts = TimeSeries(window_size=4, models=["autoregressive"], n_bins=3).fit(y)

    forecast = ts.forecast(steps=6)

    assert isinstance(forecast, np.ndarray)
    assert forecast.shape == (6,)
    assert np.all(np.isfinite(forecast))


def test_predict_accepts_lagged_feature_matrix():
    y = make_series()
    ts = TimeSeries(window_size=4, models=["autoregressive"], n_bins=3).fit(y)

    predictions = ts.predict(ts.X_supervised_[:8])

    assert predictions.shape == (8,)


def test_score_uses_selected_model_on_supplied_series():
    y = make_series()
    ts = TimeSeries(window_size=4, models=["autoregressive"], n_bins=3).fit(y)

    score = ts.score(y)

    assert isinstance(score, float)
    assert np.isfinite(score)


def test_results_dataframe_is_sorted_and_has_expected_columns():
    y = make_series()
    ts = TimeSeries(window_size=4, n_bins=3).fit(y)

    df = ts.results_dataframe()

    expected = {
        "model_name",
        "model_family",
        "window_size",
        "nescience",
        "estimator_score",
        "n_selected_features",
        "description_length",
        "selected_feature_indices",
        "selected_feature_names",
        "deficiency",
        "surplus",
        "miscoding",
        "inaccuracy",
        "surfeit",
    }
    assert expected.issubset(df.columns)
    assert df["nescience"].is_monotonic_increasing
    assert df.iloc[0]["model_name"] == ts.model_name_


def test_components_nescience_and_model_string():
    y = make_series()
    ts = TimeSeries(window_size=4, n_bins=3).fit(y)

    components = ts.components()
    model_string = ts.model_string()

    assert set(components) == {"deficiency", "surplus", "inaccuracy", "surfeit"}
    assert ts.nescience_score() == pytest.approx(ts.best_result_.nescience)
    assert model_string.startswith("SCHEMA canonical_nescience_time_series_model_v1")
    assert "TASK forecasting" in model_string
    assert "RULE" in model_string


def test_explain_contains_time_series_details():
    y = make_series()
    ts = TimeSeries(window_size=4, n_bins=3).fit(y)

    explanation = ts.explain()

    assert explanation["time_series_model"] == ts.model_name_
    assert explanation["window_size"] == ts.window_size_
    assert explanation["selected_feature_names"] == ts.selected_feature_names_
    assert "components" in explanation
    assert "dominant_component" in explanation


def test_lag_analysis_methods_without_exogenous_data():
    y = make_series()
    ts = TimeSeries(window_size=5, max_lag=6, n_bins=3).fit(y)

    auto = ts.auto_lag_analysis(max_lag=4)
    all_lags = ts.lag_analysis(max_lag=4)

    assert list(auto["lag"]) == [1, 2, 3, 4]
    assert {"lag", "feature_name", "deficiency", "surplus", "miscoding"}.issubset(auto.columns)
    assert len(all_lags) == len(auto)



def test_exogenous_data_feature_names_forecast_and_cross_lag_analysis():
    y, X = make_exogenous_series()
    ts = TimeSeries(window_size=4, n_bins=3).fit(y, X)

    assert list(ts.exogenous_feature_names_) == ["temperature", "demand"]
    assert "temperature_lag_1" in list(ts.feature_names_in_)

    future = X.tail(3).to_numpy()
    forecast = ts.forecast(steps=3, X_future=future)
    cross = ts.cross_lag_analysis("temperature", max_lag=3)
    all_lags = ts.lag_analysis(max_lag=2)

    assert forecast.shape == (3,)
    assert list(cross["lag"]) == [1, 2, 3]
    assert "attribute" in cross.columns
    assert set(all_lags.get("attribute", pd.Series(dtype=object)).dropna()).issubset({"temperature", "demand"})


def test_model_family_filtering():
    y = make_series()
    ts = TimeSeries(window_size=4, models=["moving_average"], n_bins=3).fit(y)

    assert set(result.model_family for result in ts.candidate_results_) == {"moving_average"}
    assert ts.model_name_.startswith("moving_average")


def test_moving_average_and_smoothing_configuration():
    y = make_series()
    ts = TimeSeries(
        window_size=5,
        models=["moving_average", "exponential_smoothing"],
        moving_average_windows=[2, 5],
        smoothing_alphas=[0.2, 0.8],
        n_bins=3,
    ).fit(y)

    names = {result.model_name for result in ts.candidate_results_}

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
        TimeSeries(min_improvement=-1).fit(y)

    with pytest.raises(ValueError, match="window_size"):
        TimeSeries(window_size=0).fit(y)

    with pytest.raises(ValueError, match="alphas"):
        TimeSeries(models=["exponential_smoothing"], smoothing_alphas=[1.5]).fit(y)


def test_unfitted_methods_raise_not_fitted_error():
    ts = TimeSeries(window_size=4)

    with pytest.raises(NotFittedError):
        ts.forecast()
    with pytest.raises(NotFittedError):
        ts.results_dataframe()
    with pytest.raises(NotFittedError):
        ts.components()
    with pytest.raises(NotFittedError):
        ts.model_string()


def test_fixed_linear_forecaster_and_weight_helpers():
    X = np.arange(20, dtype=float).reshape(10, 2)
    y = X @ np.array([0.25, 0.75])
    model = FixedLinearForecaster(weights=[0.25, 0.75]).fit(X, y)

    assert np.allclose(model.predict(X), y)
    assert model.score(X, y) == pytest.approx(1.0)
    assert np.allclose(moving_average_weights(3), [1 / 3, 1 / 3, 1 / 3])
    assert np.allclose(exponential_smoothing_weights(3, 0.5).sum(), 1.0)


def test_canonical_fixed_model_string_is_stable():
    text = canonical_fixed_model_string(
        model_type="moving_average",
        model_name="moving_average_2",
        feature_names=["y_lag_1", "y_lag_2"],
        weights=np.array([0.5, 0.5]),
        precision=3,
    )

    assert text.startswith("SCHEMA canonical_nescience_time_series_model_v1")
    assert "MODEL moving_average" in text
    assert "INPUTS y_lag_1, y_lag_2" in text
    assert "y_hat += 0.5 * y_lag_1" in text



def test_timeseries_uses_latest_miscoding_api_name():
    y = make_series()
    ts = TimeSeries(window_size=4, models=["autoregressive"], n_bins=3, min_improvement=0.0).fit(y)

    assert hasattr(ts.miscoding_, "min_improvement")
    assert not hasattr(ts, "surplus_penalty")
    assert ts.miscoding_.min_improvement == pytest.approx(0.0)


def test_candidate_components_are_computed_from_explicit_artifacts():
    y = make_series()
    ts = TimeSeries(window_size=4, models=["autoregressive"], n_bins=3).fit(y)
    result = ts.best_result_

    direct_components = {
        "deficiency": ts.miscoding_.miscoding_subset(result.subset, mode="deficiency"),
        "surplus": ts.miscoding_.miscoding_subset(result.subset, mode="surplus"),
        "inaccuracy": ts.inaccuracy_.inaccuracy_predictions(result.predictions),
        "surfeit": ts.surfeit_.surfeit_string(result.model_string),
    }

    assert result.components == pytest.approx(direct_components)
    assert result.nescience == pytest.approx(ts.nescience_.aggregate_components(**direct_components))
