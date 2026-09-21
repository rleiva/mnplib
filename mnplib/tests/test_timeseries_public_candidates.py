import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from mnplib.timeseries import TimeSeries


def test_capabilities_are_public_and_independent():
    capabilities = TimeSeries.family_capabilities()
    assert capabilities['autoregressive']['supports_exogenous']
    assert capabilities['arima']['subset_semantics'] == 'diagnostic_proxy'
    assert capabilities['moving_average']['subset_semantics'] == 'lag_inputs'
    capabilities['arima']['subset_semantics'] = 'changed'
    assert TimeSeries.family_capabilities()['arima']['subset_semantics'] == 'diagnostic_proxy'


@pytest.mark.parametrize('family', list(TimeSeries.family_capabilities()))
def test_named_candidate_forecasts_and_fitted_values_do_not_change_best(family):
    y = np.sin(np.arange(100) / 5) + np.arange(100) / 100
    ts = TimeSeries(window_size=3, models=[family], n_bins=2).fit(y)
    best = ts.best_result_
    for result in ts.candidate_results_:
        predictions = ts.fitted_values(candidate=result.name)
        assert np.isnan(predictions[:3]).all()
        np.testing.assert_allclose(predictions[3:], result.artifacts.predictions)
        predictions[:] = 0
        assert np.isnan(ts.fitted_values(candidate=result.name)[:3]).all()
        forecast = ts.forecast(4, candidate=result.name)
        assert forecast.shape == (4,)
        assert np.isfinite(forecast).all()
        if family in {'arima', 'state_space'}:
            np.testing.assert_allclose(forecast, result.model.forecast(4))
            assert isinstance(result.metadata['converged'], bool)
        else:
            history = list(y)
            expected = []
            for _ in range(4):
                row = np.asarray(history[-3:][::-1]).reshape(1, -1)
                value = result.model.predict(row[:, list(result.artifacts.subset)])[0]
                expected.append(value)
                history.append(value)
            np.testing.assert_allclose(forecast, expected)
        assert ts.best_result_ is best
        assert ts.model_ is best.model
    np.testing.assert_allclose(ts.fitted_values(), ts.fitted_values_, equal_nan=True)
    with pytest.raises(ValueError, match='Unknown'):
        ts.forecast(candidate='missing')
    with pytest.raises(ValueError, match='Unknown'):
        ts.fitted_values(candidate='missing')


def test_unfitted_candidate_methods():
    ts = TimeSeries()
    with pytest.raises(NotFittedError):
        ts.forecast(candidate='autoregressive')
    with pytest.raises(NotFittedError):
        ts.fitted_values()


def test_nonconvergence_is_reported_numerically():
    y = np.random.default_rng(0).normal(size=100).cumsum()
    ts = TimeSeries(window_size=3, models=['arima'], n_bins=2,
                    search_options={'arima': {'max_iter': 1}}).fit(y)
    rows = ts.results_dataframe().to_dict('records')
    failed = [row for row in rows if row['metadata']['converged'] is False]
    assert failed
    assert {row['candidate'] for row in failed} == {
        item['candidate'] for item in ts.diagnostics_ if item['reason'] == 'not_converged'
    }
