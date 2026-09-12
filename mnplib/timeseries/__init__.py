"""Time-series forecasting with minimum nescience."""

from .estimator import TimeSeries
from .models import (
    FixedLinearForecaster,
    StatsmodelsForecastModel,
    canonical_arima_model_string,
    canonical_fixed_model_string,
    canonical_linear_model_string,
    canonical_state_space_model_string,
    exponential_smoothing_weights,
    moving_average_weights,
)
from .selection import TimeSeriesCandidateResult
from .lagged import LaggedRepresentation, LaggedRepresentationBuilder

__all__ = [
    "TimeSeries",
    "FixedLinearForecaster",
    "StatsmodelsForecastModel",
    "TimeSeriesCandidateResult",
    "LaggedRepresentation",
    "LaggedRepresentationBuilder",
    "canonical_arima_model_string",
    "canonical_fixed_model_string",
    "canonical_linear_model_string",
    "canonical_state_space_model_string",
    "exponential_smoothing_weights",
    "moving_average_weights",
]
