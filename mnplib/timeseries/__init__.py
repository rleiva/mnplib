"""Time-series forecasting with minimum nescience."""

from .estimator import TimeSeries
from .models import (
    FixedLinearForecaster, StatsmodelsForecastModel,
    exponential_smoothing_weights, moving_average_weights,
)
from .selection import TimeSeriesCandidateResult
from .lagged import LaggedRepresentation, LaggedRepresentationBuilder

__all__ = [
    "TimeSeries", "FixedLinearForecaster", "StatsmodelsForecastModel",
    "TimeSeriesCandidateResult", "LaggedRepresentation", "LaggedRepresentationBuilder",
    "exponential_smoothing_weights", "moving_average_weights",
]
