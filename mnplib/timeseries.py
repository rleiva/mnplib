"""
Time-series forecasting public interface.

The estimator and helper objects are organized in the package modules under
``mnplib.timeseries``. This file exposes the public time-series API from that
implementation so source distributions present one current interface.
"""

from __future__ import annotations

from mnplib.timeseries.estimator import TimeSeries
from mnplib.timeseries.lagged import LaggedRepresentation, LaggedRepresentationBuilder
from mnplib.timeseries.models import (
    FixedLinearForecaster,
    StatsmodelsForecastModel,
    canonical_arima_model_string,
    canonical_fixed_model_string,
    canonical_linear_model_string,
    canonical_state_space_model_string,
    exponential_smoothing_weights,
    moving_average_weights,
)
from mnplib.timeseries.selection import TimeSeriesCandidateResult


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
