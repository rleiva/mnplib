"""
Model adapters for mnplib.

The adapter layer converts concrete fitted models into the explicit artifacts
required by the simplified metric classes.
"""

from .artifacts import ModelArtifacts
from .serializers.time_series import (
    TIME_SERIES_SCHEMA,
    canonical_arima_model_string,
    canonical_fixed_model_string,
    canonical_linear_model_string,
    canonical_state_space_model_string,
    time_series_model_artifacts,
)
from .sklearn import (
    sklearn_model_artifacts,
)

__all__ = [
    "ModelArtifacts",
    "sklearn_model_artifacts",
    "TIME_SERIES_SCHEMA",
    "canonical_arima_model_string",
    "canonical_fixed_model_string",
    "canonical_linear_model_string",
    "canonical_state_space_model_string",
    "time_series_model_artifacts",
]
