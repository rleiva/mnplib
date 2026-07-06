"""
Forecasting models and canonical time-series descriptions.

Model descriptions are explicit strings consumed by the surfeit component of
nescience. They should be stable, readable, and semantically richer than the
ordinary Python ``repr`` of a fitted estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted


TIME_SERIES_SCHEMA = "canonical_nescience_time_series_model_v1"
ModelFamily = Literal["autoregressive", "moving_average", "exponential_smoothing"]


class FixedLinearForecaster(BaseEstimator, RegressorMixin):
    """Linear forecaster with fixed user-supplied coefficients.

    This estimator is used for moving-average and exponential-smoothing
    candidates. It behaves like a scikit-learn regressor but does not learn
    coefficients from data.
    """

    def __init__(self, weights, intercept: float = 0.0, name: str = "fixed_linear"):
        self.weights = weights
        self.intercept = intercept
        self.name = name

    def fit(self, X, y=None):
        X = check_array(X, dtype=float)
        weights = np.asarray(self.weights, dtype=float)
        if weights.ndim != 1:
            raise ValueError("weights must be one-dimensional.")
        if X.shape[1] != weights.shape[0]:
            raise ValueError(
                f"weights length {weights.shape[0]} does not match X with {X.shape[1]} columns."
            )
        self.weights_ = weights
        self.intercept_ = float(self.intercept)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self)
        return check_array(X, dtype=float) @ self.weights_ + self.intercept_

    def score(self, X, y):
        check_is_fitted(self)
        y = np.asarray(y, dtype=float).ravel()
        prediction = self.predict(X)
        denominator = float(np.sum((y - np.mean(y)) ** 2))
        if denominator == 0.0:
            return 0.0
        numerator = float(np.sum((y - prediction) ** 2))
        return 1.0 - numerator / denominator

    def __repr__(self):
        weights = np.array2string(np.asarray(self.weights), precision=6)
        return f"FixedLinearForecaster(name={self.name!r}, weights={weights})"


@dataclass(frozen=True)
class TimeSeriesCandidateSpec:
    """Fitted candidate model ready for nescience evaluation."""

    model_name: str
    model_family: ModelFamily
    model: object
    subset: np.ndarray
    window_size: int
    model_string: str


def moving_average_weights(window: int) -> np.ndarray:
    """Return normalized moving-average weights for a lag window."""
    if int(window) < 1:
        raise ValueError("window must be positive.")
    return np.repeat(1.0 / int(window), int(window))


def exponential_smoothing_weights(window: int, alpha: float) -> np.ndarray:
    """Return normalized finite-window exponential-smoothing weights."""
    window = int(window)
    alpha = float(alpha)
    if window < 1:
        raise ValueError("window must be positive.")
    if alpha <= 0.0 or alpha >= 1.0:
        raise ValueError("alpha must lie in the open interval (0, 1).")
    weights = alpha * (1.0 - alpha) ** np.arange(window)
    return weights / np.sum(weights)


def canonical_linear_model_string(
    *,
    model: LinearRegression,
    model_name: str,
    feature_names: list[str] | tuple[str, ...],
    precision: int = 6,
) -> str:
    """Serialize a fitted linear autoregressive model."""
    check_is_fitted(model)
    coefficients = np.asarray(model.coef_, dtype=float).ravel()
    intercept = float(np.asarray(model.intercept_).ravel()[0])
    return canonical_weighted_model_string(
        model_type="autoregressive_linear",
        model_name=model_name,
        feature_names=feature_names,
        weights=coefficients,
        intercept=intercept,
        precision=precision,
        learned=True,
    )


def canonical_fixed_model_string(
    *,
    model_type: str,
    model_name: str,
    feature_names: list[str] | tuple[str, ...],
    weights: np.ndarray,
    intercept: float = 0.0,
    precision: int = 6,
) -> str:
    """Serialize a fixed-coefficient forecasting model."""
    return canonical_weighted_model_string(
        model_type=model_type,
        model_name=model_name,
        feature_names=feature_names,
        weights=np.asarray(weights, dtype=float),
        intercept=float(intercept),
        precision=precision,
        learned=False,
    )


def canonical_weighted_model_string(
    *,
    model_type: str,
    model_name: str,
    feature_names: list[str] | tuple[str, ...],
    weights: np.ndarray,
    intercept: float,
    precision: int,
    learned: bool,
) -> str:
    """Serialize a weighted one-step forecasting rule."""
    feature_names = [str(name) for name in feature_names]
    weights = np.asarray(weights, dtype=float).ravel()
    if len(feature_names) != len(weights):
        raise ValueError("feature_names and weights must have the same length.")

    lines = [
        f"SCHEMA {TIME_SERIES_SCHEMA}",
        f"MODEL {model_type}",
        "TASK forecasting",
        f"NAME {model_name}",
        f"INPUTS {', '.join(feature_names) if feature_names else '<none>'}",
        "PARAMETERS",
        f"    n_features = {len(feature_names)}",
        f"    learned_coefficients = {str(bool(learned)).lower()}",
        "RULE",
        f"    y_hat = {format_number(intercept, precision)}",
    ]

    for weight, name in zip(weights, feature_names):
        lines.append(f"    y_hat += {format_number(float(weight), precision)} * {name}")

    lines.append("    return y_hat")
    return "\n".join(lines) + "\n"


def format_number(value: float, precision: int) -> str:
    """Format numbers in canonical model descriptions."""
    if not np.isfinite(value):
        raise ValueError("Model descriptions require finite numeric coefficients.")
    rounded = f"{float(value):.{int(precision)}f}"
    # Keep at least one decimal place to make numeric constants visually clear.
    if "." in rounded:
        rounded = rounded.rstrip("0").rstrip(".")
    return rounded if "." in rounded else f"{rounded}.0"
