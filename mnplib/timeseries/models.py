"""
Forecasting models and canonical time-series descriptions.

Model descriptions are explicit strings consumed by the surfeit component of
nescience. They should be stable, readable, and semantically richer than the
ordinary Python ``repr`` of a fitted estimator.
"""

from __future__ import annotations

import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from mnplib.models.serializers.time_series import (
    TIME_SERIES_SCHEMA,
    canonical_arima_model_string,
    canonical_fixed_model_string,
    canonical_linear_model_string,
    canonical_state_space_model_string,
)

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


class StatsmodelsForecastModel:
    """Forecasting facade around a fitted statsmodels result object."""

    def __init__(
        self,
        result,
        *,
        name: str,
        family: str,
        training_predictions,
    ):
        self.result_ = result
        self.name = str(name)
        self.family = str(family)
        self.training_predictions_ = np.asarray(training_predictions, dtype=float).ravel()
        self.n_features_in_ = 1
        self.is_fitted_ = True

    def predict(self, X):
        X_checked = check_array(X, dtype=float, ensure_2d=True)
        n_rows = X_checked.shape[0]
        if n_rows > self.training_predictions_.shape[0]:
            raise ValueError(
                "predict requires no more rows than the fitted one-step prediction path."
            )
        return self.training_predictions_[:n_rows].copy()

    def score(self, X, y):
        y_array = np.asarray(y, dtype=float).ravel()
        prediction = self.predict(X)
        if prediction.shape[0] != y_array.shape[0]:
            raise ValueError("X and y have inconsistent lengths.")
        denominator = float(np.sum((y_array - np.mean(y_array)) ** 2))
        if denominator == 0.0:
            return 0.0
        numerator = float(np.sum((y_array - prediction) ** 2))
        return 1.0 - numerator / denominator

    def forecast(self, steps: int = 1, X_future=None) -> np.ndarray:
        if X_future is not None:
            raise ValueError("This forecasting model does not use future exogenous values.")
        steps = int(steps)
        if steps < 1:
            raise ValueError("steps must be positive.")
        return np.asarray(self.result_.forecast(steps=steps), dtype=float).ravel()

    def __repr__(self):
        return f"StatsmodelsForecastModel(name={self.name!r}, family={self.family!r})"


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
