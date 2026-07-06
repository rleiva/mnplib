"""
Lagged representations for time-series forecasting.

The functions in this module perform one task: transform an ordered target
series, and optionally a matrix of exogenous variables, into a supervised
lagged representation. Keeping this logic outside the estimator makes the
forecasting class easier to read, test, and extend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from sklearn.utils import check_array, column_or_1d


WindowSize = int | Literal["auto"]


@dataclass(frozen=True)
class LaggedRepresentation:
    """Supervised representation built from a time series.

    Parameters
    ----------
    X : numpy.ndarray of shape (n_samples - window_size, n_features)
        Lagged feature matrix.

    y : numpy.ndarray of shape (n_samples - window_size,)
        Target values aligned with the lagged feature rows.

    feature_names : tuple of str
        Human-readable feature names such as ``y_lag_1`` or
        ``temperature_lag_3``.

    feature_metadata : tuple of dict
        Structured metadata for each feature. The estimator uses this metadata
        to report selected lags in a stable, user-facing format.
    """

    X: np.ndarray
    y: np.ndarray
    feature_names: tuple[str, ...]
    feature_metadata: tuple[dict[str, object], ...]


class LaggedRepresentationBuilder:
    """Build lagged supervised representations.

    The builder contains no metric or model-selection logic. It only validates
    ordered arrays and creates lagged rows using the convention that
    ``lag=1`` refers to the immediately preceding observation.
    """

    def __init__(self, window_size: WindowSize = "auto"):
        self.window_size = window_size

    def build(self, y, X=None) -> tuple[LaggedRepresentation, np.ndarray | None, tuple[str, ...]]:
        """Validate inputs and build a lagged supervised representation.

        Returns the representation, the validated exogenous matrix, and the
        exogenous feature names. Returning all three objects keeps the public
        estimator free from duplicate validation code.
        """
        y_array = self.validate_y(y)
        X_array, exogenous_names = self.validate_exogenous_X(X, n_samples=len(y_array))
        window_size = self.resolve_window_size(len(y_array))
        representation = self.to_supervised(
            y=y_array,
            X=X_array,
            window_size=window_size,
            exogenous_feature_names=exogenous_names,
        )
        return representation, X_array, exogenous_names

    def resolve_window_size(self, n_samples: int) -> int:
        """Resolve the effective window size for a series length."""
        if n_samples < 3:
            raise ValueError("y must contain at least three observations.")

        if self.window_size == "auto":
            window_size = max(1, int(np.sqrt(n_samples)))
        else:
            window_size = int(self.window_size)

        if window_size < 1:
            raise ValueError("window_size must be a positive integer or 'auto'.")
        if window_size >= n_samples:
            raise ValueError(
                f"window_size={window_size} must be smaller than n_samples={n_samples}."
            )
        return window_size

    @staticmethod
    def validate_y(y) -> np.ndarray:
        """Validate and return a numeric one-dimensional target series."""
        y_array = column_or_1d(y).astype(float, copy=False)
        if y_array.size < 3:
            raise ValueError("y must contain at least three observations.")
        if not np.all(np.isfinite(y_array)):
            raise ValueError("y must contain only finite numeric values.")
        return y_array

    @staticmethod
    def validate_exogenous_X(X, *, n_samples: int) -> tuple[np.ndarray | None, tuple[str, ...]]:
        """Validate optional exogenous variables.

        The current implementation is intentionally numeric. Categorical
        exogenous variables should be encoded before fitting the estimator.
        """
        if X is None:
            return None, tuple()

        if isinstance(X, pd.DataFrame):
            if len(X) != n_samples:
                raise ValueError(f"X and y have inconsistent lengths: {len(X)} != {n_samples}.")
            if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
                raise ValueError("TimeSeries requires numeric exogenous variables.")
            return X.to_numpy(dtype=float), tuple(str(column) for column in X.columns)

        X_array = check_array(X, dtype=float, ensure_2d=True)
        if X_array.shape[0] != n_samples:
            raise ValueError(f"X and y have inconsistent lengths: {X_array.shape[0]} != {n_samples}.")
        if not np.all(np.isfinite(X_array)):
            raise ValueError("X must contain only finite numeric values.")
        return X_array, tuple(f"x{j}" for j in range(X_array.shape[1]))

    @classmethod
    def to_supervised(
        cls,
        *,
        y: np.ndarray,
        X: np.ndarray | None,
        window_size: int,
        exogenous_feature_names: tuple[str, ...] = tuple(),
    ) -> LaggedRepresentation:
        """Convert a validated series into supervised lagged arrays."""
        if window_size >= len(y):
            raise ValueError(
                f"window_size={window_size} must be smaller than n_samples={len(y)}."
            )

        rows = [cls.lagged_row(y=y, X=X, t=t, window_size=window_size) for t in range(window_size, len(y))]
        names, metadata = cls.feature_description(
            has_exogenous=X is not None,
            window_size=window_size,
            exogenous_feature_names=exogenous_feature_names,
        )

        return LaggedRepresentation(
            X=np.asarray(rows, dtype=float),
            y=np.asarray(y[window_size:], dtype=float),
            feature_names=names,
            feature_metadata=metadata,
        )

    @staticmethod
    def lagged_row(*, y: np.ndarray, X: np.ndarray | None, t: int, window_size: int) -> list[float]:
        """Build one lagged row for target index ``t``."""
        row = [float(y[t - lag]) for lag in range(1, window_size + 1)]
        if X is not None:
            for j in range(X.shape[1]):
                row.extend(float(X[t - lag, j]) for lag in range(1, window_size + 1))
        return row

    @staticmethod
    def single_forecast_row(
        *,
        y_history: np.ndarray,
        X_history: np.ndarray | None,
        window_size: int,
    ) -> np.ndarray:
        """Build the current one-step-ahead lagged row from available history."""
        y_history = np.asarray(y_history, dtype=float)
        if len(y_history) < window_size:
            raise ValueError("Not enough target history to build a lagged row.")

        row = [float(y_history[-lag]) for lag in range(1, window_size + 1)]

        if X_history is not None:
            X_history = np.asarray(X_history, dtype=float)
            if len(X_history) < window_size:
                raise ValueError("Not enough exogenous history to build a lagged row.")
            for j in range(X_history.shape[1]):
                row.extend(float(X_history[-lag, j]) for lag in range(1, window_size + 1))

        return np.asarray(row, dtype=float).reshape(1, -1)

    @staticmethod
    def feature_description(
        *,
        has_exogenous: bool,
        window_size: int,
        exogenous_feature_names: tuple[str, ...] = tuple(),
    ) -> tuple[tuple[str, ...], tuple[dict[str, object], ...]]:
        """Return stable feature names and metadata for the lagged matrix."""
        names: list[str] = []
        metadata: list[dict[str, object]] = []

        for lag in range(1, window_size + 1):
            names.append(f"y_lag_{lag}")
            metadata.append({"source": "target", "attribute": "y", "lag": lag})

        if has_exogenous:
            for attribute_index, attribute_name in enumerate(exogenous_feature_names):
                for lag in range(1, window_size + 1):
                    names.append(f"{attribute_name}_lag_{lag}")
                    metadata.append(
                        {
                            "source": "exogenous",
                            "attribute": str(attribute_name),
                            "attribute_index": int(attribute_index),
                            "lag": int(lag),
                        }
                    )

        return tuple(names), tuple(metadata)
