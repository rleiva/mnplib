"""
Model-family searchers for nescience-based time-series forecasting.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np

from sklearn.linear_model import LinearRegression

from mnplib.automl.evaluator import CandidateEvaluator
from mnplib.automl.searchers.base import ModelFamilySearcher, search_report
from mnplib.models.serializers.time_series import time_series_model_artifacts

from .models import (
    FixedLinearForecaster,
    exponential_smoothing_weights,
    moving_average_weights,
)
from .selection import TimeSeriesCandidateResult


@dataclass(frozen=True)
class TimeSeriesSearchContext:
    """
    Shared context supplied to time-series model-family searchers.
    """

    X: np.ndarray
    y: np.ndarray
    feature_names: tuple[str, ...]
    evaluator: CandidateEvaluator
    window_size: int
    moving_average_windows: tuple[int, ...]
    smoothing_alphas: tuple[float, ...]
    min_improvement: float
    smoothing_windows: tuple[int, ...]


class AutoregressiveSearcher(ModelFamilySearcher):
    """
    Evaluate a linear autoregressive candidate selected by miscoding.
    """

    family = "autoregressive"

    def search(self, context: TimeSeriesSearchContext):
        selection = context.evaluator.nescience.miscoding_.select_features(
            return_details=True, min_improvement=context.min_improvement
        )
        subset = ensure_non_empty_subset(selection["mask"], context)
        selected = np.flatnonzero(subset)

        model = LinearRegression().fit(context.X[:, selected], context.y)
        model_name = "autoregressive"
        artifacts = time_series_model_artifacts(
            model, context.X[:, selected],
            feature_names=selected_feature_names(context, selected),
            feature_indices=selected,
        )
        result = context.evaluator.evaluate_artifacts(
            name=model_name,
            family=self.family,
            model=model,
            artifacts=artifacts,
            estimator_score=float(model.score(context.X[:, selected], context.y)),
            metadata=candidate_metadata(
                context,
                family=self.family,
                model_name=model_name,
                selected=selected,
                extra={"selection_method": "miscoding"},
            ),
            result_factory=TimeSeriesCandidateResult,
        )
        return search_report(self.family, [result])


class MovingAverageSearcher(ModelFamilySearcher):
    """
    Evaluate fixed-window moving-average candidates.
    """

    family = "moving_average"

    def search(self, context: TimeSeriesSearchContext):
        results = []
        for window in context.moving_average_windows:
            subset = target_lag_subset(context, window)
            selected = np.flatnonzero(subset)
            model_name = f"moving_average_{window}"
            weights = moving_average_weights(window)
            model = FixedLinearForecaster(weights=weights, name=model_name)
            model.fit(context.X[:, selected], context.y)
            artifacts = time_series_model_artifacts(
                model, context.X[:, selected],
                feature_names=selected_feature_names(context, selected),
                feature_indices=selected,
            )
            results.append(
                context.evaluator.evaluate_artifacts(
                    name=model_name,
                    family=self.family,
                    model=model,
                    artifacts=artifacts,
                    estimator_score=float(model.score(context.X[:, selected], context.y)),
                    hyperparameters={"window": int(window)},
                    metadata=candidate_metadata(
                        context,
                        family=self.family,
                        model_name=model_name,
                        selected=selected,
                    ),
                    result_factory=TimeSeriesCandidateResult,
                )
            )
        return search_report(self.family, results)


class ExponentialSmoothingSearcher(ModelFamilySearcher):
    """
    Evaluate fixed finite-window exponential-smoothing candidates.
    """

    family = "exponential_smoothing"

    def search(self, context: TimeSeriesSearchContext):
        results = []
        for window in context.smoothing_windows:
            subset = target_lag_subset(context, window)
            selected = np.flatnonzero(subset)
            for alpha in context.smoothing_alphas:
                model_name = f"exponential_smoothing_w{window}_a{alpha:g}"
                weights = exponential_smoothing_weights(window, alpha)
                model = FixedLinearForecaster(weights=weights, name=model_name)
                model.fit(context.X[:, selected], context.y)
                artifacts = time_series_model_artifacts(
                    model, context.X[:, selected],
                    feature_names=selected_feature_names(context, selected),
                    feature_indices=selected,
                )
                results.append(
                    context.evaluator.evaluate_artifacts(
                        name=model_name,
                        family=self.family,
                        model=model,
                        artifacts=artifacts,
                        estimator_score=float(model.score(context.X[:, selected], context.y)),
                        hyperparameters={"window": int(window), "alpha": float(alpha)},
                        metadata=candidate_metadata(
                            context,
                            family=self.family,
                            model_name=model_name,
                            selected=selected,
                        ),
                        result_factory=TimeSeriesCandidateResult,
                    )
                )
        return search_report(self.family, results)


class ARIMASearcher(ModelFamilySearcher):
    """Report that stateful ARIMA descriptions are outside schema version 1."""

    family = "arima"

    def search(self, context):
        return search_report(self.family, [], [{
            "family": self.family,
            "reason": "unsupported_model_description",
            "error": f"{self.family} requires a stateful executable description; "
                     "model-language schema version 1 does not support it.",
        }])


class StateSpaceSearcher(ARIMASearcher):
    """Report unsupported structural state-space descriptions."""

    family = "state_space"


def target_lag_subset(context: TimeSeriesSearchContext, n_lags: int) -> np.ndarray:
    """
    Return a feature mask selecting target-history lags.
    """
    n_lags = min(max(1, int(n_lags)), int(context.window_size))
    subset = np.zeros(context.X.shape[1], dtype=bool)
    subset[:n_lags] = True
    return subset


def ensure_non_empty_subset(subset, context: TimeSeriesSearchContext) -> np.ndarray:
    """
    Return a non-empty subset for candidates that require fitted inputs.
    """
    subset = np.asarray(subset, dtype=bool).copy()
    if np.any(subset):
        return subset

    table = context.evaluator.nescience.miscoding_.feature_analysis()
    sort_column = "miscoding" if "miscoding" in table.columns else "deficiency"
    best_feature = int(table.sort_values(sort_column).iloc[0]["feature_index"])
    subset[best_feature] = True
    return subset


def selected_feature_names(
    context: TimeSeriesSearchContext,
    selected: Sequence[int],
) -> tuple[str, ...]:
    """
    Return public names for selected lagged features.
    """
    return tuple(str(context.feature_names[int(index)]) for index in selected)


def candidate_metadata(
    context: TimeSeriesSearchContext,
    *,
    family: str,
    model_name: str,
    selected: Sequence[int],
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    """
    Return metadata shared by time-series candidate results.
    """
    selected = tuple(int(index) for index in selected)
    metadata: dict[str, object] = {
        "schema": "mnplib-model",
        "schema_version": 1,
        "family": str(family),
        "model_name": str(model_name),
        "window_size": int(context.window_size),
        "n_selected_features": int(len(selected)),
        "selected_feature_names": selected_feature_names(context, selected),
    }
    if extra:
        metadata.update(extra)
    return metadata
