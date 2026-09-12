"""
Model-family searchers for nescience-based time-series forecasting.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
import warnings

import numpy as np

from sklearn.linear_model import LinearRegression

from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.structural import UnobservedComponents

from mnplib.automl.evaluator import CandidateEvaluator
from mnplib.automl.searchers.base import ModelFamilySearcher, search_report
from mnplib.models.serializers.time_series import (
    TIME_SERIES_SCHEMA,
    canonical_arima_model_string,
    canonical_fixed_model_string,
    canonical_linear_model_string,
    canonical_state_space_model_string,
    time_series_model_artifacts,
)

from .models import (
    FixedLinearForecaster,
    StatsmodelsForecastModel,
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
    original_y: np.ndarray
    feature_names: tuple[str, ...]
    evaluator: CandidateEvaluator
    window_size: int
    description_precision: int
    moving_average_windows: tuple[int, ...]
    smoothing_alphas: tuple[float, ...]
    arima_orders: tuple[tuple[int, int, int], ...]
    state_space_models: tuple[str, ...]
    statsmodels_maxiter: int
    verbose: int = 0


class AutoregressiveSearcher(ModelFamilySearcher):
    """
    Evaluate a linear autoregressive candidate selected by miscoding.
    """

    family = "autoregressive"

    def search(self, context: TimeSeriesSearchContext):
        selection = context.evaluator.nescience.miscoding_.select_features(
            return_details=True
        )
        subset = ensure_non_empty_subset(selection["selected_features"], context)
        selected = np.flatnonzero(subset)

        model = LinearRegression().fit(context.X[:, selected], context.y)
        predictions = model.predict(context.X[:, selected])
        model_name = "autoregressive"
        model_string = canonical_linear_model_string(
            model=model,
            model_name=model_name,
            feature_names=selected_feature_names(context, selected),
            precision=context.description_precision,
        )

        artifacts = time_series_model_artifacts(
            model=model,
            subset=selected,
            predictions=predictions,
            model_string=model_string,
            model_type=type(model).__name__,
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
            predictions = model.predict(context.X[:, selected])
            model_string = canonical_fixed_model_string(
                model_type=self.family,
                model_name=model_name,
                feature_names=selected_feature_names(context, selected),
                weights=weights,
                precision=context.description_precision,
            )
            artifacts = time_series_model_artifacts(
                model=model,
                subset=selected,
                predictions=predictions,
                model_string=model_string,
                model_type=type(model).__name__,
            )
            results.append(
                context.evaluator.evaluate_artifacts(
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
                        extra={"window": int(window)},
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
        for window in context.moving_average_windows:
            subset = target_lag_subset(context, window)
            selected = np.flatnonzero(subset)
            for alpha in context.smoothing_alphas:
                model_name = f"exponential_smoothing_w{window}_a{alpha:g}"
                weights = exponential_smoothing_weights(window, alpha)
                model = FixedLinearForecaster(weights=weights, name=model_name)
                model.fit(context.X[:, selected], context.y)
                predictions = model.predict(context.X[:, selected])
                model_string = canonical_fixed_model_string(
                    model_type=self.family,
                    model_name=model_name,
                    feature_names=selected_feature_names(context, selected),
                    weights=weights,
                    precision=context.description_precision,
                )
                artifacts = time_series_model_artifacts(
                    model=model,
                    subset=selected,
                    predictions=predictions,
                    model_string=model_string,
                    model_type=type(model).__name__,
                )
                results.append(
                    context.evaluator.evaluate_artifacts(
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
                            extra={"window": int(window), "alpha": float(alpha)},
                        ),
                        result_factory=TimeSeriesCandidateResult,
                    )
                )
        return search_report(self.family, results)


class ARIMASearcher(ModelFamilySearcher):
    """
    Evaluate statsmodels ARIMA candidates through the SARIMAX state-space backend.
    """

    family = "arima"

    def search(self, context: TimeSeriesSearchContext):
        results = []
        diagnostics = []
        for order in context.arima_orders:
            order = tuple(int(value) for value in order)
            model_name = "arima_{}_{}_{}".format(*order)
            trend = "c" if order[1] == 0 else "n"
            try:
                result = fit_arima_result(
                    context.original_y,
                    order=order,
                    trend=trend,
                    maxiter=context.statsmodels_maxiter,
                )
                predictions = prediction_path(
                    result,
                    start=context.window_size,
                    expected_length=context.y.shape[0],
                )
                subset = target_lag_subset(context, max(1, max(order)))
                selected = np.flatnonzero(subset)
                model_string = canonical_arima_model_string(
                    result=result,
                    model_name=model_name,
                    order=order,
                    trend=trend,
                    precision=context.description_precision,
                )
                model = StatsmodelsForecastModel(
                    result,
                    name=model_name,
                    family=self.family,
                    training_predictions=predictions,
                )
                artifacts = time_series_model_artifacts(
                    model=model,
                    subset=selected,
                    predictions=predictions,
                    model_string=model_string,
                    model_type=type(result).__name__,
                )
                results.append(
                    context.evaluator.evaluate_artifacts(
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
                            extra={"order": order, "trend": trend},
                        ),
                        result_factory=TimeSeriesCandidateResult,
                    )
                )
            except Exception as exc:
                diagnostics.append(
                    {
                        "family": self.family,
                        "candidate": model_name,
                        "reason": "fit_failed",
                        "error": str(exc),
                    }
                )
        return search_report(self.family, results, diagnostics)


class StateSpaceSearcher(ModelFamilySearcher):
    """
    Evaluate structural state-space candidates using statsmodels.
    """

    family = "state_space"

    _SPECIFICATIONS = {
        "local_level": "local level",
        "local_linear_trend": "local linear trend",
    }

    def search(self, context: TimeSeriesSearchContext):
        results = []
        diagnostics = []
        for specification_name in context.state_space_models:
            model_name = f"state_space_{specification_name}"
            specification = self._SPECIFICATIONS.get(str(specification_name))
            if specification is None:
                diagnostics.append(
                    {
                        "family": self.family,
                        "candidate": model_name,
                        "reason": "invalid_specification",
                        "specification": str(specification_name),
                    }
                )
                continue

            try:
                result = fit_state_space_result(
                    context.original_y,
                    specification=specification,
                    maxiter=context.statsmodels_maxiter,
                )
                predictions = prediction_path(
                    result,
                    start=context.window_size,
                    expected_length=context.y.shape[0],
                )
                n_state_lags = max(1, int(getattr(result.model, "k_states", 1)))
                subset = target_lag_subset(context, n_state_lags)
                selected = np.flatnonzero(subset)
                model_string = canonical_state_space_model_string(
                    result=result,
                    model_name=model_name,
                    specification=specification_name,
                    precision=context.description_precision,
                )
                model = StatsmodelsForecastModel(
                    result,
                    name=model_name,
                    family=self.family,
                    training_predictions=predictions,
                )
                artifacts = time_series_model_artifacts(
                    model=model,
                    subset=selected,
                    predictions=predictions,
                    model_string=model_string,
                    model_type=type(result).__name__,
                )
                results.append(
                    context.evaluator.evaluate_artifacts(
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
                            extra={"specification": specification_name},
                        ),
                        result_factory=TimeSeriesCandidateResult,
                    )
                )
            except Exception as exc:
                diagnostics.append(
                    {
                        "family": self.family,
                        "candidate": model_name,
                        "reason": "fit_failed",
                        "error": str(exc),
                    }
                )
        return search_report(self.family, results, diagnostics)


def fit_arima_result(y, *, order: tuple[int, int, int], trend: str, maxiter: int):
    """
    Fit a SARIMAX-backed ARIMA model and return the statsmodels result.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = SARIMAX(
            np.asarray(y, dtype=float),
            order=order,
            trend=trend,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        return model.fit(disp=False, maxiter=int(maxiter))


def fit_state_space_result(y, *, specification: str, maxiter: int):
    """
    Fit a structural state-space model and return the statsmodels result.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = UnobservedComponents(
            np.asarray(y, dtype=float),
            level=str(specification),
        )
        return model.fit(disp=False, maxiter=int(maxiter))


def prediction_path(result, *, start: int, expected_length: int) -> np.ndarray:
    """
    Return one-step predictions aligned with the supervised target.
    """
    predicted = result.get_prediction(
        start=int(start),
        end=int(start) + int(expected_length) - 1,
        dynamic=False,
    ).predicted_mean
    values = np.asarray(predicted, dtype=float).ravel()
    if values.shape[0] != int(expected_length):
        raise ValueError(
            f"Expected {expected_length} predictions, got {values.shape[0]}."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("Predictions must be finite.")
    return values


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
        "schema": TIME_SERIES_SCHEMA,
        "model_family": str(family),
        "model_name": str(model_name),
        "window_size": int(context.window_size),
        "n_selected_features": int(len(selected)),
        "selected_feature_names": selected_feature_names(context, selected),
    }
    if extra:
        metadata.update(extra)
    return metadata
