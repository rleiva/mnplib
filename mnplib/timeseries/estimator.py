"""
Public time-series estimator based on the Minimum Nescience Principle.

The estimator treats forecasting as model selection over lagged representations.
It builds a supervised lagged matrix, evaluates a compact set of forecasting
families, and selects the candidate with minimum nescience.

The implementation follows the latest explicit-artifact API of mnplib: candidate
models are evaluated through ``subset``, ``predictions``, and ``model_string``.
The public estimator never asks a metric object to inspect a fitted model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from ..inaccuracy import Inaccuracy
from ..miscoding import Miscoding
from ..nescience import Nescience
from ..surfeit import Surfeit
from .lagged import LaggedRepresentationBuilder
from .models import (
    FixedLinearForecaster,
    TimeSeriesCandidateSpec,
    canonical_fixed_model_string,
    canonical_linear_model_string,
    exponential_smoothing_weights,
    moving_average_weights,
)
from .selection import (
    TimeSeriesCandidateEvaluator,
    TimeSeriesCandidateResult,
    candidate_results_dataframe,
)


XType = Literal["auto", "numeric", "categorical"]
YType = Literal["auto", "numeric"]
BinSpec = int | Literal["auto"]
Aggregation = Literal[
    "euclidean",
    "arithmetic",
    "geometric",
    "harmonic",
    "maximum",
    "addition",
    "product",
]
ModelName = Literal["autoregressive", "moving_average", "exponential_smoothing"]


class TimeSeries(BaseEstimator, RegressorMixin):
    """
    Forecast a numeric time series using nescience-based model selection.

    The class converts a sequence into a lagged supervised representation and
    evaluates forecasting candidates on the full representation. No train/test or
    holdout split is introduced.

    Parameters
    ----------
    y_type : {"auto", "numeric"}, default="numeric"
        Target encoding used by inaccuracy, surfeit, and miscoding.

    X_type : {"auto", "numeric", "categorical"}, default="numeric"
        Feature encoding used by miscoding. Lagged time-series features are
        numeric by construction; ``"numeric"`` is the recommended setting.

    window_size : int or "auto", default="auto"
        Number of past observations used to build lagged features. If
        ``"auto"``, the window size is ``floor(sqrt(n_samples))`` with a lower
        bound of one.

    max_lag : int, optional
        Maximum lag used by diagnostic lag-analysis methods. If omitted, the
        resolved window size is used.

    models : sequence of {"autoregressive", "moving_average", "exponential_smoothing"}, optional
        Candidate model families to evaluate. If omitted, all supported families
        are evaluated.

    moving_average_windows : sequence of int, optional
        Windows evaluated for moving-average candidates. If omitted, all
        windows from one to ``window_size`` are evaluated.

    smoothing_alphas : sequence of float, optional
        Alpha values in ``(0, 1)`` evaluated for finite-window exponential
        smoothing.

    aggregation, weights, n_bins, zlib_level, zlib_overhead :
        Parameters forwarded to the current nescience component API.

    min_improvement : float, default=0.0
        Minimum greedy feature-selection improvement used by ``Miscoding`` for
        the autoregressive candidate.

    description_precision : int, default=6
        Number of decimal places used in canonical model descriptions.

    random_state : int or None, default=None
        Stored for estimator reproducibility and future candidate families.
    """

    _VALID_Y_TYPES = ("auto", "numeric")
    _VALID_X_TYPES = ("auto", "numeric", "categorical")
    _VALID_MODELS = ("autoregressive", "moving_average", "exponential_smoothing")

    def __init__(
        self,
        *,
        y_type: YType = "numeric",
        X_type: XType = "numeric",
        window_size: int | Literal["auto"] = "auto",
        max_lag: int | None = None,
        models: Sequence[ModelName] | None = None,
        moving_average_windows: Sequence[int] | None = None,
        smoothing_alphas: Sequence[float] | None = None,
        aggregation: Aggregation = "euclidean",
        weights: Mapping[str, float] | Sequence[float] | None = None,
        n_bins: BinSpec = "auto",
        min_improvement: float = 0.0,
        zlib_level: int = 9,
        zlib_overhead: int = 6,
        description_precision: int = 6,
        random_state: int | None = None,
        verbose: int = 0,
    ):
        self.y_type = y_type
        self.X_type = X_type
        self.window_size = window_size
        self.max_lag = max_lag
        self.models = models
        self.moving_average_windows = moving_average_windows
        self.smoothing_alphas = smoothing_alphas
        self.aggregation = aggregation
        self.weights = weights
        self.n_bins = n_bins
        self.min_improvement = min_improvement
        self.zlib_level = zlib_level
        self.zlib_overhead = zlib_overhead
        self.description_precision = description_precision
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, y, X=None):
        """Fit forecasting candidates and select the one with minimum nescience."""
        self._validate_configuration()

        builder = LaggedRepresentationBuilder(window_size=self.window_size)
        representation, X_exogenous, exogenous_names = builder.build(y, X)

        self.y_ = builder.validate_y(y)
        self.X_exogenous_ = X_exogenous
        self.exogenous_feature_names_ = exogenous_names
        self.window_size_ = builder.resolve_window_size(len(self.y_))
        self.max_lag_ = self.window_size_ if self.max_lag is None else int(self.max_lag)
        self.X_supervised_ = representation.X
        self.y_supervised_ = representation.y
        self.feature_names_in_ = np.asarray(representation.feature_names, dtype=object)
        self.feature_metadata_ = tuple(representation.feature_metadata)

        self.miscoding_ = self._make_miscoding().fit(self.X_supervised_, self.y_supervised_)
        self.inaccuracy_ = self._make_inaccuracy().fit_y(self.y_supervised_)
        self.surfeit_ = self._make_surfeit().fit_y(self.y_supervised_)
        self.nescience_ = self._make_aggregator()

        specs = self._candidate_specs()
        if not specs:
            raise RuntimeError("No candidate time-series models were evaluated.")

        evaluator = TimeSeriesCandidateEvaluator(
            miscoding=self.miscoding_,
            inaccuracy=self.inaccuracy_,
            surfeit=self.surfeit_,
            aggregator=self.nescience_,
            X=self.X_supervised_,
            y=self.y_supervised_,
            feature_names=self.feature_names_in_,
        )
        results = [evaluator.evaluate(spec) for spec in specs]
        results.sort(key=lambda result: result.nescience)

        self.candidate_results_ = results
        self.results_ = candidate_results_dataframe(results)
        self._set_selected_result(results[0])

        if self.verbose:
            for result in results:
                print(
                    f"{result.model_name}: nescience={result.nescience:.6f}, "
                    f"estimator_score={result.estimator_score:.6f}"
                )

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """Predict from an already-built lagged feature matrix."""
        check_is_fitted(self)
        X_checked = check_array(X, dtype=float, ensure_2d=True)
        selected = np.flatnonzero(self.subset_)
        return self.model_.predict(X_checked[:, selected])

    def forecast(self, steps: int = 1, X_future=None) -> np.ndarray:
        """Produce recursive forecasts for a positive number of future steps."""
        check_is_fitted(self)
        steps = int(steps)
        if steps < 1:
            raise ValueError("steps must be positive.")

        y_history = list(np.asarray(self.y_, dtype=float))
        X_history, X_future_array = self._prepare_future_exogenous(steps, X_future)
        selected = np.flatnonzero(self.subset_)
        forecasts: list[float] = []

        for step in range(steps):
            row = LaggedRepresentationBuilder.single_forecast_row(
                y_history=np.asarray(y_history, dtype=float),
                X_history=None if X_history is None else np.asarray(X_history, dtype=float),
                window_size=self.window_size_,
            )
            forecast_value = float(self.model_.predict(row[:, selected])[0])
            forecasts.append(forecast_value)
            y_history.append(forecast_value)

            if X_history is not None:
                X_history.append(np.asarray(X_future_array[step], dtype=float))

        return np.asarray(forecasts, dtype=float)

    def score(self, y, X=None) -> float:
        """Return the native score of the selected forecaster on a series."""
        check_is_fitted(self)
        y_array = LaggedRepresentationBuilder.validate_y(y)
        X_array, exogenous_names = LaggedRepresentationBuilder.validate_exogenous_X(X, n_samples=len(y_array))
        representation = LaggedRepresentationBuilder.to_supervised(
            y=y_array,
            X=X_array,
            window_size=self.window_size_,
            exogenous_feature_names=exogenous_names,
        )
        selected = np.flatnonzero(self.subset_)
        return float(self.model_.score(representation.X[:, selected], representation.y))

    def get_model(self):
        """Return the selected fitted forecasting model."""
        check_is_fitted(self)
        return self.model_

    def nescience_score(self) -> float:
        """Return the selected candidate's nescience value."""
        check_is_fitted(self)
        return float(self.best_result_.nescience)

    def components(self) -> dict[str, float]:
        """Return the four nescience components of the selected candidate."""
        check_is_fitted(self)
        return dict(self.best_result_.components)

    def model_string(self) -> str:
        """Return the selected candidate's canonical model description."""
        check_is_fitted(self)
        return str(self.best_result_.model_string)

    def results_dataframe(self) -> pd.DataFrame:
        """Return a copy of the candidate comparison table."""
        check_is_fitted(self)
        return self.results_.copy()

    def explain(self) -> dict[str, object]:
        """Return a structured explanation of the selected forecasting model."""
        check_is_fitted(self)
        components = self.components()
        dominant = max(components, key=components.get)
        return {
            "nescience": self.nescience_score(),
            "aggregation": self.aggregation,
            "components": components,
            "miscoding": max(components["deficiency"], components["surplus"]),
            "dominant_component": dominant,
            "profile": self._profile_from_components(components),
            "recommendation": self._recommendation_from_dominant_component(dominant, components),
            "time_series_model": self.model_name_,
            "model_family": self.best_result_.model_family,
            "window_size": self.window_size_,
            "selected_lags": self.selected_lags_,
            "selected_feature_indices": self.selected_feature_indices_,
            "selected_feature_names": self.selected_feature_names_,
            "n_selected_features": self.best_result_.n_selected_features,
            "model_metadata": self.best_result_.metadata,
        }

    def auto_lag_analysis(self, *, min_lag: int = 1, max_lag: int | None = None) -> pd.DataFrame:
        """Analyze target autocoding diagnostics by lag."""
        check_is_fitted(self)
        values = np.asarray(self.y_, dtype=float)
        return self._lag_analysis(values=values, target=values, prefix="y", min_lag=min_lag, max_lag=max_lag)

    def cross_lag_analysis(
        self,
        attribute: int | str,
        *,
        min_lag: int = 1,
        max_lag: int | None = None,
    ) -> pd.DataFrame:
        """Analyze lag diagnostics between an exogenous attribute and the target."""
        check_is_fitted(self)
        if self.X_exogenous_ is None:
            raise ValueError("cross_lag_analysis requires exogenous X data.")
        index = self._resolve_attribute(attribute)
        name = self.exogenous_feature_names_[index]
        return self._lag_analysis(
            values=np.asarray(self.X_exogenous_[:, index], dtype=float),
            target=np.asarray(self.y_, dtype=float),
            prefix=name,
            min_lag=min_lag,
            max_lag=max_lag,
            attribute=name,
        )

    def lag_analysis(self, *, max_lag: int | None = None) -> pd.DataFrame:
        """Return target and exogenous lag diagnostics in one table."""
        check_is_fitted(self)
        tables = [self.auto_lag_analysis(max_lag=max_lag)]
        if self.X_exogenous_ is not None:
            tables.extend(
                self.cross_lag_analysis(name, max_lag=max_lag)
                for name in self.exogenous_feature_names_
            )
        return pd.concat(tables, ignore_index=True)

    # ------------------------------------------------------------------
    # Candidate creation
    # ------------------------------------------------------------------

    def _candidate_specs(self) -> list[TimeSeriesCandidateSpec]:
        """Create and fit all configured candidate forecasting models."""
        specs: list[TimeSeriesCandidateSpec] = []
        for model_name in self._resolved_model_names():
            if model_name == "autoregressive":
                specs.append(self._autoregressive_spec())
            elif model_name == "moving_average":
                specs.extend(self._fixed_specs("moving_average"))
            elif model_name == "exponential_smoothing":
                specs.extend(self._fixed_specs("exponential_smoothing"))
        return specs

    def _autoregressive_spec(self) -> TimeSeriesCandidateSpec:
        """Fit the linear autoregressive candidate selected by miscoding."""
        selection = self.miscoding_.select_features(return_details=True)
        subset = np.asarray(selection["selected_features"], dtype=bool)
        subset = self._ensure_non_empty_subset(subset)
        selected = np.flatnonzero(subset)

        model = LinearRegression().fit(self.X_supervised_[:, selected], self.y_supervised_)
        selected_names = tuple(str(self.feature_names_in_[j]) for j in selected)
        model_string = canonical_linear_model_string(
            model=model,
            model_name="autoregressive",
            feature_names=selected_names,
            precision=self.description_precision,
        )

        return TimeSeriesCandidateSpec(
            model_name="autoregressive",
            model_family="autoregressive",
            model=model,
            subset=subset,
            window_size=self.window_size_,
            model_string=model_string,
        )

    def _fixed_specs(self, family: str) -> list[TimeSeriesCandidateSpec]:
        """Fit fixed-coefficient moving-average or smoothing candidates."""
        specs: list[TimeSeriesCandidateSpec] = []
        for window in self._moving_average_windows():
            subset = self._target_lag_subset(window)
            selected = np.flatnonzero(subset)
            selected_names = tuple(str(self.feature_names_in_[j]) for j in selected)

            for model_name, weights in self._fixed_weight_configs(family, window):
                model = FixedLinearForecaster(weights=weights, name=model_name)
                model.fit(self.X_supervised_[:, selected], self.y_supervised_)
                model_type = "moving_average" if family == "moving_average" else "exponential_smoothing"
                model_string = canonical_fixed_model_string(
                    model_type=model_type,
                    model_name=model_name,
                    feature_names=selected_names,
                    weights=weights,
                    precision=self.description_precision,
                )
                specs.append(
                    TimeSeriesCandidateSpec(
                        model_name=model_name,
                        model_family=family,
                        model=model,
                        subset=subset,
                        window_size=window,
                        model_string=model_string,
                    )
                )
        return specs

    def _fixed_weight_configs(self, family: str, window: int):
        """Return model names and weights for fixed-coefficient families."""
        if family == "moving_average":
            return [(f"moving_average_{window}", moving_average_weights(window))]
        if family == "exponential_smoothing":
            return [
                (
                    f"exponential_smoothing_w{window}_a{alpha:g}",
                    exponential_smoothing_weights(window, alpha),
                )
                for alpha in self._smoothing_alphas()
            ]
        raise ValueError(f"Unknown fixed model family {family!r}.")

    def _ensure_non_empty_subset(self, subset: np.ndarray) -> np.ndarray:
        """Fallback to the best single feature if greedy selection selects none."""
        subset = np.asarray(subset, dtype=bool).copy()
        if np.any(subset):
            return subset

        table = self.miscoding_.feature_analysis()
        sort_column = "miscoding" if "miscoding" in table.columns else "deficiency"
        best_feature = int(table.sort_values(sort_column).iloc[0]["feature_index"])
        subset[best_feature] = True
        return subset

    def _target_lag_subset(self, window: int) -> np.ndarray:
        """Return a feature mask selecting the first target lags."""
        window = int(window)
        subset = np.zeros(self.X_supervised_.shape[1], dtype=bool)
        subset[:window] = True
        return subset

    # ------------------------------------------------------------------
    # Lag diagnostics
    # ------------------------------------------------------------------

    def _lag_analysis(
        self,
        *,
        values,
        target,
        prefix: str,
        min_lag: int,
        max_lag: int | None,
        attribute: str | None = None,
    ) -> pd.DataFrame:
        if int(min_lag) < 1:
            raise ValueError("min_lag must be positive.")
        effective_max_lag = self.max_lag_ if max_lag is None else int(max_lag)
        effective_max_lag = min(effective_max_lag, len(target) - 1)

        rows = []
        for lag in range(int(min_lag), effective_max_lag + 1):
            diagnostic = self._single_lag_miscoding(values=values, target=target, lag=lag)
            deficiency = float(diagnostic["deficiency"])
            surplus = float(diagnostic["surplus"])
            row = {
                "lag": int(lag),
                "feature_name": f"{prefix}_lag_{lag}",
                "deficiency": deficiency,
                "surplus": surplus,
                "miscoding": float(diagnostic.get("miscoding", max(deficiency, surplus))),
            }
            if attribute is not None:
                row["attribute"] = attribute
            rows.append(row)

        return pd.DataFrame(rows)

    def _single_lag_miscoding(self, *, values, target, lag: int) -> pd.Series:
        metric = self._make_miscoding()
        metric.fit(
            np.asarray(values[:-lag], dtype=float).reshape(-1, 1),
            np.asarray(target[lag:], dtype=float),
        )
        return metric.feature_analysis().iloc[0]

    # ------------------------------------------------------------------
    # Metric factories
    # ------------------------------------------------------------------

    def _make_aggregator(self) -> Nescience:
        """Return an unfitted Nescience instance used only for aggregation."""
        return Nescience(
            X_type=self.X_type,
            y_type=self.y_type,
            aggregation=self.aggregation,
            weights=self.weights,
            n_bins=self.n_bins,
            zlib_level=self.zlib_level,
            zlib_overhead=self.zlib_overhead,
        )

    def _make_miscoding(self) -> Miscoding:
        """Return a Miscoding instance configured with the latest API."""
        return Miscoding(
            X_type=self.X_type,
            y_type=self.y_type,
            n_bins=self.n_bins,
            min_improvement=self.min_improvement,
        )

    def _make_inaccuracy(self) -> Inaccuracy:
        """Return an Inaccuracy instance configured for the target series."""
        return Inaccuracy(y_type=self.y_type, n_bins=self.n_bins)

    def _make_surfeit(self) -> Surfeit:
        """Return a Surfeit instance configured for canonical model strings."""
        return Surfeit(
            y_type=self.y_type,
            n_bins=self.n_bins,
            zlib_level=self.zlib_level,
            zlib_overhead=self.zlib_overhead,
        )

    # ------------------------------------------------------------------
    # Configuration and helpers
    # ------------------------------------------------------------------

    def _resolved_model_names(self) -> tuple[str, ...]:
        if self.models is None:
            return self._VALID_MODELS
        return tuple(str(name) for name in self.models)

    def _moving_average_windows(self) -> list[int]:
        if self.moving_average_windows is None:
            values = range(1, self.window_size_ + 1)
        else:
            values = (int(value) for value in self.moving_average_windows)
        windows = sorted({value for value in values if 1 <= value <= self.window_size_})
        if not windows:
            raise ValueError("No valid moving-average windows to evaluate.")
        return windows

    def _smoothing_alphas(self) -> list[float]:
        if self.smoothing_alphas is None:
            alphas = [0.1, 0.2, 0.3, 0.5, 0.8]
        else:
            alphas = [float(value) for value in self.smoothing_alphas]
        if any(alpha <= 0.0 or alpha >= 1.0 for alpha in alphas):
            raise ValueError("All smoothing alphas must lie in the open interval (0, 1).")
        return alphas

    def _prepare_future_exogenous(self, steps: int, X_future):
        if self.X_exogenous_ is None:
            return None, None

        history = [row.copy() for row in np.asarray(self.X_exogenous_, dtype=float)]
        n_exogenous = len(self.exogenous_feature_names_)

        if X_future is None:
            future = np.repeat(np.asarray(history[-1]).reshape(1, -1), steps, axis=0)
        else:
            future = check_array(X_future, dtype=float, ensure_2d=True)
            if future.shape != (steps, n_exogenous):
                raise ValueError(
                    f"X_future must have shape ({steps}, {n_exogenous}). Got {future.shape}."
                )
        return history, future

    def _resolve_attribute(self, attribute) -> int:
        if isinstance(attribute, str):
            if attribute not in self.exogenous_feature_names_:
                raise ValueError(f"Unknown exogenous attribute {attribute!r}.")
            return self.exogenous_feature_names_.index(attribute)
        index = int(attribute)
        if index < 0 or index >= len(self.exogenous_feature_names_):
            raise ValueError(f"attribute index {index} is outside the valid range.")
        return index

    def _selected_lags_from_indices(self, indices) -> list[dict[str, object]]:
        selected_lags = []
        for index in indices:
            metadata = dict(self.feature_metadata_[int(index)])
            metadata["feature_index"] = int(index)
            metadata["feature_name"] = str(self.feature_names_in_[int(index)])
            selected_lags.append(metadata)
        return selected_lags

    def _set_selected_result(self, result: TimeSeriesCandidateResult) -> None:
        self.best_result_ = result
        self.model_ = result.model
        self.subset_ = np.asarray(result.subset, dtype=bool)
        self.model_name_ = result.model_name
        self.best_nescience_ = float(result.nescience)
        self.best_components_ = dict(result.components)
        self.best_model_string_ = str(result.model_string)
        self.selected_feature_indices_ = list(result.selected_feature_indices)
        self.selected_feature_names_ = list(result.selected_feature_names)
        self.selected_lags_ = self._selected_lags_from_indices(result.selected_feature_indices)

    def _validate_configuration(self) -> None:
        if self.y_type not in self._VALID_Y_TYPES:
            raise ValueError(f"Valid options for y_type are {self._VALID_Y_TYPES}. Got {self.y_type!r}.")
        if self.X_type not in self._VALID_X_TYPES:
            raise ValueError(f"Valid options for X_type are {self._VALID_X_TYPES}. Got {self.X_type!r}.")
        if self.window_size != "auto" and int(self.window_size) < 1:
            raise ValueError("window_size must be a positive integer or 'auto'.")
        if self.max_lag is not None and int(self.max_lag) < 1:
            raise ValueError("max_lag must be positive when provided.")
        if self.models is not None:
            unknown = set(map(str, self.models)) - set(self._VALID_MODELS)
            if unknown:
                raise ValueError(f"Unknown model names {sorted(unknown)}.")
        if self.min_improvement < 0:
            raise ValueError("min_improvement must be non-negative.")
        if int(self.zlib_level) < 0 or int(self.zlib_level) > 9:
            raise ValueError("zlib_level must be an integer between 0 and 9.")
        if int(self.zlib_overhead) < 0:
            raise ValueError("zlib_overhead must be non-negative.")
        if int(self.description_precision) < 0:
            raise ValueError("description_precision must be non-negative.")

    @staticmethod
    def _profile_from_components(components: Mapping[str, float]) -> str:
        """Return a compact qualitative profile for the selected candidate."""
        dominant = max(components, key=components.get)
        return {
            "deficiency": "under_informed_forecaster",
            "surplus": "over_fed_forecaster",
            "inaccuracy": "inaccurate_forecaster",
            "surfeit": "over_complex_forecaster",
        }.get(dominant, "mixed_nescience_profile")

    @staticmethod
    def _recommendation_from_dominant_component(
        dominant_component: str,
        components: Mapping[str, float],
    ) -> str:
        """Return a practical recommendation from the dominant component."""
        value = float(components[dominant_component])
        if dominant_component == "deficiency":
            return (
                f"Dominant source of nescience: deficiency ({value:.4f}). "
                "Increase the information available to the forecaster by adding relevant "
                "lags, exogenous variables, or more informative transformations."
            )
        if dominant_component == "surplus":
            return (
                f"Dominant source of nescience: surplus ({value:.4f}). "
                "Reduce target-irrelevant lagged information by limiting the window, "
                "removing weak exogenous variables, or increasing feature-selection strictness."
            )
        if dominant_component == "inaccuracy":
            return (
                f"Dominant source of nescience: inaccuracy ({value:.4f}). "
                "Try a richer forecasting family, a different lag window, or additional "
                "predictive variables."
            )
        if dominant_component == "surfeit":
            return (
                f"Dominant source of nescience: surfeit ({value:.4f}). "
                "Prefer a simpler forecasting rule, fewer lags, or a more compact model family."
            )
        return "Inspect the four nescience components to identify the limiting factor."
