"""
Public time-series estimator based on the Minimum Nescience Principle.

The estimator treats forecasting as model selection over lagged representations.
It builds a supervised lagged matrix, evaluates a compact set of forecasting
families, and selects the candidate with minimum nescience.

The implementation follows the API of mnplib: candidate models are evaluated through
``subset``, ``predictions``, and ``model_string``. The public estimator never asks
a metric object to inspect a fitted model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from mnplib.automl import CandidateEvaluator
from mnplib.automl.descriptions import describe_candidate_model
from mnplib.automl.configuration import validated_search_options
from mnplib.automl.results import candidate_results_dataframe

from ..inaccuracy import Inaccuracy
from ..miscoding import Miscoding
from ..nescience import Nescience
from ..surfeit import Surfeit
from .lagged import LaggedRepresentationBuilder
from .searchers import (
    ARIMASearcher,
    AutoregressiveSearcher,
    ExponentialSmoothingSearcher,
    MovingAverageSearcher,
    StateSpaceSearcher,
    TimeSeriesSearchContext,
)
from .selection import (
    TimeSeriesCandidateResult,
)


XType = Literal["auto", "numeric", "categorical"]
BinSpec = int | Literal["auto", "adaptive"]
Aggregation = Literal[
    "euclidean",
    "arithmetic",
    "geometric",
    "harmonic",
    "maximum",
    "addition",
    "product",
]
ModelName = Literal[
    "autoregressive",
    "moving_average",
    "exponential_smoothing",
    "arima",
    "state_space",
]


class TimeSeries(BaseEstimator):
    """
    Forecast a numeric time series using nescience-based model selection.

    The class converts a sequence into a lagged supervised representation and
    evaluates forecasting candidates on the full representation. No train/test or
    holdout split is introduced.

    Parameters
    ----------
    X_type : {"auto", "numeric", "categorical"}, default="numeric"
        Feature encoding used by miscoding. Lagged time-series features are
        numeric by construction; ``"numeric"`` is the recommended setting.

    window_size : int or "auto", default="auto"
        Number of past observations used to build lagged features. If
        ``"auto"``, the window size is ``floor(sqrt(n_samples))`` with a lower
        bound of one.

    models : sequence of {"autoregressive", "moving_average", "exponential_smoothing", "arima", "state_space"}, optional
        Candidate model families to evaluate. If omitted, all supported families
        are evaluated.

    search_options : mapping, optional
        Per-family search settings. Autoregressive accepts ``min_improvement``;
        moving_average accepts ``windows``; exponential_smoothing accepts
        ``windows`` and ``alphas``; arima accepts ``orders`` and ``max_iter``;
        state_space accepts ``models`` and ``max_iter``. Structural model names
        are ``"local_level"`` and ``"local_linear_trend"``.

    aggregation, weights, n_bins, zlib_level, zlib_overhead :
        Parameters forwarded to the current nescience component API.

    random_state : int or None, default=None
        Stored for estimator reproducibility and future candidate families.

    Attributes
    ----------
    fitted_values_ : ndarray of shape (n_samples,)
        Training predictions aligned with the supplied series. Initial positions
        lacking a complete lag window contain NaN. Forecasts are produced by
        ``forecast()``; ``score()`` evaluates subsequent observed values.
    """

    _VALID_X_TYPES = ("auto", "numeric", "categorical")
    _VALID_MODELS = (
        "autoregressive",
        "moving_average",
        "exponential_smoothing",
        "arima",
        "state_space",
    )
    _VALID_STATE_SPACE_MODELS = ("local_level", "local_linear_trend")
    _DEFAULT_ARIMA_ORDERS = ((1, 0, 0), (2, 0, 0), (1, 1, 0), (0, 1, 1))

    def __init__(
        self,
        *,
        X_type: XType = "numeric",
        window_size: int | Literal["auto"] = "auto",
        models: Sequence[ModelName] | None = None,
        search_options: Mapping[str, Mapping[str, object]] | None = None,
        aggregation: Aggregation = "euclidean",
        weights: Mapping[str, float] | Sequence[float] | None = None,
        n_bins: BinSpec = "adaptive",
        zlib_level: int = 9,
        zlib_overhead: int = 6,
        random_state: int | None = None,
        verbose: int = 0,
    ):
        self.X_type = X_type
        self.window_size = window_size
        self.models = models
        self.search_options = search_options
        self.aggregation = aggregation
        self.weights = weights
        self.n_bins = n_bins
        self.zlib_level = zlib_level
        self.zlib_overhead = zlib_overhead
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
        self.X_supervised_ = representation.X
        self.y_supervised_ = representation.y
        self.feature_names_in_ = np.asarray(representation.feature_names, dtype=object)
        self.feature_metadata_ = tuple(representation.feature_metadata)

        self.miscoding_ = self._make_miscoding().fit(self.X_supervised_, self.y_supervised_)
        self.inaccuracy_ = self._make_inaccuracy().fit_y(self.y_supervised_)
        self.surfeit_ = self._make_surfeit().fit(self.X_supervised_, self.y_supervised_)
        self.nescience_ = self._make_fitted_aggregator()

        self.evaluator_ = CandidateEvaluator(
            X=self.X_supervised_,
            y=self.y_supervised_,
            nescience=self.nescience_,
            feature_names=list(self.feature_names_in_),
        )
        self.results_ = []
        self.diagnostics_ = []
        self.searchers_ = self._resolve_searchers()
        self._fit_searchers()

        results = list(self.results_)
        valid_results = [
            result
            for result in results
            if result.is_reliable and np.isfinite(result.nescience)
        ]
        if not valid_results:
            raise ValueError(
                "No reliable time-series candidate subset could be evaluated "
                "with the available sample size and discretization."
            )

        results.sort(key=self._candidate_sort_key)

        self.candidate_results_ = results
        self.results_ = candidate_results_dataframe(results, self.feature_names_in_)
        self._set_selected_result(results[0])

        if self.verbose:
            for result in results:
                print(
                    f"{result.name}: nescience={result.nescience:.6f}, "
                    f"estimator_score={result.estimator_score:.6f}"
                )

        self.is_fitted_ = True
        return self


    def forecast(self, steps: int = 1, *, X_future=None) -> np.ndarray:
        """Produce recursive forecasts for a positive number of future steps."""
        check_is_fitted(self)
        if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, (int, np.integer)) or steps < 1:
            raise ValueError("steps must be a positive integer.")

        if self.best_result_.family in {"arima", "state_space"}:
            return self.model_.forecast(steps=steps, X_future=X_future)

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

    def score(self, y_future, *, X_future=None) -> float:
        """Return forecast R-squared for observations immediately after training."""
        check_is_fitted(self)
        values = np.asarray(y_future, dtype=float)
        if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
            raise ValueError("y_future must contain at least two finite observations.")
        return float(r2_score(values, self.forecast(len(values), X_future=X_future)))


    def nescience(self) -> float:
        """Return the selected candidate's nescience value."""
        check_is_fitted(self)
        return float(self.best_result_.nescience)

    def components(self) -> dict[str, float]:
        """Return the four nescience components of the selected candidate."""
        check_is_fitted(self)
        return dict(self.best_result_.components)


    def model_description(
        self,
        candidate: str | None = None,
    ) -> dict[str, object]:
        """Return model-description diagnostics for an evaluated candidate."""
        check_is_fitted(self)
        return describe_candidate_model(
            self.candidate_results_,
            candidate,
            best_result=self.best_result_,
            surfeit=self.surfeit_,
        )

    def results_dataframe(self) -> pd.DataFrame:
        """Return a copy of the candidate comparison table."""
        check_is_fitted(self)
        return self.results_.copy()

    def explain(self) -> dict[str, object]:
        """Return a structured explanation of the selected forecasting model.

        The native estimator score is R-squared recorded on the lagged training
        representation. It is not future-forecast performance; use ``score()``
        with subsequent observations to evaluate forecasts.
        """
        check_is_fitted(self)
        explanation = self.nescience_.explain(**self.best_artifacts_.to_nescience_kwargs())
        explanation.update({
            "candidate": self.model_name_,
            "family": self.best_result_.family,
            "model_type": self.best_artifacts_.model_type,
            "hyperparameters": dict(self.best_result_.hyperparameters),
            "task": "forecasting",
            "native_estimator_score": float(self.best_result_.estimator_score),
            "evaluation_context": "lagged_training",
            "window_size": self.window_size_,
            "selected_lags": self.selected_lags_,
            "selected_features": self.selected_feature_indices_,
            "selected_feature_names": self.selected_feature_names_,
            "n_selected_features": self.best_result_.n_selected_features,
        })
        explanation["metadata"] = {key: value for key, value in self.best_result_.metadata.items()
                                   if key not in explanation and key != "model_name"}
        return explanation

    def _auto_lag_analysis(self, *, min_lag: int = 1, max_lag: int | None = None) -> pd.DataFrame:
        """Analyze target autocoding diagnostics by lag."""
        check_is_fitted(self)
        values = np.asarray(self.y_, dtype=float)
        return self._lag_analysis(values=values, target=values, prefix="y", min_lag=min_lag, max_lag=max_lag)

    def _cross_lag_analysis(
        self,
        attribute: int | str,
        *,
        min_lag: int = 1,
        max_lag: int | None = None,
    ) -> pd.DataFrame:
        """Analyze lag diagnostics between an exogenous attribute and the target."""
        check_is_fitted(self)
        if self.X_exogenous_ is None:
            raise ValueError("Exogenous lag analysis requires X data.")
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
        tables = [self._auto_lag_analysis(max_lag=max_lag)]
        if self.X_exogenous_ is not None:
            tables.extend(
                self._cross_lag_analysis(name, max_lag=max_lag)
                for name in self.exogenous_feature_names_
            )
        return pd.concat(tables, ignore_index=True)

    #
    # Searcher orchestration
    #

    def _resolve_searchers(self):
        """Build searchers in the configured model-family order."""
        return [self._make_searcher(name) for name in self._resolved_model_names()]

    def _make_searcher(self, name: str):
        """Instantiate one time-series model-family searcher."""
        if name == "autoregressive":
            return AutoregressiveSearcher()
        if name == "moving_average":
            return MovingAverageSearcher()
        if name == "exponential_smoothing":
            return ExponentialSmoothingSearcher()
        if name == "arima":
            return ARIMASearcher()
        if name == "state_space":
            return StateSpaceSearcher()
        raise RuntimeError(f"Validated unsupported model family {name!r}.")

    def _fit_searchers(self) -> None:
        """Execute configured searchers and collect shared candidate results."""
        model_names = set(self._resolved_model_names())
        context = TimeSeriesSearchContext(
            X=self.X_supervised_,
            y=self.y_supervised_,
            original_y=self.y_,
            feature_names=tuple(str(name) for name in self.feature_names_in_),
            evaluator=self.evaluator_,
            window_size=self.window_size_,
            description_precision=6,
            moving_average_windows=(
                tuple(self._moving_average_windows())
                if model_names & {"moving_average", "exponential_smoothing"}
                else tuple()
            ),
            smoothing_alphas=(
                tuple(self._smoothing_alphas())
                if "exponential_smoothing" in model_names
                else tuple()
            ),
            arima_orders=(
                tuple(self._arima_orders())
                if "arima" in model_names
                else tuple()
            ),
            state_space_models=(
                tuple(self._state_space_models())
                if "state_space" in model_names
                else tuple()
            ),
            arima_max_iter=int(self._search_options_["arima"].get("max_iter", 50)),
            state_space_max_iter=int(self._search_options_["state_space"].get("max_iter", 50)),
            min_improvement=float(self._search_options_["autoregressive"].get("min_improvement", 0.0)),
            smoothing_windows=tuple(self._moving_average_windows("exponential_smoothing")),
            verbose=self.verbose,
        )

        for searcher in self.searchers_:
            report = searcher.search(context)
            self.results_.extend(report.results)
            self.diagnostics_.extend(report.diagnostics)

    #
    # Lag diagnostics
    #

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
        if max_lag is not None and (isinstance(max_lag, bool) or not isinstance(max_lag, (int, np.integer)) or max_lag < 1):
            raise ValueError("max_lag must be a positive integer.")
        effective_max_lag = self.window_size_ if max_lag is None else int(max_lag)
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

    #
    # Metric factories
    #

    def _make_aggregator(self) -> Nescience:
        """Return an unfitted Nescience instance used only for aggregation."""
        return Nescience(
            X_type=self.X_type,
            y_type="numeric",
            aggregation=self.aggregation,
            weights=self.weights,
            n_bins=self.n_bins,
            zlib_level=self.zlib_level,
            zlib_overhead=self.zlib_overhead,
        )

    def _make_fitted_aggregator(self) -> Nescience:
        """Return an aggregation object with fitted component metrics attached."""
        metric = self._make_aggregator()
        metric.X_ = self.X_supervised_
        metric._model_X_ = self.X_supervised_
        metric.feature_names_in_ = self.feature_names_in_.copy()
        metric.y_ = self.y_supervised_
        metric.n_samples_in_, metric.n_features_in_ = self.X_supervised_.shape
        metric.weights_ = metric._resolve_weights()
        metric.miscoding_ = self.miscoding_
        metric.inaccuracy_ = self.inaccuracy_
        metric.surfeit_ = self.surfeit_
        metric.is_fitted_ = True
        return metric

    def _make_miscoding(self) -> Miscoding:
        """Return the miscoding metric for the lagged representation."""
        return Miscoding(
            X_type=self.X_type,
            y_type="numeric",
            n_bins=self.n_bins,
        )

    def _make_inaccuracy(self) -> Inaccuracy:
        """Return an Inaccuracy instance configured for the target series."""
        return Inaccuracy(y_type="numeric", n_bins=self.n_bins)

    def _make_surfeit(self) -> Surfeit:
        """Return a Surfeit instance configured for canonical model strings."""
        return Surfeit(
            y_type="numeric",
            n_bins=self.n_bins,
            zlib_level=self.zlib_level,
            zlib_overhead=self.zlib_overhead,
        )

    #
    # Configuration and helpers
    #

    def _resolved_model_names(self) -> tuple[str, ...]:
        if self.models is None:
            return self._VALID_MODELS
        return tuple(str(name) for name in self.models)

    def _moving_average_windows(self, family="moving_average") -> list[int]:
        configured = self._search_options_[family].get("windows")
        if configured is None:
            values = range(1, self.window_size_ + 1)
        else:
            values = (int(value) for value in configured)
        windows = sorted({value for value in values if 1 <= value <= self.window_size_})
        if not windows:
            raise ValueError("No valid moving-average windows to evaluate.")
        return windows

    def _smoothing_alphas(self) -> list[float]:
        configured = self._search_options_["exponential_smoothing"].get("alphas")
        if configured is None:
            alphas = [0.1, 0.2, 0.3, 0.5, 0.8]
        else:
            alphas = [float(value) for value in configured]
        if any(alpha <= 0.0 or alpha >= 1.0 for alpha in alphas):
            raise ValueError("All smoothing alphas must lie in the open interval (0, 1).")
        return alphas

    def _arima_orders(self) -> list[tuple[int, int, int]]:
        configured = self._search_options_["arima"].get("orders")
        if configured is None:
            orders = list(self._DEFAULT_ARIMA_ORDERS)
        else:
            orders = [tuple(int(value) for value in order) for order in configured]
        if not orders:
            raise ValueError("At least one ARIMA order must be configured.")
        return orders

    def _state_space_models(self) -> list[str]:
        configured = self._search_options_["state_space"].get("models")
        if configured is None:
            return list(self._VALID_STATE_SPACE_MODELS)
        models = [str(name) for name in configured]
        if not models:
            raise ValueError("At least one state-space model must be configured.")
        return models

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
        self.best_artifacts_ = result.artifacts
        self.fitted_values_ = np.full(len(self.y_), np.nan)
        self.fitted_values_[self.window_size_:] = result.artifacts.predictions
        self.model_ = result.model
        self.subset_ = self._subset_mask(result.artifacts.subset)
        self.model_name_ = result.name
        self.best_nescience_ = float(result.nescience)
        self.best_components_ = dict(result.components)
        self.best_model_string_ = str(result.artifacts.model_string)
        self.selected_feature_indices_ = [int(index) for index in result.artifacts.subset]
        self.selected_feature_names_ = [
            str(self.feature_names_in_[index])
            for index in self.selected_feature_indices_
        ]
        self.selected_lags_ = self._selected_lags_from_indices(
            self.selected_feature_indices_
        )

    @staticmethod
    def _candidate_sort_key(result: TimeSeriesCandidateResult) -> tuple[object, ...]:
        if result.is_reliable and np.isfinite(result.nescience):
            return (0, float(result.nescience), result.n_selected_features, result.name)
        return (1, float("inf"), result.n_selected_features, result.name)

    def _subset_mask(self, selected_indices) -> np.ndarray:
        mask = np.zeros(self.X_supervised_.shape[1], dtype=bool)
        mask[[int(index) for index in selected_indices]] = True
        return mask

    def _validate_configuration(self) -> None:
        self._search_options_ = validated_search_options(self.search_options, {
            "autoregressive": {"min_improvement"},
            "moving_average": {"windows"},
            "exponential_smoothing": {"windows", "alphas"},
            "arima": {"orders", "max_iter"},
            "state_space": {"models", "max_iter"},
        })
        if self.X_type not in self._VALID_X_TYPES:
            raise ValueError(f"Valid options for X_type are {self._VALID_X_TYPES}. Got {self.X_type!r}.")
        if self.window_size != "auto" and int(self.window_size) < 1:
            raise ValueError("window_size must be a positive integer or 'auto'.")
        if self.models is not None:
            if isinstance(self.models, str) or len(self.models) == 0:
                raise ValueError("models must be a non-empty sequence of model-family names.")
            unknown = set(map(str, self.models)) - set(self._VALID_MODELS)
            if unknown:
                raise ValueError(f"Unknown model names {sorted(unknown)}.")
        arima_orders = self._search_options_["arima"].get("orders")
        if arima_orders is not None:
            for order in arima_orders:
                try:
                    values = tuple(int(value) for value in order)
                except (TypeError, ValueError) as exc:
                    raise ValueError("Each ARIMA order must be a three-integer tuple.") from exc
                if len(values) != 3:
                    raise ValueError("Each ARIMA order must be a three-integer tuple.")
                if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in order):
                    raise ValueError("ARIMA order values must be integers.")
                if any(value < 0 for value in values):
                    raise ValueError("ARIMA order values must be non-negative.")
        state_space_models = self._search_options_["state_space"].get("models")
        if state_space_models is not None:
            unknown_state_models = (
                set(map(str, state_space_models))
                - set(self._VALID_STATE_SPACE_MODELS)
            )
            if unknown_state_models:
                raise ValueError(
                    f"Unknown state-space model names {sorted(unknown_state_models)}."
                )
        for family in ("arima", "state_space"):
            iterations = self._search_options_[family].get("max_iter", 50)
            if isinstance(iterations, bool) or not isinstance(iterations, (int, np.integer)) or iterations < 1:
                raise ValueError(f"{family} max_iter must be a positive integer.")
        improvement = self._search_options_["autoregressive"].get("min_improvement", 0.0)
        if not np.isfinite(improvement) or improvement < 0:
            raise ValueError("min_improvement must be finite and non-negative.")
        if int(self.zlib_level) < 0 or int(self.zlib_level) > 9:
            raise ValueError("zlib_level must be an integer between 0 and 9.")
        if int(self.zlib_overhead) < 0:
            raise ValueError("zlib_overhead must be non-negative.")
