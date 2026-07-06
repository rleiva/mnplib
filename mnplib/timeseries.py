"""
Simplified TimeSeries class for mnplib/nescience.

Forecasting is treated as model selection over lagged representations.
Public lag diagnostics are deficiency, surplus, and miscoding, where

    miscoding = max(deficiency, surplus).

Candidate models are evaluated with four-component nescience:
deficiency, surplus, inaccuracy, and surfeit.
"""

from __future__ import annotations

from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_array, column_or_1d
from sklearn.utils.validation import check_is_fitted

from .miscoding import Miscoding
from .nescience import Nescience


XType = Literal["auto", "numeric", "mixed", "categorical"]
YType = Literal["auto", "numeric", "categorical"]
BinSpec = int | Literal["auto"]
Validation = Literal["none", "holdout"]
Aggregation = Literal[
    "euclidean",
    "arithmetic",
    "geometric",
    "harmonic",
    "maximum",
    "addition",
    "product",
]


class FixedLinearForecaster(BaseEstimator, RegressorMixin):
    """Linear forecaster with fixed coefficients."""

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

    def __repr__(self):
        weights = np.array2string(np.asarray(self.weights), precision=6)
        return f"FixedLinearForecaster(name={self.name!r}, weights={weights})"


class TimeSeries(BaseEstimator, RegressorMixin):
    """Forecast time series using nescience-based lagged representations."""

    _VALID_Y_TYPES = ("auto", "numeric", "categorical")
    _VALID_X_TYPES = ("auto", "numeric", "mixed", "categorical")
    _VALID_VALIDATION = ("none", "holdout")
    _VALID_MODELS = ("autoregressive", "moving_average", "exponential_smoothing")

    def __init__(
        self,
        *,
        y_type: YType = "numeric",
        X_type: XType = "auto",
        window_size: int | Literal["auto"] = "auto",
        max_lag: int | None = None,
        models: Sequence[str] | None = None,
        auto: bool = True,
        validation: Validation = "none",
        test_size: float = 0.2,
        moving_average_windows: Sequence[int] | None = None,
        smoothing_alphas: Sequence[float] | None = None,
        aggregation: Aggregation = "euclidean",
        weights: Mapping[str, float] | Sequence[float] | None = None,
        n_bins: BinSpec = "auto",
        min_bins: int = 2,
        max_bins: int | None = None,
        missing: str = "raise",
        base: float = 2.0,
        correction: Literal["none", "miller_madow", "dirichlet"] = "none",
        alpha: float = 0.5,
        alphabet_size: int | None = None,
        threshold: float | Literal["auto"] = "auto",
        threshold_fraction: float = 0.01,
        surplus_penalty: float = 0.0,
        high_surplus_threshold: float = 0.75,
        coef_tol: float = 1e-12,
        zlib_level: int = 9,
        zlib_overhead: int = 6,
        random_state: int | None = None,
    ):
        self._validate_init(
            y_type, X_type, window_size, max_lag, models, validation,
            test_size, surplus_penalty
        )
        for name, value in locals().items():
            if name != "self":
                setattr(self, name, value)

    def fit(self, y, X=None):
        y = column_or_1d(y).astype(float, copy=False)
        if len(y) < 3:
            raise ValueError("y must contain at least three observations.")

        X, names = self._validate_exogenous_X(X, len(y))
        self.y_ = y
        self.X_exogenous_ = X
        self.exogenous_feature_names_ = names
        self.y_isnumeric_ = self.y_type != "categorical"
        self.window_size_ = self._resolve_window_size(len(y))
        self.max_lag_ = self.window_size_ if self.max_lag is None else int(self.max_lag)

        data = self._to_supervised(y, X)
        self.X_supervised_ = data["X"]
        self.y_supervised_ = data["y"]
        self.feature_names_in_ = np.asarray(data["feature_names"], dtype=object)
        self.feature_metadata_ = data["feature_metadata"]

        split = self._train_validation_split(self.X_supervised_, self.y_supervised_)
        self.X_train_, self.y_train_, self.X_validation_, self.y_validation_ = split

        self.nescience_ = self._make_nescience().fit(self.X_train_, self.y_train_)
        self.miscoding_ = self.nescience_.miscoding_
        self.results_ = pd.DataFrame()
        self.candidate_results_ = []

        self._fit_auto() if self.auto else self._clear_selected_model()
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self)
        self._require_model()
        X = check_array(X, dtype=float)
        return self.model_.predict(X[:, np.flatnonzero(self.subset_)])

    def forecast(self, steps: int = 1, X_future=None) -> np.ndarray:
        check_is_fitted(self)
        self._require_model()

        steps = int(steps)
        if steps < 1:
            raise ValueError("steps must be positive.")

        y_hist = list(np.asarray(self.y_, dtype=float))
        X_hist, X_future = self._prepare_future_exogenous(steps, X_future)
        selected = np.flatnonzero(self.subset_)
        forecasts = []

        for i in range(steps):
            X_context = None if X_hist is None else np.asarray(X_hist, dtype=float)
            row = self._single_lag_row(np.asarray(y_hist), X_context)
            y_hat = float(self.model_.predict(row[:, selected])[0])
            forecasts.append(y_hat)
            y_hist.append(y_hat)
            if X_hist is not None:
                X_hist.append(np.asarray(X_future[i], dtype=float))

        return np.asarray(forecasts, dtype=float)

    def score(self, y, X=None):
        check_is_fitted(self)
        self._require_model()
        y = column_or_1d(y).astype(float, copy=False)
        X, _ = self._validate_exogenous_X(X, len(y))
        data = self._to_supervised(y, X)
        selected = np.flatnonzero(self.subset_)
        return self.model_.score(data["X"][:, selected], data["y"])

    def get_model(self):
        check_is_fitted(self)
        return self.model_

    def explain(self) -> dict[str, object]:
        check_is_fitted(self)
        self._require_model()

        details = self._evaluate_components(self.model_, self.subset_)
        components = {
            "deficiency": float(details["deficiency"]),
            "surplus": float(details["surplus"]),
            "inaccuracy": float(details["inaccuracy"]),
            "surfeit": float(details["surfeit"]),
        }
        dominant = max(components, key=components.get)

        return {
            "nescience": float(details["nescience"]),
            "aggregation": details.get("aggregation", self.aggregation),
            "components": components,
            "miscoding": max(components["deficiency"], components["surplus"]),
            "dominant_component": dominant,
            "time_series_model": self.model_name_,
            "window_size": self.window_size_,
            "selected_lags": self.selected_lags_,
            "selected_feature_indices": self.selected_feature_indices_,
            "selected_feature_names": self.selected_feature_names_,
            "n_features_in_use": int(np.sum(self.subset_)),
            "validation": self.validation,
            "details": details,
        }

    def auto_lag_analysis(self, *, min_lag: int = 1, max_lag: int | None = None):
        check_is_fitted(self)
        y = np.asarray(self.y_, dtype=float)
        return self._lag_analysis(values=y, target=y, prefix="y",
                                  min_lag=min_lag, max_lag=max_lag)

    def cross_lag_analysis(
        self,
        attribute: int | str,
        *,
        min_lag: int = 1,
        max_lag: int | None = None,
    ):
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

    def lag_analysis(self, *, max_lag: int | None = None):
        check_is_fitted(self)
        tables = [self.auto_lag_analysis(max_lag=max_lag)]
        if self.X_exogenous_ is not None:
            tables += [
                self.cross_lag_analysis(name, max_lag=max_lag)
                for name in self.exogenous_feature_names_
            ]
        return pd.concat(tables, ignore_index=True)

    # ------------------------------------------------------------------
    # Candidate models
    # ------------------------------------------------------------------

    def _fit_auto(self):
        candidates = []
        for name in self.models or self._VALID_MODELS:
            if name == "autoregressive":
                candidates.append(self._autoregressive_candidate())
            elif name == "moving_average":
                candidates.extend(self._fixed_candidates("moving_average"))
            elif name == "exponential_smoothing":
                candidates.extend(self._fixed_candidates("exponential_smoothing"))

        if not candidates:
            raise RuntimeError("No candidate time-series models were evaluated.")

        candidates.sort(key=lambda row: row["nescience"])
        self.candidate_results_ = candidates
        self.results_ = self._candidate_frame(candidates)
        self._set_selected_candidate(candidates[0])

    def _autoregressive_candidate(self):
        selection = self.miscoding_.select_features(
            threshold=self.threshold,
            surplus_penalty=self.surplus_penalty,
            return_details=True,
        )
        subset = np.asarray(selection["selected_features"], dtype=int)

        if np.sum(subset) == 0:
            table = self.miscoding_.feature_analysis()
            sort_col = "miscoding" if "miscoding" in table.columns else "deficiency"
            subset[int(table.sort_values(sort_col).iloc[0]["feature_index"])] = 1

        selected = np.flatnonzero(subset)
        model = LinearRegression().fit(self.X_train_[:, selected], self.y_train_)
        return self._evaluate_candidate("autoregressive", model, subset, self.window_size_)

    def _fixed_candidates(self, kind: str):
        candidates = []
        for window in self._moving_average_windows():
            subset = self._target_lag_subset(window)
            selected = np.flatnonzero(subset)

            for name, weights in self._fixed_configs(kind, window):
                model = FixedLinearForecaster(weights=weights, name=name)
                model.fit(self.X_train_[:, selected], self.y_train_)
                candidates.append(self._evaluate_candidate(name, model, subset, window))

        return candidates

    def _fixed_configs(self, kind: str, window: int):
        if kind == "moving_average":
            return [(f"moving_average_{window}", np.repeat(1.0 / window, window))]

        if kind == "exponential_smoothing":
            return [
                (f"exponential_smoothing_w{window}_a{a:g}", self._smoothing_weights(window, a))
                for a in self._smoothing_alphas()
            ]

        raise ValueError(f"Unknown fixed-weight candidate kind {kind!r}.")

    def _evaluate_candidate(self, model_name: str, model, subset: np.ndarray, window_size: int):
        details = self._evaluate_components(model, subset)
        selected = np.flatnonzero(subset)
        deficiency = float(details["deficiency"])
        surplus = float(details["surplus"])

        return {
            "model_name": model_name,
            "window_size": int(window_size),
            "selected_feature_indices": tuple(int(i) for i in selected),
            "selected_feature_names": tuple(str(self.feature_names_in_[i]) for i in selected),
            "n_features_in_use": int(len(selected)),
            "nescience": float(details["nescience"]),
            "deficiency": deficiency,
            "surplus": surplus,
            "miscoding": max(deficiency, surplus),
            "inaccuracy": float(details["inaccuracy"]),
            "surfeit": float(details["surfeit"]),
            "validation_score": self._validation_score(model, subset),
            "model": model,
            "subset": subset.astype(int),
        }

    def _evaluate_components(self, model, subset: np.ndarray):
        X_eval, y_eval = self._evaluation_data()
        selected = np.flatnonzero(subset)
        if len(selected) == 0:
            raise ValueError("Candidate subset must select at least one feature.")
        predictions = model.predict(X_eval[:, selected])
        nsc = self._make_nescience().fit(X_eval, y_eval)
        return nsc.components(
            model=model,
            subset=subset,
            predictions=predictions,
            model_string=repr(model),
        )

    def _validation_score(self, model, subset):
        if self.validation != "holdout":
            return None
        X_eval, y_eval = self._evaluation_data()
        return float(model.score(X_eval[:, np.flatnonzero(subset)], y_eval))

    @staticmethod
    def _candidate_frame(candidates):
        return pd.DataFrame(
            [
                {k: v for k, v in row.items() if k not in {"model", "subset"}}
                for row in candidates
            ]
        ).sort_values("nescience", ignore_index=True)

    def _set_selected_candidate(self, candidate):
        self.best_result_ = candidate
        self.model_ = candidate["model"]
        self.subset_ = candidate["subset"].astype(int)
        self.model_name_ = candidate["model_name"]
        self.selected_feature_indices_ = list(candidate["selected_feature_indices"])
        self.selected_feature_names_ = list(candidate["selected_feature_names"])
        self.selected_lags_ = self._selected_lags_from_indices(self.selected_feature_indices_)

    # ------------------------------------------------------------------
    # Lag analysis and representation
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
    ):
        if min_lag < 1:
            raise ValueError("min_lag must be positive.")
        max_lag = self.max_lag_ if max_lag is None else int(max_lag)
        max_lag = min(max_lag, len(target) - 1)

        rows = []
        for lag in range(min_lag, max_lag + 1):
            row = self._single_lag_miscoding(values, target, lag)
            deficiency = float(row["deficiency"])
            surplus = float(row["surplus"])
            result = {
                "lag": lag,
                "feature_name": f"{prefix}_lag_{lag}",
                "deficiency": deficiency,
                "surplus": surplus,
                "miscoding": float(row.get("miscoding", max(deficiency, surplus))),
            }
            if attribute is not None:
                result["attribute"] = attribute
            rows.append(result)

        return pd.DataFrame(rows)

    def _single_lag_miscoding(self, values, target, lag):
        metric = self._make_miscoding()
        metric.fit(np.asarray(values[:-lag]).reshape(-1, 1), target[lag:])
        return metric.feature_analysis().iloc[0]

    def _to_supervised(self, y, X):
        p = self.window_size_
        if p >= len(y):
            raise ValueError(f"window_size={p} must be smaller than n_samples={len(y)}.")

        rows = [self._lagged_row(y, X, t, p) for t in range(p, len(y))]
        names, metadata = self._feature_description(X is not None, p)

        return {
            "X": np.asarray(rows, dtype=float),
            "y": np.asarray(y[p:], dtype=float),
            "feature_names": names,
            "feature_metadata": metadata,
        }

    @staticmethod
    def _lagged_row(y, X, t: int, p: int):
        row = [y[t - lag] for lag in range(1, p + 1)]
        if X is not None:
            for j in range(X.shape[1]):
                row.extend(X[t - lag, j] for lag in range(1, p + 1))
        return row

    def _single_lag_row(self, y_history, X_history):
        p = self.window_size_
        if len(y_history) < p:
            raise ValueError("Not enough target history to build a lagged row.")
        row = [y_history[-lag] for lag in range(1, p + 1)]

        if X_history is not None:
            if len(X_history) < p:
                raise ValueError("Not enough exogenous history to build a lagged row.")
            for j in range(X_history.shape[1]):
                row.extend(X_history[-lag, j] for lag in range(1, p + 1))

        return np.asarray(row, dtype=float).reshape(1, -1)

    def _feature_description(self, has_exogenous: bool, p: int):
        names = [f"y_lag_{lag}" for lag in range(1, p + 1)]
        metadata = [{"source": "target", "attribute": "y", "lag": lag} for lag in range(1, p + 1)]

        if has_exogenous:
            for j, name in enumerate(self.exogenous_feature_names_):
                for lag in range(1, p + 1):
                    names.append(f"{name}_lag_{lag}")
                    metadata.append({
                        "source": "exogenous",
                        "attribute": str(name),
                        "attribute_index": j,
                        "lag": lag,
                    })

        return names, metadata

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def _metric_params(self):
        return {
            "y_type": self.y_type,
            "n_bins": self.n_bins,
            "min_bins": self.min_bins,
            "max_bins": self.max_bins,
            "missing": self.missing,
            "base": self.base,
            "correction": self.correction,
            "alpha": self.alpha,
            "alphabet_size": self.alphabet_size,
            "threshold": self.threshold,
            "threshold_fraction": self.threshold_fraction,
            "surplus_penalty": self.surplus_penalty,
            "high_surplus_threshold": self.high_surplus_threshold,
            "coef_tol": self.coef_tol,
        }

    def _make_nescience(self):
        params = self._metric_params()
        params.update({
            "X_type": "numeric",
            "aggregation": self.aggregation,
            "weights": self.weights,
            "zlib_level": self.zlib_level,
            "zlib_overhead": self.zlib_overhead,
        })
        return Nescience(**params)

    def _make_miscoding(self):
        params = self._metric_params()
        params["X_type"] = "numeric"
        return Miscoding(**params)

    def _resolve_window_size(self, n: int):
        p = max(1, int(np.sqrt(n))) if self.window_size == "auto" else int(self.window_size)
        if p >= n:
            raise ValueError("window_size must be smaller than the number of observations.")
        return p

    def _moving_average_windows(self):
        values = (
            range(1, self.window_size_ + 1)
            if self.moving_average_windows is None
            else (int(w) for w in self.moving_average_windows)
        )
        windows = sorted({w for w in values if 1 <= w <= self.window_size_})
        if not windows:
            raise ValueError("No valid moving-average windows to evaluate.")
        return windows

    def _smoothing_alphas(self):
        alphas = [0.1, 0.2, 0.3, 0.5, 0.8] if self.smoothing_alphas is None else list(map(float, self.smoothing_alphas))
        if any(a <= 0 or a >= 1 for a in alphas):
            raise ValueError("All smoothing alphas must lie in (0, 1).")
        return alphas

    @staticmethod
    def _smoothing_weights(window: int, alpha: float):
        weights = alpha * (1.0 - alpha) ** np.arange(window)
        return weights / np.sum(weights)

    def _target_lag_subset(self, window: int):
        subset = np.zeros(self.X_train_.shape[1], dtype=int)
        subset[:window] = 1
        return subset

    def _train_validation_split(self, X, y):
        if self.validation == "none":
            return X, y, None, None
        split = int(np.floor((1.0 - self.test_size) * len(X)))
        if split <= 1 or split >= len(X):
            raise ValueError("The holdout split is invalid. Adjust test_size or provide more data.")
        return X[:split], y[:split], X[split:], y[split:]

    def _evaluation_data(self):
        return (self.X_validation_, self.y_validation_) if self.validation == "holdout" else (self.X_train_, self.y_train_)

    def _validate_exogenous_X(self, X, n: int):
        if X is None:
            return None, []
        if isinstance(X, pd.DataFrame):
            if len(X) != n:
                raise ValueError(f"X and y have inconsistent lengths: {len(X)} != {n}.")
            if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
                raise ValueError("The current TimeSeries implementation requires numeric exogenous variables.")
            return X.to_numpy(dtype=float), [str(c) for c in X.columns]

        X = check_array(X, dtype=float, ensure_2d=True)
        if X.shape[0] != n:
            raise ValueError(f"X and y have inconsistent lengths: {X.shape[0]} != {n}.")
        return X, [f"x{i}" for i in range(X.shape[1])]

    def _resolve_attribute(self, attribute):
        if isinstance(attribute, str):
            if attribute not in self.exogenous_feature_names_:
                raise ValueError(f"Unknown exogenous attribute {attribute!r}.")
            return self.exogenous_feature_names_.index(attribute)
        index = int(attribute)
        if index < 0 or index >= len(self.exogenous_feature_names_):
            raise ValueError(f"attribute index {index} is outside the valid range.")
        return index

    def _prepare_future_exogenous(self, steps: int, X_future):
        if self.X_exogenous_ is None:
            return None, None

        history = [row.copy() for row in np.asarray(self.X_exogenous_, dtype=float)]
        n_exog = len(self.exogenous_feature_names_)

        if X_future is None:
            future = np.repeat(np.asarray(history[-1]).reshape(1, -1), steps, axis=0)
        else:
            future = check_array(X_future, dtype=float, ensure_2d=True)
            if future.shape != (steps, n_exog):
                raise ValueError(f"X_future must have shape ({steps}, {n_exog}). Got {future.shape}.")

        return history, future

    def _selected_lags_from_indices(self, indices):
        result = []
        for index in indices:
            meta = dict(self.feature_metadata_[int(index)])
            meta["feature_index"] = int(index)
            meta["feature_name"] = str(self.feature_names_in_[int(index)])
            result.append(meta)
        return result

    def _clear_selected_model(self):
        self.model_ = None
        self.subset_ = np.zeros(self.X_supervised_.shape[1], dtype=int)
        self.selected_feature_indices_ = []
        self.selected_feature_names_ = []
        self.selected_lags_ = []
        self.best_result_ = None
        self.model_name_ = None

    def _require_model(self):
        if self.model_ is None:
            raise ValueError("No forecasting model has been fitted because auto=False.")

    @classmethod
    def _validate_init(cls, y_type, X_type, window_size, max_lag, models, validation, test_size, surplus_penalty):
        if y_type not in cls._VALID_Y_TYPES:
            raise ValueError(f"Valid options for 'y_type' are {cls._VALID_Y_TYPES}. Got {y_type!r}.")
        if X_type not in cls._VALID_X_TYPES:
            raise ValueError(f"Valid options for 'X_type' are {cls._VALID_X_TYPES}. Got {X_type!r}.")
        if validation not in cls._VALID_VALIDATION:
            raise ValueError(f"Valid options for 'validation' are {cls._VALID_VALIDATION}. Got {validation!r}.")
        if models is not None and (set(models) - set(cls._VALID_MODELS)):
            raise ValueError(f"Unknown model names {sorted(set(models) - set(cls._VALID_MODELS))}.")
        if window_size != "auto" and int(window_size) < 1:
            raise ValueError("window_size must be a positive integer or 'auto'.")
        if max_lag is not None and int(max_lag) < 1:
            raise ValueError("max_lag must be positive when provided.")
        if not 0 < test_size < 1:
            raise ValueError("test_size must lie in the open interval (0, 1).")
        if surplus_penalty < 0:
            raise ValueError("surplus_penalty must be non-negative.")

