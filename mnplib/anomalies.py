"""
Model-relative anomaly detection with the Minimum Nescience Principle.

The detector identifies observations that are not reconstructed by a predictive
model. Classification anomalies are misclassified samples. Regression anomalies
are samples whose observed and predicted target values fall in different bins of
a common uniform discretization of the target domain.

Mismatch is the anomaly criterion. Information-theoretic quantities are used
only to characterize identified anomalies. The class also applies supervised
miscoding to the anomalous subset to identify attributes associated with
systematic anomaly patterns, and measures how compressible the predicted target
states of anomalous observations are.

@author:    Rafael Garcia Leiva
@mail:      rgarcialeiva@gmail.com
@copyright: GNU GPLv3
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, clone
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted

from .classifier import NescienceClassifier
from .miscoding import Miscoding
from .regressor import NescienceRegressor


Task = Literal["auto", "classification", "regression"]
ResolvedTask = Literal["classification", "regression"]
XType = Literal["auto", "numeric", "categorical"]
YType = Literal["auto", "numeric", "categorical"]
BinSpec = int | Literal["auto"]
AnomalyKind = Literal["all", "misclassified", "under_predicted", "over_predicted"]


class AnomalyDetector(BaseEstimator):
    """
    Detect and explain model-relative anomalies.

    Parameters
    ----------
    task : {"auto", "classification", "regression"}, default="auto"
        Predictive task. If ``"auto"``, the task is inferred from ``y``.

    X_type : {"auto", "numeric", "categorical"}, default="auto"
        Feature encoding strategy passed to the nescience-based auto estimators
        and to ``Miscoding`` during anomaly explanation.

    y_type : {"auto", "numeric", "categorical"}, default="auto"
        Target encoding strategy passed to the nescience-based auto estimators.

    n_bins : int or "auto", default="auto"
        Number of bins used for regression anomaly detection and by ``Miscoding``
        when numeric attributes are analyzed. ``"auto"`` uses Rice's rule,
        ``ceil(2 * n_samples**(1/3))``.

    fit_model : bool, default=False
        If ``True`` and a model is supplied to ``fit``, clone and fit that model
        on ``(X, y)`` before producing predictions. If ``False``, the supplied
        model is assumed to be already fitted.

    auto_model_kwargs : mapping, optional
        Additional keyword arguments passed to ``NescienceClassifier`` or
        ``NescienceRegressor`` when no model and no predictions are supplied.

    random_state : int, optional
        Random seed passed to the nescience-based auto estimators.

    Notes
    -----
    Anomaly detection is intentionally threshold-free. A classification sample
    is anomalous exactly when its predicted class differs from the observed
    class. A regression sample is anomalous exactly when its observed and
    predicted values fall in different bins of one common discretization.

    Local correction information and negative local explanatory gain are
    diagnostics computed after anomaly detection. They never affect the anomaly
    mask.
    """

    _VALID_TASKS = ("auto", "classification", "regression")
    _VALID_X_TYPES = ("auto", "numeric", "categorical")
    _VALID_Y_TYPES = ("auto", "numeric", "categorical")
    _VALID_KINDS = ("all", "misclassified", "under_predicted", "over_predicted")

    def __init__(
        self,
        task: Task = "auto",
        X_type: XType = "auto",
        y_type: YType = "auto",
        n_bins: BinSpec = "auto",
        fit_model: bool = False,
        auto_model_kwargs: Mapping[str, Any] | None = None,
        random_state: int | None = None,
    ):
        self.task = task
        self.X_type = X_type
        self.y_type = y_type
        self.n_bins = n_bins
        self.fit_model = fit_model
        self.auto_model_kwargs = auto_model_kwargs
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, X, y, *, model=None, predictions=None):
        """
        Fit the anomaly detector.

        Parameters
        ----------
        X : array-like or pandas.DataFrame of shape (n_samples, n_features)
            Feature matrix.

        y : array-like of shape (n_samples,)
            Observed target values.

        model : object, optional
            Predictive model implementing ``predict(X)``. If omitted and
            ``predictions`` is also omitted, the appropriate nescience-based
            auto estimator is fitted.

        predictions : array-like of shape (n_samples,), optional
            Precomputed predictions.

        Returns
        -------
        self : AnomalyDetector
            Fitted detector.
        """
        self._validate_configuration()
        X_checked, y_checked, feature_names, X_frame = self._prepare_X_y(X, y)

        self.X_ = X_checked
        self.X_frame_ = X_frame
        self.y_ = y_checked
        self.feature_names_in_ = np.asarray(feature_names, dtype=object)
        self.n_samples_in_, self.n_features_in_ = self.X_.shape
        self.task_ = self._resolve_task(self.y_)
        self.model_ = None

        self.y_pred_ = self._resolve_predictions(
            X=self.X_,
            y=self.y_,
            model=model,
            predictions=predictions,
        )

        self._compute_anomalies()
        self._compute_information_diagnostics()
        self.anomaly_compressibility_ = self._compute_anomaly_compressibility()
        self.is_fitted_ = True

        return self

    def fit_predictions(self, X, y, predictions):
        """Fit the detector directly from a prediction vector."""
        return self.fit(X, y, predictions=predictions)

    def _resolve_predictions(self, *, X, y, model, predictions) -> np.ndarray:
        """Return validated predictions and store a fitted model when present."""
        if predictions is not None and model is not None:
            raise ValueError("Provide either model or predictions, not both.")

        if predictions is not None:
            return self._validate_predictions(predictions)

        if model is None:
            model = self._fit_auto_model(X, y)
        elif self.fit_model:
            model = clone(model)
            model.fit(X, y)

        if not hasattr(model, "predict"):
            raise TypeError("model must implement a predict(X) method.")

        self.model_ = model
        return self._validate_predictions(model.predict(X))

    def _fit_auto_model(self, X, y):
        """Fit the appropriate nescience-based auto estimator."""
        kwargs = dict(self.auto_model_kwargs or {})
        kwargs.setdefault("X_type", self.X_type)
        kwargs.setdefault("n_bins", self.n_bins)
        kwargs.setdefault("random_state", self.random_state)

        if self.task_ == "classification":
            model = NescienceClassifier(**kwargs)
            model.fit(X, y)
            return model

        model = NescienceRegressor(**kwargs)
        model.fit(X, y)
        return model

    # ------------------------------------------------------------------
    # Public anomaly outputs
    # ------------------------------------------------------------------

    def anomalies(self, kind: AnomalyKind = "all") -> np.ndarray:
        """
        Return sample indices identified as anomalous.

        Parameters
        ----------
        kind : {"all", "misclassified", "under_predicted", "over_predicted"},
               default="all"
            Subset of anomalies to return. ``"misclassified"`` is available
            only for classification. ``"under_predicted"`` and
            ``"over_predicted"`` are available only for regression.
        """
        check_is_fitted(self)
        return np.flatnonzero(self._mask_for_kind(kind)).astype(int)

    def anomaly_table(self, *, only_anomalies: bool = True) -> pd.DataFrame:
        """
        Return row-level anomaly diagnostics.

        Parameters
        ----------
        only_anomalies : bool, default=True
            If ``True``, return only anomalous samples. Otherwise return every
            sample. Information diagnostics are defined only for identified
            anomalies and are ``NaN`` for regular observations.

        Returns
        -------
        pandas.DataFrame
            Per-sample detection and information-theoretic diagnostics.
        """
        check_is_fitted(self)

        table = pd.DataFrame(
            {
                "sample_index": np.arange(self.n_samples_in_, dtype=int),
                "y_true": self.y_,
                "y_pred": self.y_pred_,
                "is_anomaly": self.anomaly_mask_,
                "anomaly_kind": self.anomaly_kind_,
                "local_correction_information": self.local_correction_information_,
                "negative_local_explanatory_gain": self.negative_local_explanatory_gain_,
            }
        )

        if self.task_ == "classification":
            table["correct"] = self.y_ == self.y_pred_
        else:
            table["residual"] = self.residual_
            table["direction"] = self.direction_
            table["y_true_bin"] = self.y_true_bin_
            table["y_pred_bin"] = self.y_pred_bin_
            table["bin_match"] = self.y_true_bin_ == self.y_pred_bin_

        if only_anomalies:
            table = table[table["is_anomaly"]].copy()

        return table.reset_index(drop=True)

    def summary(self) -> dict[str, object]:
        """Return compact summary statistics for the fitted detector."""
        check_is_fitted(self)

        anomaly_values = self.local_correction_information_[self.anomaly_mask_]
        negative_gain_values = self.negative_local_explanatory_gain_[self.anomaly_mask_]
        compressibility = self.anomaly_compressibility_

        result: dict[str, object] = {
            "task": self.task_,
            "n_samples": int(self.n_samples_in_),
            "n_features": int(self.n_features_in_),
            "n_anomalies": int(np.sum(self.anomaly_mask_)),
            "anomaly_rate": float(np.mean(self.anomaly_mask_)),
            "model_type": None if self.model_ is None else type(self.model_).__name__,
            "mean_local_correction_information": self._safe_mean(anomaly_values),
            "mean_negative_local_explanatory_gain": self._safe_mean(negative_gain_values),
            "anomaly_compressibility": float(compressibility["compressibility"]),
            "anomaly_compression_ratio": float(compressibility["compression_ratio"]),
            "anomaly_optimal_code_length": float(
                compressibility["optimal_code_length"]
            ),
            "anomaly_uniform_code_length": float(
                compressibility["uniform_code_length"]
            ),
            "n_target_states": int(compressibility["n_states"]),
            "n_anomaly_predicted_states": int(
                compressibility["n_anomaly_predicted_states"]
            ),
        }

        if hasattr(self.model_, "nescience_score"):
            result["model_nescience"] = float(self.model_.nescience_score())

        if self.task_ == "classification":
            result["n_misclassified"] = int(np.sum(self.anomaly_mask_))
        else:
            result.update(
                {
                    "n_bins": int(self.n_bins_),
                    "n_bin_mismatches": int(np.sum(self.anomaly_mask_)),
                    "n_under_predicted": int(
                        np.sum(self.anomaly_mask_ & self._under_prediction_mask())
                    ),
                    "n_over_predicted": int(
                        np.sum(self.anomaly_mask_ & self._over_prediction_mask())
                    ),
                }
            )

        return result

    # ------------------------------------------------------------------
    # Miscoding-based anomaly explanation
    # ------------------------------------------------------------------

    def explain_anomalies(
        self,
        *,
        kind: AnomalyKind = "all",
        max_features: int | None = None,
    ) -> dict[str, object]:
        """
        Identify attributes associated with the correction patterns of anomalies.

        Miscoding is fitted only on the requested anomalous observations. The
        supervised target is the correction state, represented by the ordered
        pair ``(predicted_state, observed_state)``. Consequently, the analysis
        searches for attributes that help distinguish different mechanisms of
        model failure inside the anomaly subset.

        Parameters
        ----------
        kind : {"all", "misclassified", "under_predicted", "over_predicted"},
               default="all"
            Anomaly subset to explain.

        max_features : int, optional
            Maximum number of attributes considered by the greedy miscoding
            selector. If omitted, every attribute is eligible.

        Returns
        -------
        dict
            Explanation diagnostics. ``feature_analysis`` ranks individual
            attributes from lowest to highest miscoding. ``selected_features``
            and ``selection_path`` contain the redundancy-aware greedy subset
            selected by ``Miscoding``.

        Notes
        -----
        At least two anomalies and at least two distinct correction patterns are
        required. If the requested anomalies all share one correction pattern,
        there is no supervised variation for miscoding to explain.
        """
        check_is_fitted(self)

        indices = self.anomalies(kind=kind)
        correction_target = self.correction_state_[indices]
        n_patterns = int(pd.Series(correction_target, dtype="object").nunique())

        base = {
            "kind": kind,
            "n_anomalies": int(indices.size),
            "n_correction_patterns": n_patterns,
        }

        if indices.size < 2:
            return {
                **base,
                "status": "insufficient_anomalies",
                "feature_analysis": self._empty_feature_analysis(),
                "selected_features": [],
                "selection_path": pd.DataFrame(),
            }

        if n_patterns < 2:
            return {
                **base,
                "status": "single_correction_pattern",
                "feature_analysis": self._empty_feature_analysis(),
                "selected_features": [],
                "selection_path": pd.DataFrame(),
            }

        metric = Miscoding(
            X_type=self.X_type,
            y_type="categorical",
            n_bins=self.n_bins,
        )
        metric.fit(self.X_frame_.iloc[indices].reset_index(drop=True), correction_target)

        features = metric.feature_analysis()
        selection = metric.select_features(
            max_features=max_features,
            return_details=True,
        )

        return {
            **base,
            "status": "ok",
            "feature_analysis": features,
            "selected_features": list(selection["selected_feature_names"]),
            "selected_feature_indices": list(selection["selected_feature_indices"]),
            "selection_path": selection["path"],
            "subset_analysis": selection["subset"],
        }

    def anomaly_compressibility(self) -> dict[str, object]:
        """
        Return compressibility of predicted target states for anomalous samples.

        The predicted states of the anomalous observations are encoded in two
        ways. The optimal code uses their empirical state probabilities, while
        the reference code assigns equal probability to every state in the
        encoded target alphabet.

        If ``L_opt`` and ``L_uniform`` are the corresponding ideal code lengths,
        the compression ratio is

        ``L_opt / L_uniform``

        and anomaly compressibility is

        ``1 - L_opt / L_uniform``.

        A value close to one indicates that anomalies are concentrated in a
        small or highly unbalanced set of predicted target states. A value close
        to zero indicates that their predicted states are approximately uniform
        over the available target alphabet.

        Returns
        -------
        dict
            Number of anomalies, target alphabet size, number of distinct
            predicted states present among anomalies, optimal code length,
            uniform code length, compression ratio, and compressibility.
        """
        check_is_fitted(self)
        return dict(self.anomaly_compressibility_)

    # ------------------------------------------------------------------
    # Anomaly detection
    # ------------------------------------------------------------------

    def _compute_anomalies(self) -> None:
        """Compute task-specific anomaly masks and symbolic target states."""
        if self.task_ == "classification":
            self._compute_classification_anomalies()
        else:
            self._compute_regression_anomalies()

        self.correction_state_ = np.asarray(
            [
                f"{int(predicted)}->{int(observed)}"
                for predicted, observed in zip(self.y_pred_state_, self.y_true_state_)
            ],
            dtype=object,
        )

    def _compute_classification_anomalies(self) -> None:
        """Mark classification samples whose predicted class is incorrect."""
        y_true_state, y_pred_state = self._common_categorical_codes(
            self.y_, self.y_pred_
        )
        misclassified = self.y_ != self.y_pred_

        self.y_true_state_ = y_true_state
        self.y_pred_state_ = y_pred_state
        self.anomaly_mask_ = np.asarray(misclassified, dtype=bool)
        self.anomaly_kind_ = np.where(self.anomaly_mask_, "misclassified", "regular")

    def _compute_regression_anomalies(self) -> None:
        """Mark regression samples whose values occupy different common bins."""
        y_true = self._numeric_vector(self.y_, name="y")
        y_pred = self._numeric_vector(self.y_pred_, name="predictions")

        true_bins, pred_bins, edges = self._common_numeric_bins(y_true, y_pred)
        residual = y_true - y_pred
        mismatched = true_bins != pred_bins

        self.bin_edges_ = edges
        self.y_true_state_ = true_bins
        self.y_pred_state_ = pred_bins
        self.y_true_bin_ = true_bins.copy()
        self.y_pred_bin_ = pred_bins.copy()
        self.residual_ = residual
        self.direction_ = np.where(
            residual > 0,
            "under_predicted",
            np.where(residual < 0, "over_predicted", "exact"),
        )
        self.anomaly_mask_ = np.asarray(mismatched, dtype=bool)
        self.anomaly_kind_ = np.where(self.anomaly_mask_, self.direction_, "regular")

    # ------------------------------------------------------------------
    # Information diagnostics
    # ------------------------------------------------------------------

    def _compute_information_diagnostics(self) -> None:
        """
        Compute correction information and negative explanatory gain.

        Probabilities are estimated from the complete fitted sample. Diagnostic
        values are retained only for observations already identified as
        anomalous by mismatch.
        """
        true_state = np.asarray(self.y_true_state_, dtype=int)
        pred_state = np.asarray(self.y_pred_state_, dtype=int)
        n_samples = int(true_state.size)

        true_counts = self._state_count_dict(true_state)
        pred_counts = self._state_count_dict(pred_state)
        pair_counts = self._pair_count_dict(pred_state, true_state)

        correction = np.full(n_samples, np.nan, dtype=float)
        negative_gain = np.full(n_samples, np.nan, dtype=float)

        for i in np.flatnonzero(self.anomaly_mask_):
            observed = int(true_state[i])
            predicted = int(pred_state[i])

            p_observed = true_counts[observed] / n_samples
            p_observed_given_prediction = (
                pair_counts[(predicted, observed)] / pred_counts[predicted]
            )

            correction[i] = -np.log2(p_observed_given_prediction)

            local_gain = np.log2(p_observed_given_prediction / p_observed)
            negative_gain[i] = max(0.0, -float(local_gain))

        self.local_correction_information_ = correction
        self.negative_local_explanatory_gain_ = negative_gain

    def _compute_anomaly_compressibility(self) -> dict[str, object]:
        """
        Compute compressibility of anomalous predicted target states.

        The optimal length is the ideal Shannon code length obtained from the
        empirical distribution of predicted states within the anomaly subset.
        The uniform length uses the complete encoded target alphabet as its
        reference distribution.
        """
        predicted = np.asarray(
            self.y_pred_state_[self.anomaly_mask_],
            dtype=int,
        )
        n_anomalies = int(predicted.size)
        n_states = self._number_of_target_states()

        if n_anomalies == 0:
            return {
                "n_anomalies": 0,
                "n_states": int(n_states),
                "n_anomaly_predicted_states": 0,
                "optimal_code_length": 0.0,
                "uniform_code_length": 0.0,
                "compression_ratio": 0.0,
                "compressibility": 0.0,
            }

        _, counts = np.unique(predicted, return_counts=True)
        counts = counts.astype(float)
        probabilities = counts / float(n_anomalies)

        optimal_length = max(
            0.0,
            -float(np.sum(counts * np.log2(probabilities))),
        )

        uniform_length = float(
            n_anomalies * np.log2(n_states)
        )

        if uniform_length == 0.0:
            compression_ratio = 0.0
            compressibility = 1.0
        else:
            compression_ratio = float(
                np.clip(optimal_length / uniform_length, 0.0, 1.0)
            )
            compressibility = 1.0 - compression_ratio

        return {
            "n_anomalies": n_anomalies,
            "n_states": int(n_states),
            "n_anomaly_predicted_states": int(counts.size),
            "optimal_code_length": optimal_length,
            "uniform_code_length": uniform_length,
            "compression_ratio": compression_ratio,
            "compressibility": float(compressibility),
        }

    def _number_of_target_states(self) -> int:
        """
        Return the size of the encoded target alphabet.

        Classification uses the common alphabet built from observed and
        predicted labels. Regression uses the common target bins and includes
        each out-of-range prediction state when that state occurs.
        """
        if self.task_ == "classification":
            states = np.concatenate(
                [
                    np.asarray(self.y_true_state_, dtype=int),
                    np.asarray(self.y_pred_state_, dtype=int),
                ]
            )
            return max(1, int(np.unique(states).size))

        n_states = int(self.n_bins_)
        predicted = np.asarray(self.y_pred_state_, dtype=int)

        if np.any(predicted < 0):
            n_states += 1

        if np.any(predicted >= self.n_bins_):
            n_states += 1

        return max(1, n_states)

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def _common_numeric_bins(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Discretize observed and predicted values with one set of uniform edges.

        The bin edges are learned once from the observed target domain and are
        then applied unchanged to both ``y`` and ``y_hat``. Predictions outside
        the observed target range receive dedicated out-of-range states, so an
        extreme prediction cannot be hidden inside an edge bin.
        """
        bins = self._resolve_bins(self.n_bins, self.n_samples_in_)

        lower = float(np.min(y_true))
        upper = float(np.max(y_true))

        if bins <= 1 or lower == upper:
            self.n_bins_ = 1
            edges = np.asarray([lower, upper], dtype=float)
            true_bins = np.zeros(self.n_samples_in_, dtype=int)
            pred_bins = np.zeros(self.n_samples_in_, dtype=int)
            pred_bins[y_pred < lower] = -1
            pred_bins[y_pred > upper] = 1
            return true_bins, pred_bins, edges

        self.n_bins_ = bins
        edges = np.linspace(lower, upper, bins + 1, dtype=float)
        internal_edges = edges[1:-1]

        true_bins = np.digitize(y_true, internal_edges, right=False).astype(int)
        pred_bins = np.digitize(y_pred, internal_edges, right=False).astype(int)

        pred_bins[y_pred < lower] = -1
        pred_bins[y_pred > upper] = bins

        return true_bins, pred_bins, edges

    @staticmethod
    def _common_categorical_codes(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
        """Encode observed and predicted categorical labels in one alphabet."""
        observed = np.asarray(y_true, dtype=object)
        predicted = np.asarray(y_pred, dtype=object)
        combined = np.concatenate([observed, predicted])
        codes, _ = pd.factorize(combined, sort=False)
        n = observed.shape[0]
        return codes[:n].astype(int), codes[n:].astype(int)

    @staticmethod
    def _resolve_bins(n_bins: BinSpec, n_samples: int) -> int:
        """Resolve an explicit bin count or Rice's automatic rule."""
        if n_bins == "auto":
            bins = int(np.ceil(2.0 * int(n_samples) ** (1.0 / 3.0)))
            return int(min(max(1, bins), int(n_samples)))

        if isinstance(n_bins, bool):
            raise ValueError("n_bins must be a positive integer or 'auto'.")

        bins = int(n_bins)
        if bins < 1:
            raise ValueError("n_bins must be a positive integer or 'auto'.")
        return bins

    # ------------------------------------------------------------------
    # Masks
    # ------------------------------------------------------------------

    def _mask_for_kind(self, kind: AnomalyKind) -> np.ndarray:
        """Return a boolean anomaly mask for the requested kind."""
        if kind not in self._VALID_KINDS:
            raise ValueError(
                f"Valid options for kind are {self._VALID_KINDS}. Got {kind!r}."
            )

        if kind == "all":
            return self.anomaly_mask_.copy()

        if kind == "misclassified":
            if self.task_ != "classification":
                raise ValueError("'misclassified' is valid only for classification.")
            return self.anomaly_mask_.copy()

        if kind == "under_predicted":
            if self.task_ != "regression":
                raise ValueError("'under_predicted' is valid only for regression.")
            return self.anomaly_mask_ & self._under_prediction_mask()

        if self.task_ != "regression":
            raise ValueError("'over_predicted' is valid only for regression.")

        return self.anomaly_mask_ & self._over_prediction_mask()

    def _under_prediction_mask(self) -> np.ndarray:
        """Return samples for which the model predicted too small a value."""
        return np.asarray(getattr(self, "residual_", np.array([]))) > 0

    def _over_prediction_mask(self) -> np.ndarray:
        """Return samples for which the model predicted too large a value."""
        return np.asarray(getattr(self, "residual_", np.array([]))) < 0

    # ------------------------------------------------------------------
    # Validation and configuration
    # ------------------------------------------------------------------

    def _validate_configuration(self) -> None:
        """Validate constructor parameters before fitting."""
        if self.task not in self._VALID_TASKS:
            raise ValueError(
                f"Valid options for task are {self._VALID_TASKS}. Got {self.task!r}."
            )

        if self.X_type not in self._VALID_X_TYPES:
            raise ValueError(
                f"Valid options for X_type are {self._VALID_X_TYPES}. "
                f"Got {self.X_type!r}."
            )

        if self.y_type not in self._VALID_Y_TYPES:
            raise ValueError(
                f"Valid options for y_type are {self._VALID_Y_TYPES}. "
                f"Got {self.y_type!r}."
            )

        self._resolve_bins(self.n_bins, 1)

    @staticmethod
    def _prepare_X_y(X, y) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
        """Validate input data while preserving feature names and DataFrame types."""
        if isinstance(X, pd.DataFrame):
            feature_names = [str(column) for column in X.columns]
            source_frame = X.copy().reset_index(drop=True)
            source_frame.columns = feature_names
        else:
            feature_names = None
            source_frame = None

        X_checked, y_checked = check_X_y(X, y, dtype=None, ensure_2d=True)

        if feature_names is None:
            feature_names = [f"x{i}" for i in range(X_checked.shape[1])]
            source_frame = pd.DataFrame(X_checked, columns=feature_names)

        return (
            X_checked,
            np.ravel(np.asarray(y_checked)),
            feature_names,
            source_frame,
        )

    def _validate_predictions(self, predictions) -> np.ndarray:
        """Validate prediction vector against the fitted target."""
        pred = np.ravel(np.asarray(predictions))

        if pred.shape[0] != self.n_samples_in_:
            raise ValueError(
                "predictions and y must have the same number of samples. "
                f"Got {pred.shape[0]} predictions and {self.n_samples_in_} targets."
            )

        return pred

    def _resolve_task(self, y: np.ndarray) -> ResolvedTask:
        """Resolve the configured task from the target vector."""
        if self.task in ("classification", "regression"):
            return self.task

        target_type = type_of_target(y)

        if target_type in ("binary", "multiclass"):
            return "classification"

        if target_type == "continuous":
            return "regression"

        raise ValueError(
            "Unsupported target type {!r}. Supported targets are binary, "
            "multiclass, and continuous.".format(target_type)
        )

    @staticmethod
    def _numeric_vector(values, *, name: str) -> np.ndarray:
        """Return a finite numeric one-dimensional vector."""
        array = np.ravel(np.asarray(values, dtype=float))

        if array.size == 0:
            raise ValueError(f"{name} must not be empty.")

        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite numeric values.")

        return array

    @staticmethod
    def _state_count_dict(states: np.ndarray) -> dict[int, int]:
        """Return empirical counts for integer states."""
        values, counts = np.unique(states, return_counts=True)
        return {int(value): int(count) for value, count in zip(values, counts)}

    @staticmethod
    def _pair_count_dict(
        first: np.ndarray,
        second: np.ndarray,
    ) -> dict[tuple[int, int], int]:
        """Return empirical counts for ordered pairs of integer states."""
        pairs = np.column_stack([first, second])
        values, counts = np.unique(pairs, axis=0, return_counts=True)
        return {
            (int(value[0]), int(value[1])): int(count)
            for value, count in zip(values, counts)
        }

    def _empty_feature_analysis(self) -> pd.DataFrame:
        """Return an empty feature-analysis table with stable columns."""
        return pd.DataFrame(
            columns=[
                "feature_index",
                "feature_name",
                "is_numeric",
                "code_length",
                "deficiency",
                "surplus",
                "miscoding",
            ]
        )

    @staticmethod
    def _safe_mean(values: np.ndarray) -> float | None:
        """Return the finite mean of a diagnostic vector, or None if empty."""
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return None
        return float(np.mean(values))


def anomaly_table(X, y, predictions, *, task: Task = "auto", **kwargs) -> pd.DataFrame:
    """Return an anomaly table directly from a prediction vector."""
    detector = AnomalyDetector(task=task, **kwargs)
    detector.fit_predictions(X, y, predictions)
    return detector.anomaly_table()