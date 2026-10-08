"""
Model-relative residual and correction analysis with the Minimum Nescience Principle.

The detector identifies observations that are not reconstructed by a predictive
model. Classification anomalies are misclassified samples. Regression anomalies
are samples whose observed and predicted target values fall in different bins of
a common uniform discretization of the target domain.

Mismatch is the anomaly criterion. Information-theoretic quantities characterize
all observations. The class also applies supervised
miscoding to the anomalous subset to identify attributes associated with
systematic anomaly patterns, and measures how compressible the predicted target
states of anomalous observations are.

@author:    Rafael Garcia Leiva
@mail:      rgarcialeiva@gmail.com
@copyright: GNU GPLv3
"""

from __future__ import annotations

from copy import deepcopy
from typing import Literal, get_args

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted

from ._types import ResolvedTask, Task, XType
from .inaccuracy import Inaccuracy
from .miscoding import Miscoding
from .utils import _resolve_bins, _resolve_feature_names, _resolve_y_isnumeric


AnomalyKind = Literal["all", "misclassified", "under_predicted", "over_predicted"]


class ResidualAnalysis(BaseEstimator):
    """Analyze empirical correction information for explicit model predictions.

    No model is trained by this class. Classification anomalies are label
    mismatches; regression anomalies are mismatches under common target-derived
    uniform bins, including separate out-of-range prediction states. Local
    information characterizes all samples; it is not an anomaly threshold.

    Inaccuracy retains the library's own encoding (independent numeric edges).
    Correction information uses common target states and is NOT a decomposition
    of that inaccuracy. Empirical conditional lengths omit codebook costs and
    do not measure generalization. Feature explanations are computed lazily.
    """

    _VALID_TASKS = get_args(Task)
    _VALID_X_TYPES = get_args(XType)
    _VALID_KINDS = get_args(AnomalyKind)

    def __init__(self, task: Task = "auto", X_type: XType = "auto"):
        self.task = task
        self.X_type = X_type

    def fit(self, X, y, *, predictions, sample_ids=None):
        """Fit with optional explanatory attributes; predictions must align with y.

        All attributes in X are available for correction-pattern explanation,
        not necessarily just those used by the predictive model.
        """
        self._validate_configuration()
        for name in list(vars(self)):
            if name.endswith("_"):
                delattr(self, name)
        y = self._vector(y, "y")
        predictions = self._vector(predictions, "predictions")
        if len(y) != len(predictions):
            raise ValueError("predictions and y must have the same number of samples.")
        self.task_ = self._resolve_task(y)
        if self.task_ == "regression":
            y = self._numeric_vector(y, name="y")
            predictions = self._numeric_vector(predictions, name="predictions")
        if X is None:
            self.X_frame_ = pd.DataFrame(index=np.arange(len(y)))
            self.X_ = self.X_frame_.to_numpy()
            names = []
        else:
            self.X_, _, names, self.X_frame_ = self._prepare_X_y(X, y)
        self.y_, self.y_pred_ = y.copy(), predictions.copy()
        self.n_samples_in_, self.n_features_in_ = self.X_.shape
        self.feature_names_in_ = np.fromiter(names, dtype=object)
        ids = np.arange(len(y)) if sample_ids is None else self._vector(sample_ids, "sample_ids")
        if len(ids) != len(y) or not pd.Index(ids).is_unique:
            raise ValueError("sample_ids must be unique and aligned with y.")
        self.sample_ids_ = ids.copy()
        self.n_bins_ = _resolve_bins("auto", len(y))
        self._compute_anomalies()
        self._compute_information_diagnostics()
        self.inaccuracy_analysis_ = Inaccuracy(
            y_type="numeric" if self.task_ == "regression" else "categorical"
        ).fit_y(self.y_).prediction_analysis(self.y_pred_)
        self._explanations_ = {}
        self.is_fitted_ = True
        return self

    def fit_y(self, y, *, predictions, sample_ids=None):
        """Fit prediction diagnostics without explanatory attributes."""
        return self.fit(None, y, predictions=predictions, sample_ids=sample_ids)

    def __sklearn_is_fitted__(self):
        return getattr(self, "is_fitted_", False)

    @staticmethod
    def _vector(values, name):
        array = np.asarray(values)
        if array.ndim != 1 or not array.size:
            raise ValueError(f"{name} must be a nonempty one-dimensional vector.")
        if pd.isna(array).any():
            raise ValueError(f"{name} must not contain missing values.")
        if any(isinstance(v, (float, np.floating)) and not np.isfinite(v) for v in array):
            raise ValueError(f"{name} must contain only finite values.")
        return array

    # ------------------------------------------------------------------
    # Public reports
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

    def results_dataframe(self, *, only_anomalies: bool = False) -> pd.DataFrame:
        """
        Return row-level correction and anomaly diagnostics.

        Parameters
        ----------
        only_anomalies : bool, default=False
            If ``True``, return only anomalous samples. Otherwise return every
            sample. Information diagnostics use empirical probabilities from all samples.

        Returns
        -------
        pandas.DataFrame
            Per-sample detection and information-theoretic diagnostics.
        """
        check_is_fitted(self)

        table = pd.DataFrame(
            {
                "sample_index": np.arange(self.n_samples_in_, dtype=int),
                "sample_id": self.sample_ids_,
                "pattern_id": self.correction_state_,
                "observed_state": self.y_true_state_,
                "predicted_state": self.y_pred_state_,
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

    def analysis(self) -> dict[str, object]:
        """Return full-population numerical diagnostics without feature fitting."""
        check_is_fitted(self)
        true_counts = np.array(list(self._state_count_dict(self.y_true_state_).values()))
        pred_counts = np.array(list(self._state_count_dict(self.y_pred_state_).values()))
        pair_counts = np.array(list(self._pair_count_dict(self.y_pred_state_, self.y_true_state_).values()))
        target = self._code_length(true_counts)
        predicted = self._code_length(pred_counts)
        joint = self._code_length(pair_counts)
        return {
            "report_kind": "residuals", "scope": "all_samples", "task": self.task_,
            "n_samples": self.n_samples_in_, "n_features": self.n_features_in_,
            "n_anomalies": int(self.anomaly_mask_.sum()),
            "anomaly_rate": float(self.anomaly_mask_.mean()),
            "n_correction_patterns": int(len(pair_counts)),
            "inaccuracy": float(self.inaccuracy_analysis_["inaccuracy"]),
            "inaccuracy_analysis": deepcopy(self.inaccuracy_analysis_),
            "inaccuracy_encoding": "independent_numeric_bins" if self.task_ == "regression" else "categorical",
            "correction_encoding": "common_target_bins" if self.task_ == "regression" else "common_labels",
            "bin_edges": getattr(self, "bin_edges_", np.array([])).copy(),
            "states": self._state_labels(),
            "target_code_length_bits": target,
            "prediction_code_length_bits": predicted,
            "joint_code_length_bits": joint,
            "target_conditional_code_length_bits": max(0., joint - predicted),
            "prediction_conditional_code_length_bits": max(0., joint - target),
            "mean_local_correction_information": float(self.local_correction_information_.mean()),
            "mean_negative_local_explanatory_gain": float(self.negative_local_explanatory_gain_.mean()),
            "n_observed_joint_states": len(pair_counts),
            "mean_joint_occupancy": float(self.n_samples_in_ / len(pair_counts)),
            "singleton_fraction": float(np.sum(pair_counts == 1) / self.n_samples_in_),
        }

    def patterns_dataframe(self, *, only_anomalies=False, kind: AnomalyKind = "all"):
        """Count common-state transitions; local probabilities always use all samples."""
        table = self.results_dataframe()
        if only_anomalies:
            table = table.loc[self._mask_for_kind(kind)]
        elif kind != "all":
            raise ValueError("kind requires only_anomalies=True.")
        return table.groupby(
            ["pattern_id", "predicted_state", "observed_state", "is_anomaly"], sort=False
        ).agg(count=("sample_index", "size"),
              local_correction_information=("local_correction_information", "first"),
              negative_local_explanatory_gain=("negative_local_explanatory_gain", "first")
        ).reset_index().sort_values("count", ascending=False, kind="stable").reset_index(drop=True)

    def feature_analysis(self, *, kind: AnomalyKind = "all", max_features=None):
        """Lazily explain variation BETWEEN anomaly correction patterns."""
        check_is_fitted(self)
        self._mask_for_kind(kind)
        if max_features is not None and (isinstance(max_features, bool) or not isinstance(max_features, int) or max_features < 1):
            raise ValueError("max_features must be a positive integer or None.")
        key = (kind, max_features)
        if key not in self._explanations_:
            self._explanations_[key] = self._explain_features(kind=kind, max_features=max_features)
        return deepcopy(self._explanations_[key])

    def compressibility(self, *, kind: AnomalyKind = "all"):
        """Empirical predicted-state compression within the requested anomalies."""
        check_is_fitted(self)
        return self._compute_anomaly_compressibility(self._mask_for_kind(kind))

    def _state_labels(self):
        if self.task_ == "classification":
            _, labels = pd.factorize(np.concatenate([self.y_, self.y_pred_]), sort=False)
            return [{"state": i, "value": value} for i, value in enumerate(labels)]
        edges = self.bin_edges_
        states = [{"state": i, "lower": float(edges[i]), "upper": float(edges[i + 1]),
                   "upper_inclusive": i == self.n_bins_ - 1} for i in range(self.n_bins_)]
        if np.any(self.y_pred_state_ < 0):
            states.insert(0, {"state": -1, "upper": float(edges[0]), "outside": "below"})
        if np.any(self.y_pred_state_ >= self.n_bins_):
            states.append({"state": self.n_bins_, "lower": float(edges[-1]), "outside": "above"})
        return states

    def attribute_distribution(self, attribute, *, kind: AnomalyKind = "all"):
        """Return shared-bin counts for all observations and selected anomalies.

        All categorical levels are preserved. Numeric edges use the complete
        fitted attribute range, never a separately fitted anomaly histogram.
        """
        check_is_fitted(self)
        if attribute not in self.X_frame_.columns:
            raise ValueError(f"Unknown explanatory attribute: {attribute!r}.")
        mask = self._mask_for_kind(kind)
        values = self.X_frame_[attribute]
        if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
            data = values.to_numpy(dtype=float)
            if not np.isfinite(data).all():
                raise ValueError("Numeric attributes must be finite.")
            lo, hi = float(data.min()), float(data.max())
            count = 1 if lo == hi else min(12, _resolve_bins("auto", len(data)))
            edges = np.linspace(lo, hi, count + 1)
            codes = np.digitize(data, edges[1:-1])
            states = [{"lower": float(edges[i]), "upper": float(edges[i + 1])}
                      for i in range(count)]
            encoding = "numeric"
        else:
            codes, labels = pd.factorize(values, sort=False)
            states = [{"value": value} for value in labels]
            encoding = "categorical"
        totals = np.bincount(codes, minlength=len(states))
        anomaly_counts = np.bincount(codes[mask], minlength=len(states))
        bins = []
        for i, state in enumerate(states):
            bins.append({**state, "total_count": int(totals[i]),
                         "anomaly_count": int(anomaly_counts[i])})
        return {"attribute": attribute, "kind": encoding, "anomaly_kind": kind,
                "n_samples": self.n_samples_in_, "n_anomalies": int(mask.sum()), "bins": bins}

    @staticmethod
    def _code_length(counts):
        return max(0., float(-np.sum(counts * np.log2(counts / counts.sum()))))

    def _explain_features(
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

        if not self.n_features_in_:
            return {**base, "status": "no_attributes", "feature_analysis": self._empty_feature_analysis(),
                    "selected_features": [], "selection_path": pd.DataFrame()}

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
            "selected_feature_names": list(selection["selected_feature_names"]),
            "selected_features": list(selection["selected_features"]),
            "selection_path": selection["path"],
            "subset_analysis": selection["subset"],
        }


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
        values are retained for every observation; mismatch alone identifies anomalies.
        """
        true_state = np.asarray(self.y_true_state_, dtype=int)
        pred_state = np.asarray(self.y_pred_state_, dtype=int)
        n_samples = int(true_state.size)

        true_counts = self._state_count_dict(true_state)
        pred_counts = self._state_count_dict(pred_state)
        pair_counts = self._pair_count_dict(pred_state, true_state)

        correction = np.full(n_samples, np.nan, dtype=float)
        negative_gain = np.full(n_samples, np.nan, dtype=float)

        for i in range(n_samples):
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

    def _compute_anomaly_compressibility(self, mask) -> dict[str, object]:
        """
        Compute compressibility of anomalous predicted target states.

        The optimal length is the ideal Shannon code length obtained from the
        empirical distribution of predicted states within the anomaly subset.
        The uniform length uses the complete encoded target alphabet as its
        reference distribution.
        """
        predicted = np.asarray(
            self.y_pred_state_[mask],
            dtype=int,
        )
        n_anomalies = int(predicted.size)
        n_states = self._number_of_target_states()

        if n_anomalies == 0:
            return {
                "scope": "anomalous_predicted_states", "status": "no_anomalies",
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
            "scope": "anomalous_predicted_states", "status": "ok",
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
        bins = self.n_bins_

        lower = float(np.min(y_true))
        upper = float(np.max(y_true))

        if lower == upper:
            self.n_bins_ = 1
            edges = np.asarray([lower, upper], dtype=float)
            true_bins = np.zeros(self.n_samples_in_, dtype=int)
            pred_bins = np.zeros(self.n_samples_in_, dtype=int)
            pred_bins[y_pred < lower] = -1
            pred_bins[y_pred > upper] = 1
            return true_bins, pred_bins, edges

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

    @staticmethod
    def _prepare_X_y(X, y) -> tuple[np.ndarray, np.ndarray, list[object], pd.DataFrame]:
        """Validate input data while preserving feature names and DataFrame types."""
        X_checked, y_checked = check_X_y(X, y, dtype=None, ensure_2d=True)
        feature_names = _resolve_feature_names(X)
        if isinstance(X, pd.DataFrame):
            source_frame = X.copy().reset_index(drop=True)
        else:
            source_frame = pd.DataFrame(X_checked, columns=feature_names)

        return (
            X_checked,
            np.ravel(np.asarray(y_checked)),
            feature_names,
            source_frame,
        )

    def _resolve_task(self, y: np.ndarray) -> ResolvedTask:
        """Resolve the configured task from the target vector."""
        if self.task in get_args(ResolvedTask):
            return self.task

        return "regression" if _resolve_y_isnumeric(y) else "classification"

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
                "code_length_bits",
                "deficiency",
                "surplus",
                "miscoding",
                "is_reliable",
                "failure_reason",
                "resolved_n_bins",
                "n_samples",
                "n_observed_joint_states",
                "mean_joint_occupancy",
                "n_singleton_joint_states",
                "singleton_fraction",
            ]
        )
