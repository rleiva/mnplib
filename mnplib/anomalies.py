"""
Model-relative anomaly detection with the Minimum Nescience Principle.

The detector implemented in this module identifies samples that are poorly
explained by a predictive model. In the terminology of the theory of nescience,
these are model-relative anomalies: observations whose target values do not
follow the regularities captured by the selected description.

For classification tasks, anomalous samples are the samples whose predicted
class differs from the observed class. For regression tasks, anomalous samples
are the samples whose observed and predicted target values fall in different
discretization bins.

The detector can use an already fitted model, fit a supplied estimator, use the
nescience-based auto estimators, or work directly from a precomputed prediction
vector.

@author:    Rafael Garcia Leiva
@mail:      rgarcialeiva@gmail.com
@copyright: GNU GPLv3
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations
from typing import Any, Literal

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, clone
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted

from .classifier import NescienceClassifier
from .miscoding import Miscoding
from .regressor import NescienceRegressor
from .utils import discretize_vector


Task = Literal["auto", "classification", "regression"]
ResolvedTask = Literal["classification", "regression"]
XType = Literal["auto", "numeric", "categorical"]
YType = Literal["auto", "numeric", "categorical"]
BinSpec = int | Literal["auto", "adaptive"]
AnomalyKind = Literal["all", "misclassified", "under_predicted", "over_predicted"]


class AnomalyDetector(BaseEstimator):
    """
    Detect and analyze model-relative anomalies.

    Parameters
    ----------
    task : {"auto", "classification", "regression"}, default="auto"
        Predictive task. If ``"auto"``, the target type is inferred from ``y``.

    X_type : {"auto", "numeric", "categorical"}, default="auto"
        Feature encoding type passed to the nescience-based auto estimators and
        to ``Miscoding`` when redundancy is used for grouping.

    y_type : {"auto", "numeric", "categorical"}, default="auto"
        Target encoding type passed to the nescience-based auto estimators and
        to ``Miscoding``.

    n_bins : int, "auto", or "adaptive", default="auto"
        Number of bins used to compare observed and predicted target values in
        regression tasks, and by ``Miscoding`` when redundancy is used for
        grouping.

    fit_model : bool, default=False
        If ``True`` and a model is supplied to ``fit``, clone and fit that model
        on ``(X, y)`` before producing predictions. If ``False``, the supplied
        model is assumed to be already fitted.

    auto_model_kwargs : mapping, optional
        Additional keyword arguments passed to ``NescienceClassifier`` or
        ``NescienceRegressor`` when no model and no predictions are supplied.

    min_cluster_fraction : float, default=0.10
        Minimum fraction of anomalies required in the smaller KMeans cluster
        when filtering anomaly group candidates.

    redundancy_threshold : float, default=0.85
        Feature-redundancy threshold used when filtering anomaly group
        candidates.

    random_state : int, optional
        Random seed used by KMeans and by the nescience-based auto estimators.
    """

    _VALID_TASKS   = ("auto", "classification", "regression")
    _VALID_X_TYPES = ("auto", "numeric", "categorical")
    _VALID_Y_TYPES = ("auto", "numeric", "categorical")
    _VALID_KINDS   = ("all", "misclassified", "under_predicted", "over_predicted")

    def __init__(
        self,
        task: Task = "auto",
        X_type: XType = "auto",
        y_type: YType = "auto",
        n_bins: BinSpec = "auto",
        fit_model: bool = False,
        auto_model_kwargs: Mapping[str, Any] | None = None,
        min_cluster_fraction: float = 0.10,
        redundancy_threshold: float = 0.85,
        random_state: int | None = None,
    ):
        self.task         = task
        self.X_type       = X_type
        self.y_type       = y_type
        self.n_bins       = n_bins
        self.fit_model    = fit_model
        self.auto_model_kwargs    = auto_model_kwargs
        self.min_cluster_fraction = min_cluster_fraction
        self.redundancy_threshold = redundancy_threshold
        self.random_state = random_state

    #
    # Fitting
    #

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
            ``predictions`` is also omitted, the detector uses the appropriate
            nescience-based auto estimator.

        predictions : array-like of shape (n_samples,), optional
            Precomputed predictions.

        Returns
        -------
        self : AnomalyDetector
            Fitted detector.
        """
        self._validate_configuration()
        X_checked, y_checked, feature_names = self._prepare_X_y(X, y)

        self.X_ = X_checked
        self.y_ = y_checked
        self.feature_names_in_ = np.asarray(feature_names, dtype=object)
        self.n_samples_in_, self.n_features_in_ = self.X_.shape
        self.task_ = self._resolve_task(self.y_)
        self.model_ = None

        # Predictions can be provided directly by the user,
        # computed using the fitted model provided by the user,
        # or computed using a fitted model provided by the AutoML
        self.y_pred_ = self._resolve_predictions(
            X=self.X_,
            y=self.y_,
            model=model,
            predictions=predictions,
        )

        self._compute_anomalies()
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
            if NescienceClassifier is None:
                raise ImportError(
                    "Automatic classification requires "
                    "mnplib.classifier.NescienceClassifier."
                )
            model = NescienceClassifier(**kwargs)
            model.fit(X, y)
            return model

        if NescienceRegressor is None:
            raise ImportError(
                "Automatic regression requires mnplib.regressor.NescienceRegressor."
            )

        model = NescienceRegressor(**kwargs)
        model.fit(X, y)

        return model

    #
    # Public anomaly outputs
    #

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
        mask = self._mask_for_kind(kind)

        return np.flatnonzero(mask).astype(int)

    def anomaly_table(self, *, only_anomalies: bool = True) -> pd.DataFrame:
        """
        Return a row-level anomaly table.

        Parameters
        ----------
        only_anomalies : bool, default=True
            If ``True``, return only anomalous samples. If ``False``, return all
            samples with their anomaly labels and task-specific diagnostics.
        """
        check_is_fitted(self)

        table = pd.DataFrame({
            "sample_index" : np.arange(self.n_samples_in_, dtype=int),
            "y_true"       : self.y_,
            "y_pred"       : self.y_pred_,
            "is_anomaly"   : self.anomaly_mask_,
            "anomaly_kind" : self.anomaly_kind_,
        })

        if self.task_ == "classification":
            table["correct"] = self.y_ == self.y_pred_
        else:
            table["residual"]   = self.residual_
            table["direction"]  = self.direction_
            table["y_true_bin"] = self.y_true_bin_
            table["y_pred_bin"] = self.y_pred_bin_
            table["bin_match"]  = self.y_true_bin_ == self.y_pred_bin_

        if only_anomalies:
            table = table[table["is_anomaly"]].copy()

        return table.reset_index(drop=True)

    def summary(self) -> dict[str, object]:
        """Return compact summary statistics for the fitted detector."""
        check_is_fitted(self)

        result: dict[str, object] = {
            "task"         : self.task_,
            "n_samples"    : int(self.n_samples_in_),
            "n_features"   : int(self.n_features_in_),
            "n_anomalies"  : int(np.sum(self.anomaly_mask_)),
            "anomaly_rate" : float(np.mean(self.anomaly_mask_)),
            "model_type"   : None if self.model_ is None else type(self.model_).__name__,
        }

        if hasattr(self.model_, "nescience_score"):
            result["model_nescience"] = float(self.model_.nescience_score())

        if self.task_ == "classification":
            result["n_misclassified"] = int(np.sum(self.y_ != self.y_pred_))
        else:
            result.update(
                {
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

    #
    # Grouping anomalous samples
    #

    def group_anomalies(
        self,
        *,
        dimensions: Literal[1, 2] = 1,
        kind: AnomalyKind = "all",
        max_groups: int | None = None,
        filter_balanced: bool = True,
        filter_redundant: bool = True,
        filter_repeated_attributes: bool = True,
        min_cluster_fraction: float | None = None,
        redundancy_threshold: float | None = None,
    ) -> pd.DataFrame:
        """
        Rank simple one- or two-dimensional clusterings of anomalous samples.

        Candidate subspaces are standardized before clustering so that
        attributes with different measurement scales are comparable.
        """
        check_is_fitted(self)

        if dimensions not in (1, 2):
            raise ValueError("dimensions must be 1 or 2.")

        anomaly_indices = self.anomalies(kind=kind)
        if anomaly_indices.size < 2:
            return self._empty_group_table(dimensions)

        rows = [
            self._evaluate_group_candidate(anomaly_indices, attributes)
            for attributes in combinations(range(self.n_features_in_), dimensions)
        ]

        table = pd.DataFrame(rows).sort_values(
            by=["inertia", "balance", "attribute_1"],
            ascending=[True, False, True],
            ignore_index=True,
        )

        if filter_balanced:
            min_fraction = (
                self.min_cluster_fraction
                if min_cluster_fraction is None
                else float(min_cluster_fraction)
            )
            table = table[table["balance"] >= min_fraction].copy()

        if filter_repeated_attributes or filter_redundant:
            threshold = (
                self.redundancy_threshold
                if redundancy_threshold is None
                else float(redundancy_threshold)
            )
            table = self._filter_group_table(
                table,
                filter_repeated_attributes=filter_repeated_attributes,
                filter_redundant=filter_redundant,
                redundancy_threshold=threshold,
            )

        if max_groups is not None:
            table = table.head(int(max_groups)).copy()

        return table.reset_index(drop=True)

    def group_points(
        self,
        attribute_1,
        attribute_2=None,
        *,
        kind: AnomalyKind = "all",
    ) -> pd.DataFrame:
        """
        Return clustered anomalous samples for a chosen one- or two-attribute view.
        """
        check_is_fitted(self)

        index_1 = self._resolve_attribute(attribute_1)
        attributes = (index_1,)

        if attribute_2 is not None:
            attributes = (index_1, self._resolve_attribute(attribute_2))

        anomaly_indices = self.anomalies(kind=kind)
        if anomaly_indices.size < 2:
            columns = ["sample_index", "cluster", "y_true", "y_pred"]
            columns.extend(str(self.feature_names_in_[index]) for index in attributes)
            return pd.DataFrame(columns=columns)

        labels, _ = self._cluster_anomaly_projection(anomaly_indices, attributes)

        result = pd.DataFrame({
            "sample_index" : anomaly_indices,
            "cluster"      : labels.astype(int),
            "y_true"       : self.y_[anomaly_indices],
            "y_pred"       : self.y_pred_[anomaly_indices],
        })

        for index in attributes:
            result[str(self.feature_names_in_[index])] = self.X_[anomaly_indices, index]

        return result.sort_values(
            by           = ["cluster", "sample_index"],
            ascending    = [True, True],
            ignore_index = True
        )

    #
    # Compute anomalies
    #

    def _compute_anomalies(self) -> None:
        """Compute task-specific anomaly masks."""
        if self.task_ == "classification":
            self._compute_classification_anomalies()
        else:
            self._compute_regression_anomalies()

    def _compute_classification_anomalies(self) -> None:
        """Classification anomalies are samples whose predicted class
            differs from the observed class."""
        misclassified = self.y_ != self.y_pred_

        self.anomaly_mask_ = np.asarray(misclassified, dtype=bool)
        self.anomaly_kind_ = np.where(self.anomaly_mask_, "misclassified", "regular")

    def _compute_regression_anomalies(self) -> None:
        """Regression anomalies are samples whose observed and predicted
           target values fall in different discretized bins."""
        y_true = self._numeric_vector(self.y_, name="y")
        y_pred = self._numeric_vector(self.y_pred_, name="predictions")

        true_bins = discretize_vector(y_true, n_bins=self.n_bins)
        pred_bins = discretize_vector(y_pred, n_bins=self.n_bins)
        residual = y_true - y_pred

        self.residual_ = residual
        self.direction_ = np.where(
            residual > 0,
            "under_predicted",
            np.where(residual < 0, "over_predicted", "exact"),
        )
        self.y_true_bin_ = np.asarray(true_bins)
        self.y_pred_bin_ = np.asarray(pred_bins)
        self.anomaly_mask_ = np.asarray(true_bins != pred_bins, dtype=bool)
        self.anomaly_kind_ = np.where(self.anomaly_mask_, self.direction_, "regular")

    #
    # Grouping internals
    #

    def _evaluate_group_candidate(
        self,
        anomaly_indices: np.ndarray,
        attributes: tuple[int, ...],
    ) -> dict[str, object]:
        """Return grouping diagnostics for one attribute subspace."""
        labels, inertia = self._cluster_anomaly_projection(anomaly_indices, attributes)

        n_cluster_0 = int(np.sum(labels == 0))
        n_cluster_1 = int(np.sum(labels == 1))
        total = int(labels.size)
        balance = min(n_cluster_0, n_cluster_1) / total if total else 0.0

        row: dict[str, object] = {
            "attribute_1"      : int(attributes[0]),
            "attribute_1_name" : str(self.feature_names_in_[attributes[0]]),
            "attribute_2"      : None,
            "attribute_2_name" : None,
            "dimensions"       : int(len(attributes)),
            "inertia"          : float(inertia),
            "cluster_0_size"   : n_cluster_0,
            "cluster_1_size"   : n_cluster_1,
            "balance"          : float(balance),
            "n_anomalies"      : total,
        }

        if len(attributes) == 2:
            row["attribute_2"]      = int(attributes[1])
            row["attribute_2_name"] = str(self.feature_names_in_[attributes[1]])

        return row

    def _cluster_anomaly_projection(
        self,
        anomaly_indices: np.ndarray,
        attributes: tuple[int, ...],
    ) -> tuple[np.ndarray, float]:
        """Cluster an anomalous subspace after standardizing the projection."""
        raw = self.X_[anomaly_indices[:, None], np.asarray(attributes, dtype=int)]
        raw = np.asarray(raw, dtype=float)
        raw = raw.reshape(len(anomaly_indices), len(attributes))

        scaled = StandardScaler().fit_transform(raw)

        model = KMeans(
            n_clusters   = 2,
            random_state = self.random_state,
            n_init       = 10,
        )
        labels = model.fit_predict(scaled)

        return labels.astype(int), float(model.inertia_)

    def _filter_group_table(
        self,
        table: pd.DataFrame,
        *,
        filter_repeated_attributes: bool,
        filter_redundant: bool,
        redundancy_threshold: float,
    ) -> pd.DataFrame:
        """Greedily filter group candidates after sorting by inertia."""
        if table.empty:
            return table

        redundancy = self._feature_redundancy_matrix() if filter_redundant else None
        accepted_rows = []
        accepted_attributes: set[int] = set()

        for _, row in table.sort_values(
            by=["inertia", "balance"],
            ascending=[True, False],
        ).iterrows():
            attributes = self._row_attributes(row)

            if filter_repeated_attributes and accepted_attributes.intersection(attributes):
                continue

            if redundancy is not None and self._is_redundant_with_accepted(
                attributes,
                accepted_attributes,
                redundancy,
                redundancy_threshold,
            ):
                continue

            accepted_rows.append(row.to_dict())
            accepted_attributes.update(attributes)

        return pd.DataFrame(accepted_rows, columns=table.columns)

    def _feature_redundancy_matrix(self) -> np.ndarray:
        """Return the latest Miscoding feature-redundancy matrix."""
        metric = Miscoding(
            X_type=self.X_type,
            y_type=self.y_type,
            n_bins=self.n_bins,
        )
        metric.fit(pd.DataFrame(self.X_, columns=self.feature_names_in_), self.y_)

        return metric.feature_redundancy().to_numpy(dtype=float)

    @staticmethod
    def _is_redundant_with_accepted(
        attributes: tuple[int, ...],
        accepted: set[int],
        redundancy: np.ndarray,
        threshold: float,
    ) -> bool:
        """Return True when a candidate is redundant with accepted attributes."""
        for attribute in attributes:
            for accepted_attribute in accepted:
                if redundancy[attribute, accepted_attribute] >= threshold:
                    return True

        return False

    @staticmethod
    def _row_attributes(row: pd.Series) -> tuple[int, ...]:
        """Extract attribute indices from a group-candidate row."""
        attributes = [int(row["attribute_1"])]
        if pd.notna(row.get("attribute_2")):
            attributes.append(int(row["attribute_2"]))

        return tuple(attributes)

    @staticmethod
    def _empty_group_table(dimensions: int) -> pd.DataFrame:
        """Return an empty group-candidate table with stable columns."""
        return pd.DataFrame(
            columns=[
                "attribute_1",
                "attribute_1_name",
                "attribute_2",
                "attribute_2_name",
                "dimensions",
                "inertia",
                "cluster_0_size",
                "cluster_1_size",
                "balance",
                "n_anomalies",
            ]
        )

    #
    # Masks and attribute handling
    #

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

    def _resolve_attribute(self, attribute) -> int:
        """Resolve an attribute name or index into a validated column index."""
        if isinstance(attribute, str):
            names = list(map(str, self.feature_names_in_))
            if attribute not in names:
                raise ValueError(f"Unknown attribute {attribute!r}.")
            return names.index(attribute)

        index = int(attribute)
        if index < 0 or index >= self.n_features_in_:
            raise ValueError(
                f"attribute index {index} is outside the valid range "
                f"[0, {self.n_features_in_ - 1}]."
            )

        return index

    #
    # Validation and configuration
    #

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

        if not 0.0 <= float(self.min_cluster_fraction) <= 0.5:
            raise ValueError("min_cluster_fraction must lie in [0, 0.5].")

        if not 0.0 <= float(self.redundancy_threshold) <= 1.0:
            raise ValueError("redundancy_threshold must lie in [0, 1].")

    @staticmethod
    def _prepare_X_y(X, y) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Validate input data and return stable feature names."""
        if isinstance(X, pd.DataFrame):
            feature_names = [str(column) for column in X.columns]
        else:
            feature_names = None

        X_checked, y_checked = check_X_y(X, y, dtype=None, ensure_2d=True)

        if feature_names is None:
            feature_names = [f"x{i}" for i in range(X_checked.shape[1])]

        return X_checked, np.ravel(np.asarray(y_checked)), feature_names

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
            "multiclass, and continuous."
            .format(target_type)
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


def anomaly_table(X, y, predictions, *, task: Task = "auto", **kwargs) -> pd.DataFrame:
    """Return an anomaly table from a prediction vector."""
    detector = AnomalyDetector(
        task=task,
        **kwargs,
    )
    detector.fit_predictions(X, y, predictions)

    return detector.anomaly_table()