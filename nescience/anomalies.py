"""
Model-relative anomaly detection with the Minimum Nescience Principle.

The detector implemented in this module identifies samples that are poorly
explained by a predictive model. In the terminology of the theory of nescience,
these are model-relative anomalies: they are observations whose target values do
not follow the regularities captured by the selected description.

The class is deliberately separated from model serialization and metric
implementation. It can use an already fitted model, fit a supplied estimator, use
the new nescience-based auto estimators when available, or work directly from a
precomputed prediction vector.

Public workflow
---------------
    detector = AnomalyDetector(task="auto")
    detector.fit(X, y)

    detector.summary()
    detector.anomalies()
    detector.anomaly_scores()
    detector.anomaly_table()
    detector.group_anomalies()
    detector.group_points("feature_name")

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
from sklearn.utils.validation import check_array, check_is_fitted

from .miscoding import Miscoding
from .utils import discretize_vector


try:  # Optional dependency on the new nescience-based classifier.
    from .classifier import NescienceClassifier
except Exception:  # pragma: no cover - exercised only when the module is absent.
    NescienceClassifier = None


try:  # Optional dependency on the new nescience-based regressor.
    from .regressor import NescienceRegressor
except Exception:  # pragma: no cover - exercised only when the module is absent.
    NescienceRegressor = None


Task = Literal["auto", "classification", "regression"]
ResolvedTask = Literal["classification", "regression"]
XType = Literal["auto", "numeric", "categorical"]
YType = Literal["auto", "numeric", "categorical"]
AnomalyRule = Literal[
    "auto",
    "misclassification",
    "probability_quantile",
    "residual_quantile",
    "standardized_residual",
    "bin_mismatch",
]
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

    anomaly_rule : {"auto", "misclassification", "probability_quantile",
                    "residual_quantile", "standardized_residual",
                    "bin_mismatch"}, default="auto"
        Rule used to turn prediction errors into anomaly labels.

        ``"auto"`` resolves to ``"misclassification"`` for classification and
        to ``"residual_quantile"`` for regression.

    anomaly_quantile : float, default=0.95
        Quantile used by ``"probability_quantile"`` and
        ``"residual_quantile"``. A value of ``0.95`` marks approximately the
        largest five percent of scores as anomalous.

    z_score_threshold : float, default=3.0
        Threshold used by ``"standardized_residual"``.

    n_bins : int or "auto", default="auto"
        Number of bins used by ``"bin_mismatch"`` and by ``Miscoding``.

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
        candidates. Larger values mean that two attributes are considered
        redundant only when they are very similar.

    random_state : int, optional
        Random seed used by KMeans and by the nescience-based auto estimators.

    Notes
    -----
    The detector does not claim that anomalous rows are invalid data. It only
    states that they are poorly explained by the model or prediction vector used
    during fitting.
    """

    _VALID_TASKS = ("auto", "classification", "regression")
    _VALID_X_TYPES = ("auto", "numeric", "categorical")
    _VALID_Y_TYPES = ("auto", "numeric", "categorical")
    _VALID_RULES = (
        "auto",
        "misclassification",
        "probability_quantile",
        "residual_quantile",
        "standardized_residual",
        "bin_mismatch",
    )
    _VALID_KINDS = ("all", "misclassified", "under_predicted", "over_predicted")

    def __init__(
        self,
        task: Task = "auto",
        X_type: XType = "auto",
        y_type: YType = "auto",
        anomaly_rule: AnomalyRule = "auto",
        anomaly_quantile: float = 0.95,
        z_score_threshold: float = 3.0,
        n_bins: int | Literal["auto"] = "auto",
        fit_model: bool = False,
        auto_model_kwargs: Mapping[str, Any] | None = None,
        min_cluster_fraction: float = 0.10,
        redundancy_threshold: float = 0.85,
        random_state: int | None = None,
    ):
        self.task = task
        self.X_type = X_type
        self.y_type = y_type
        self.anomaly_rule = anomaly_rule
        self.anomaly_quantile = anomaly_quantile
        self.z_score_threshold = z_score_threshold
        self.n_bins = n_bins
        self.fit_model = fit_model
        self.auto_model_kwargs = auto_model_kwargs
        self.min_cluster_fraction = min_cluster_fraction
        self.redundancy_threshold = redundancy_threshold
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
            ``predictions`` is also omitted, the detector uses the appropriate
            nescience-based auto estimator.

        predictions : array-like of shape (n_samples,), optional
            Precomputed predictions. Supplying predictions lets the detector be
            used independently of any specific model implementation.

        Returns
        -------
        self : AnomalyDetector
            Fitted detector.
        """
        self._validate_configuration()

        X_checked, y_checked, feature_names = self._validate_X_y(X, y)

        self.X_ = X_checked
        self.y_ = y_checked
        self.feature_names_in_ = np.asarray(feature_names, dtype=object)
        self.n_samples_in_, self.n_features_in_ = self.X_.shape
        self.task_ = self._resolve_task(self.y_)
        self.anomaly_rule_ = self._resolve_rule()
        self.model_ = None

        self.y_pred_ = self._resolve_predictions(
            X=self.X_,
            y=self.y_,
            model=model,
            predictions=predictions,
        )

        self._compute_anomaly_statistics()
        self.is_fitted_ = True
        return self

    def fit_predictions(self, X, y, predictions):
        """
        Fit the detector directly from a prediction vector.

        This is useful when predictions are produced outside ``mnplib`` or when
        anomaly analysis is performed after a separate modeling workflow.
        """
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
        """
        Fit the appropriate nescience-based auto estimator.

        Automatic model selection is delegated to ``NescienceClassifier`` or
        ``NescienceRegressor``. This class should not duplicate model-selection
        logic.
        """
        kwargs = dict(self.auto_model_kwargs or {})
        kwargs.setdefault("X_type", self.X_type)
        kwargs.setdefault("n_bins", self.n_bins)
        kwargs.setdefault("random_state", self.random_state)

        if self.task_ == "classification":
            if NescienceClassifier is None:
                raise ImportError(
                    "Automatic classification requires mnplib.classifier."
                    "NescienceClassifier."
                )
            model = NescienceClassifier(**kwargs)
            model.fit(X, y)
            return model

        if NescienceRegressor is None:
            raise ImportError(
                "Automatic regression requires mnplib.regressor."
                "NescienceRegressor."
            )

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
            Subset of anomalies to return. ``"under_predicted"`` and
            ``"over_predicted"`` are available only for regression tasks.
        """
        check_is_fitted(self)
        mask = self._mask_for_kind(kind)
        return np.flatnonzero(mask).astype(int)

    def anomaly_scores(self) -> np.ndarray:
        """Return one anomaly score per sample."""
        check_is_fitted(self)
        return self.anomaly_score_.copy()

    def anomaly_table(self, *, only_anomalies: bool = True) -> pd.DataFrame:
        """
        Return a row-level anomaly table.

        Parameters
        ----------
        only_anomalies : bool, default=True
            If ``True``, return only anomalous samples. If ``False``, return all
            samples with their scores and labels.
        """
        check_is_fitted(self)

        table = pd.DataFrame(
            {
                "sample_index": np.arange(self.n_samples_in_, dtype=int),
                "y_true": self.y_,
                "y_pred": self.y_pred_,
                "anomaly_score": self.anomaly_score_,
                "is_anomaly": self.anomaly_mask_,
                "anomaly_kind": self.anomaly_kind_,
            }
        )

        if self.task_ == "classification":
            table["correct"] = self.y_ == self.y_pred_
            if self.true_class_probability_ is not None:
                table["true_class_probability"] = self.true_class_probability_
        else:
            table["residual"] = self.residual_
            table["absolute_residual"] = self.absolute_residual_
            table["standardized_residual"] = self.standardized_residual_
            table["direction"] = self.direction_

        if only_anomalies:
            table = table[table["is_anomaly"]].copy()

        return table.reset_index(drop=True)

    def summary(self) -> dict[str, object]:
        """
        Return compact summary statistics for the fitted detector.
        """
        check_is_fitted(self)

        result: dict[str, object] = {
            "task": self.task_,
            "anomaly_rule": self.anomaly_rule_,
            "n_samples": int(self.n_samples_in_),
            "n_features": int(self.n_features_in_),
            "n_anomalies": int(np.sum(self.anomaly_mask_)),
            "anomaly_rate": float(np.mean(self.anomaly_mask_)),
            "model_type": None if self.model_ is None else type(self.model_).__name__,
            "threshold": None if self.anomaly_threshold_ is None else float(self.anomaly_threshold_),
        }

        if hasattr(self.model_, "nescience_score"):
            result["model_nescience"] = float(self.model_.nescience_score())

        if self.task_ == "classification":
            result["n_misclassified"] = int(np.sum(self.y_ != self.y_pred_))
        else:
            result.update(
                {
                    "mean_absolute_residual": float(np.mean(self.absolute_residual_)),
                    "median_absolute_residual": float(np.median(self.absolute_residual_)),
                    "n_under_predicted": int(np.sum(self._under_prediction_mask())),
                    "n_over_predicted": int(np.sum(self._over_prediction_mask())),
                }
            )

        return result

    def explain(self) -> dict[str, object]:
        """
        Return a structured explanation of the detector's current state.
        """
        check_is_fitted(self)

        summary = self.summary()
        recommendation = (
            "Inspect the anomaly table and group candidates. High-scoring rows "
            "are observations least explained by the fitted model or supplied "
            "prediction vector."
        )

        if self.task_ == "regression":
            recommendation += (
                " For regression, compare under-predicted and over-predicted "
                "subsets separately because they often correspond to different "
                "mechanisms."
            )

        return {
            "summary": summary,
            "feature_names": list(map(str, self.feature_names_in_)),
            "recommendation": recommendation,
        }

    # ------------------------------------------------------------------
    # Grouping anomalous samples
    # ------------------------------------------------------------------

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

        The method fits a two-cluster KMeans model to each candidate attribute
        subspace and ranks candidates by standardized inertia. Candidate
        subspaces are standardized before clustering so that attributes with
        different measurement scales are comparable.

        Parameters
        ----------
        dimensions : {1, 2}, default=1
            Number of attributes used for each candidate grouping.

        kind : {"all", "misclassified", "under_predicted", "over_predicted"},
               default="all"
            Anomaly subset to group.

        max_groups : int, optional
            Maximum number of group candidates to return.

        filter_balanced : bool, default=True
            If ``True``, discard candidates whose smaller cluster is too small.

        filter_redundant : bool, default=True
            If ``True``, greedily remove candidates using feature redundancy.

        filter_repeated_attributes : bool, default=True
            If ``True`` and ``dimensions=2``, avoid returning multiple groups
            that reuse the same attribute.

        min_cluster_fraction : float, optional
            Overrides the detector's configured ``min_cluster_fraction``.

        redundancy_threshold : float, optional
            Overrides the detector's configured ``redundancy_threshold``.

        Returns
        -------
        pandas.DataFrame
            Candidate anomaly-group descriptions sorted by increasing inertia.
        """
        check_is_fitted(self)

        if dimensions not in (1, 2):
            raise ValueError("dimensions must be 1 or 2.")

        anomaly_indices = self.anomalies(kind=kind)
        if anomaly_indices.size < 2:
            return self._empty_group_table(dimensions)

        rows = []
        for attributes in combinations(range(self.n_features_in_), dimensions):
            rows.append(self._evaluate_group_candidate(anomaly_indices, attributes))

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

        The returned table contains original sample indices, cluster labels,
        attribute values, observed and predicted targets, and anomaly scores.
        """
        check_is_fitted(self)

        index_1 = self._resolve_attribute(attribute_1)
        attributes = (index_1,)

        if attribute_2 is not None:
            attributes = (index_1, self._resolve_attribute(attribute_2))

        anomaly_indices = self.anomalies(kind=kind)
        if anomaly_indices.size < 2:
            return pd.DataFrame(
                columns=[
                    "sample_index",
                    "cluster",
                    "y_true",
                    "y_pred",
                    "anomaly_score",
                ]
            )

        labels, _ = self._cluster_anomaly_projection(anomaly_indices, attributes)

        result = pd.DataFrame(
            {
                "sample_index": anomaly_indices,
                "cluster": labels.astype(int),
                "y_true": self.y_[anomaly_indices],
                "y_pred": self.y_pred_[anomaly_indices],
                "anomaly_score": self.anomaly_score_[anomaly_indices],
            }
        )

        for index in attributes:
            result[str(self.feature_names_in_[index])] = self.X_[anomaly_indices, index]

        return result.sort_values(
            by=["cluster", "anomaly_score"],
            ascending=[True, False],
            ignore_index=True,
        )

    # ------------------------------------------------------------------
    # Anomaly scoring
    # ------------------------------------------------------------------

    def _compute_anomaly_statistics(self) -> None:
        """Compute task-specific anomaly scores and masks."""
        if self.task_ == "classification":
            self._compute_classification_anomalies()
        else:
            self._compute_regression_anomalies()

    def _compute_classification_anomalies(self) -> None:
        """Compute anomaly scores for a classification task."""
        misclassified = self.y_ != self.y_pred_
        probabilities = self._true_class_probabilities()

        if self.anomaly_rule_ == "misclassification":
            scores = misclassified.astype(float)
            threshold = None
            mask = misclassified

        elif self.anomaly_rule_ == "probability_quantile":
            if probabilities is None:
                raise ValueError(
                    "probability_quantile requires a model implementing "
                    "predict_proba(X)."
                )
            scores = 1.0 - probabilities
            threshold = float(np.quantile(scores, self.anomaly_quantile))
            mask = scores >= threshold

        else:
            raise ValueError(
                f"Rule {self.anomaly_rule_!r} is not valid for classification."
            )

        self.true_class_probability_ = None if probabilities is None else probabilities.copy()
        self.anomaly_score_ = np.asarray(scores, dtype=float)
        self.anomaly_threshold_ = threshold
        self.anomaly_mask_ = np.asarray(mask, dtype=bool)
        self.anomaly_kind_ = np.where(self.anomaly_mask_, "misclassified", "regular")

    def _compute_regression_anomalies(self) -> None:
        """Compute anomaly scores for a regression task."""
        y_true = self._numeric_vector(self.y_, name="y")
        y_pred = self._numeric_vector(self.y_pred_, name="predictions")

        residual = y_true - y_pred
        absolute = np.abs(residual)
        standardized = absolute / self._robust_scale(residual)

        if self.anomaly_rule_ == "residual_quantile":
            scores = absolute
            threshold = float(np.quantile(scores, self.anomaly_quantile))
            mask = scores >= threshold

        elif self.anomaly_rule_ == "standardized_residual":
            scores = standardized
            threshold = float(self.z_score_threshold)
            mask = scores >= threshold

        elif self.anomaly_rule_ == "bin_mismatch":
            true_bins = discretize_vector(y_true, n_bins=self.n_bins)
            pred_bins = discretize_vector(y_pred, n_bins=self.n_bins)
            scores = (true_bins != pred_bins).astype(float)
            threshold = None
            mask = true_bins != pred_bins

        else:
            raise ValueError(
                f"Rule {self.anomaly_rule_!r} is not valid for regression."
            )

        self.residual_ = residual
        self.absolute_residual_ = absolute
        self.standardized_residual_ = standardized
        self.direction_ = np.where(
            residual > 0,
            "under_predicted",
            np.where(residual < 0, "over_predicted", "exact"),
        )
        self.true_class_probability_ = None
        self.anomaly_score_ = np.asarray(scores, dtype=float)
        self.anomaly_threshold_ = threshold
        self.anomaly_mask_ = np.asarray(mask, dtype=bool)
        self.anomaly_kind_ = np.where(self.anomaly_mask_, self.direction_, "regular")

    def _true_class_probabilities(self) -> np.ndarray | None:
        """
        Return the probability assigned to each sample's observed class.

        If the model does not expose ``predict_proba``, return ``None`` and let
        the caller use a classification rule that does not require probabilities.
        """
        if self.model_ is None or not hasattr(self.model_, "predict_proba"):
            return None

        probabilities = np.asarray(self.model_.predict_proba(self.X_), dtype=float)
        classes = getattr(self.model_, "classes_", None)

        if classes is None:
            return None

        class_to_position = {
            label: position
            for position, label in enumerate(np.asarray(classes, dtype=object))
        }

        result = np.full(self.n_samples_in_, np.nan, dtype=float)
        for i, observed in enumerate(np.asarray(self.y_, dtype=object)):
            position = class_to_position.get(observed)
            if position is not None and position < probabilities.shape[1]:
                result[i] = probabilities[i, position]

        if np.any(np.isnan(result)):
            return None

        return result

    # ------------------------------------------------------------------
    # Grouping internals
    # ------------------------------------------------------------------

    def _evaluate_group_candidate(
        self,
        anomaly_indices: np.ndarray,
        attributes: tuple[int, ...],
    ) -> dict[str, object]:
        """Return the grouping diagnostics for one attribute subspace."""
        labels, inertia = self._cluster_anomaly_projection(anomaly_indices, attributes)

        n_cluster_0 = int(np.sum(labels == 0))
        n_cluster_1 = int(np.sum(labels == 1))
        total = int(labels.size)
        balance = min(n_cluster_0, n_cluster_1) / total if total else 0.0

        row: dict[str, object] = {
            "attribute_1": int(attributes[0]),
            "attribute_1_name": str(self.feature_names_in_[attributes[0]]),
            "attribute_2": None,
            "attribute_2_name": None,
            "dimensions": int(len(attributes)),
            "inertia": float(inertia),
            "cluster_0_size": n_cluster_0,
            "cluster_1_size": n_cluster_1,
            "balance": float(balance),
            "n_anomalies": total,
        }

        if len(attributes) == 2:
            row["attribute_2"] = int(attributes[1])
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
            n_clusters=2,
            random_state=self.random_state,
            n_init=10,
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
        """
        Greedily filter group candidates after sorting by inertia.

        Redundancy is computed once from the full representation and target.
        A candidate is discarded when any of its attributes is highly redundant
        with an attribute already accepted.
        """
        if table.empty:
            return table

        redundancy = None
        if filter_redundant:
            redundancy = self._feature_redundancy_matrix()

        accepted_rows = []
        accepted_attributes: set[int] = set()

        for _, row in table.sort_values(by=["inertia", "balance"], ascending=[True, False]).iterrows():
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

    # ------------------------------------------------------------------
    # Masks and attribute handling
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

        if self.anomaly_rule not in self._VALID_RULES:
            raise ValueError(
                f"Valid options for anomaly_rule are {self._VALID_RULES}. "
                f"Got {self.anomaly_rule!r}."
            )

        if not 0.0 < float(self.anomaly_quantile) < 1.0:
            raise ValueError("anomaly_quantile must lie in the open interval (0, 1).")

        if float(self.z_score_threshold) <= 0:
            raise ValueError("z_score_threshold must be positive.")

        if not 0.0 <= float(self.min_cluster_fraction) <= 0.5:
            raise ValueError("min_cluster_fraction must lie in [0, 0.5].")

        if not 0.0 <= float(self.redundancy_threshold) <= 1.0:
            raise ValueError("redundancy_threshold must lie in [0, 1].")

    @staticmethod
    def _validate_X_y(X, y) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Validate input data and preserve DataFrame feature names."""
        if isinstance(X, pd.DataFrame):
            if len(X) != len(y):
                raise ValueError(f"X and y have inconsistent lengths: {len(X)} != {len(y)}.")
            X_checked = X.to_numpy()
            y_checked = np.ravel(np.asarray(y))
            feature_names = [str(column) for column in X.columns]
            if X_checked.ndim != 2:
                raise ValueError("X must be two-dimensional.")
            if y_checked.size == 0:
                raise ValueError("y must not be empty.")
            return X_checked, y_checked, feature_names

        X_checked, y_checked = check_X_y(X, y, dtype=None, ensure_2d=True)
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

    def _resolve_rule(self) -> str:
        """Resolve the anomaly rule after task inference."""
        if self.anomaly_rule != "auto":
            return self.anomaly_rule

        return (
            "misclassification"
            if self.task_ == "classification"
            else "residual_quantile"
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
    def _robust_scale(values) -> float:
        """
        Return a robust scale estimate for residual standardization.

        The median absolute deviation is used first. If it degenerates to zero,
        the ordinary standard deviation is used. As a final fallback, the scale
        is set to one.
        """
        values = np.ravel(np.asarray(values, dtype=float))
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        scale = 1.4826 * mad

        if scale > 0.0 and np.isfinite(scale):
            return scale

        std = float(np.std(values))
        if std > 0.0 and np.isfinite(std):
            return std

        return 1.0


def anomaly_table(
    X,
    y,
    predictions,
    *,
    task: Task = "auto",
    anomaly_rule: AnomalyRule = "auto",
    **kwargs,
) -> pd.DataFrame:
    """
    Functional interface returning an anomaly table from predictions.
    """
    detector = AnomalyDetector(
        task=task,
        anomaly_rule=anomaly_rule,
        **kwargs,
    )
    detector.fit_predictions(X, y, predictions)
    return detector.anomaly_table()
