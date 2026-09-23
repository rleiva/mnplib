"""
Labeled miscoding based on empirical code lengths.

This module provides the :class:`Miscoding` estimator, a scikit-learn-compatible
utility for measuring how well a set of features represents a target variable.

The estimator provides feature-level diagnostics and subset-level diagnostics.
Feature-level diagnostics are computed from empirical code lengths. Subset-level
diagnostics are computed from empirical joint code lengths for the selected
feature subset and target.

The code-length estimates are computed through the stateless empirical
distribution utilities.

@author:    Rafael Garcia Leiva
@mail:      rgarcialeiva@gmail.com
"""

from __future__ import annotations

from typing import Literal, get_args

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted

from ._types import BinSpec, XType, YType
from .utils import _resolve_bins, _validate_vector, empirical_distribution_array
from .models.inputs import model_artifacts
from ._diagnostics import warn_nan_model


RankingCriterion = Literal["deficiency", "miscoding"]

_SPARSE_JOINT_FAILURE = "joint_distribution_too_sparse"


class Miscoding(BaseEstimator):
    """
    Analyze supervised miscoding through deficiency and surplus.

    For each feature ``X_j`` and target ``Y``, the estimator computes

        deficiency_j = K(Y | X_j) / K(Y)
        surplus_j    = K(X_j | Y) / K(X_j)
        miscoding_j  = max(deficiency_j, surplus_j)

    For a subset of features ``S``, the estimator computes empirical subset
    quantities:

        deficiency(S) = (K(X_S, Y) - K(X_S)) / K(Y)
        surplus(S)    = (K(X_S, Y) - K(Y)) / K(X_S)
        miscoding(S)  = max(deficiency(S), surplus(S))

    Numeric variables use uniform discretization. ``n_bins="auto"`` uses
    ``max(2, floor(2 * n_samples**(1/3)))``. ``n_bins="adaptive"`` uses
    ``max(2, floor(2 * n_samples**(1/3) / log2(|S| + 1)))`` for empirical
    subset quantities, so one-feature subsets match ``"auto"`` and larger
    subsets use coarser bins to reduce joint sparsity.

    ``select_features()`` greedily selects a strict subset by requiring subset
    miscoding improvement. ``rank_features()`` greedily orders features for
    model construction while reliable candidate extensions remain.
    """

    _VALID_X_TYPES = get_args(XType)
    _VALID_Y_TYPES = get_args(YType)
    _VALID_RANKING_CRITERIA = get_args(RankingCriterion)

    def __init__(
        self,
        X_type: XType   = "auto",
        y_type: YType   = "auto",
        n_bins: BinSpec = "adaptive",
    ):
        """
        Initialize the estimator.

        Parameters
        ----------
        X_type : {"auto", "numeric", "categorical"}, default="auto"
            Encoding strategy for the feature variables.

        y_type : {"auto", "numeric", "categorical"}, default="auto"
            Encoding strategy for the target variable.

        n_bins : int, "auto", or "adaptive", default="adaptive"
            Number of uniform bins used to discretize numeric variables.
            Integer counts must be at least two and are validated during fit.
            ``"auto"`` uses ``max(2, floor(2 * n_samples**(1/3)))``.
            ``"adaptive"`` matches ``"auto"`` for feature-level diagnostics
            and uses ``max(2, floor(2 * n_samples**(1/3) / log2(|S| + 1)))``
            for empirical subset diagnostics.

        """
        self._validate_init(
            X_type=X_type,
            y_type=y_type,
        )

        self.X_type = X_type
        self.y_type = y_type
        self.n_bins = n_bins

    def fit(self, X, y):
        """
        Estimate feature-level code lengths, feature miscoding values, and
        pairwise feature redundancies.

        Parameters
        ----------
        X : array-like or pandas.DataFrame of shape (n_samples, n_features)
            Feature matrix. pandas DataFrames preserve column names and allow
            automatic per-column type inference.

        y : array-like of shape (n_samples,)
            Target vector.

        Returns
        -------
        self : Miscoding
            Fitted estimator.
        """
        if y is None:
            raise ValueError("Miscoding.fit requires a target vector y.")

        self.X_, self.y_ = self._validate_X_y(X, y)
        self._model_X_ = X
        self.n_samples_in_, self.n_features_in_ = self.X_.shape
        self.X_isnumeric_ = self._infer_X_isnumeric(X, self.X_)
        self.y_isnumeric_ = self._infer_y_isnumeric(self.y_)

        self._code_length_cache_ = {}
        self._empirical_summary_cache_ = {}
        self.target_code_length_ = self._code_length_for_indices(y_included=True)

        self.feature_code_lengths_ = np.array(
            [
                self._code_length_for_indices(features=[j])
                for j in range(self.n_features_in_)
            ],
            dtype=float,
        )

        target_given_feature = np.array(
            [
                self._conditional_target_length([j])
                for j in range(self.n_features_in_)
            ],
            dtype=float,
        )

        feature_given_target = np.array(
            [
                self._conditional_feature_length(j, selected=[], y_included=True)
                for j in range(self.n_features_in_)
            ],
            dtype=float,
        )

        self.deficiency_ = np.clip(
            self._safe_divide(
                target_given_feature,
                self.target_code_length_,
                default=0.0,
            ),
            0.0,
            1.0,
        )

        self.surplus_ = np.clip(
            self._safe_divide(
                feature_given_target,
                self.feature_code_lengths_,
                default=0.0,
            ),
            0.0,
            1.0,
        )

        self.miscoding_ = np.maximum(self.deficiency_, self.surplus_)
        self.redundancy_ = self._feature_redundancy_matrix()

        self.is_fitted_ = True
        return self

    #
    # Public feature-level diagnostics
    #

    def deficiency_feature(self, feature=None):
        """
        Return deficiency for one feature or all features.

        Parameters
        ----------
        feature : int, str, or None, default=None
            Feature index or column name. None requests all features.

        Returns
        -------
        float or numpy.ndarray of shape (n_features,)
            Values of ``K(Y | X_j) / K(Y)`` for each feature.
        """
        check_is_fitted(self)
        return self._feature_value(self.deficiency_, feature)

    def surplus_feature(self, feature=None):
        """
        Return surplus for one feature or all features.

        Parameters
        ----------
        feature : int, str, or None, default=None
            Feature index or column name. None requests all features.

        Returns
        -------
        float or numpy.ndarray of shape (n_features,)
            Values of ``K(X_j | Y) / K(X_j)`` for each feature.
        """
        check_is_fitted(self)
        return self._feature_value(self.surplus_, feature)

    def miscoding_feature(self, feature=None):
        """
        Return miscoding for one feature or all features.

        Parameters
        ----------
        feature : int, str, or None, default=None
            Feature index or column name. None requests all features.

        Returns
        -------
        float or numpy.ndarray of shape (n_features,)
            Values of ``max(deficiency, surplus)`` for each feature.
        """
        check_is_fitted(self)
        return self._feature_value(self.miscoding_, feature)

    def _feature_value(self, values, feature):
        """Resolve a feature name or integer position, or return all values."""
        if feature is None:
            return values.copy()
        if isinstance(feature, str):
            matches = np.flatnonzero(self.feature_names_in_ == feature)
            if len(matches) != 1:
                raise ValueError(f"Feature name {feature!r} must identify exactly one column.")
            feature = int(matches[0])
        if isinstance(feature, (bool, np.bool_)) or not isinstance(feature, (int, np.integer)):
            raise ValueError("feature must be an integer index or column name.")
        if not 0 <= feature < self.n_features_in_:
            raise ValueError("feature is outside the fitted feature dimension.")
        return float(values[feature])

    def redundancy_matrix(self) -> pd.DataFrame:
        """
        Return the pairwise redundancy matrix between features.

        Returns
        -------
        pandas.DataFrame
            Square matrix indexed and labeled by feature name. Values close to
            one indicate highly redundant features. Values close to zero
            indicate little shared information according to the empirical
            code-length approximation.
        """
        check_is_fitted(self)
        return pd.DataFrame(
            self.redundancy_.copy(),
            index=self.feature_names_in_,
            columns=self.feature_names_in_,
        )

    def feature_analysis(self) -> pd.DataFrame:
        """
        Return feature-level diagnostics in tabular form.

        Returns
        -------
        pandas.DataFrame
            Table with one row per feature and the columns ``feature_index``,
            ``feature_name``, ``is_numeric``, ``code_length_bits``, ``deficiency``,
            ``surplus``, and ``miscoding``. Rows are sorted from lowest to
            highest miscoding.
        """
        check_is_fitted(self)

        table = pd.DataFrame(
            {
                "feature_index": np.arange(self.n_features_in_),
                "feature_name": self.feature_names_in_,
                "is_numeric": self.X_isnumeric_,
                "code_length_bits": self.feature_code_lengths_,
                "deficiency": self.deficiency_,
                "surplus": self.surplus_,
                "miscoding": self.miscoding_,
            }
        )
        return table.sort_values(
            by=["miscoding", "deficiency", "surplus"],
            ascending=[True, True, True],
            ignore_index=True,
        )

    #
    # Subset-level diagnostics
    #

    def miscoding_subset(self, subset) -> float:
        """Return subset miscoding, or NaN when joint counts are unreliable."""
        return float(self.subset_analysis(subset)["miscoding"])

    def deficiency_subset(self, subset) -> float:
        """Return deficiency for integer feature indices or a Boolean mask."""
        return float(self.subset_analysis(subset)["deficiency"])

    def surplus_subset(self, subset) -> float:
        """Return surplus for integer feature indices or a Boolean mask."""
        return float(self.subset_analysis(subset)["surplus"])

    def miscoding_model(self, model, *, X=None, feature_names=None, feature_indices=None) -> float:
        """Evaluate the feature subset effectively used by a fitted model.

        X defaults to fitted evaluation data. Explicit X contains estimator
        input columns; feature_indices maps those columns to fitted features.
        Unreliable subsets return NaN with a RuntimeWarning. Model analysis
        returns diagnostics without issuing this warning.
        """
        report = self.model_analysis(model, X=X, feature_names=feature_names,
                                     feature_indices=feature_indices)
        value = float(report["miscoding"])
        if np.isnan(value):
            warn_nan_model("miscoding_model", report)
        return value

    def model_analysis(self, model, *, X=None, feature_names=None,
                       feature_indices=None) -> dict[str, object]:
        """Analyze the effective feature subset of a canonically serialized model.

        The report has the same fields as ``subset_analysis()``. Explicit X is
        in estimator coordinates; feature_indices maps those columns to the
        fitted feature space. The model is evaluated without refitting it.
        """
        artifacts = model_artifacts(self, model, X=X, feature_names=feature_names,
                                    feature_indices=feature_indices)
        return self.subset_analysis(artifacts.subset)

    def subset_analysis(self, subset) -> dict[str, object]:
        """
        Return detailed empirical diagnostics for a feature subset.

        Parameters
        ----------
        subset : array-like
            Boolean mask of selected features or list of selected feature indices.

        Returns
        -------
        dict
            Dictionary containing deficiency, surplus, miscoding, selected
            feature metadata, redundancy weights, and feature weights.
            Reliability diagnostics include ``resolved_n_bins``, the numeric
            bin count for the subset, or None when no numeric discretization is
            applied, including an empty subset. Categorical variables retain
            their observed categories without binning.
        """
        check_is_fitted(self)
        return self._subset_measures(subset)

    #
    # Feature selection and ordering
    #

    def select_features(self, *, max_features: int | None = None,
                        min_improvement: float = 0.0, return_details: bool = False,
    ):
        """
        Select features by strict subset-miscoding improvement.

        At each step, the method evaluates every candidate feature not yet
        selected and adds the feature that produces the lowest subset miscoding.
        Selection stops when the best candidate does not reduce subset
        miscoding by more than ``min_improvement``.

        Parameters
        ----------
        max_features : int, optional
            Maximum number of features to select. If omitted, all features are
            eligible.

        min_improvement : float, default=0.0
            Minimum reduction in subset miscoding required to accept a feature.

        return_details : bool, default=False
            If ``False``, return a Boolean selection mask. If ``True``, return a
            dictionary with the mask, selected indices, selected names, selection
            path, and final subset diagnostics.

        Returns
        -------
        numpy.ndarray or dict
            Binary selection mask by default, or detailed selection output when
            ``return_details=True``.
        """
        check_is_fitted(self)

        improvement_threshold = float(min_improvement)
        if not np.isfinite(improvement_threshold) or improvement_threshold < 0:
            raise ValueError("min_improvement must be non-negative.")

        max_features = self._validate_max_features(max_features)

        selected : list[int] = []
        path     : list[dict[str, object]] = []
        current  = self._subset_measures(selected)

        while len(selected) < max_features:

            candidates = self._sort_candidates(
                self._candidate_extensions(selected, current),
                criterion="miscoding",
            )
            if candidates.empty:
                break

            best = candidates.iloc[0]
            if not bool(best["is_reliable"]):
                break

            improvement = float(best["miscoding_improvement"])

            if (not np.isfinite(improvement)) or improvement <= improvement_threshold:
                break

            feature = int(best["feature_index"])
            selected.append(feature)
            current = self._subset_measures(selected)

            path.append(
                {
                    "step": len(path) + 1,
                    "feature_index": feature,
                    "feature_name": str(self.feature_names_in_[feature]),
                    "deficiency": float(current["deficiency"]),
                    "surplus": float(current["surplus"]),
                    "miscoding": float(current["miscoding"]),
                    "is_reliable": bool(current["is_reliable"]),
                    "failure_reason": current["failure_reason"],
                    "resolved_n_bins": current["resolved_n_bins"],
                    "n_samples": current["n_samples"],
                    "n_observed_joint_states": current["n_observed_joint_states"],
                    "mean_joint_occupancy": current["mean_joint_occupancy"],
                    "n_singleton_joint_states": current["n_singleton_joint_states"],
                    "singleton_fraction": current["singleton_fraction"],
                    "deficiency_improvement": float(best["deficiency_improvement"]),
                    "surplus_change": float(best["surplus_change"]),
                    "miscoding_improvement": improvement,
                    "improvement": improvement,
                    "selected_features": tuple(selected),
                    "selected_feature_names": tuple(
                        str(self.feature_names_in_[j]) for j in selected
                    ),
                }
            )

        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[selected] = True

        if not return_details:
            return mask

        return {
            "mask"                     : mask,
            "selected_features" : selected,
            "selected_feature_names"   : [str(self.feature_names_in_[j]) for j in selected],
            "min_improvement"          : float(improvement_threshold),
            "path"                     : pd.DataFrame(path),
            "subset"                   : self._subset_measures(selected),
            "features"                 : self.feature_analysis(),
            "redundancy"               : self.redundancy_matrix(),
        }

    def rank_features(
        self,
        *,
        max_features: int | None = None,
        criterion: RankingCriterion = "deficiency",
        return_details: bool = False,
    ):
        """
        Rank features for model construction.

        The ranking is greedy and uses the same empirical subset diagnostics as
        ``miscoding_subset``. Unlike ``select_features()``, this method keeps
        adding reliable features to the order even when subset miscoding stops
        improving, until the requested count is reached or every remaining
        candidate is unreliable.

        Parameters
        ----------
        max_features : int, optional
            Maximum number of features to rank. If omitted, every feature is
            ranked.

        criterion : {"deficiency", "miscoding"}, default="deficiency"
            Candidate ordering criterion. ``"deficiency"`` prioritizes the
            lowest resulting subset deficiency, then miscoding, surplus, and
            feature index. ``"miscoding"`` prioritizes the lowest resulting
            subset miscoding, then deficiency, surplus, and feature index.

        return_details : bool, default=False
            If ``False``, return ordered feature indices. If ``True``, return a
            dictionary with the feature order, feature names, ranking path, and
            supporting diagnostics.

        Returns
        -------
        list[int] or dict
            Ordered feature indices by default, or detailed ranking output when
            ``return_details=True``.
        """
        check_is_fitted(self)
        self._validate_ranking_criterion(criterion)
        max_features = self._validate_max_features(max_features)

        selected: list[int] = []
        path: list[dict[str, object]] = []
        current = self._subset_measures(selected)

        while len(selected) < max_features:
            candidates = self._sort_candidates(
                self._candidate_extensions(selected, current),
                criterion=criterion,
            )
            if candidates.empty:
                break

            best = candidates.iloc[0]
            if not bool(best["is_reliable"]):
                break

            feature = int(best["feature_index"])
            selected.append(feature)
            current = self._subset_measures(selected)

            path.append(
                {
                    "step": len(path) + 1,
                    "feature_index": feature,
                    "feature_name": str(self.feature_names_in_[feature]),
                    "deficiency": float(best["deficiency"]),
                    "surplus": float(best["surplus"]),
                    "miscoding": float(best["miscoding"]),
                    "is_reliable": bool(best["is_reliable"]),
                    "failure_reason": best["failure_reason"],
                    "resolved_n_bins": best["resolved_n_bins"],
                    "n_samples": best["n_samples"],
                    "n_observed_joint_states": best["n_observed_joint_states"],
                    "mean_joint_occupancy": best["mean_joint_occupancy"],
                    "n_singleton_joint_states": best["n_singleton_joint_states"],
                    "singleton_fraction": best["singleton_fraction"],
                    "deficiency_improvement": float(best["deficiency_improvement"]),
                    "surplus_change": float(best["surplus_change"]),
                    "miscoding_improvement": float(best["miscoding_improvement"]),
                    "selected_features": tuple(selected),
                    "selected_feature_names": tuple(
                        str(self.feature_names_in_[j]) for j in selected
                    ),
                }
            )

        if not return_details:
            return selected

        return {
            "feature_order": selected,
            "feature_names": [str(self.feature_names_in_[j]) for j in selected],
            "path": pd.DataFrame(path),
            "features": self.feature_analysis(),
            "redundancy": self.redundancy_matrix(),
        }

    #
    # Validation and type inference
    #

    def _validate_X_y(self, X, y) -> tuple[np.ndarray, np.ndarray]:
        """
        Validate inputs and establish feature names.

        pandas DataFrames preserve their column names; other array-like inputs
        receive generated names ``x0``, ``x1``, and so on.
        """
        y_arr = _validate_vector(y, name="y")
        X_arr, y_arr = check_X_y(X, y_arr, dtype=None, ensure_2d=True)
        self.feature_names_in_ = np.asarray(
            getattr(X, "columns", [f"x{i}" for i in range(X_arr.shape[1])]),
            dtype=object,
        )
        return X_arr, y_arr

    def _infer_X_isnumeric(self, X_original, X_array: np.ndarray) -> list[bool]:
        """
        Infer whether each feature should be treated as numeric.

        Explicit ``X_type`` values override automatic inference. DataFrames
        allow automatic per-column numeric/categorical inference.
        """
        if self.X_type == "numeric":
            return [True] * X_array.shape[1]
        if self.X_type == "categorical":
            return [False] * X_array.shape[1]
        if isinstance(X_original, pd.DataFrame):
            return [
                bool(pd.api.types.is_numeric_dtype(dtype))
                for dtype in X_original.dtypes
            ]
        return (
            [True] * X_array.shape[1]
            if np.issubdtype(X_array.dtype, np.number)
            else [
                bool(np.issubdtype(np.asarray(X_array[:, j]).dtype, np.number))
                for j in range(X_array.shape[1])
            ]
        )

    def _infer_y_isnumeric(self, y: np.ndarray) -> bool:
        """Infer whether the target should be encoded as numeric or categorical."""
        if self.y_type == "numeric":
            return True
        if self.y_type == "categorical":
            return False

        target_type = type_of_target(y)
        if target_type in ("binary", "multiclass"):
            return False
        if target_type == "continuous":
            return True

        raise ValueError(
            "Unsupported target type {!r}. Supported one-dimensional target "
            "types are binary, multiclass, and continuous."
            .format(target_type)
        )

    #
    # Code-length computations
    #

    def _code_length(self, columns, numeric, *, n_bins: BinSpec | int) -> float:
        """
        Compute an empirical joint code length.

        Parameters
        ----------
        columns : sequence of iterable
            Variables to include in the joint code-length computation.

        numeric : sequence of bool
            Flags indicating whether each variable should be treated as numeric.

        n_bins : int, "auto", or "adaptive"
            Bin specification used for numeric variables.

        Returns
        -------
        float
            Empirical code length of the supplied variables.
        """
        return float(
            empirical_distribution_array(
                np.asarray(columns, dtype=object).T,
                numeric=numeric,
                n_bins=n_bins,
            ).code_length
        )

    def _code_length_for_indices(
        self,
        features: list[int] | tuple[int, ...] | None = None,
        y_included: bool = False,
        *,
        n_bins: int | None = None,
        subset_size: int | None = None,
    ) -> float:
        """
        Compute and cache a code length for a feature subset and optional target.
        """
        features = [] if features is None else list(features)
        feature_tuple = tuple(sorted(int(j) for j in features))
        if n_bins is None:
            effective_subset_size = (
                max(1, len(feature_tuple))
                if subset_size is None
                else int(subset_size)
            )
            n_bins = self._resolve_n_bins_for_subset(effective_subset_size)

        key = (feature_tuple, bool(y_included), int(n_bins))

        if key in self._code_length_cache_:
            return self._code_length_cache_[key]

        columns = [self.X_[:, j] for j in feature_tuple]
        numeric = [self.X_isnumeric_[j] for j in feature_tuple]

        if y_included:
            columns.append(self.y_)
            numeric.append(self.y_isnumeric_)

        value = (
            0.0
            if not columns
            else self._code_length(columns, numeric, n_bins=int(n_bins))
        )
        self._code_length_cache_[key] = value
        return value

    def _empirical_summary_for_indices(
        self,
        features: list[int] | tuple[int, ...] | None = None,
        y_included: bool = False,
        *,
        n_bins: int | None = None,
        subset_size: int | None = None,
    ):
        """
        Return the empirical summary for a feature subset and optional target.
        """
        features = [] if features is None else list(features)
        feature_tuple = tuple(sorted(int(j) for j in features))
        if n_bins is None:
            effective_subset_size = (
                max(1, len(feature_tuple))
                if subset_size is None
                else int(subset_size)
            )
            n_bins = self._resolve_n_bins_for_subset(effective_subset_size)

        key = (feature_tuple, bool(y_included), int(n_bins))
        if key in self._empirical_summary_cache_:
            return self._empirical_summary_cache_[key]

        columns = [self.X_[:, j] for j in feature_tuple]
        numeric = [self.X_isnumeric_[j] for j in feature_tuple]

        if y_included:
            columns.append(self.y_)
            numeric.append(self.y_isnumeric_)

        if not columns:
            raise ValueError("At least one random variable must be provided.")

        summary = empirical_distribution_array(
            np.asarray(columns, dtype=object).T,
            numeric=numeric,
            n_bins=int(n_bins),
        )
        self._empirical_summary_cache_[key] = summary
        self._code_length_cache_[key] = float(summary.code_length)
        return summary

    def _conditional_target_length(self, selected) -> float:
        """Return ``K(Y | X_S)`` for a selected feature subset ``S``."""
        selected = list(selected)
        subset_size = max(1, len(selected))
        n_bins = self._resolve_n_bins_for_subset(subset_size)
        return max(
            0.0,
            self._code_length_for_indices(
                features=selected,
                y_included=True,
                n_bins=n_bins,
            )
            - self._code_length_for_indices(
                features=selected,
                y_included=False,
                n_bins=n_bins,
            ),
        )

    def _conditional_feature_length(
        self,
        feature: int,
        *,
        selected,
        y_included: bool,
    ) -> float:
        """
        Estimate the conditional code length of one feature.

        If ``y_included`` is false, the conditioning set is ``X_S``. If
        ``y_included`` is true, the conditioning set is ``(X_S, Y)``.
        """
        selected = list(selected)
        if feature in selected:
            return 0.0

        subset_size = max(1, len(selected) + 1)
        n_bins = self._resolve_n_bins_for_subset(subset_size)
        return max(
            0.0,
            self._code_length_for_indices(
                features=selected + [int(feature)],
                y_included=y_included,
                n_bins=n_bins,
            )
            - self._code_length_for_indices(
                features=selected,
                y_included=y_included,
                n_bins=n_bins,
                subset_size=subset_size,
            ),
        )

    #
    # Redundancy and empirical subset diagnostics
    #

    def _feature_redundancy_matrix(self) -> np.ndarray:
        """
        Estimate pairwise redundancy between features.

        Redundancy is defined as ``1 - mu(X_i, X_j)``, where ``mu`` is the
        symmetric normalized code-length distance between the two feature
        strings. The diagonal is set to one.
        """
        redundancy = np.eye(self.n_features_in_, dtype=float)

        for i in range(self.n_features_in_):
            for j in range(i + 1, self.n_features_in_):
                value = self._feature_pair_redundancy(i, j)
                redundancy[i, j] = value
                redundancy[j, i] = value

        return redundancy

    def _feature_pair_redundancy(self, i: int, j: int) -> float:
        """
        Estimate the redundancy between two features.
        """
        n_bins = self._resolve_n_bins_for_subset(2)
        k_i = float(self._code_length_for_indices(features=[i], n_bins=n_bins))
        k_j = float(self._code_length_for_indices(features=[j], n_bins=n_bins))
        k_ij = float(self._code_length_for_indices(features=[i, j], n_bins=n_bins))

        denominator = max(k_i, k_j)
        if denominator <= 0.0:
            return 1.0

        miscoding = (k_ij - min(k_i, k_j)) / denominator
        return float(np.clip(1.0 - miscoding, 0.0, 1.0))

    def _redundancy_weights(self, selected: list[int]) -> np.ndarray:
        """
        Compute redundancy-discounting exponents for a selected subset.
        """
        if len(selected) == 0:
            return np.array([], dtype=float)

        matrix = self.redundancy_[np.ix_(selected, selected)]
        off_diagonal_sum = np.sum(matrix, axis=1) - np.diag(matrix)
        return 1.0 / (1.0 + off_diagonal_sum)
    

    def _subset_measures(self, subset) -> dict[str, object]:
        """
        Compute empirical deficiency, surplus, and miscoding for a selected
        feature subset.
        """

        selected = self._normalize_indices(subset)

        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[selected] = True

        if len(selected) == 0:
            deficiency = 0.0 if self.target_code_length_ <= 0.0 else 1.0
            return {
                "deficiency"               : deficiency,
                "surplus"                  : 0.0,
                "miscoding"                : deficiency,
                "is_reliable"              : True,
                "failure_reason"           : None,
                "resolved_n_bins"          : None,
                "n_samples"                : int(self.n_samples_in_),
                "n_observed_joint_states"  : None,
                "mean_joint_occupancy"     : None,
                "n_singleton_joint_states" : None,
                "singleton_fraction"       : None,
                "mask"          : mask,
                "n_selected_features"      : 0,
                "selected_features" : [],
                "selected_feature_names"   : [],
                "redundancy_weights"       : np.array([], dtype=float),
                "feature_weights"          : np.array([], dtype=float),
            }

        selected_array = np.asarray(selected, dtype=int)
        alpha = self._redundancy_weights(selected)
        feature_lengths = self.feature_code_lengths_[selected_array]
        feature_weights = alpha * feature_lengths
        values = self._empirical_subset_measures(selected)

        return {
            "deficiency"               : float(values["deficiency"]),
            "surplus"                  : float(values["surplus"]),
            "miscoding"                : float(values["miscoding"]),
            "is_reliable"              : bool(values["is_reliable"]),
            "failure_reason"           : values["failure_reason"],
            "resolved_n_bins"          : values["resolved_n_bins"],
            "n_samples"                : values["n_samples"],
            "n_observed_joint_states"  : values["n_observed_joint_states"],
            "mean_joint_occupancy"     : values["mean_joint_occupancy"],
            "n_singleton_joint_states" : values["n_singleton_joint_states"],
            "singleton_fraction"       : values["singleton_fraction"],
            "mask"          : mask,
            "n_selected_features"      : int(np.sum(mask)),
            "selected_features" : selected,
            "selected_feature_names"   : [str(self.feature_names_in_[j]) for j in selected],
            "redundancy_weights"       : alpha,
            "feature_weights"          : feature_weights,
        }

    def _empirical_subset_deficiency(self, selected: list[int]) -> float:
        """
        Compute empirical ``K(Y | X_S) / K(Y)`` for a non-empty subset.
        """
        return self._empirical_subset_measures(selected)["deficiency"]

    def _empirical_subset_surplus(self, selected: list[int]) -> float:
        """
        Compute empirical ``K(X_S | Y) / K(X_S)`` for a non-empty subset.
        """
        return self._empirical_subset_measures(selected)["surplus"]

    def _empirical_subset_measures(self, selected: list[int]) -> dict[str, object]:
        """
        Compute empirical subset deficiency, surplus, and miscoding.
        """
        selected = list(selected)
        if len(selected) == 0:
            deficiency = 0.0 if self.target_code_length_ <= 0.0 else 1.0
            return {
                "deficiency": deficiency,
                "surplus": 0.0,
                "miscoding": deficiency,
                "is_reliable": True,
                "failure_reason": None,
                "resolved_n_bins": None,
                "n_samples": int(self.n_samples_in_),
                "n_observed_joint_states": None,
                "mean_joint_occupancy": None,
                "n_singleton_joint_states": None,
                "singleton_fraction": None,
            }

        n_bins = self._resolve_n_bins_for_subset(len(selected))
        joint_summary = self._empirical_summary_for_indices(
            features=selected,
            y_included=True,
            n_bins=n_bins,
        )
        reliability = self._joint_reliability_diagnostics(joint_summary)
        reliability["resolved_n_bins"] = (
            n_bins if self.y_isnumeric_ or any(self.X_isnumeric_[j] for j in selected)
            else None
        )

        if not reliability["is_reliable"]:
            return {
                "deficiency": float("nan"),
                "surplus": float("nan"),
                "miscoding": float("nan"),
                **reliability,
            }

        x_summary = self._empirical_summary_for_indices(
            features=selected,
            y_included=False,
            n_bins=n_bins,
        )
        y_summary = self._empirical_summary_for_indices(
            features=[],
            y_included=True,
            n_bins=n_bins,
            subset_size=len(selected),
        )
        k_xy = float(joint_summary.code_length)
        k_x = float(x_summary.code_length)
        k_y = float(y_summary.code_length)

        deficiency = (
            0.0
            if k_y <= 0.0
            else float(np.clip((k_xy - k_x) / k_y, 0.0, 1.0))
        )
        surplus = (
            0.0
            if k_x <= 0.0
            else float(np.clip((k_xy - k_y) / k_x, 0.0, 1.0))
        )

        return {
            "deficiency": deficiency,
            "surplus": surplus,
            "miscoding": max(deficiency, surplus),
            **reliability,
        }

    @staticmethod
    def _joint_reliability_diagnostics(summary) -> dict[str, object]:
        """
        Return reliability metadata for an observed joint distribution.
        """
        n_samples = int(summary.n_samples)
        n_states = int(summary.n_states)
        mean_occupancy = float(n_samples / n_states)
        n_singletons = int(np.sum(np.asarray(summary.counts, dtype=float) == 1.0))
        singleton_fraction = float(n_singletons / n_states)
        is_reliable = (
            mean_occupancy >= 2.0
            and singleton_fraction <= 0.5
        )

        return {
            "is_reliable": bool(is_reliable),
            "failure_reason": None if is_reliable else _SPARSE_JOINT_FAILURE,
            "n_samples": n_samples,
            "n_observed_joint_states": n_states,
            "mean_joint_occupancy": mean_occupancy,
            "n_singleton_joint_states": n_singletons,
            "singleton_fraction": singleton_fraction,
        }

    def _candidate_extensions(
        self,
        selected: list[int],
        current: dict[str, object],
    ) -> pd.DataFrame:
        """
        Evaluate all one-feature extensions of the current selected set.
        """
        selected_set = set(selected)
        rows: list[dict[str, object]] = []

        for feature in range(self.n_features_in_):

            if feature in selected_set:
                continue

            candidate_subset = selected + [feature]
            values           = self._subset_measures(candidate_subset)

            rows.append({
                "feature_index": feature,
                "feature_name": str(self.feature_names_in_[feature]),
                "deficiency": float(values["deficiency"]),
                "surplus": float(values["surplus"]),
                "miscoding": float(values["miscoding"]),
                "is_reliable": bool(values["is_reliable"]),
                "failure_reason": values["failure_reason"],
                "resolved_n_bins": values["resolved_n_bins"],
                "n_samples": values["n_samples"],
                "n_observed_joint_states": values["n_observed_joint_states"],
                "mean_joint_occupancy": values["mean_joint_occupancy"],
                "n_singleton_joint_states": values["n_singleton_joint_states"],
                "singleton_fraction": values["singleton_fraction"],
                "deficiency_improvement": self._finite_difference(
                    current["deficiency"],
                    values["deficiency"],
                ),
                "surplus_change": self._finite_difference(
                    values["surplus"],
                    current["surplus"],
                ),
                "miscoding_improvement": self._finite_difference(
                    current["miscoding"],
                    values["miscoding"],
                ),
                "candidate_subset": tuple(candidate_subset),
            })

        if not rows:
            return pd.DataFrame(
                columns=[
                    "feature_index",
                    "feature_name",
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
                    "deficiency_improvement",
                    "surplus_change",
                    "miscoding_improvement",
                    "candidate_subset",
                ]
            )

        return pd.DataFrame(rows)

    def _sort_candidates(
        self,
        candidates: pd.DataFrame,
        *,
        criterion: RankingCriterion,
    ) -> pd.DataFrame:
        """
        Sort candidate extensions according to the requested ranking criterion.
        """
        self._validate_ranking_criterion(criterion)

        if candidates.empty:
            return candidates

        if criterion == "deficiency":
            columns = [
                "is_reliable",
                "deficiency",
                "miscoding",
                "surplus",
                "feature_index",
            ]
        else:
            columns = [
                "is_reliable",
                "miscoding",
                "deficiency",
                "surplus",
                "feature_index",
            ]

        return candidates.sort_values(
            by=columns,
            ascending=[False, True, True, True, True],
            ignore_index=True,
            na_position="last",
        )

    def _validate_max_features(self, max_features: int | None) -> int:
        """
        Validate and cap a requested feature count.
        """
        if max_features is None:
            return int(self.n_features_in_)

        max_features = int(max_features)
        if max_features < 0:
            raise ValueError("max_features must be non-negative.")

        return min(max_features, int(self.n_features_in_))

    @classmethod
    def _validate_ranking_criterion(cls, criterion: str) -> None:
        """
        Validate ranking criterion values.
        """
        if criterion not in cls._VALID_RANKING_CRITERIA:
            raise ValueError(
                "Valid options for 'criterion' are {}. Got criterion={!r} instead."
                .format(cls._VALID_RANKING_CRITERIA, criterion)
            )

    def _resolve_n_bins_for_subset(self, subset_size: int) -> int:
        """
        Resolve the numeric bin count for a feature subset size.
        """
        return _resolve_bins(
            self.n_bins, self.n_samples_in_, subset_size=subset_size
        )

    #
    # Index handling and numerical helpers
    #

    def _normalize_indices(self, selected) -> list[int]:
        """
        Normalize a Boolean mask or index list into validated feature indices.
        """

        if selected is None:
            return []

        arr = np.asarray(selected)
        if arr.size == 0:
            return []
        if arr.ndim != 1:
            raise ValueError("selected must be a one-dimensional mask or index list.")

        if arr.dtype.kind == "b":
            if arr.size != self.n_features_in_:
                raise ValueError("Boolean masks must match the fitted feature dimension.")
            indices = np.flatnonzero(arr).tolist()
        elif arr.dtype.kind in "iu":
            indices = arr.tolist()
        else:
            raise ValueError("subset must contain integer indices or Boolean mask values.")

        if len(indices) != len(set(indices)):
            raise ValueError("selected contains duplicate feature indices.")
        if any(j < 0 or j >= self.n_features_in_ for j in indices):
            raise ValueError(
                f"selected indices must lie in [0, {self.n_features_in_ - 1}]."
            )

        return indices

    @staticmethod
    def _finite_difference(left, right) -> float:
        """
        Return ``left - right`` when both operands are finite.
        """
        left = float(left)
        right = float(right)
        if not (np.isfinite(left) and np.isfinite(right)):
            return float("nan")
        return left - right

    @staticmethod
    def _safe_divide(numerator, denominator, *, default: float) -> np.ndarray:
        """
        Safely divide arrays, assigning ``default`` where division is invalid.
        """
        numerator, denominator = np.broadcast_arrays(
            np.asarray(numerator, dtype=float),
            np.asarray(denominator, dtype=float),
        )
        result = np.full_like(numerator, default, dtype=float)
        valid = (
            (denominator > 0)
            & np.isfinite(denominator)
            & np.isfinite(numerator)
        )
        np.divide(numerator, denominator, out=result, where=valid)
        return result

    @classmethod
    def _validate_init(
        cls,
        *,
        X_type,
        y_type,
    ):
        """
        Validate constructor arguments before storing them on the estimator.
        """
        if X_type not in cls._VALID_X_TYPES:
            raise ValueError(
                f"Valid options for 'X_type' are {cls._VALID_X_TYPES}. "
                f"Got {X_type!r}."
            )
        if y_type not in cls._VALID_Y_TYPES:
            raise ValueError(
                f"Valid options for 'y_type' are {cls._VALID_Y_TYPES}. "
                f"Got {y_type!r}."
            )


#
# Functional interface
#


def feature_analysis(*, X, y, X_type: XType = "auto", y_type: YType = "auto",
                     n_bins: BinSpec = "adaptive") -> pd.DataFrame:
    """
    Return feature analysis using a functional interface.
    """
    metric = Miscoding(X_type=X_type, y_type=y_type, n_bins=n_bins).fit(X, y)
    return metric.feature_analysis()


def redundancy_matrix(*, X, y, X_type: XType = "auto", y_type: YType = "auto",
                      n_bins: BinSpec = "adaptive") -> pd.DataFrame:
    """
    Return pairwise feature redundancy using a functional interface.
    """
    metric = Miscoding(X_type=X_type, y_type=y_type, n_bins=n_bins).fit(X, y)
    return metric.redundancy_matrix()


def miscoding_subset(subset, *, X, y, X_type: XType = "auto", y_type: YType = "auto",
                     n_bins: BinSpec = "adaptive") -> float:
    """
    Return a subset-level miscoding quantity using a functional interface.
    """
    metric = Miscoding(X_type=X_type, y_type=y_type, n_bins=n_bins).fit(X, y)
    return metric.miscoding_subset(subset)


def deficiency_subset(subset, *, X, y, X_type: XType = "auto", y_type: YType = "auto",
                       n_bins: BinSpec = "adaptive") -> float:
    """Compute subset deficiency from evaluation data."""
    return Miscoding(X_type=X_type, y_type=y_type, n_bins=n_bins).fit(X, y).deficiency_subset(subset)


def surplus_subset(subset, *, X, y, X_type: XType = "auto", y_type: YType = "auto",
                    n_bins: BinSpec = "adaptive") -> float:
    """Compute subset surplus from evaluation data."""
    return Miscoding(X_type=X_type, y_type=y_type, n_bins=n_bins).fit(X, y).surplus_subset(subset)


def miscoding_feature(feature=None, *, X, y, X_type: XType = "auto",
                      y_type: YType = "auto", n_bins: BinSpec = "adaptive"):
    """Compute one feature's miscoding or all feature values in input order."""
    return Miscoding(X_type=X_type, y_type=y_type, n_bins=n_bins).fit(X, y).miscoding_feature(feature)


def deficiency_feature(feature=None, *, X, y, X_type: XType = "auto",
                        y_type: YType = "auto", n_bins: BinSpec = "adaptive"):
    """Compute one feature's deficiency or all values in input order."""
    return Miscoding(X_type=X_type, y_type=y_type, n_bins=n_bins).fit(X, y).deficiency_feature(feature)


def surplus_feature(feature=None, *, X, y, X_type: XType = "auto",
                     y_type: YType = "auto", n_bins: BinSpec = "adaptive"):
    """Compute one feature's surplus or all values in input order."""
    return Miscoding(X_type=X_type, y_type=y_type, n_bins=n_bins).fit(X, y).surplus_feature(feature)


def subset_analysis(subset, *, X, y, X_type: XType = "auto", y_type: YType = "auto",
                    n_bins: BinSpec = "adaptive") -> dict[str, object]:
    """Return empirical subset diagnostics, including reliability information."""
    return Miscoding(X_type=X_type, y_type=y_type, n_bins=n_bins).fit(X, y).subset_analysis(subset)


def miscoding_model(model, *, X, y, feature_names=None, feature_indices=None,
                    X_type: XType = "auto", y_type: YType = "auto",
                    n_bins: BinSpec = "adaptive") -> float:
    """Compute fitted-model miscoding from evaluation data."""
    return Miscoding(X_type=X_type, y_type=y_type, n_bins=n_bins).fit(X, y).miscoding_model(
        model, feature_names=feature_names, feature_indices=feature_indices)


def model_analysis(model, *, X, y, feature_names=None, feature_indices=None,
                   X_type: XType = "auto", y_type: YType = "auto",
                   n_bins: BinSpec = "adaptive") -> dict[str, object]:
    """Analyze the effective feature subset of a fitted model."""
    return Miscoding(X_type=X_type, y_type=y_type, n_bins=n_bins).fit(X, y).model_analysis(
        model, feature_names=feature_names, feature_indices=feature_indices)


def select_features(
    *,
    X,
    y,
    max_features: int | None = None,
    min_improvement: float = 0.0,
    return_details: bool = False,
    X_type: XType = "auto",
    y_type: YType = "auto",
    n_bins: BinSpec = "adaptive",
):
    """
    Select features using a functional interface.
    """
    metric = Miscoding(X_type=X_type, y_type=y_type, n_bins=n_bins).fit(X, y)
    return metric.select_features(
        max_features=max_features,
        min_improvement=min_improvement,
        return_details=return_details,
    )


def rank_features(
    *,
    X,
    y,
    max_features: int | None = None,
    criterion: RankingCriterion = "deficiency",
    return_details: bool = False,
    X_type: XType = "auto",
    y_type: YType = "auto",
    n_bins: BinSpec = "adaptive",
):
    """
    Rank features using a functional interface.
    """
    metric = Miscoding(X_type=X_type, y_type=y_type, n_bins=n_bins).fit(X, y)
    return metric.rank_features(
        max_features=max_features,
        criterion=criterion,
        return_details=return_details,
    )
