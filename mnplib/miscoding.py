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

from dataclasses import dataclass
from typing import Literal, get_args

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted

from ._types import XType, YType
from .utils import _resolve_bins, _validate_vector, empirical_distribution_array
from .models.inputs import model_artifacts
from ._diagnostics import warn_nan_model


RankingCriterion = Literal["deficiency", "miscoding"]

_SPARSE_JOINT_FAILURE = "joint_distribution_too_sparse"


@dataclass(frozen=True)
class _EmpiricalStatistics:
    """Scalar distribution statistics for code lengths and reliability checks."""

    code_length  : float
    n_samples    : int
    n_states     : int
    n_singletons : int


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

    Numeric variables use uniform discretization with
    ``max(2, floor(2 * n_samples**(1/3) / log2(|S| + 1)))`` bins. The subset
    size counts selected features, excluding the target. Joint and marginal
    distributions use the same resolved count. Feature diagnostics use a
    subset size of one; pairwise redundancy uses two. Larger subsets use
    coarser bins to reduce joint sparsity.

    ``select_features()`` greedily selects features by requiring subset
    miscoding improvement. ``rank_features()`` greedily orders features for
    model construction while reliable candidate extensions remain.

    Feature-level diagnostics are computed during ``fit()``. Pairwise
    redundancy is computed on demand and cached as a full matrix.
    ``redundancy_matrix()`` and ``redundancy_`` request all pairs. Detailed
    selection and ranking reports include the matrix by default.

    Code lengths and scalar state counts share a bin-aware cache. Distribution
    arrays are discarded after these statistics have been extracted.

    Fitted data and variable types are snapshots. Input changes and parameter
    changes take effect only after fitting again.
    """

    _VALID_X_TYPES = get_args(XType)
    _VALID_Y_TYPES = get_args(YType)
    _VALID_RANKING_CRITERIA = get_args(RankingCriterion)

    def __init__(
        self,
        X_type: XType   = "auto",
        y_type: YType   = "auto",
    ):
        """
        Initialize the estimator.

        Parameters
        ----------
        X_type : {"auto", "numeric", "categorical"}, default="auto"
            Encoding strategy for the feature variables.

        y_type : {"auto", "numeric", "categorical"}, default="auto"
            Encoding strategy for the target variable.
        """
        self._validate_init(
            X_type=X_type,
            y_type=y_type,
        )

        self.X_type = X_type
        self.y_type = y_type

        # List of attributes

        # self.X_type                    # Encoding strategy configured for the feature variables.
        # self.y_type                    # Encoding strategy configured for the target variable.

        # self.is_fitted_                # Indicates whether fitting completed successfully.
        # self._empirical_cache_         # Caches scalar distribution statistics for each feature, target, and bin context.
        # self._redundancy_matrix_       # Stores the complete feature redundancy matrix once it has been computed.

        # self.X_                        # Validated copy of the feature matrix used to fit the estimator.
        # self.y_                        # Validated copy of the target vector used to fit the estimator.
        # self._model_X_                 # Feature data preserved in a form suitable for model-artifact analysis.
        # self.n_samples_in_             # Number of samples in the fitted dataset.
        # self.n_features_in_            # Number of features in the fitted dataset.
        # self.feature_names_in_         # Names assigned to the fitted features, preserving DataFrame column names when available.
        # self.X_isnumeric_              # Boolean flags indicating which fitted features are treated as numeric.
        # self.y_isnumeric_              # Indicates whether the fitted target is treated as numeric.

        # self.target_code_length_       # Empirical code length of the target variable.
        # self.feature_code_lengths_     # Empirical code length of each individual feature.
        # self.deficiency_               # Feature-level deficiency values.
        # self.surplus_                  # Feature-level surplus values.
        # self.miscoding_                # Feature-level miscoding values, computed as max(deficiency, surplus).

        # self.redundancy_               # Lazily computed public property returning the complete feature redundancy matrix.

    def fit(self, X, y):
        """
        Store evaluation data and compute feature-level diagnostics.

        Reset diagnostic caches and defer pairwise redundancy computations
        until requested. Input arrays are copied to isolate fitted results
        from changes to the supplied data.

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
        self.is_fitted_ = False
        self._empirical_cache_ = {}
        self._redundancy_matrix_ = None
        self._validate_init(X_type=self.X_type, y_type=self.y_type)
        if y is None:
            raise ValueError("Miscoding.fit requires a target vector y.")

        self.X_, self.y_ = self._validate_X_y(X, y)
        self._model_X_ = X.copy(deep=True) if isinstance(X, pd.DataFrame) else self.X_
        self.n_samples_in_, self.n_features_in_ = self.X_.shape
        self.X_isnumeric_ = self._infer_X_isnumeric(X, self.X_)
        self.y_isnumeric_ = self._infer_y_isnumeric(self.y_)

        n_bins = self._resolve_n_bins_for_subset(1)

        self.target_code_length_ = self._empirical_statistics_for_indices(
            y_included=True, n_bins=n_bins,
        ).code_length

        self.feature_code_lengths_ = np.array(
            [
                self._empirical_statistics_for_indices(features=[j], n_bins=n_bins).code_length
                for j in range(self.n_features_in_)
            ],
            dtype=float,
        )

        joint_lengths = np.array(
            [
                self._empirical_statistics_for_indices(
                    features=[j], y_included=True, n_bins=n_bins,
                ).code_length
                for j in range(self.n_features_in_)
            ],
            dtype=float,
        )

        # ( K(Y, X_j) - K(X_j) ) / K(Y)
        self.deficiency_ = np.clip(
            self._safe_divide(
                joint_lengths - self.feature_code_lengths_,
                self.target_code_length_,
                default=0.0,
            ),
            0.0,
            1.0,
        )

        # ( K(X_j, Y) - K(Y) ) / K(X_j)
        self.surplus_ = np.clip(
            self._safe_divide(
                joint_lengths - self.target_code_length_,
                self.feature_code_lengths_,
                default=0.0,
            ),
            0.0,
            1.0,
        )

        self.miscoding_ = np.maximum(self.deficiency_, self.surplus_)

        self.is_fitted_ = True
        return self

    def __sklearn_is_fitted__(self) -> bool:
        """Report whether fitting completed successfully."""
        return getattr(self, "is_fitted_", False)

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

    @property
    def redundancy_(self) -> np.ndarray:
        """Compute the full redundancy matrix on first access and return a copy."""
        check_is_fitted(self)
        if self._redundancy_matrix_ is None:
            self._redundancy_matrix_ = self._feature_redundancy_matrix()
        return self._redundancy_matrix_.copy()

    def redundancy_matrix(self) -> pd.DataFrame:
        """
        Return the pairwise redundancy matrix between features.

        Compute any missing pairs and cache the complete matrix. Each call
        returns an independent DataFrame.

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
            self.redundancy_,
            index   = self.feature_names_in_,
            columns = self.feature_names_in_,
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

        table = pd.DataFrame({
            "feature_index"    : np.arange(self.n_features_in_),
            "feature_name"     : self.feature_names_in_,
            "is_numeric"       : self.X_isnumeric_,
            "code_length_bits" : self.feature_code_lengths_,
            "deficiency"       : self.deficiency_,
            "surplus"          : self.surplus_,
            "miscoding"        : self.miscoding_,
        })

        return table.sort_values(
            by           = ["miscoding", "deficiency", "surplus"],
            ascending    = [True, True, True],
            ignore_index = True,
        )

    #
    # Subset-level diagnostics
    #

    def miscoding_subset(self, subset) -> float:
        """Return subset miscoding, or NaN when joint counts are unreliable."""
        return float(self.subset_analysis(subset)["miscoding"])

    def deficiency_subset(self, subset) -> float:
        """Return subset deficiency, or NaN when joint counts are unreliable."""
        return float(self.subset_analysis(subset)["deficiency"])

    def surplus_subset(self, subset) -> float:
        """Return subset surplus, or NaN when joint counts are unreliable."""
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
            Masks must match the fitted feature dimension. An empty index list
            requests the empty subset.

        Returns
        -------
        dict
            Dictionary containing deficiency, surplus, miscoding, selected
            feature metadata, and reliability diagnostics. Pairwise redundancy
            is available separately through ``redundancy_matrix()``.
            Reliability diagnostics include ``resolved_n_bins``, the numeric
            bin count for the subset, or None when no numeric discretization is
            applied, including an empty subset. Categorical variables retain
            their observed categories without binning.
            Unreliable subsets have NaN deficiency, surplus, and miscoding.
            The empty subset is reliable and has no joint-state diagnostics.
        """
        check_is_fitted(self)
        return self._subset_measures(subset)

    #
    # Feature selection and ordering
    #

    def select_features(self, *, max_features: int | None = None,
                        min_improvement: float = 0.0, return_details: bool = False,
                        include_redundancy: bool = True,
    ):
        """
        Select features by strict subset-miscoding improvement.

        At each step, the method evaluates every candidate feature not yet
        selected and adds the feature that produces the lowest subset miscoding.
        Selection stops when the best candidate does not reduce subset
        miscoding by more than ``min_improvement``.

        Parameters
        ----------
        max_features : non-negative int, optional
            Maximum number of features to select. If omitted, all features are
            eligible.

        min_improvement : float, default=0.0
            Minimum reduction in subset miscoding required to accept a feature.

        return_details : bool, default=False
            If ``False``, return a Boolean selection mask. If ``True``, return a
            dictionary with the mask, selected indices, selected names, selection
            path, and final subset diagnostics.
        include_redundancy : bool, default=True
            Include the full redundancy matrix in detailed output.
            False preserves the selection and path without computing pairwise
            redundancy. Ignored when return_details=False.

        Returns
        -------
        numpy.ndarray or dict
            Binary selection mask by default, or detailed selection output when
            ``return_details=True``.
        """
        check_is_fitted(self)

        improvement_threshold = float(min_improvement)
        if not np.isfinite(improvement_threshold) or improvement_threshold < 0:
            raise ValueError("min_improvement must be finite and non-negative.")

        max_features = self._validate_max_features(max_features)

        selected : list[int] = []
        path     : list[dict[str, object]] = []
        current  = self._empirical_subset_measures(selected)

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
            current = self._empirical_subset_measures(selected)

            path.append(
                {
                    "step"                   : len(path) + 1,
                    "feature_index"          : feature,
                    "feature_name"           : str(self.feature_names_in_[feature]),
                    **current,
                    "deficiency_improvement" : float(best["deficiency_improvement"]),
                    "surplus_change"         : float(best["surplus_change"]),
                    "miscoding_improvement"  : improvement,
                    "improvement"            : improvement,
                    "selected_features"      : tuple(selected),
                    "selected_feature_names" : tuple(
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
            "selected_features"        : selected,
            "selected_feature_names"   : [str(self.feature_names_in_[j]) for j in selected],
            "min_improvement"          : float(improvement_threshold),
            "path"                     : pd.DataFrame(path, columns=[
                "step", "feature_index", "feature_name", *current,
                "deficiency_improvement", "surplus_change", "miscoding_improvement",
                "improvement", "selected_features", "selected_feature_names",
            ]),
            "subset"                   : self._subset_measures(selected),
            "features"                 : self.feature_analysis(),
            **({"redundancy": self.redundancy_matrix()} if include_redundancy else {}),
        }

    def rank_features(
        self,
        *,
        max_features: int | None = None,
        criterion: RankingCriterion = "deficiency",
        return_details: bool = False,
        include_redundancy: bool = True,
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
        max_features : non-negative int, optional
            Maximum number of features to rank. If omitted, every feature is
            eligible. Ranking stops when no reliable extension remains.

        criterion : {"deficiency", "miscoding"}, default="deficiency"
            Candidate ordering criterion. ``"deficiency"`` prioritizes the
            lowest resulting subset deficiency, then miscoding, surplus, and
            feature index. ``"miscoding"`` prioritizes the lowest resulting
            subset miscoding, then deficiency, surplus, and feature index.

        return_details : bool, default=False
            If ``False``, return ordered feature indices. If ``True``, return a
            dictionary with the feature order, feature names, ranking path, and
            supporting diagnostics.
        include_redundancy : bool, default=True
            Include the full redundancy matrix in detailed output. False
            preserves the order and path without computing pairwise redundancy.
            Ignored when return_details=False.

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
        current = self._empirical_subset_measures(selected)

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
            current = self._empirical_subset_measures(selected)

            path.append(
                {
                    "step": len(path) + 1,
                    "feature_index": feature,
                    "feature_name": str(self.feature_names_in_[feature]),
                    **{name: best[name] for name in current},
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
            "path": pd.DataFrame(path, columns=[
                "step", "feature_index", "feature_name", *current,
                "deficiency_improvement", "surplus_change", "miscoding_improvement",
                "selected_features", "selected_feature_names",
            ]),
            "features": self.feature_analysis(),
            **({"redundancy": self.redundancy_matrix()} if include_redundancy else {}),
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
        ).copy()
        return X_arr.copy(), y_arr.copy()

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
    # Empirical distribution statistics
    #

    def _empirical_statistics_for_indices(
        self,
        features: list[int] | tuple[int, ...] = (),
        y_included: bool = False,
        *,
        n_bins: int,
    ) -> _EmpiricalStatistics:
        """
        Cache scalar distribution statistics using the resolved bin count.

        Bin-aware keys distinguish marginals evaluated in different contexts.
        Full distribution arrays are used only to extract these statistics.
        """
        feature_tuple = tuple(sorted(features))
        key = (feature_tuple, y_included, n_bins)
        if key in self._empirical_cache_:
            return self._empirical_cache_[key]

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
            n_bins=n_bins,
        )
        statistics = _EmpiricalStatistics(
            code_length  = float(summary.code_length),
            n_samples    = int(summary.n_samples),
            n_states     = int(summary.n_states),
            n_singletons = int(np.count_nonzero(summary.counts == 1)),
        )
        self._empirical_cache_[key] = statistics
        return statistics

    #
    # Redundancy and empirical subset diagnostics
    #

    def _feature_redundancy_matrix(self) -> np.ndarray:
        """
        Assemble pairwise redundancy for all features.

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
        Compute symmetric redundancy using cached empirical statistics.
        """
        if i == j:
            return 1.0
        n_bins = self._resolve_n_bins_for_subset(2)
        k_i = self._empirical_statistics_for_indices(features=[i], n_bins=n_bins).code_length
        k_j = self._empirical_statistics_for_indices(features=[j], n_bins=n_bins).code_length
        k_ij = self._empirical_statistics_for_indices(features=[i, j], n_bins=n_bins).code_length

        denominator = max(k_i, k_j)
        if denominator <= 0.0:
            value = 1.0
        else:
            miscoding = (k_ij - min(k_i, k_j)) / denominator
            value = float(np.clip(1.0 - miscoding, 0.0, 1.0))
        return value

    def _subset_measures(self, subset) -> dict[str, object]:
        """
        Compute empirical deficiency, surplus, and miscoding for a selected
        feature subset.
        """

        selected = self._normalize_indices(subset)

        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[selected] = True

        values = self._empirical_subset_measures(selected)

        return {
            **values,
            "mask"                   : mask,
            "n_selected_features"    : len(selected),
            "selected_features"      : selected,
            "selected_feature_names" : [str(self.feature_names_in_[j]) for j in selected],
        }

    def _empirical_subset_measures(self, selected: list[int]) -> dict[str, object]:
        """
        Compute empirical subset deficiency, surplus, and miscoding.
        """
        selected = list(selected)
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
            }

        n_bins = self._resolve_n_bins_for_subset(len(selected))
        joint_statistics = self._empirical_statistics_for_indices(
            features   = selected,
            y_included = True,
            n_bins     = n_bins,
        )
        reliability = self._joint_reliability_diagnostics(
            joint_statistics,
            resolved_n_bins=(
                n_bins if self.y_isnumeric_ or any(self.X_isnumeric_[j] for j in selected)
                else None
            ),
        )

        if not reliability["is_reliable"]:
            return {
                "deficiency" : float("nan"),
                "surplus"    : float("nan"),
                "miscoding"  : float("nan"),
                **reliability,
            }

        k_x = self._empirical_statistics_for_indices(features=selected, n_bins=n_bins).code_length
        k_y = self._empirical_statistics_for_indices(y_included=True,   n_bins=n_bins).code_length
        k_xy = joint_statistics.code_length

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
            "deficiency" : deficiency,
            "surplus"    : surplus,
            "miscoding"  : max(deficiency, surplus),
            **reliability,
        }

    @staticmethod
    def _joint_reliability_diagnostics(
        statistics: _EmpiricalStatistics, *, resolved_n_bins: int | None,
    ) -> dict[str, object]:
        """
        Return reliability metadata for an observed joint distribution.
        """
        n_samples          = statistics.n_samples
        n_states           = statistics.n_states
        mean_occupancy     = float(n_samples / n_states)
        n_singletons       = statistics.n_singletons
        singleton_fraction = float(n_singletons / n_states)
        is_reliable        = (
            mean_occupancy >= 2.0
            and singleton_fraction <= 0.5
        )

        return {
            "is_reliable"              : bool(is_reliable),
            "failure_reason"           : None if is_reliable else _SPARSE_JOINT_FAILURE,
            "resolved_n_bins"          : resolved_n_bins,
            "n_samples"                : n_samples,
            "n_observed_joint_states"  : n_states,
            "mean_joint_occupancy"     : mean_occupancy,
            "n_singleton_joint_states" : n_singletons,
            "singleton_fraction"       : singleton_fraction,
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
            values           = self._empirical_subset_measures(candidate_subset)

            rows.append({
                "feature_index": feature,
                "feature_name": str(self.feature_names_in_[feature]),
                **values,
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

        if (
            isinstance(max_features, (bool, np.bool_))
            or not isinstance(max_features, (int, np.integer))
            or max_features < 0
        ):
            raise ValueError("max_features must be a non-negative integer or None.")

        return min(int(max_features), int(self.n_features_in_))

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
            "adaptive", self.n_samples_in_, subset_size=subset_size
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
        if arr.ndim != 1:
            raise ValueError("selected must be a one-dimensional mask or index list.")

        if arr.dtype.kind == "b":
            if arr.size != self.n_features_in_:
                raise ValueError("Boolean masks must match the fitted feature dimension.")
            indices = np.flatnonzero(arr).tolist()
        elif arr.size == 0:
            return []
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


def feature_analysis(*, X, y, X_type: XType = "auto", y_type: YType = "auto") -> pd.DataFrame:
    """
    Return feature analysis using a functional interface.
    """
    metric = Miscoding(X_type=X_type, y_type=y_type).fit(X, y)
    return metric.feature_analysis()


def redundancy_matrix(*, X, y, X_type: XType = "auto", y_type: YType = "auto") -> pd.DataFrame:
    """
    Return pairwise feature redundancy using a functional interface.
    """
    metric = Miscoding(X_type=X_type, y_type=y_type).fit(X, y)
    return metric.redundancy_matrix()


def miscoding_subset(subset, *, X, y, X_type: XType = "auto", y_type: YType = "auto") -> float:
    """
    Return a subset-level miscoding quantity using a functional interface.
    """
    metric = Miscoding(X_type=X_type, y_type=y_type).fit(X, y)
    return metric.miscoding_subset(subset)


def deficiency_subset(subset, *, X, y, X_type: XType = "auto", y_type: YType = "auto") -> float:
    """Compute subset deficiency from evaluation data."""
    return Miscoding(X_type=X_type, y_type=y_type).fit(X, y).deficiency_subset(subset)


def surplus_subset(subset, *, X, y, X_type: XType = "auto", y_type: YType = "auto") -> float:
    """Compute subset surplus from evaluation data."""
    return Miscoding(X_type=X_type, y_type=y_type).fit(X, y).surplus_subset(subset)


def miscoding_feature(feature=None, *, X, y, X_type: XType = "auto",
                      y_type: YType = "auto"):
    """Compute one feature's miscoding or all feature values in input order."""
    return Miscoding(X_type=X_type, y_type=y_type).fit(X, y).miscoding_feature(feature)


def deficiency_feature(feature=None, *, X, y, X_type: XType = "auto",
                        y_type: YType = "auto"):
    """Compute one feature's deficiency or all values in input order."""
    return Miscoding(X_type=X_type, y_type=y_type).fit(X, y).deficiency_feature(feature)


def surplus_feature(feature=None, *, X, y, X_type: XType = "auto",
                     y_type: YType = "auto"):
    """Compute one feature's surplus or all values in input order."""
    return Miscoding(X_type=X_type, y_type=y_type).fit(X, y).surplus_feature(feature)


def subset_analysis(subset, *, X, y, X_type: XType = "auto", y_type: YType = "auto") -> dict[str, object]:
    """Return empirical subset diagnostics, including reliability information."""
    return Miscoding(X_type=X_type, y_type=y_type).fit(X, y).subset_analysis(subset)


def miscoding_model(model, *, X, y, feature_names=None, feature_indices=None,
                    X_type: XType = "auto", y_type: YType = "auto") -> float:
    """Compute fitted-model miscoding from evaluation data."""
    return Miscoding(X_type=X_type, y_type=y_type).fit(X, y).miscoding_model(
        model, feature_names=feature_names, feature_indices=feature_indices)


def model_analysis(model, *, X, y, feature_names=None, feature_indices=None,
                   X_type: XType = "auto", y_type: YType = "auto") -> dict[str, object]:
    """Analyze the effective feature subset of a fitted model."""
    return Miscoding(X_type=X_type, y_type=y_type).fit(X, y).model_analysis(
        model, feature_names=feature_names, feature_indices=feature_indices)


def select_features(
    *,
    X,
    y,
    max_features: int | None = None,
    min_improvement: float = 0.0,
    return_details: bool = False,
    include_redundancy: bool = True,
    X_type: XType = "auto",
    y_type: YType = "auto",
):
    """
    Select features using a functional interface.
    """
    metric = Miscoding(X_type=X_type, y_type=y_type).fit(X, y)
    return metric.select_features(
        max_features       = max_features,
        min_improvement    = min_improvement,
        return_details     = return_details,
        include_redundancy = include_redundancy,
    )


def rank_features(
    *,
    X,
    y,
    max_features: int | None = None,
    criterion: RankingCriterion = "deficiency",
    return_details: bool = False,
    include_redundancy: bool = True,
    X_type: XType = "auto",
    y_type: YType = "auto",
):
    """
    Rank features using a functional interface.
    """
    metric = Miscoding(X_type=X_type, y_type=y_type).fit(X, y)
    return metric.rank_features(
        max_features=max_features,
        criterion=criterion,
        return_details=return_details,
        include_redundancy=include_redundancy,
    )
