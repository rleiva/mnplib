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
from sklearn.utils.validation import check_is_fitted

from ._types import XType, YType
from ._rms import _rms_pair
from .utils import (
    _resolve_bins,
    _resolve_feature_names,
    _resolve_y_isnumeric,
    _validate_vector,
    _validate_y_type,
    empirical_distribution_array,
)
from .models.inputs import model_artifacts
from ._diagnostics import warn_unreliable_estimate


RankingCriterion = Literal["deficiency", "miscoding"]

_SPARSE_JOINT_FAILURE = "joint_distribution_too_sparse"


def miscoding(*, deficiency: float, surplus: float) -> float:
    """Return equal-weight RMS of deficiency and surplus.

    Nonfinite components produce NaN. Finite components must be nonnegative.
    Normalized inputs yield a scalar in [0, 1].
    """
    return float(_rms_pair(float(deficiency), float(surplus)))


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
        miscoding_j  = sqrt((deficiency_j**2 + surplus_j**2) / 2)

    For a subset of features ``S``, the estimator computes empirical subset
    quantities:

        deficiency(S) = (K(X_S, Y) - K(X_S)) / K(Y)
        surplus(S)    = (K(X_S, Y) - K(Y)) / K(X_S)
        miscoding(S)  = sqrt((deficiency(S)**2 + surplus(S)**2) / 2)

    Numeric variables use uniform discretization with
    ``max(2, floor(2 * n_samples**(1/3) / log2(|S| + 1)))`` bins. The subset
    size counts selected features, excluding the target. Joint and marginal
    distributions use the same resolved count. Feature diagnostics use a
    subset size of one; pairwise miscoding uses two. Larger subsets use
    coarser bins to reduce joint sparsity.

    ``select_features()`` greedily selects features by requiring subset
    miscoding improvement. ``rank_features()`` greedily orders features for
    model construction while finite candidate extensions remain.

    Reliability describes statistical support, not mathematical availability.
    Computable estimates remain numerical. Feature, subset, and model scalar
    methods emit RuntimeWarning for sparse joint states; analysis methods are
    quiet. Selection and ranking warn once if their returned path contains an
    unreliable step, unless ``return_details=True`` exposes its diagnostics.

    Feature-level diagnostics are computed during ``fit()``. Pairwise
    miscoding is computed on demand and cached as a full matrix.
    ``pairwise_miscoding_matrix()`` and ``pairwise_miscoding_`` request all
    pairs. Detailed selection and ranking reports include the matrix by default.

    Code lengths and scalar state counts share a bin-aware cache. Distribution
    arrays are discarded after these statistics have been extracted.

    Fitted data and variable types are snapshots. Input and encoding changes
    take effect only after fitting again. ``debug=True`` adds raw code lengths,
    discretization details, and observed-state counts to analysis reports.
    It does not change scores, selection, or ranking, and does not print.
    """

    _VALID_X_TYPES = get_args(XType)
    _VALID_RANKING_CRITERIA = get_args(RankingCriterion)

    def __init__(
        self,
        X_type: XType   = "auto",
        y_type: YType   = "auto",
        *,
        debug: bool = False,
    ):
        """
        Initialize the estimator.

        Parameters
        ----------
        X_type : {"auto", "numeric", "categorical"}, default="auto"
            Encoding strategy for the feature variables.

        y_type : {"auto", "numeric", "categorical"}, default="auto"
            Encoding strategy for the target variable.

        debug : bool, default=False
            Include empirical calculation details in feature, subset, and model
            analysis, and in detailed selection and ranking reports. Code lengths
            are in bits. Sparse subsets retain their scores and reliability diagnostics.
        """
        self._validate_X_type(X_type)
        _validate_y_type(y_type)
        self._validate_debug(debug)

        self.X_type = X_type
        self.y_type = y_type
        self.debug = debug

        # List of attributes

        # self.X_type                    # Encoding strategy configured for the feature variables.
        # self.y_type                    # Encoding strategy configured for the target variable.
        # self.debug                     # Includes empirical calculation details in analysis reports.

        # self.is_fitted_                # Indicates whether fitting completed successfully.
        # self._empirical_cache_         # Caches scalar distribution statistics for each feature, target, and bin context.
        # self._pairwise_miscoding_matrix_ # Stores the complete pairwise miscoding matrix once it has been computed.

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
        # self.miscoding_                # Equal-weight RMS of feature deficiency and surplus.

        # self.pairwise_miscoding_       # Lazily computed public property returning the complete pairwise miscoding matrix.

    def fit(self, X, y):
        """
        Store evaluation data and compute feature-level diagnostics.

        Reset diagnostic caches and defer pairwise miscoding computations
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
        self._pairwise_miscoding_matrix_ = None
        self._validate_X_type(self.X_type)
        self._validate_debug(self.debug)
        if y is None:
            raise ValueError("Miscoding.fit requires a target vector y.")

        self.X_, self.y_ = self._validate_X_y(X, y)
        self._model_X_ = X.copy(deep=True) if isinstance(X, pd.DataFrame) else self.X_
        self.n_samples_in_, self.n_features_in_ = self.X_.shape
        self.X_isnumeric_ = self._infer_X_isnumeric(X, self.X_)
        self.y_isnumeric_ = _resolve_y_isnumeric(self.y_, y_type=self.y_type)

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

        self.miscoding_ = np.array([
            self.aggregate_components(deficiency=d, surplus=s)
            for d, s in zip(self.deficiency_, self.surplus_)
        ])

        self.is_fitted_ = True
        return self

    def __sklearn_is_fitted__(self) -> bool:
        """Report whether fitting completed successfully."""
        return getattr(self, "is_fitted_", False)

    @staticmethod
    def aggregate_components(*, deficiency: float, surplus: float) -> float:
        """Combine supplied primitive components without fitting."""
        return miscoding(deficiency=deficiency, surplus=surplus)

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
        return self._feature_value(self.deficiency_, feature, "deficiency_feature")

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
        return self._feature_value(self.surplus_, feature, "surplus_feature")

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
            Equal-weight RMS of deficiency and surplus for each feature.
        """
        check_is_fitted(self)
        return self._feature_value(self.miscoding_, feature, "miscoding_feature")

    def _feature_value(self, values, feature, operation):
        """Resolve a feature name or integer position, or return all values."""
        if feature is None:
            for j in range(self.n_features_in_):
                diagnostics = self._empirical_subset_measures([j])
                if not diagnostics["is_reliable"]:
                    warn_unreliable_estimate(operation, diagnostics)
                    break
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
        warn_unreliable_estimate(operation, self._empirical_subset_measures([feature]))
        return float(values[feature])

    @property
    def pairwise_miscoding_(self) -> np.ndarray:
        """Compute the pairwise miscoding matrix on first access and return a copy."""
        check_is_fitted(self)
        if self._pairwise_miscoding_matrix_ is None:
            self._pairwise_miscoding_matrix_ = self._compute_pairwise_miscoding_matrix()
        return self._pairwise_miscoding_matrix_.copy()

    def pairwise_miscoding_matrix(self) -> pd.DataFrame:
        """
        Return the pairwise miscoding matrix between features.

        Compute any missing pairs and cache the complete matrix. Each call
        returns an independent DataFrame. Each pair uses

            d = (K(X_i, X_j) - K(X_i)) / K(X_j)
            s = (K(X_i, X_j) - K(X_j)) / K(X_i)
            miscoding = sqrt((d**2 + s**2) / 2)

        with a shared discretization for the joint and marginal distributions.
        A zero denominator contributes zero to its primitive component.
        Primitive values are clipped to [0, 1]. The diagonal is zero, and the
        target does not enter this calculation.

        Returns
        -------
        pandas.DataFrame
            Square matrix indexed and labeled by feature name. Values close to
            zero indicate that each feature describes the other well. Values
            close to one indicate little shared information according to the
            empirical code-length approximation.
        """
        check_is_fitted(self)
        return pd.DataFrame(
            self.pairwise_miscoding_,
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
            highest miscoding. Joint-state counts, bin counts, and reliability
            diagnostics are included without warnings.

            With ``debug=True``, ``code_length_bits`` is K(X_j); additional
            columns expose K(Y), K(X_j, Y), K(Y | X_j), and K(X_j | Y), as
            documented in ``subset_analysis()``. Marginal-state counts are
            included. Feature scores remain numeric even when the subset fails the
            joint reliability check; ``is_reliable`` reports that check.
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

        diagnostics = pd.DataFrame([
            self._empirical_subset_measures([j]) for j in range(self.n_features_in_)
        ])
        table = pd.concat([
            table, diagnostics.drop(columns=table.columns, errors="ignore"),
        ], axis=1)

        return table.sort_values(
            by           = ["miscoding", "deficiency", "surplus"],
            ascending    = [True, True, True],
            ignore_index = True,
        )

    #
    # Subset-level diagnostics
    #

    def miscoding_subset(self, subset) -> float:
        """Return subset miscoding, warning if empirical support is weak."""
        report = self.subset_analysis(subset)
        warn_unreliable_estimate("miscoding_subset", report)
        return float(report["miscoding"])

    def deficiency_subset(self, subset) -> float:
        """Return subset deficiency, warning if empirical support is weak."""
        report = self.subset_analysis(subset)
        warn_unreliable_estimate("deficiency_subset", report)
        return float(report["deficiency"])

    def surplus_subset(self, subset) -> float:
        """Return subset surplus, warning if empirical support is weak."""
        report = self.subset_analysis(subset)
        warn_unreliable_estimate("surplus_subset", report)
        return float(report["surplus"])

    def miscoding_model(self, model, *, X=None, feature_names=None, feature_indices=None) -> float:
        """Evaluate the feature subset effectively used by a fitted model.

        X defaults to fitted evaluation data. Explicit X contains estimator
        input columns; feature_indices maps those columns to fitted features.
        Unreliable estimates are returned with a RuntimeWarning. Model analysis
        returns diagnostics without issuing this warning.
        """
        report = self.model_analysis(model, X=X, feature_names=feature_names,
                                     feature_indices=feature_indices)
        value = float(report["miscoding"])
        warn_unreliable_estimate("miscoding_model", report)
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
            feature metadata, and reliability diagnostics. Pairwise miscoding
            is available separately through ``pairwise_miscoding_matrix()``.
            Reliability diagnostics include ``resolved_n_bins``, the numeric
            bin count for the subset, or None when no numeric discretization is
            applied, including an empty subset. Categorical variables retain
            their observed categories without binning.
            Unreliable estimates remain numerical and are reported quietly.
            The empty subset is reliable and has no joint-state diagnostics.

            With ``debug=True``, the report also includes:

            - ``code_length_bits``: K(X_S).
            - ``target_code_length_bits``: K(Y) in the subset's bin context.
            - ``joint_code_length_bits``: K(X_S, Y).
            - ``target_conditional_code_length_bits``: K(X_S, Y) - K(X_S).
            - ``feature_conditional_code_length_bits``: K(X_S, Y) - K(Y).
            - ``n_observed_feature_states`` and ``n_observed_target_states``.
            - ``target_n_bins``: the numeric target's bin count, or None.

            Conditional code lengths are raw empirical differences, without
            clipping. They remain available for sparse subsets and do not
            override reliability decisions. Numeric bins are resolved counts,
            not occupied-state counts. For the empty subset, K(X_S)=0,
            K(X_S, Y)=K(Y), and the target uses the feature-level bin context.
        """
        check_is_fitted(self)
        return self._subset_measures(subset)

    #
    # Feature selection and ordering
    #

    def select_features(self, *, max_features: int | None = None,
                        min_improvement: float = 0.0, return_details: bool = False,
                        include_pairwise_miscoding: bool = True,
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
        include_pairwise_miscoding : bool, default=True
            Include the full matrix under ``pairwise_miscoding`` in detailed
            output.
            False preserves the selection and path without computing pairwise
            miscoding. Ignored when return_details=False.

        Returns
        -------
        numpy.ndarray or dict
            Binary selection mask by default, or detailed selection output when
            ``return_details=True``. Detailed output is quiet; otherwise an
            unreliable selected step produces one RuntimeWarning.
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
            for step in path:
                if not step["is_reliable"]:
                    warn_unreliable_estimate("select_features", step)
                    break
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
            **({"pairwise_miscoding": self.pairwise_miscoding_matrix()}
               if include_pairwise_miscoding else {}),
        }

    def rank_features(
        self,
        *,
        max_features: int | None = None,
        criterion: RankingCriterion = "deficiency",
        return_details: bool = False,
        include_pairwise_miscoding: bool = True,
    ):
        """
        Rank features for model construction.

        The ranking is greedy and uses the same empirical subset diagnostics as
        ``miscoding_subset``. Unlike ``select_features()``, this method keeps
        adding features to the order even when subset miscoding stops improving,
        until the requested count is reached or no finite candidate remains.
        Reliability is diagnostic and does not affect candidate ordering.

        Parameters
        ----------
        max_features : non-negative int, optional
            Maximum number of features to rank. If omitted, every feature is
            eligible. Ranking stops when no finite extension remains.

        criterion : {"deficiency", "miscoding"}, default="deficiency"
            Candidate ordering criterion. ``"deficiency"`` prioritizes the
            lowest resulting subset deficiency, then miscoding, surplus, and
            feature index. ``"miscoding"`` prioritizes the lowest resulting
            subset miscoding, then deficiency, surplus, and feature index.

        return_details : bool, default=False
            If ``False``, return ordered feature indices. If ``True``, return a
            dictionary with the feature order, feature names, ranking path, and
            supporting diagnostics.
        include_pairwise_miscoding : bool, default=True
            Include the full matrix under ``pairwise_miscoding`` in detailed
            output. False preserves the order and path without computing
            pairwise miscoding. Ignored when return_details=False.

        Returns
        -------
        list[int] or dict
            Ordered feature indices by default, or detailed ranking output when
            ``return_details=True``. Detailed output is quiet; otherwise an
            unreliable selected step produces one RuntimeWarning.
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
            if not np.isfinite(best[criterion]):
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
            for step in path:
                if not step["is_reliable"]:
                    warn_unreliable_estimate("rank_features", step)
                    break
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
            **({"pairwise_miscoding": self.pairwise_miscoding_matrix()}
               if include_pairwise_miscoding else {}),
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
        self.feature_names_in_ = np.fromiter(_resolve_feature_names(X), dtype=object)
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
    # Pairwise miscoding and empirical subset diagnostics
    #

    def _debug_subset_diagnostics(self, selected: list[int], *, n_bins: int) -> dict[str, object]:
        """Expose raw code lengths and marginal counts in a shared bin context."""
        target = self._empirical_statistics_for_indices(y_included=True, n_bins=n_bins)
        features = (
            self._empirical_statistics_for_indices(features=selected, n_bins=n_bins)
            if selected else None
        )
        joint = (
            self._empirical_statistics_for_indices(
                features=selected, y_included=True, n_bins=n_bins,
            ) if selected else target
        )
        k_x = features.code_length if features is not None else 0.0
        return {
            "code_length_bits": k_x,
            "target_code_length_bits": target.code_length,
            "joint_code_length_bits": joint.code_length,
            "target_conditional_code_length_bits": joint.code_length - k_x,
            "feature_conditional_code_length_bits": joint.code_length - target.code_length,
            "n_observed_feature_states": features.n_states if features is not None else 1,
            "n_observed_target_states": target.n_states,
            "target_n_bins": n_bins if self.y_isnumeric_ else None,
        }

    def _compute_pairwise_miscoding_matrix(self) -> np.ndarray:
        """Assemble symmetric pairwise miscoding with a zero diagonal."""
        matrix = np.zeros((self.n_features_in_, self.n_features_in_), dtype=float)

        for i in range(self.n_features_in_):
            for j in range(i + 1, self.n_features_in_):
                value = self._feature_pair_miscoding(i, j)
                matrix[i, j] = value
                matrix[j, i] = value

        return matrix

    def _feature_pair_miscoding(self, i: int, j: int) -> float:
        """
        Compute symmetric miscoding using cached empirical statistics.
        """
        if i == j:
            return 0.0
        n_bins = self._resolve_n_bins_for_subset(2)
        k_i = self._empirical_statistics_for_indices(features=[i], n_bins=n_bins).code_length
        k_j = self._empirical_statistics_for_indices(features=[j], n_bins=n_bins).code_length
        k_ij = self._empirical_statistics_for_indices(features=[i, j], n_bins=n_bins).code_length

        deficiency = 0.0 if k_j <= 0.0 else float(np.clip((k_ij - k_i) / k_j, 0.0, 1.0))
        surplus = 0.0 if k_i <= 0.0 else float(np.clip((k_ij - k_j) / k_i, 0.0, 1.0))
        return self.aggregate_components(deficiency=deficiency, surplus=surplus)

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
                "miscoding"                : self.aggregate_components(deficiency=deficiency, surplus=0.0),
                "is_reliable"              : True,
                "failure_reason"           : None,
                "resolved_n_bins"          : None,
                "n_samples"                : int(self.n_samples_in_),
                "n_observed_joint_states"  : None,
                "mean_joint_occupancy"     : None,
                "n_singleton_joint_states" : None,
                "singleton_fraction"       : None,
                **(self._debug_subset_diagnostics(
                    selected, n_bins=self._resolve_n_bins_for_subset(1),
                ) if self.debug else {}),
            }

        n_bins = self._resolve_n_bins_for_subset(len(selected))
        joint_statistics = self._empirical_statistics_for_indices(
            features   = selected,
            y_included = True,
            n_bins     = n_bins,
        )
        diagnostics = self._joint_reliability_diagnostics(
            joint_statistics,
            resolved_n_bins=(
                n_bins if self.y_isnumeric_ or any(self.X_isnumeric_[j] for j in selected)
                else None
            ),
        )
        if self.debug:
            diagnostics.update(self._debug_subset_diagnostics(selected, n_bins=n_bins))

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
            "miscoding"  : self.aggregate_components(deficiency=deficiency, surplus=surplus),
            **diagnostics,
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
                    *current,
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
                "deficiency",
                "miscoding",
                "surplus",
                "feature_index",
            ]
        else:
            columns = [
                "miscoding",
                "deficiency",
                "surplus",
                "feature_index",
            ]

        return candidates.sort_values(
            by=columns,
            ascending=True,
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
    def _validate_X_type(cls, X_type: XType) -> None:
        """Validate the feature encoding policy before construction or fitting."""
        if X_type not in cls._VALID_X_TYPES:
            raise ValueError(
                f"Valid options for 'X_type' are {cls._VALID_X_TYPES}. "
                f"Got {X_type!r}."
            )

    @staticmethod
    def _validate_debug(debug: bool) -> None:
        """Require an explicit boolean for diagnostic reporting."""
        if not isinstance(debug, (bool, np.bool_)):
            raise TypeError("debug must be a boolean.")


#
# Functional interface
#

def feature_analysis(*, X, y, X_type: XType = "auto", y_type: YType = "auto",
                     debug: bool = False) -> pd.DataFrame:
    """
    Return feature analysis, including empirical details when debug=True.
    """
    metric = Miscoding(X_type=X_type, y_type=y_type, debug=debug).fit(X, y)
    return metric.feature_analysis()


def pairwise_miscoding_matrix(*, X, y, X_type: XType = "auto", y_type: YType = "auto") -> pd.DataFrame:
    """
    Return pairwise feature miscoding using a functional interface.
    """
    metric = Miscoding(X_type=X_type, y_type=y_type).fit(X, y)
    return metric.pairwise_miscoding_matrix()


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


def subset_analysis(subset, *, X, y, X_type: XType = "auto", y_type: YType = "auto",
                    debug: bool = False) -> dict[str, object]:
    """Return subset diagnostics, including empirical details when debug=True."""
    return Miscoding(X_type=X_type, y_type=y_type, debug=debug).fit(X, y).subset_analysis(subset)


def miscoding_model(model, *, X, y, feature_names=None, feature_indices=None,
                    X_type: XType = "auto", y_type: YType = "auto") -> float:
    """Compute fitted-model miscoding from evaluation data."""
    return Miscoding(X_type=X_type, y_type=y_type).fit(X, y).miscoding_model(
        model, feature_names=feature_names, feature_indices=feature_indices)


def model_analysis(model, *, X, y, feature_names=None, feature_indices=None,
                   X_type: XType = "auto", y_type: YType = "auto",
                   debug: bool = False) -> dict[str, object]:
    """Analyze a fitted model's feature subset with optional empirical details."""
    return Miscoding(X_type=X_type, y_type=y_type, debug=debug).fit(X, y).model_analysis(
        model, feature_names=feature_names, feature_indices=feature_indices)


def select_features(
    *,
    X,
    y,
    max_features: int | None = None,
    min_improvement: float = 0.0,
    return_details: bool = False,
    include_pairwise_miscoding: bool = True,
    X_type: XType = "auto",
    y_type: YType = "auto",
    debug: bool = False,
):
    """
    Select features, with optional empirical details in return_details output.
    """
    metric = Miscoding(X_type=X_type, y_type=y_type, debug=debug).fit(X, y)
    return metric.select_features(
        max_features              = max_features,
        min_improvement           = min_improvement,
        return_details            = return_details,
        include_pairwise_miscoding = include_pairwise_miscoding,
    )


def rank_features(
    *,
    X,
    y,
    max_features: int | None = None,
    criterion: RankingCriterion = "deficiency",
    return_details: bool = False,
    include_pairwise_miscoding: bool = True,
    X_type: XType = "auto",
    y_type: YType = "auto",
    debug: bool = False,
):
    """
    Rank features, with optional empirical details in return_details output.
    """
    metric = Miscoding(X_type=X_type, y_type=y_type, debug=debug).fit(X, y)
    return metric.rank_features(
        max_features=max_features,
        criterion=criterion,
        return_details=return_details,
        include_pairwise_miscoding=include_pairwise_miscoding,
    )
