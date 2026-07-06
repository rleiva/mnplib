"""
Labeled miscoding based on empirical code lengths.

This module provides the :class:`Miscoding` estimator, a scikit-learn-compatible
utility for measuring how well a set of features represents a target variable.

The estimator provides feature-level diagnostics and subset-level diagnostics.
Feature-level diagnostics are computed from empirical code lengths. Subset-level
diagnostics aggregate the feature-level quantities through a redundancy-discounted
product of deficiencies and a redundancy-weighted surplus average.

The code-length estimates are computed through the stateless empirical
distribution utilities.

@author:    Rafael Garcia Leiva
@mail:      rgarcialeiva@gmail.com
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted

from .utils import empirical_distribution


XType = Literal["auto", "numeric", "categorical"]
YType = Literal["auto", "numeric", "categorical"]
BinSpec = int | Literal["auto"]
SubsetMode = Literal["deficiency", "surplus", "miscoding"]


class Miscoding(BaseEstimator):
    """
    Analyze supervised miscoding through deficiency and surplus.

    For each feature ``X_j`` and target ``Y``, the estimator computes

        deficiency_j = K(Y | X_j) / K(Y)
        surplus_j    = K(X_j | Y) / K(X_j)
        miscoding_j  = max(deficiency_j, surplus_j)

    For a subset of features ``S``, the estimator uses a
    redundancy-discounted aggregation:

        D(S) = product_j deficiency_j ** alpha_j,

    where

        alpha_j = 1 / (1 + sum_{k != j} rho_{jk}).

    Here ``rho_{jk}`` is the pairwise redundancy between features ``X_j`` and
    ``X_k``. The subset surplus is computed as a redundancy-weighted average of
    the individual feature surpluses. Subset miscoding is the maximum of the
    aggregated deficiency and surplus.

    Feature selection is performed greedily. At each step, the estimator adds
    the feature whose inclusion produces the largest reduction in subset
    miscoding according to the same redundancy-discounted aggregation.
    """

    _VALID_X_TYPES = ("auto", "numeric", "categorical")
    _VALID_Y_TYPES = ("auto", "numeric", "categorical")
    _VALID_SUBSET_MODES = ("deficiency", "surplus", "miscoding")

    def __init__(
        self,
        X_type: XType = "auto",
        y_type: YType = "auto",
        n_bins: BinSpec = "auto",
        min_improvement: float = 0.0,
    ):
        """
        Initialize the estimator.

        Parameters
        ----------
        X_type : {"auto", "numeric", "categorical"}, default="auto"
            Encoding strategy for the feature variables.

        y_type : {"auto", "numeric", "categorical"}, default="auto"
            Encoding strategy for the target variable.

        n_bins : int or "auto", default="auto"
            Number of uniform bins used to discretize numeric variables.

        min_improvement : float, default=0.0
            Minimum reduction in subset miscoding required to accept a feature
            during greedy feature selection. The value is measured on the
            normalized miscoding scale.
        """
        self._validate_init(
            X_type=X_type,
            y_type=y_type,
            min_improvement=min_improvement,
        )

        self.X_type = X_type
        self.y_type = y_type
        self.n_bins = n_bins
        self.min_improvement = min_improvement

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
        self.n_samples_in_, self.n_features_in_ = self.X_.shape
        self.X_isnumeric_ = self._infer_X_isnumeric(X, self.X_)
        self.y_isnumeric_ = self._infer_y_isnumeric(self.y_)

        self._code_length_cache_ = {}
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

    def feature_deficiency(self) -> np.ndarray:
        """
        Return the deficiency of each feature.

        Returns
        -------
        numpy.ndarray of shape (n_features,)
            Values of ``K(Y | X_j) / K(Y)`` for each feature.
        """
        check_is_fitted(self)
        return self.deficiency_.copy()

    def feature_surplus(self) -> np.ndarray:
        """
        Return the surplus of each feature.

        Returns
        -------
        numpy.ndarray of shape (n_features,)
            Values of ``K(X_j | Y) / K(X_j)`` for each feature.
        """
        check_is_fitted(self)
        return self.surplus_.copy()

    def feature_miscoding(self) -> np.ndarray:
        """
        Return the miscoding of each feature.

        Returns
        -------
        numpy.ndarray of shape (n_features,)
            Values of ``max(deficiency, surplus)`` for each feature.
        """
        check_is_fitted(self)
        return self.miscoding_.copy()

    def feature_redundancy(self) -> pd.DataFrame:
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
            ``feature_name``, ``is_numeric``, ``code_length``, ``deficiency``,
            ``surplus``, and ``miscoding``. Rows are sorted from lowest to
            highest miscoding.
        """
        check_is_fitted(self)

        table = pd.DataFrame(
            {
                "feature_index": np.arange(self.n_features_in_),
                "feature_name": self.feature_names_in_,
                "is_numeric": self.X_isnumeric_,
                "code_length": self.feature_code_lengths_,
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

    def miscoding_subset(
        self,
        subset,
        mode: SubsetMode = "miscoding",
    ) -> float:
        """
        Compute a redundancy-discounted subset-level miscoding quantity.

        Parameters
        ----------
        subset : array-like
            Binary mask of selected features or list of selected feature indices.

        mode : {"deficiency", "surplus", "miscoding"}, default="miscoding"
            Quantity to return.

        Returns
        -------
        float
            Requested subset-level value.
        """
        check_is_fitted(self)

        if mode not in self._VALID_SUBSET_MODES:
            raise ValueError(
                "Valid options for 'mode' are {}. Got mode={!r} instead."
                .format(self._VALID_SUBSET_MODES, mode)
            )

        return float(self._subset_measures(subset)[mode])

    def subset_analysis(self, subset) -> dict[str, object]:
        """
        Return detailed redundancy-discounted diagnostics for a feature subset.

        Parameters
        ----------
        subset : array-like
            Binary mask of selected features or list of selected feature indices.

        Returns
        -------
        dict
            Dictionary containing deficiency, surplus, miscoding, selected
            feature metadata, redundancy weights, and feature weights.
        """
        check_is_fitted(self)
        return self._subset_measures(subset)

    #
    # Greedy feature selection
    #

    def select_features(
        self,
        *,
        max_features: int | None = None,
        min_improvement: float | None = None,
        return_details: bool = False,
    ):
        """
        Select features by greedy redundancy-penalized aggregation.

        At each step, the method evaluates every candidate feature not yet
        selected and adds the feature that produces the largest reduction in
        subset miscoding. The subset score is computed with the same
        redundancy-discounted aggregation used by :meth:`miscoding_subset`.

        Parameters
        ----------
        max_features : int, optional
            Maximum number of features to select. If omitted, all features are
            eligible.

        min_improvement : float, optional
            Minimum reduction in subset miscoding required to accept a feature.
            If omitted, the estimator's configured ``min_improvement`` is used.

        return_details : bool, default=False
            If ``False``, return a binary selection mask. If ``True``, return a
            dictionary with the mask, selected indices, selected names, selection
            path, and final subset diagnostics.

        Returns
        -------
        numpy.ndarray or dict
            Binary selection mask by default, or detailed selection output when
            ``return_details=True``.
        """
        check_is_fitted(self)

        improvement_threshold = (
            self.min_improvement
            if min_improvement is None
            else float(min_improvement)
        )
        if improvement_threshold < 0:
            raise ValueError("min_improvement must be non-negative.")

        max_features = (
            self.n_features_in_
            if max_features is None
            else min(int(max_features), self.n_features_in_)
        )
        if max_features < 0:
            raise ValueError("max_features must be non-negative.")

        selected: list[int] = []
        path: list[dict[str, object]] = []
        current = self._subset_measures(selected)

        while len(selected) < max_features:
            candidates = self._selection_candidates(selected, current["miscoding"])
            if candidates.empty:
                break

            best = candidates.iloc[0]
            improvement = float(best["improvement"])

            if improvement <= improvement_threshold:
                break

            feature = int(best["feature_index"])
            selected.append(feature)
            current = self._subset_measures(selected)

            path.append(
                {
                    "step": len(path) + 1,
                    "feature_index": feature,
                    "feature_name": str(self.feature_names_in_[feature]),
                    "deficiency": current["deficiency"],
                    "surplus": current["surplus"],
                    "miscoding": current["miscoding"],
                    "improvement": improvement,
                    "selected_feature_indices": tuple(selected),
                    "selected_feature_names": tuple(
                        str(self.feature_names_in_[j]) for j in selected
                    ),
                }
            )

        mask = np.zeros(self.n_features_in_, dtype=int)
        mask[selected] = 1

        if not return_details:
            return mask

        return {
            "selected_features": mask,
            "selected_feature_indices": selected,
            "selected_feature_names": [
                str(self.feature_names_in_[j])
                for j in selected
            ],
            "min_improvement": float(improvement_threshold),
            "path": pd.DataFrame(path),
            "subset": self._subset_measures(selected),
            "features": self.feature_analysis(),
            "redundancy": self.feature_redundancy(),
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
        y_arr = np.ravel(np.asarray(y))
        if y_arr.size == 0:
            raise ValueError("y must not be empty.")

        if isinstance(X, pd.DataFrame):
            if len(X) != len(y_arr):
                raise ValueError(
                    f"X and y have inconsistent lengths: {len(X)} != {len(y_arr)}."
                )
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
            return X.to_numpy(), y_arr

        X_arr, y_arr = check_X_y(X, y_arr, dtype=None, ensure_2d=True)
        self.feature_names_in_ = np.asarray(
            [f"x{i}" for i in range(X_arr.shape[1])],
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

    def _code_length(self, columns, numeric) -> float:
        """
        Compute an empirical joint code length.

        Parameters
        ----------
        columns : sequence of iterable
            Variables to include in the joint code-length computation.

        numeric : sequence of bool
            Flags indicating whether each variable should be treated as numeric.

        Returns
        -------
        float
            Empirical code length of the supplied variables.
        """
        return float(
            empirical_distribution(
                columns=columns,
                numeric=numeric,
                n_bins=self.n_bins,
            ).code_length
        )

    def _code_length_for_indices(
        self,
        features: list[int] | tuple[int, ...] | None = None,
        y_included: bool = False,
    ) -> float:
        """
        Compute and cache a code length for a feature subset and optional target.
        """
        features = [] if features is None else list(features)
        feature_tuple = tuple(sorted(int(j) for j in features))
        key = feature_tuple + ((-1,) if y_included else tuple())

        if key in self._code_length_cache_:
            return self._code_length_cache_[key]

        columns = [self.X_[:, j] for j in feature_tuple]
        numeric = [self.X_isnumeric_[j] for j in feature_tuple]

        if y_included:
            columns.append(self.y_)
            numeric.append(self.y_isnumeric_)

        value = 0.0 if not columns else self._code_length(columns, numeric)
        self._code_length_cache_[key] = value
        return value

    def _conditional_target_length(self, selected) -> float:
        """Return ``K(Y | X_S)`` for a selected feature subset ``S``."""
        selected = list(selected)
        return max(
            0.0,
            self._code_length_for_indices(features=selected, y_included=True)
            - self._code_length_for_indices(features=selected, y_included=False),
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

        return max(
            0.0,
            self._code_length_for_indices(
                features=selected + [int(feature)],
                y_included=y_included,
            )
            - self._code_length_for_indices(
                features=selected,
                y_included=y_included,
            ),
        )

    #
    # Redundancy-discounted aggregation
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
        k_i = float(self.feature_code_lengths_[i])
        k_j = float(self.feature_code_lengths_[j])
        k_ij = float(self._code_length_for_indices(features=[i, j]))

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
        Compute redundancy-discounted deficiency, surplus, and miscoding for a
        selected feature subset.
        """
        selected = self._selected_indices(subset)

        mask = np.zeros(self.n_features_in_, dtype=int)
        mask[selected] = 1

        if len(selected) == 0:
            deficiency = 0.0 if self.target_code_length_ <= 0.0 else 1.0
            return {
                "deficiency": deficiency,
                "surplus": 0.0,
                "miscoding": deficiency,
                "features_in_use": mask,
                "n_features_in_use": 0,
                "selected_feature_indices": [],
                "selected_feature_names": [],
                "redundancy_weights": np.array([], dtype=float),
                "feature_weights": np.array([], dtype=float),
            }

        selected_array = np.asarray(selected, dtype=int)
        alpha = self._redundancy_weights(selected)
        feature_lengths = self.feature_code_lengths_[selected_array]

        deficiency_values = np.clip(self.deficiency_[selected_array], 0.0, 1.0)
        surplus_values = np.clip(self.surplus_[selected_array], 0.0, 1.0)

        deficiency = float(
            np.prod(np.power(deficiency_values, alpha))
        )

        feature_weights = alpha * feature_lengths
        weight_sum = float(np.sum(feature_weights))
        surplus = (
            0.0
            if weight_sum <= 0.0
            else float(np.sum(feature_weights * surplus_values) / weight_sum)
        )

        deficiency = float(np.clip(deficiency, 0.0, 1.0))
        surplus = float(np.clip(surplus, 0.0, 1.0))

        return {
            "deficiency": deficiency,
            "surplus": surplus,
            "miscoding": max(deficiency, surplus),
            "features_in_use": mask,
            "n_features_in_use": int(np.sum(mask)),
            "selected_feature_indices": selected,
            "selected_feature_names": [
                str(self.feature_names_in_[j])
                for j in selected
            ],
            "redundancy_weights": alpha,
            "feature_weights": feature_weights,
        }

    def _selection_candidates(
        self,
        selected: list[int],
        current_miscoding: float,
    ) -> pd.DataFrame:
        """
        Evaluate all candidate features for the next greedy selection step.
        """
        selected_set = set(selected)
        rows: list[dict[str, object]] = []

        for feature in range(self.n_features_in_):
            if feature in selected_set:
                continue

            candidate_subset = selected + [feature]
            values = self._subset_measures(candidate_subset)
            improvement = current_miscoding - float(values["miscoding"])

            rows.append(
                {
                    "feature_index": feature,
                    "feature_name": str(self.feature_names_in_[feature]),
                    "deficiency": float(values["deficiency"]),
                    "surplus": float(values["surplus"]),
                    "miscoding": float(values["miscoding"]),
                    "improvement": float(improvement),
                    "candidate_subset": tuple(candidate_subset),
                }
            )

        if not rows:
            return pd.DataFrame(
                columns=[
                    "feature_index",
                    "feature_name",
                    "deficiency",
                    "surplus",
                    "miscoding",
                    "improvement",
                    "candidate_subset",
                ]
            )

        return pd.DataFrame(rows).sort_values(
            by=["miscoding", "deficiency", "surplus", "feature_index"],
            ascending=[True, True, True, True],
            ignore_index=True,
        )

    #
    # Index handling and numerical helpers
    #

    def _selected_indices(self, selected) -> list[int]:
        """
        Normalize a binary mask or index list into validated feature indices.
        """
        if selected is None:
            return []

        arr = np.asarray(selected)
        if arr.size == 0:
            return []
        if arr.ndim != 1:
            raise ValueError("selected must be a one-dimensional mask or index list.")

        is_mask = (
            arr.shape[0] == self.n_features_in_
            and np.all(np.isin(arr, [0, 1, False, True]))
        )

        indices = (
            [int(j) for j in np.flatnonzero(arr.astype(int))]
            if is_mask
            else [int(j) for j in arr.tolist()]
        )

        if len(indices) != len(set(indices)):
            raise ValueError("selected contains duplicate feature indices.")
        if any(j < 0 or j >= self.n_features_in_ for j in indices):
            raise ValueError(
                f"selected indices must lie in [0, {self.n_features_in_ - 1}]."
            )

        return indices

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
        min_improvement,
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
        if min_improvement < 0:
            raise ValueError("min_improvement must be non-negative.")


#
# Functional interface
#


def feature_analysis(X, y, **kwargs) -> pd.DataFrame:
    """
    Return feature analysis using a functional interface.
    """
    metric = Miscoding(**kwargs).fit(X, y)
    return metric.feature_analysis()


def feature_redundancy(X, y, **kwargs) -> pd.DataFrame:
    """
    Return pairwise feature redundancy using a functional interface.
    """
    metric = Miscoding(**kwargs).fit(X, y)
    return metric.feature_redundancy()


def miscoding_subset(
    X,
    y,
    subset,
    *,
    mode: SubsetMode = "miscoding",
    **kwargs,
) -> float:
    """
    Return a subset-level miscoding quantity using a functional interface.
    """
    metric = Miscoding(**kwargs).fit(X, y)
    return metric.miscoding_subset(subset, mode=mode)


def select_features(
    X,
    y,
    *,
    max_features: int | None = None,
    min_improvement: float | None = None,
    return_details: bool = False,
    **kwargs,
):
    """
    Select features using a functional interface.
    """
    metric = Miscoding(**kwargs).fit(X, y)
    return metric.select_features(
        max_features=max_features,
        min_improvement=min_improvement,
        return_details=return_details,
    )
