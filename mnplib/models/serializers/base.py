"""
Base classes and formatting helpers for canonical scikit-learn serializers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np

from sklearn.utils.validation import check_is_fitted

from ..artifacts import ModelArtifacts

Task = Literal["classification", "regression"]

# Fixed canonical model-string policy used by all sklearn serializers.
indent                 : str   = " "
feature_token_template : str   = "X{index}"
class_token_template   : str   = "C{index}"
numeric_token_prefix   : str   = ""
zero_tolerance         : float = 0

class SklearnSerializer(ABC):
    """
    Abstract base class for scikit-learn model serializers.

    A serializer is responsible for extracting the selected feature subset and
    generating a canonical model string for one estimator family.
    """

    name            : str = "base"
    supported_types : tuple[type, ...] = ()

    def supports(self, model) -> bool:
        """
        Return ``True`` if this serializer supports the fitted estimator.
        """
        return isinstance(model, self.supported_types)

    def artifacts(self, model, X, *, feature_names=None, feature_indices=None) -> ModelArtifacts:
        """
        Extract explicit nescience artifacts from a fitted estimator.
        """
        require_fitted(model)

        n_features = int(model.n_features_in_)
        if feature_names is not None:
            resolve_feature_names(
                X,
                feature_names=feature_names,
                n_features=n_features,
            )

        indices = resolve_feature_indices(
            feature_indices,
            n_features=n_features,
        )
        tokens = [feature_token(index) for index in indices]

        subset       = self.subset(model)
        predictions  = np.asarray(model.predict(X))
        model_string = self.serialize(model, feature_names=tokens)

        return ModelArtifacts(
            subset       = subset,
            predictions  = predictions,
            model_string = model_string,
            model_type   = type(model).__name__,
        )

    @abstractmethod
    def task(self, model) -> Task:
        """
        Return the model task: ``"classification"`` or ``"regression"``.
        """

    @abstractmethod
    def subset(self, model) -> list[int]:
        """
        Return feature indices used by the fitted estimator.
        """

    @abstractmethod
    def serialize(self, model, *, feature_names: list[str]) -> str:
        """
        Return a canonical string description of the fitted estimator.
        """

def format_number(value: float) -> str:
    """
    Return a compact discretized representation of a floating-point value.
    """

    value = float(value)

    if not np.isfinite(value):
        if np.isnan(value):
            return f"{numeric_token_prefix}nan"
        sign = "+" if value > 0 else "-"
        return f"{numeric_token_prefix}{sign}inf"

    if abs(value) <= zero_tolerance:
        return f"{numeric_token_prefix}0"

    # Sort of discretization of real values
    scientific = '{:.2e}'.format(value)

    return f"{numeric_token_prefix}{scientific}"


def nonzero_mask(values) -> np.ndarray:
    """
    Return a fixed-policy non-zero mask for fitted numeric parameters.
    """
    return np.abs(np.asarray(values, dtype=float)) > zero_tolerance


def feature_token(index: int) -> str:
    """
    Return the compact feature reference for an original feature index.
    """
    return feature_token_template.format(index=int(index))


def class_token(index: int) -> str:
    """
    Return the compact class reference for a class index.
    """
    return class_token_template.format(index=int(index))


def resolve_feature_names(X, *, feature_names=None, n_features: int | None = None) -> list[str]:
    """
    Resolve feature names from explicit names, a DataFrame, or generated names.
    """
    if n_features is None:
        n_features = int(getattr(X, "shape")[1])

    if feature_names is None:
        if hasattr(X, "columns"):
            names = [str(name) for name in X.columns]
        else:
            names = [f"x{i}" for i in range(n_features)]
    else:
        names = [str(name) for name in feature_names]

    if len(names) != n_features:
        raise ValueError(
            "feature_names must have length {}. Got {} names instead."
            .format(n_features, len(names))
        )

    return names


def resolve_feature_indices(feature_indices=None, *, n_features: int) -> list[int]:
    """
    Resolve local adapter columns to canonical feature-token indices.
    """
    if feature_indices is None:
        indices = list(range(int(n_features)))
    else:
        indices = [int(index) for index in feature_indices]

    if len(indices) != int(n_features):
        raise ValueError(
            "feature_indices must have length {}. Got {} indices instead."
            .format(n_features, len(indices))
        )

    if len(indices) != len(set(indices)):
        raise ValueError("feature_indices must not contain duplicates.")

    if any(index < 0 for index in indices):
        raise ValueError("feature_indices must be non-negative.")

    return indices


def require_fitted(model) -> None:
    """
    Raise an informative error if the estimator is not fitted.
    """
    check_is_fitted(model)
