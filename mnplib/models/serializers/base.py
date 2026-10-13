"""Fitted-model validation and the common AST-to-description boundary."""

from abc import ABC, abstractmethod
from ..._types import ResolvedTask
from numbers import Integral
import numpy as np
from sklearn.utils.validation import check_is_fitted
from ...utils import _resolve_feature_names
from ..artifacts import ModelArtifacts, _fitted_subset
from ..description import ModelDescription
from ..language import And, Binary, Conditional, Feature


def score_conditionals(scores, labels):
    """Build explicit comparisons, choosing the first label on score ties."""
    if not labels or len(scores) != len(labels):
        raise ValueError("Class scores must match a non-empty sequence of labels.")
    result = labels[-1]
    for index in reversed(range(len(labels) - 1)):
        comparisons = tuple(
            Binary(">=", scores[index], scores[later])
            for later in range(index + 1, len(labels))
        )
        condition = comparisons[0] if len(comparisons) == 1 else And(comparisons)
        result = Conditional(condition, labels[index], result)
    return result


class SklearnSerializer(ABC):
    """Base class for serializers of supported scikit-learn estimators."""

    name = "base"
    supported_types = ()

    def supports(self, model):
        """Return whether this serializer supports the estimator type."""
        return isinstance(model, self.supported_types)

    def description(self, model, *, feature_names=None, feature_indices=None):
        """Build the semantic description of a fitted estimator."""
        check_is_fitted(model)

        count = int(model.n_features_in_)
        indices = resolve_feature_indices(feature_indices, n_features=count)

        if feature_names is None:
            feature_names = getattr(model, "feature_names_in_", None)

        names    = _resolve_feature_names(feature_names=feature_names, n_features=count)
        features = tuple(Feature(index, str(name)) for index, name in zip(indices, names))
        subset   = [indices[index] for index in self.subset(model)]
        metadata = {**self.metadata(model), "features_used": subset}

        return ModelDescription(
            type(model).__name__,
            self.serialize(model, features=features),
            tuple(indices),
            tuple(str(name) for name in names),
            metadata,
        )

    def artifacts(self, model, X, *, feature_names=None, feature_indices=None,):
        """Return the description, effective subset, and model predictions."""
        if feature_names is None and hasattr(X, "columns"):
            feature_names = list(X.columns)

        description = self.description(
            model,
            feature_names   = feature_names,
            feature_indices = feature_indices,
        )

        return ModelArtifacts(
            _fitted_subset(model, description.feature_indices),
            np.asarray(model.predict(X)),
            description,
        )

    def metadata(self, model):
        """Return compact descriptive hints, never execution or scoring inputs."""
        return {}

    @abstractmethod
    def task(self, model) -> ResolvedTask:
        """Return whether the estimator performs classification or regression."""

    @abstractmethod
    def subset(self, model):
        """Return the local indices of features used by the prediction rule."""

    @abstractmethod
    def serialize(self, model, *, features):
        """Translate the fitted estimator into a semantic AST."""


def resolve_feature_indices(feature_indices=None, *, n_features):
    """Validate and return the mapping to the original feature space."""

    indices = (
        list(range(n_features))
        if feature_indices is None
        else list(feature_indices)
    )

    if len(indices) != n_features:
        raise ValueError(
            f"feature_indices must have length {n_features}."
        )

    if any(
        isinstance(index, bool) or not isinstance(index, Integral)
        for index in indices
    ):
        raise ValueError("feature_indices must contain integers.")

    indices = [int(index) for index in indices]

    if len(set(indices)) != len(indices):
        raise ValueError(
            "feature_indices must not contain duplicates."
        )

    if any(index < 0 for index in indices):
        raise ValueError(
            "feature_indices must be non-negative."
        )

    return indices
    
