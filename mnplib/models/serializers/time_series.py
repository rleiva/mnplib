"""AST construction for finite-window linear forecasting rules."""

import numpy as np
from sklearn.utils.validation import check_is_fitted
from ..artifacts import ModelArtifacts
from ..description import ModelDescription
from ..language import Feature
from ...utils import _resolve_feature_names
from .base import resolve_feature_indices
from .linear import linear_expression


def fixed_model_description(model, *, feature_names=None, feature_indices=None):
    """Describe a fitted finite-window forecaster in original lag coordinates."""
    check_is_fitted(model)
    indices = resolve_feature_indices(feature_indices, n_features=int(model.n_features_in_))
    names = _resolve_feature_names(feature_names=feature_names, n_features=len(indices))
    features = tuple(Feature(index, str(name)) for index, name in zip(indices, names))
    weights = np.asarray(model.weights_, dtype=float)
    subset = [index for index, weight in zip(indices, weights) if weight != 0]
    return ModelDescription(
        type(model).__name__, linear_expression(weights, model.intercept_, features),
        tuple(indices), tuple(str(name) for name in names),
        {"features_used": subset, "n_terms": int(np.count_nonzero(weights))},
    )


def time_series_model_artifacts(model, X, *, feature_names=None, feature_indices=None):
    """Use the shared canonical description for lag-based evaluation artifacts."""
    from ..description import describe_model
    description = describe_model(model, feature_names=feature_names, feature_indices=feature_indices)
    return ModelArtifacts(list(description.metadata["features_used"]),
                          np.asarray(model.predict(X)), description)
