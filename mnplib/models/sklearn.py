"""
scikit-learn adapter dispatch for supported mnplib model families.

The adapter layer converts fitted, library-supported scikit-learn estimators
into the explicit artifacts consumed by nescience metrics.

Unsupported estimators fail explicitly.
"""

from __future__ import annotations

from .artifacts                  import ModelArtifacts
from .serializers.linear         import LinearModelSerializer, LogisticRegressionSerializer
from .serializers.naive_bayes    import NaiveBayesSerializer
from .serializers.neural_network import MLPSerializer
from .serializers.svm            import LinearSVMSerializer
from .serializers.tree           import DecisionTreeSerializer

_SUPPORTED_SERIALIZERS = (
    DecisionTreeSerializer(),
    LinearModelSerializer(),
    LogisticRegressionSerializer(),
    LinearSVMSerializer(),
    NaiveBayesSerializer(),
    MLPSerializer(),
)

def _supported_model_type_names() -> tuple[str, ...]:
    names: list[str] = []
    for serializer in _SUPPORTED_SERIALIZERS:
        names.extend(model_type.__name__ for model_type in serializer.supported_types)
    return tuple(names)


def _find_serializer(model):
    """
    Return the static serializer for a supported fitted estimator.
    """
    for serializer in _SUPPORTED_SERIALIZERS:
        if serializer.supports(model):
            return serializer

    supported = ", ".join(_supported_model_type_names())
    raise ValueError(
        "Unsupported scikit-learn model type {}. Supported model types are: {}. "
        "No generic or repr-based model serialization is available."
        .format(type(model).__name__, supported)
    )


def sklearn_model_artifacts(model, X, *, feature_names=None, feature_indices=None) -> ModelArtifacts:
    """
    Extract explicit nescience artifacts from a supported scikit-learn model.
    """
    serializer = _find_serializer(model)
    return serializer.artifacts(
        model,
        X,
        feature_names=feature_names,
        feature_indices=feature_indices,
    )


def nescience_model(metric, model, X, *, feature_names=None) -> float:
    """
    Compute nescience for a supported scikit-learn model.
    """
    artifacts = sklearn_model_artifacts(
        model,
        X,
        feature_names=feature_names,
    )
    return metric.nescience(**artifacts.to_nescience_kwargs())


def components_model(metric, model, X, *, feature_names=None) -> dict[str, float]:
    """
    Compute the four nescience components for a supported scikit-learn model.
    """
    artifacts = sklearn_model_artifacts(
        model,
        X,
        feature_names=feature_names,
    )
    return metric.components(**artifacts.to_nescience_kwargs())


def explain_model(metric, model, X, *, feature_names=None) -> dict[str, object]:
    """
    Explain nescience for a supported scikit-learn model.
    """
    artifacts = sklearn_model_artifacts(
        model,
        X,
        feature_names=feature_names,
    )
    explanation = metric.explain(**artifacts.to_nescience_kwargs())
    explanation["model_type"] = artifacts.model_type
    return explanation


def score_model(metric, model, X, *, feature_names=None) -> float:
    """
    Return ``1 - nescience`` for a supported scikit-learn model.
    """
    artifacts = sklearn_model_artifacts(
        model,
        X,
        feature_names=feature_names,
    )
    return metric.score(**artifacts.to_nescience_kwargs())
