"""
Public scikit-learn adapter API.

This module exposes convenience functions that convert supported scikit-learn
estimators into explicit artifacts and apply the simplified ``Nescience`` API.
"""

from __future__ import annotations

from .artifacts import ModelArtifacts, SerializationConfig
from .registry import SklearnModelRegistry
from .serializers.linear import LinearModelSerializer, LogisticRegressionSerializer
from .serializers.tree import DecisionTreeSerializer


def create_default_registry() -> SklearnModelRegistry:
    """
    Create a registry with all stable built-in scikit-learn serializers.
    """
    return SklearnModelRegistry(
        serializers=[
            DecisionTreeSerializer(),
            LinearModelSerializer(),
            LogisticRegressionSerializer(),
        ]
    )


registry = create_default_registry()


def sklearn_model_artifacts(
    model,
    X,
    *,
    feature_names=None,
    config: SerializationConfig | None = None,
) -> ModelArtifacts:
    """
    Extract explicit nescience artifacts from a supported scikit-learn model.
    """
    return registry.artifacts(
        model,
        X,
        feature_names=feature_names,
        config=config,
    )


def nescience_model(
    metric,
    model,
    X,
    *,
    feature_names=None,
    config: SerializationConfig | None = None,
) -> float:
    """
    Compute nescience for a supported scikit-learn model.
    """
    artifacts = sklearn_model_artifacts(
        model,
        X,
        feature_names=feature_names,
        config=config,
    )
    return metric.nescience(**artifacts.to_nescience_kwargs())


def components_model(
    metric,
    model,
    X,
    *,
    feature_names=None,
    config: SerializationConfig | None = None,
) -> dict[str, float]:
    """
    Compute the four nescience components for a supported scikit-learn model.
    """
    artifacts = sklearn_model_artifacts(
        model,
        X,
        feature_names=feature_names,
        config=config,
    )
    return metric.components(**artifacts.to_nescience_kwargs())


def explain_model(
    metric,
    model,
    X,
    *,
    feature_names=None,
    config: SerializationConfig | None = None,
) -> dict[str, object]:
    """
    Explain nescience for a supported scikit-learn model.
    """
    artifacts = sklearn_model_artifacts(
        model,
        X,
        feature_names=feature_names,
        config=config,
    )
    explanation = metric.explain(**artifacts.to_nescience_kwargs())
    explanation["model_type"] = artifacts.model_type
    explanation["model_metadata"] = artifacts.metadata
    return explanation


def score_model(
    metric,
    model,
    X,
    *,
    feature_names=None,
    config: SerializationConfig | None = None,
) -> float:
    """
    Return ``1 - nescience`` for a supported scikit-learn model.
    """
    artifacts = sklearn_model_artifacts(
        model,
        X,
        feature_names=feature_names,
        config=config,
    )
    return metric.score(**artifacts.to_nescience_kwargs())


def register_sklearn_serializer(serializer) -> None:
    """
    Register a new serializer in the global default registry.
    """
    registry.register(serializer)
