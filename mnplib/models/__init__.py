"""
Model adapters for mnplib.

The adapter layer converts concrete fitted models into the explicit artifacts
required by the simplified metric classes.
"""

from .artifacts import ModelArtifacts, SerializationConfig
from .registry import SklearnModelRegistry
from .sklearn import (
    components_model,
    create_default_registry,
    explain_model,
    nescience_model,
    register_sklearn_serializer,
    score_model,
    sklearn_model_artifacts,
)

__all__ = [
    "ModelArtifacts",
    "SerializationConfig",
    "SklearnModelRegistry",
    "create_default_registry",
    "sklearn_model_artifacts",
    "nescience_model",
    "components_model",
    "explain_model",
    "score_model",
    "register_sklearn_serializer",
]
