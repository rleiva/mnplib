"""
Model adapters for mnplib.

The adapter layer converts concrete fitted models into the explicit artifacts
required by the simplified metric classes.
"""

from .artifacts import ModelArtifacts, SerializationConfig
from .sklearn import (
    components_model,
    explain_model,
    nescience_model,
    score_model,
    sklearn_model_artifacts,
)

__all__ = [
    "ModelArtifacts",
    "SerializationConfig",
    "sklearn_model_artifacts",
    "nescience_model",
    "components_model",
    "explain_model",
    "score_model",
]
