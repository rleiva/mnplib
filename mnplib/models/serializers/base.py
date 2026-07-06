"""
Base classes and formatting helpers for canonical scikit-learn serializers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np

from sklearn.utils.validation import check_is_fitted

from ..artifacts import ModelArtifacts, SerializationConfig, SupportLevel


Task = Literal["classification", "regression"]


class SklearnSerializer(ABC):
    """
    Abstract base class for scikit-learn model serializers.

    A serializer is responsible for extracting the selected feature subset and
    generating a canonical model string for one estimator family.
    """

    name: str = "base"
    support_level: SupportLevel = "experimental"
    supported_types: tuple[type, ...] = ()

    def supports(self, model) -> bool:
        """
        Return ``True`` if this serializer supports the fitted estimator.
        """
        return isinstance(model, self.supported_types)

    def artifacts(
        self,
        model,
        X,
        *,
        feature_names=None,
        config: SerializationConfig | None = None,
    ) -> ModelArtifacts:
        """
        Extract explicit nescience artifacts from a fitted estimator.
        """
        require_fitted(model)
        config = SerializationConfig() if config is None else config

        names = resolve_feature_names(
            X,
            feature_names=feature_names,
            n_features=int(model.n_features_in_),
        )

        subset = self.subset(model, config=config)
        predictions = np.asarray(model.predict(X))
        model_string = self.serialize(
            model,
            feature_names=names,
            config=config,
        )

        metadata = self.metadata(
            model,
            feature_names=names,
            subset=subset,
            config=config,
        )
        metadata.setdefault("task", self.task(model))
        metadata.setdefault("schema", config.schema_name)
        metadata.setdefault("support_level", self.support_level)
        metadata.setdefault("serializer", self.name)
        metadata.setdefault("n_features_in", int(model.n_features_in_))
        metadata.setdefault("n_features_in_use", int(len(subset)))
        metadata.setdefault("selected_feature_names", [names[j] for j in subset])

        return ModelArtifacts(
            subset=subset,
            predictions=predictions,
            model_string=model_string,
            model_type=type(model).__name__,
            metadata=metadata,
        )

    @abstractmethod
    def task(self, model) -> Task:
        """
        Return the model task: ``"classification"`` or ``"regression"``.
        """

    @abstractmethod
    def subset(self, model, *, config: SerializationConfig) -> list[int]:
        """
        Return feature indices used by the fitted estimator.
        """

    @abstractmethod
    def serialize(
        self,
        model,
        *,
        feature_names: list[str],
        config: SerializationConfig,
    ) -> str:
        """
        Return a canonical string description of the fitted estimator.
        """

    def metadata(
        self,
        model,
        *,
        feature_names: list[str],
        subset: list[int],
        config: SerializationConfig,
    ) -> dict:
        """
        Return optional model-specific metadata.
        """
        return {}


def canonical_header(
    *,
    model_type: str,
    task: Task,
    feature_names: list[str],
    config: SerializationConfig,
) -> list[str]:
    """
    Return the shared canonical header used by every serializer.
    """
    inputs = ", ".join(feature_names) if feature_names else "<none>"

    return [
        f"SCHEMA {config.schema_name}",
        f"MODEL {model_type}",
        f"TASK {task}",
        f"INPUTS {inputs}",
    ]


def format_number(value: float, config: SerializationConfig) -> str:
    """
    Return a stable canonical representation of a floating-point value.
    """
    value = float(value)

    if abs(value) <= config.zero_tolerance:
        value = 0.0

    text = f"{value:.{config.precision}f}"

    if text == "-0." + ("0" * config.precision):
        text = "0." + ("0" * config.precision)

    return text


def format_label(value) -> str:
    """
    Return a stable canonical representation of a target label.
    """
    if isinstance(value, np.generic):
        value = value.item()

    return repr(value)


def resolve_feature_names(
    X,
    *,
    feature_names=None,
    n_features: int | None = None,
) -> list[str]:
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


def require_fitted(model) -> None:
    """
    Raise an informative error if the estimator is not fitted.
    """
    check_is_fitted(model)
