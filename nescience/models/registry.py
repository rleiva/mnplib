"""
Registry for scikit-learn model serializers.
"""

from __future__ import annotations

from collections.abc import Iterable

from .artifacts import ModelArtifacts, SerializationConfig
from .serializers.base import SklearnSerializer


class SklearnModelRegistry:
    """
    Registry of scikit-learn serializers.

    Serializers are tried in registration order. A serializer is selected when
    ``serializer.supports(model)`` returns ``True``.
    """

    def __init__(self, serializers: Iterable[SklearnSerializer] | None = None):
        """
        Create a registry.

        Parameters
        ----------
        serializers : iterable of SklearnSerializer, optional
            Initial serializers to register.
        """
        self._serializers: list[SklearnSerializer] = []

        if serializers is not None:
            for serializer in serializers:
                self.register(serializer)

    def register(self, serializer: SklearnSerializer) -> None:
        """
        Register a serializer instance.
        """
        if not isinstance(serializer, SklearnSerializer):
            raise TypeError("serializer must be an instance of SklearnSerializer.")

        self._serializers.append(serializer)

    def artifacts(
        self,
        model,
        X,
        *,
        feature_names=None,
        config: SerializationConfig | None = None,
    ) -> ModelArtifacts:
        """
        Extract explicit nescience artifacts from a supported fitted estimator.
        """
        config = SerializationConfig() if config is None else config

        for serializer in self._serializers:
            if serializer.supports(model):
                return serializer.artifacts(
                    model,
                    X,
                    feature_names=feature_names,
                    config=config,
                )

        raise NotImplementedError(
            "No scikit-learn serializer registered for {}."
            .format(type(model).__name__)
        )

    def supported_model_types(self) -> tuple[type, ...]:
        """
        Return all estimator classes supported by the registered serializers.
        """
        supported: list[type] = []

        for serializer in self._serializers:
            supported.extend(serializer.supported_types)

        return tuple(supported)

    def serializer_names(self) -> tuple[str, ...]:
        """
        Return names of registered serializers.
        """
        return tuple(serializer.name for serializer in self._serializers)
