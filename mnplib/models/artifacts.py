"""Evaluation data paired with an immutable canonical model description."""

from dataclasses import dataclass
import numpy as np
from .description import ModelDescription


def _fitted_subset(model, feature_indices):
    """Extract effective inputs from fitted prediction parameters, not hints."""
    from ..timeseries.models import FixedLinearForecaster

    if isinstance(model, FixedLinearForecaster):
        local_indices = np.flatnonzero(model.weights_ != 0)
    else:
        from .sklearn import _find_serializer
        local_indices = _find_serializer(model).subset(model)
    return [feature_indices[index] for index in local_indices]


@dataclass(frozen=True)
class ModelArtifacts:
    """Evaluation inputs kept independent of descriptive model metadata."""

    subset: list[int]
    predictions: np.ndarray
    description: ModelDescription

    @property
    def model_string(self):
        return self.description.canonical

    @property
    def model_type(self):
        return self.description.model_type

    def to_nescience_kwargs(self):
        return {"subset": self.subset, "predictions": self.predictions,
                "model_string": self.model_string}
