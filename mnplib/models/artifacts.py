"""Evaluation data paired with an immutable canonical model description."""

from dataclasses import dataclass
import numpy as np
from .description import ModelDescription


@dataclass(frozen=True)
class ModelArtifacts:
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
