"""
Shared artifacts and internal serialization policy for model adapters.

The nescience metrics require explicit artifacts rather than fitted model
objects. Model strings affect surfeit, so the serialization policy is fixed by
the library and is not exposed as a user preference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

@dataclass(frozen=True)
class ModelArtifacts:
    """
    Explicit artifacts required to compute model nescience.

    Parameters
    ----------
    subset : list of int
        Indices of the input features used by the fitted model.

    predictions : numpy.ndarray
        Predictions produced by the model on the evaluation data.

    model_string : str
        Canonical string description of the fitted model.

    model_type : str
        Name of the estimator class.
    """

    subset       : list[int]
    predictions  : np.ndarray
    model_string : str
    model_type   : str

    def to_nescience_kwargs(self) -> dict[str, Any]:
        """
        Return the keyword arguments expected by the simplified ``Nescience`` API.
        """
        return {
            "subset"       : self.subset,
            "predictions"  : self.predictions,
            "model_string" : self.model_string,
        }
