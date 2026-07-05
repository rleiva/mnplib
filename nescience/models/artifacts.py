"""
Shared artifacts and serialization configuration for model adapters.

The simplified nescience metrics require explicit artifacts rather than fitted
model objects. This module defines those artifacts and the configuration used
by canonical model serializers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np


SupportLevel = Literal["stable", "beta", "experimental"]


@dataclass(frozen=True)
class SerializationConfig:
    """
    Configuration for canonical model serialization.

    Parameters
    ----------
    precision : int, default=6
        Number of decimal places used for floating-point values.

    zero_tolerance : float, default=1e-12
        Coefficients with absolute value less than or equal to this tolerance
        are treated as zero.

    include_metadata : bool, default=True
        Whether serializers should include compact structural information in
        the ``PARAMETERS`` section.

    schema_name : str, default="canonical_nescience_model_v1"
        Name of the canonical description schema.

    indent : str, default="    "
        Indentation used in nested canonical descriptions.
    """

    precision: int = 6
    zero_tolerance: float = 1e-12
    include_metadata: bool = True
    schema_name: str = "canonical_nescience_model_v1"
    indent: str = "    "

    def __post_init__(self) -> None:
        """Validate the serialization configuration."""
        if self.precision < 0:
            raise ValueError("precision must be non-negative.")

        if self.zero_tolerance < 0:
            raise ValueError("zero_tolerance must be non-negative.")

        if not isinstance(self.schema_name, str) or not self.schema_name:
            raise ValueError("schema_name must be a non-empty string.")

        if not isinstance(self.indent, str):
            raise TypeError("indent must be a string.")


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

    metadata : dict, default={}
        Optional model-specific diagnostics. The metric classes do not depend
        on this field; it is provided for reporting and debugging.
    """

    subset: list[int]
    predictions: np.ndarray
    model_string: str
    model_type: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_nescience_kwargs(self) -> dict[str, Any]:
        """
        Return the keyword arguments expected by the simplified ``Nescience`` API.
        """
        return {
            "subset": self.subset,
            "predictions": self.predictions,
            "model_string": self.model_string,
        }
