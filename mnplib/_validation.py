"""Shared validation for empirical metric inputs."""

import numpy as np


def validate_vector(values, *, name):
    """Require a non-empty one-dimensional vector without reshaping it."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return array
