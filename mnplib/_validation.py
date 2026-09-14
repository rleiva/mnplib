"""Shared validation for empirical metric inputs and discretization settings."""

import numpy as np


def validate_vector(values, *, name):
    """Require a non-empty one-dimensional vector without reshaping it."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return array


def validate_n_bins(n_bins):
    """Accept an integer bin count or a supported automatic bin policy."""
    if isinstance(n_bins, str) and n_bins in ("auto", "adaptive"):
        return n_bins
    if (isinstance(n_bins, (bool, np.bool_))
            or not isinstance(n_bins, (int, np.integer)) or n_bins < 2):
        raise ValueError("n_bins must be an integer >= 2, 'auto', or 'adaptive'.")
    return int(n_bins)
