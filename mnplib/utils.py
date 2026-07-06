"""
Empirical code-length utilities for the Minimum Nescience Principle.

This module provides stateless utilities for estimating empirical probability
distributions, empirical entropies, and empirical code lengths from observed
variables.

The implementation uses NumPy, pandas, and scikit-learn utilities for the
standard parts of the workflow:

    - scikit-learn validates one-dimensional inputs, numeric arrays, and
      consistent sample lengths;
    - pandas detects missing categorical values and factorizes categorical
      equality classes;
    - NumPy computes uniform bin edges, bin labels, joint states, counts,
      entropy, and code length.

Numeric variables are discretized independently using uniform bin edges. When
``n_bins='auto'``, the number of bins is selected with Rice's rule,

    ``ceil(2 * n_samples**(1/3))``,

bounded above by the number of samples.

Categorical variables are encoded according to order of first appearance. The
actual integer labels are not meaningful; only equality classes and empirical
frequencies are used.

The public API is intentionally small:

    - ``discretize_vector``
    - ``empirical_distribution``
    - ``empirical_entropy``
    - ``empirical_code_length``
    - ``entropy_from_counts``
    - ``code_length_from_counts``

@copyright: Rafael Garcia Leiva
@mail:      rgarcialeiva@gmail.com
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.utils.validation import (
    check_array,
    check_consistent_length,
    column_or_1d,
)


BinSpec = int | Literal["auto"]

__all__ = [
    "EmpiricalSummary",
    "discretize_vector",
    "empirical_distribution",
    "empirical_entropy",
    "empirical_code_length",
    "entropy_from_counts",
    "code_length_from_counts",
]


@dataclass(frozen=True)
class EmpiricalSummary:
    """
    Summary of an empirical distribution over jointly encoded variables.

    Parameters
    * states : numpy.ndarray
          Two-dimensional integer array containing the observed joint states.
          Each row represents one distinct state of the jointly encoded
          variables.
    * counts : numpy.ndarray
          Number of observations associated with each state.
    * probabilities : numpy.ndarray
          Empirical probabilities associated with each state.
    * entropy : float
          Empirical entropy of the distribution, measured in bits.
    * code_length : float
          Total empirical code length in bits, equal to ``n_samples * entropy``.
    * n_samples : int
          Number of observations used to estimate the empirical distribution.
    * n_states : int
          Number of distinct observed states.
    """
    states:        np.ndarray
    counts:        np.ndarray
    probabilities: np.ndarray
    entropy:       float
    code_length:   float
    n_samples:     int
    n_states:      int


#
# Validation and encoding
#


def _as_1d_array(x, name: str) -> np.ndarray:
    """
    Convert an input object to a non-empty one-dimensional NumPy array.

    Parameters
    * x : array-like
          Input values.
    * name : str
          Name used in error messages.

    Returns
    * numpy.ndarray
          One-dimensional NumPy array.

    Raises
    * ValueError
          If the input is not one-dimensional or is empty.
    """
    try:
        values = column_or_1d(x, warn=False)
    except ValueError as exc:
        raise ValueError(f"{name} must be a one-dimensional array.") from exc

    values = np.asarray(values)
    if values.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return values


def _validate_columns(
    columns: Sequence[Iterable],
    numeric: Sequence[bool],
) -> tuple[list[np.ndarray], list[bool], int]:
    """
    Validate jointly observed variables.

    Parameters
    * columns : sequence of iterable
          Observed variables. All variables must be one-dimensional and must have
          the same number of observations.
    * numeric : sequence of bool
          Flags indicating whether each variable should be treated as numeric.

    Returns
    * arrays : list of numpy.ndarray
          Validated one-dimensional arrays.
    * numeric_flags : list of bool
          Normalized numeric flags.
    * n_samples : int
          Common number of observations.

    Raises
    * ValueError
          If no variables are provided, if the number of flags does not match the
          number of variables, or if variables have inconsistent lengths.
    """
    if len(columns) == 0:
        raise ValueError("At least one random variable must be provided.")
    if len(columns) != len(numeric):
        raise ValueError("columns and numeric must have the same length.")

    arrays = [_as_1d_array(column, name=f"columns[{i}]") for i, column in enumerate(columns)]

    try:
        check_consistent_length(*arrays)
    except ValueError as exc:
        raise ValueError("All random variables must have the same number of samples.") from exc

    return arrays, [bool(flag) for flag in numeric], arrays[0].size


def _numeric_array(x, name: str) -> np.ndarray:
    """
    Validate a numeric vector and return it as finite floats.

    Parameters
    * x : array-like
          Numeric values.
    * name : str
          Name used in error messages.

    Returns
    * numpy.ndarray
          Floating-point vector.

    Raises
    * ValueError
          If the vector contains non-numeric, NaN, or infinite values.
    """
    values = _as_1d_array(x, name=name).reshape(-1, 1)

    try:
        values = check_array(
            values,
            dtype=float,
            ensure_2d=True,
            ensure_all_finite=True,
            input_name=name,
        )
    except TypeError:
        # Compatibility with older scikit-learn versions.
        values = check_array(
            values,
            dtype=float,
            ensure_2d=True,
            force_all_finite=True,
            input_name=name,
        )
    except ValueError as exc:
        raise ValueError(f"{name} must contain only finite numeric values.") from exc

    return values.ravel()


def _categorical_codes(x, name: str) -> np.ndarray:
    """
    Encode a categorical vector as integer symbols.

    Missing categorical values are rejected.

    Encoding follows the order of first appearance. This is sufficient for
    empirical entropy and code-length computations, because only equality
    classes and empirical frequencies are used.

    Parameters
    * x : array-like
          Categorical values.
    * name : str
          Name used in error messages.

    Returns
    * numpy.ndarray
          Integer-encoded categorical vector.

    Raises
    * ValueError
          If a missing categorical value is found.
    * TypeError
          If a categorical value is unhashable.
    """
    values = _as_1d_array(x, name=name)

    if pd.isna(values).any():
        raise ValueError(f"{name} contains missing categorical values.")

    try:
        codes, _ = pd.factorize(values, sort=False)
    except TypeError as exc:
        raise TypeError(f"{name} contains unhashable categorical values.") from exc

    return codes.astype(int, copy=False)


def _encode_columns(
    columns: Sequence[Iterable],
    numeric: Sequence[bool],
    n_bins: BinSpec = "auto"
) -> np.ndarray:
    """
    Encode variables as integer columns.

    Parameters
    * columns : sequence of iterable
          Variables to encode jointly. All must be non-empty, one-dimensional,
          and have the same length.
    * numeric : sequence of bool
          Whether each variable is numeric. Numeric variables are discretized;
          categorical variables are factorized into integer codes.
    * n_bins : int or "auto", default="auto"
          Number of bins for numeric variables. If ``"auto"``, Rice's rule is used.
    Returns
    * numpy.ndarray
          Integer array of shape ``(n_samples, n_variables)``.

    Raises
    * ValueError
          If the inputs are inconsistent, invalid, missing, non-finite, or if the
          bin specification is invalid.
    *  TypeError
          If categorical values cannot be factorized.
    """
    arrays, numeric_flags, n_samples = _validate_columns(columns, numeric)
    bins   = _resolve_bins(n_bins, n_samples=n_samples)

    encoded = [
        discretize_vector(column, n_bins=bins)
        if is_numeric
        else _categorical_codes(column, name=f"columns[{i}]")
        for i, (column, is_numeric) in enumerate(zip(arrays, numeric_flags))
    ]

    return np.column_stack(encoded).astype(int)


#
# Numeric discretization
#


def _resolve_bins(
    n_bins: BinSpec,
    n_samples: int,
) -> int:
    """
    Resolve an explicit or automatic bin specification.

    The automatic rule is Rice's rule:

        ceil(2 * n_samples**(1/3))

    The returned value is constrained to be at most ``n_samples``.

    Parameters
    * n_bins : int or "auto"
          Requested bin specification.
    * n_samples : int
          Number of observations.

    Returns
    * int
          Resolved number of bins.

    Raises
    * ValueError
          If ``n_samples`` is not positive, or if ``n_bins`` is neither
          ``"auto"`` nor a positive integer.
    """
    n_samples = int(n_samples)

    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    if n_bins == "auto":
        bins = int(np.ceil(2.0 * n_samples ** (1.0 / 3.0)))
        return int(min(bins, n_samples))

    bins = int(n_bins)
    if bins < 1:
        raise ValueError("n_bins must be a positive integer or 'auto'.")
    return bins


def discretize_vector(
    x,
    n_bins: BinSpec = "auto"
) -> np.ndarray:
    """
    Discretize a finite numeric vector using uniform bins.

    This is a stateless one-shot discretization function. Bin edges are learned
    from the input vector and are not returned or stored.

    Parameters
    * x : array-like
          Numeric vector to discretize.
    * n_bins : int or "auto", default="auto"
          Number of uniform bins. If ``"auto"``, Rice's rule is used.

    Returns
    * numpy.ndarray
          Integer bin labels.
    """
    values = _numeric_array(x, name="x")
    bins   = _resolve_bins(n_bins, n_samples=values.size)
    
    if bins <= 1 or float(np.min(values)) == float(np.max(values)):
        return np.zeros(values.size, dtype=int)

    edges = np.histogram_bin_edges(values, bins=bins)
    return np.digitize(values, edges[1:-1], right=False).astype(int)


#
# Counts, entropy, and code length
#


def _unique_states_and_counts(encoded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return observed joint states and their counts.

    Parameters
    * encoded : numpy.ndarray
          Two-dimensional array of encoded variables.

    Returns
    * states : numpy.ndarray
          Distinct observed states.
    * counts : numpy.ndarray
          Frequency of each observed state.

    Raises
    * ValueError
          If ``encoded`` is not two-dimensional.
    """
    encoded = check_array(encoded, dtype=int, ensure_2d=True)
    states, counts = np.unique(encoded, axis=0, return_counts=True)
    return states.astype(int), counts.astype(float)


def _validate_counts(counts: Sequence[float]) -> tuple[np.ndarray, float]:
    """
    Validate empirical state counts.

    Parameters
    * counts : sequence of float
          Counts associated with observed states.

    Returns
    * counts_array : numpy.ndarray
          One-dimensional floating-point array containing the validated counts.
    * n_samples : float
          Total number of observations, equal to the sum of ``counts_array``.

    Raises
    * ValueError
          If ``counts`` is empty, is not one-dimensional, contains non-finite values,
          contains negative values, or has a sum less than or equal to zero.
    """

    counts_array = check_array(
        counts,
        dtype=float,
        ensure_2d=False,
        ensure_min_samples=1,
        ensure_all_finite=True,
    )

    if counts_array.ndim != 1:
        raise ValueError("counts must be one-dimensional.")
    if np.any(counts_array < 0):
        raise ValueError("counts must be non-negative.")

    n_samples = float(np.sum(counts_array))
    if n_samples <= 0.0:
        raise ValueError("The sum of counts must be positive.")

    return counts_array, n_samples


def entropy_from_counts(counts: Sequence[float]) -> float:
    """
    Compute plug-in empirical entropy from state counts, measured in bits.

    Parameters
    * counts : sequence of float
          State counts.

    Returns
    * float
          Plug-in empirical entropy.

    Raises
    * ValueError
          If counts are invalid.
    """
    counts_array, n_samples = _validate_counts(counts)
    probabilities           = counts_array[counts_array > 0.0] / n_samples
    return -float(np.sum(probabilities * np.log2(probabilities)))


def code_length_from_counts(counts: Sequence[float]) -> float:
    """
    Compute empirical code length from state counts, measured in bits.

    Parameters
    * counts : sequence of float
          State counts.

    Returns
    * float
          Empirical code length.
    """
    counts_array, n_samples = _validate_counts(counts)
    nonzero_counts          = counts_array[counts_array > 0.0]
    probabilities           = nonzero_counts / n_samples
    return float(np.sum(nonzero_counts * (-np.log2(probabilities))))


#
# Public empirical summaries
#

def empirical_distribution(
    columns:  Sequence[Iterable],
    numeric:  Sequence[bool],
    n_bins:   BinSpec = "auto"
) -> EmpiricalSummary:
    """
    Estimate the empirical distribution of jointly encoded variables.

    Parameters
    * columns : sequence of iterable
          Variables observed on the same samples. Each element must be a
          one-dimensional sequence.
    * numeric : sequence of bool
          Flags indicating whether each variable is numeric. Numeric variables
          are discretized; categorical variables are symbolically encoded.
    * n_bins : int or "auto", default="auto"
          Number of uniform bins for numeric variables.
          If ``"auto"``, Rice's rule is used.

    Returns
    * EmpiricalSummary
          Summary containing observed states, counts, probabilities, entropy,
          code length, and metadata.
    """
    encoded = _encode_columns(
        columns,
        numeric,
        n_bins=n_bins
    )

    states, counts = _unique_states_and_counts(encoded)
    n_samples      = int(np.sum(counts))
    probabilities  = counts / n_samples
    entropy        = entropy_from_counts(counts)

    return EmpiricalSummary(
        states        = states,
        counts        = counts,
        probabilities = probabilities,
        entropy       = entropy,
        code_length   = float(n_samples * entropy),
        n_samples     = n_samples,
        n_states      = int(states.shape[0]),
    )
