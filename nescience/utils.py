"""
utils.py

Machine learning with the Minimum Nescience Principle.

This module provides utilities for estimating empirical distributions,
empirical entropies, and empirical code lengths from observed random
variables. Numeric variables are discretized using uniform binning;
categorical variables are encoded as integer symbols; joint empirical
frequencies are then used to compute Shannon-type quantities.

@author:    Rafael Garcia Leiva
@mail:      rgarcialeiva@gmail.com
@web:       http://www.mathematicsunknown.com/
@copyright: GNU GPLv3
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np


BinSpec = int | Literal["auto"]
MissingPolicy = Literal["raise"]
EntropyCorrection = Literal["none", "miller_madow", "dirichlet"]


__all__ = [
    "EmpiricalSummary",
    "optimal_number_of_bins",
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
    Detailed empirical summary of one or more jointly encoded random variables.
    """

    states:        np.ndarray
    counts:        np.ndarray
    probabilities: np.ndarray
    entropy:       float
    code_length:   float
    n_samples:     int
    n_states:      int
    base:          float
    correction:    str


def _validate_base(base: float) -> float:
    base = float(base)

    if base <= 0.0 or base == 1.0:
        raise ValueError("base must be positive and different from 1.")

    return base


def _log_base(x: np.ndarray | float, base: float) -> np.ndarray | float:
    return np.log(x) / np.log(base)


def _as_1d_array(x, name: str = "x") -> np.ndarray:
    """Convert input into a non-empty one-dimensional NumPy array."""
    arr = np.asarray(x)

    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")

    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")

    return arr


def _validate_missing_policy(missing: str) -> None:
    if missing != "raise":
        raise NotImplementedError(
            "Only missing='raise' is currently implemented."
        )


def _contains_missing_or_infinite(arr: np.ndarray, *, numeric: bool) -> bool:
    """
    Return True if the array contains missing or invalid values.

    For numeric variables, NaN and infinite values are considered invalid.
    For categorical variables, None and NaN are considered missing.
    """
    if numeric:
        try:
            arr_float = arr.astype(float)
        except (TypeError, ValueError):
            return True

        return bool(np.any(~np.isfinite(arr_float)))

    for value in arr:
        if value is None:
            return True

        try:
            if bool(np.isnan(value)):
                return True
        except (TypeError, ValueError):
            pass

    return False


def _validate_columns(
    columns: Sequence[Iterable],
    numeric: Sequence[bool],
) -> tuple[list[np.ndarray], list[bool], int]:
    """Validate a collection of random variables."""
    if len(columns) == 0:
        raise ValueError("At least one random variable must be provided.")

    if len(columns) != len(numeric):
        raise ValueError("columns and numeric must have the same length.")

    arrays: list[np.ndarray] = []
    n_samples: int | None = None

    for i, col in enumerate(columns):
        arr = _as_1d_array(col, name=f"columns[{i}]")

        if n_samples is None:
            n_samples = arr.size
        elif arr.size != n_samples:
            raise ValueError(
                "All random variables must have the same number of samples."
            )

        arrays.append(arr)

    assert n_samples is not None

    return arrays, [bool(flag) for flag in numeric], n_samples


def _normalize_n_bins(n_bins: BinSpec) -> int | Literal["auto"]:
    if n_bins == "auto":
        return "auto"

    bins = int(n_bins)

    if bins < 1:
        raise ValueError("n_bins must be a positive integer or 'auto'.")

    return bins


def _encode_categorical(
    x,
    *,
    name: str = "x",
    missing: MissingPolicy = "raise",
) -> np.ndarray:
    """
    Encode a categorical one-dimensional vector as integer symbols.

    The encoding follows the order of first appearance. This avoids sorting
    requirements and is sufficient because only equality classes and empirical
    frequencies are used later.
    """
    _validate_missing_policy(missing)

    arr = _as_1d_array(x, name=name)

    if _contains_missing_or_infinite(arr, numeric=False):
        raise ValueError(f"{name} contains missing categorical values.")

    mapping: dict[object, int] = {}
    encoded = np.empty(arr.size, dtype=int)

    for i, value in enumerate(arr):
        if not isinstance(value, Hashable):
            raise TypeError(
                f"{name} contains an unhashable categorical value at "
                f"position {i}: {value!r}"
            )

        if value not in mapping:
            mapping[value] = len(mapping)

        encoded[i] = mapping[value]

    return encoded


def optimal_number_of_bins(
    n_samples: int,
    n_numeric: int = 1,
    n_categorical_states: int = 1,
    *,
    min_samples_per_cell: int = 5,
    min_bins: int = 2,
    max_bins: int | None = None,
) -> int:
    """
    Compute an occupancy-based number of bins per numeric random variable.

    The rule chooses b so that:

        n_categorical_states * b**n_numeric ~= n_samples / min_samples_per_cell
    """
    n_samples = int(n_samples)
    n_numeric = int(n_numeric)
    n_categorical_states = int(n_categorical_states)
    min_samples_per_cell = int(min_samples_per_cell)
    min_bins = int(min_bins)

    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    if n_numeric < 0:
        raise ValueError("n_numeric must be non-negative.")

    if n_numeric == 0:
        return 1

    if n_categorical_states <= 0:
        raise ValueError("n_categorical_states must be positive.")

    if min_samples_per_cell <= 0:
        raise ValueError("min_samples_per_cell must be positive.")

    if min_bins < 1:
        raise ValueError("min_bins must be positive.")

    if max_bins is not None:
        max_bins = int(max_bins)
        if max_bins < min_bins:
            raise ValueError("max_bins must be greater than or equal to min_bins.")

    effective_cells = n_samples / (min_samples_per_cell * n_categorical_states)

    if effective_cells <= 1.0:
        bins = min_bins
    else:
        bins = int(np.floor(effective_cells ** (1.0 / n_numeric)))
        bins = max(min_bins, bins)

    bins = min(bins, n_samples)

    if max_bins is not None:
        bins = min(bins, max_bins)

    return int(bins)


def _compute_uniform_edges(x: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Compute internal bin edges for uniform discretization.

    Returns n_bins - 1 internal edges. Constant vectors return an empty edge
    array, corresponding to one effective bin.
    """
    x_min = float(np.min(x))
    x_max = float(np.max(x))

    if x_min == x_max:
        return np.array([], dtype=float)

    n_bins = max(1, int(n_bins))

    if n_bins == 1:
        return np.array([], dtype=float)

    return np.linspace(x_min, x_max, n_bins + 1, dtype=float)[1:-1]


def _transform_with_edges(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Transform a numeric vector using internal bin edges."""
    return np.searchsorted(edges, x, side="right").astype(int)


def _validate_numeric(
    x,
    *,
    name: str = "x",
    missing: MissingPolicy = "raise",
) -> np.ndarray:
    _validate_missing_policy(missing)

    arr = _as_1d_array(x, name=name)

    if _contains_missing_or_infinite(arr, numeric=True):
        raise ValueError(f"{name} must contain only finite numeric values.")

    return arr.astype(float, copy=False)


def discretize_vector(
    x,
    *,
    n_bins: BinSpec = "auto",
    missing: MissingPolicy = "raise",
    min_samples_per_cell: int = 5,
    max_bins: int | None = None,
) -> np.ndarray:
    """
    Fit uniform bin edges on a numeric vector and discretize it.

    This is a stateless one-shot discretization function. The bin edges are
    learned from the vector being discretized and are not stored.
    """
    arr = _validate_numeric(x, name="x", missing=missing)

    normalized_bins = _normalize_n_bins(n_bins)

    if normalized_bins == "auto":
        bins = optimal_number_of_bins(
            n_samples=arr.size,
            n_numeric=1,
            n_categorical_states=1,
            min_samples_per_cell=min_samples_per_cell,
            min_bins=2,
            max_bins=max_bins,
        )
    else:
        bins = normalized_bins

    edges = _compute_uniform_edges(arr, bins)

    return _transform_with_edges(arr, edges)


def _count_joint_states(encoded_columns: Sequence[np.ndarray]) -> int:
    """Count observed joint states in already-encoded columns."""
    if len(encoded_columns) == 0:
        return 1

    if len(encoded_columns) == 1:
        return int(np.unique(encoded_columns[0]).size)

    joint = np.column_stack(encoded_columns)
    return int(np.unique(joint, axis=0).shape[0])


def _encode_columns(
    columns: Sequence[Iterable],
    numeric: Sequence[bool],
    *,
    n_bins: BinSpec = "auto",
    missing: MissingPolicy = "raise",
    min_samples_per_cell: int = 5,
    min_bins: int = 2,
    max_bins: int | None = None,
) -> np.ndarray:
    """Encode categorical and numeric columns into integer states."""
    arrays, numeric_flags, n_samples = _validate_columns(columns, numeric)

    categorical_encoded: list[np.ndarray] = []

    for i, (arr, is_numeric) in enumerate(zip(arrays, numeric_flags)):
        if not is_numeric:
            categorical_encoded.append(
                _encode_categorical(
                    arr,
                    name=f"columns[{i}]",
                    missing=missing,
                )
            )

    n_categorical_states = _count_joint_states(categorical_encoded)
    n_numeric = int(sum(numeric_flags))

    normalized_bins = _normalize_n_bins(n_bins)

    if normalized_bins == "auto":
        bins_for_numeric = optimal_number_of_bins(
            n_samples=n_samples,
            n_numeric=n_numeric,
            n_categorical_states=n_categorical_states,
            min_samples_per_cell=min_samples_per_cell,
            min_bins=min_bins,
            max_bins=max_bins,
        )
    else:
        bins_for_numeric = normalized_bins

    encoded_columns: list[np.ndarray] = []
    categorical_index = 0

    for i, (arr, is_numeric) in enumerate(zip(arrays, numeric_flags)):
        if is_numeric:
            numeric_arr = _validate_numeric(
                arr,
                name=f"columns[{i}]",
                missing=missing,
            )
            edges = _compute_uniform_edges(numeric_arr, bins_for_numeric)
            encoded_columns.append(_transform_with_edges(numeric_arr, edges))
        else:
            encoded_columns.append(categorical_encoded[categorical_index])
            categorical_index += 1

    return np.column_stack(encoded_columns).astype(int)


def _unique_states_and_counts(encoded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute observed states and counts from encoded random variables."""
    if encoded.ndim != 2:
        raise ValueError("encoded must be a two-dimensional array.")

    states, counts = np.unique(encoded, axis=0, return_counts=True)

    return states.astype(int), counts.astype(float)


def entropy_from_counts(
    counts: Sequence[float],
    *,
    base: float = 2.0,
    correction: EntropyCorrection = "none",
    alpha: float = 0.5,
    alphabet_size: int | None = None,
    normalized: bool = False,
) -> float:
    """Compute entropy from empirical counts."""
    base = _validate_base(base)

    counts_arr = np.asarray(counts, dtype=float)

    if counts_arr.ndim != 1:
        raise ValueError("counts must be one-dimensional.")

    if counts_arr.size == 0:
        raise ValueError("counts must not be empty.")

    if np.any(counts_arr < 0):
        raise ValueError("counts must be non-negative.")

    n_samples = float(np.sum(counts_arr))

    if n_samples <= 0.0:
        raise ValueError("The sum of counts must be positive.")

    observed_states = int(np.count_nonzero(counts_arr))

    if alphabet_size is None:
        alphabet_size = observed_states
    else:
        alphabet_size = int(alphabet_size)
        if alphabet_size < observed_states:
            raise ValueError("alphabet_size cannot be smaller than observed states.")

    if correction == "none":
        positive_counts = counts_arr[counts_arr > 0.0]
        probabilities = positive_counts / n_samples
        entropy = -float(np.sum(probabilities * _log_base(probabilities, base)))

    elif correction == "miller_madow":
        positive_counts = counts_arr[counts_arr > 0.0]
        probabilities = positive_counts / n_samples
        plugin_entropy = -float(
            np.sum(probabilities * _log_base(probabilities, base))
        )
        entropy = plugin_entropy + (
            (observed_states - 1) / (2.0 * n_samples * np.log(base))
        )

    elif correction == "dirichlet":
        if alpha <= 0:
            raise ValueError("alpha must be positive for Dirichlet correction.")

        if alphabet_size > counts_arr.size:
            padded = np.zeros(alphabet_size, dtype=float)
            padded[: counts_arr.size] = counts_arr
            counts_arr = padded

        smoothed = counts_arr + float(alpha)
        probabilities = smoothed / np.sum(smoothed)
        entropy = -float(np.sum(probabilities * _log_base(probabilities, base)))

    else:
        raise ValueError(
            "correction must be 'none', 'miller_madow', or 'dirichlet'."
        )

    if normalized:
        if alphabet_size <= 1:
            return 0.0

        max_entropy = _log_base(alphabet_size, base)
        entropy = entropy / max_entropy

    return float(entropy)


def code_length_from_counts(
    counts: Sequence[float],
    *,
    base: float = 2.0,
    correction: EntropyCorrection = "none",
    alpha: float = 0.5,
    alphabet_size: int | None = None,
    per_sample: bool = False,
    normalized: bool = False,
) -> float:
    """Compute empirical code length from counts."""
    counts_arr = np.asarray(counts, dtype=float)
    n_samples = float(np.sum(counts_arr))

    entropy = entropy_from_counts(
        counts_arr,
        base=base,
        correction=correction,
        alpha=alpha,
        alphabet_size=alphabet_size,
        normalized=normalized,
    )

    if per_sample:
        return float(entropy)

    return float(n_samples * entropy)


def empirical_distribution(
    columns: Sequence[Iterable],
    numeric: Sequence[bool],
    *,
    n_bins: BinSpec = "auto",
    missing: MissingPolicy = "raise",
    min_samples_per_cell: int = 5,
    min_bins: int = 2,
    max_bins: int | None = None,
    base: float = 2.0,
    correction: EntropyCorrection = "none",
    alpha: float = 0.5,
    alphabet_size: int | None = None,
    normalized: bool = False,
    per_sample: bool = False,
) -> EmpiricalSummary:
    """Estimate the empirical distribution of one or more random variables."""
    base = _validate_base(base)

    encoded = _encode_columns(
        columns,
        numeric,
        n_bins=n_bins,
        missing=missing,
        min_samples_per_cell=min_samples_per_cell,
        min_bins=min_bins,
        max_bins=max_bins,
    )

    states, counts = _unique_states_and_counts(encoded)
    n_samples = int(np.sum(counts))
    probabilities = counts / n_samples

    entropy = entropy_from_counts(
        counts,
        base=base,
        correction=correction,
        alpha=alpha,
        alphabet_size=alphabet_size,
        normalized=normalized,
    )

    if per_sample:
        code_length = entropy
    else:
        code_length = n_samples * entropy

    return EmpiricalSummary(
        states=states,
        counts=counts,
        probabilities=probabilities,
        entropy=float(entropy),
        code_length=float(code_length),
        n_samples=n_samples,
        n_states=int(states.shape[0]),
        base=base,
        correction=correction,
    )


def empirical_entropy(
    columns: Sequence[Iterable],
    numeric: Sequence[bool],
    *,
    n_bins: BinSpec = "auto",
    missing: MissingPolicy = "raise",
    min_samples_per_cell: int = 5,
    min_bins: int = 2,
    max_bins: int | None = None,
    base: float = 2.0,
    correction: EntropyCorrection = "none",
    alpha: float = 0.5,
    alphabet_size: int | None = None,
    normalized: bool = False,
) -> float:
    """Estimate empirical entropy of one or more jointly encoded variables."""
    summary = empirical_distribution(
        columns,
        numeric,
        n_bins=n_bins,
        missing=missing,
        min_samples_per_cell=min_samples_per_cell,
        min_bins=min_bins,
        max_bins=max_bins,
        base=base,
        correction=correction,
        alpha=alpha,
        alphabet_size=alphabet_size,
        normalized=normalized,
        per_sample=True,
    )

    return float(summary.entropy)


def empirical_code_length(
    columns: Sequence[Iterable],
    numeric: Sequence[bool],
    *,
    n_bins: BinSpec = "auto",
    missing: MissingPolicy = "raise",
    min_samples_per_cell: int = 5,
    min_bins: int = 2,
    max_bins: int | None = None,
    base: float = 2.0,
    correction: EntropyCorrection = "none",
    alpha: float = 0.5,
    alphabet_size: int | None = None,
    per_sample: bool = False,
    normalized: bool = False,
) -> float:
    """Estimate empirical code length of one or more jointly encoded variables."""
    summary = empirical_distribution(
        columns,
        numeric,
        n_bins=n_bins,
        missing=missing,
        min_samples_per_cell=min_samples_per_cell,
        min_bins=min_bins,
        max_bins=max_bins,
        base=base,
        correction=correction,
        alpha=alpha,
        alphabet_size=alphabet_size,
        normalized=normalized,
        per_sample=per_sample,
    )

    return float(summary.code_length)
