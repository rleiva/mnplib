"""
Surfeit based on compressed model descriptions.

This module implements the surfeit component of nescience. Surfeit measures
how much unnecessary structure appears to be present in a model description.

The class is intentionally string-based: it computes surfeit from an explicit
model description string. Model-specific serialization is deliberately kept
outside this class, so the metric remains independent of any particular machine
learning library or estimator type.

@author:    Rafael Garcia Leiva
@mail:      rgarcialeiva@gmail.com
@copyright: GNU GPLv3
"""

from __future__ import annotations

from typing import Literal

import zlib

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted

from .utils import empirical_distribution

YType = Literal["auto", "numeric", "categorical"]
BinSpec = int | Literal["auto"]

class Surfeit(BaseEstimator):
    """
    Compute the surfeit of a model description string.

    Surfeit is estimated by comparing the raw length of a model description with
    a compressed reference length. The compressed length is corrected by
    subtracting the fixed zlib wrapper overhead, then bounded by the target code
    length. This prevents the reference description from exceeding the amount of
    information available in the target representation.

    Parameters
    ----------
    y_type : {"auto", "numeric", "categorical"}, default="auto"
        Encoding strategy for the target variable.

    n_bins : int or "auto", default="auto"
        Number of uniform bins used for numeric targets. If ``"auto"``,
        Rice's rule is used by the empirical-distribution utilities.

    zlib_level : int, default=9
        Compression level passed to ``zlib.compress``. Must be between 0 and 9.

    zlib_overhead : int, default=6
        Estimated zlib wrapper overhead, in bytes, subtracted from the raw
        compressed length.
    """

    _VALID_Y_TYPES = ("auto", "numeric", "categorical")

    def __init__(
        self,
        y_type: YType = "auto",
        n_bins: BinSpec = "auto",
        zlib_level: int = 9,
        zlib_overhead: int = 6,
    ):
        """Initialize the estimator and validate configuration parameters."""
        self._validate_init(
            y_type=y_type,
            zlib_level=zlib_level,
            zlib_overhead=zlib_overhead,
        )

        self.y_type = y_type
        self.n_bins = n_bins
        self.zlib_level = int(zlib_level)
        self.zlib_overhead = int(zlib_overhead)

    def fit(self, X, y):
        """
        Fit the surfeit object with a dataset.

        The feature matrix is stored only to preserve familiar estimator shape
        attributes. Surfeit itself depends on the target representation and on a
        model description string.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : Surfeit
            Fitted estimator.
        """
        self.X_, self.y_ = check_X_y(X, y, dtype=None, ensure_2d=True)
        self._fit_target(self.y_)
        self.n_features_in_ = self.X_.shape[1]

        return self

    def fit_y(self, y):
        """
        Fit the surfeit object with only a target vector.

        This method is useful when the model description is already available as
        a string and no feature matrix is needed.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : Surfeit
            Fitted estimator.
        """
        self.X_ = None
        self._fit_target(y)

        return self

    def surfeit_string(self, model_string: str) -> float:
        """
        Compute the surfeit of a model description string.

        Parameters
        ----------
        model_string : str
            String representation of a model or description.

        Returns
        -------
        float
            Surfeit value in the interval [0, 1].
        """
        check_is_fitted(self)

        model_bytes = self._validate_model_string(model_string)
        model_length = len(model_bytes)
        compressed_length = len(self._compress_bytes(model_bytes))

        return self._surfeit_from_lengths(
            model_length=model_length,
            compressed_length=compressed_length,
        )

    def _fit_target(self, y) -> None:
        """Fit target-dependent attributes."""
        self.y_ = self._validate_1d_vector(y, name="y")
        self.y_isnumeric_ = self._infer_y_isnumeric(self.y_)
        self.len_y_ = self._target_code_length()
        self.n_samples_in_ = self.y_.shape[0]
        self.is_fitted_ = True

    def _target_code_length(self) -> float:
        """Return the empirical code length of the fitted target in bits."""
        return float(
            empirical_distribution(
                columns=[self.y_],
                numeric=[self.y_isnumeric_],
                n_bins=self.n_bins,
            ).code_length
        )

    def _surfeit_from_lengths(self, model_length: int, compressed_length: int) -> float:
        """Compute surfeit from raw and compressed model-description lengths."""
        effective_length = self._effective_compressed_length(
            compressed_length=compressed_length,
            model_length=model_length,
        )

        reference_length = min(float(self.len_y_), float(effective_length))
        value = 1.0 - reference_length / float(model_length)

        return float(np.clip(value, 0.0, 1.0))

    def _effective_compressed_length(self, compressed_length: int, model_length: int) -> int:
        """
        Return zlib-compressed length after overhead correction.

        The corrected compressed length is clipped to the interval
        ``[0, model_length]`` so that compression never increases the reference
        description length.
        """
        effective_length = int(compressed_length) - int(self.zlib_overhead)
        effective_length = max(0, effective_length)
        effective_length = min(int(model_length), effective_length)

        return effective_length

    def _compress_bytes(self, data: bytes) -> bytes:
        """Compress bytes using the configured zlib level."""
        return zlib.compress(data, level=self.zlib_level)

    @staticmethod
    def _validate_model_string(model_string: str) -> bytes:
        """Validate a model description string and return its UTF-8 bytes."""
        if not isinstance(model_string, str):
            raise TypeError("model_string must be a string.")

        model_bytes = model_string.encode("utf-8")

        if len(model_bytes) == 0:
            raise ValueError("model_string must not be empty.")

        return model_bytes

    def _infer_y_isnumeric(self, y: np.ndarray) -> bool:
        """
        Infer whether the target should be treated as numeric.

        Returns
        -------
        bool
            True for numeric/regression targets, False for categorical targets.
        """
        if self.y_type == "numeric":
            return True

        if self.y_type == "categorical":
            return False

        target_type = type_of_target(y)

        if target_type in ("binary", "multiclass"):
            return False

        if target_type == "continuous":
            return True

        raise ValueError(
            "Unsupported target type {!r}. Supported one-dimensional target "
            "types are binary, multiclass, and continuous. You may also set "
            "y_type explicitly to 'numeric' or 'categorical'."
            .format(target_type)
        )

    @staticmethod
    def _validate_1d_vector(values, *, name: str) -> np.ndarray:
        """Convert an input vector into a non-empty one-dimensional NumPy array."""
        arr = np.asarray(values)

        if arr.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional array.")

        if arr.shape[0] == 0:
            raise ValueError(f"{name} must not be empty.")

        return arr

    @classmethod
    def _validate_init(
        cls,
        *,
        y_type,
        zlib_level,
        zlib_overhead,
    ) -> None:
        """Validate constructor arguments before storing them on the estimator."""
        if y_type not in cls._VALID_Y_TYPES:
            raise ValueError(
                "Valid options for 'y_type' are {}. Got y_type={!r} instead."
                .format(cls._VALID_Y_TYPES, y_type)
            )

        zlib_level = int(zlib_level)
        if zlib_level < 0 or zlib_level > 9:
            raise ValueError(
                "zlib_level must be an integer between 0 and 9. "
                f"Got zlib_level={zlib_level!r} instead."
            )

        zlib_overhead = int(zlib_overhead)
        if zlib_overhead < 0:
            raise ValueError("zlib_overhead must be non-negative.")


def surfeit_score(
    model_string: str,
    y,
    *,
    y_type: YType = "auto",
    n_bins: BinSpec = "auto",
    zlib_level: int = 9,
    zlib_overhead: int = 6,
) -> float:
    """
    Compute the surfeit of a model description string.

    This functional interface does not require an explicit ``Surfeit`` object.

    Parameters
    ----------
    model_string : str
        String representation of a model or description.

    y : array-like of shape (n_samples,)
        Target values.

    y_type : {"auto", "numeric", "categorical"}, default="auto"
        Encoding strategy for the target variable.

    n_bins : int or "auto", default="auto"
        Number of uniform bins used for numeric targets.

    zlib_level : int, default=9
        Compression level passed to ``zlib.compress``.

    zlib_overhead : int, default=6
        Estimated zlib wrapper overhead, in bytes.

    Returns
    -------
    float
        Surfeit value in the interval [0, 1].
    """
    metric = Surfeit(
        y_type=y_type,
        n_bins=n_bins,
        zlib_level=zlib_level,
        zlib_overhead=zlib_overhead,
    )

    metric.fit_y(y)

    return metric.surfeit_string(model_string)
