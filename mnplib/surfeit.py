"""
Surfeit based on compressed model descriptions.

This module implements the surfeit component of nescience. Surfeit measures
how much unnecessary structure appears to be present in a model description.

The string-based computation is the primitive metric operation. Fitted-model
methods obtain canonical model descriptions through the library serializer
layer and then delegate to the same string-based computation.

@author:    Rafael Garcia Leiva
@mail:      rgarcialeiva@gmail.com
@copyright: GNU GPLv3
"""

from __future__ import annotations

import zlib
from typing import get_args

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted

from ._types import BinSpec, YType
from .models.inputs import model_artifacts
from .utils import _validate_vector, empirical_distribution_vector


class Surfeit(BaseEstimator):
    """
    Compute the surfeit of a model description or fitted estimator.

    Surfeit is estimated by comparing the raw length of a model description with
    a compressed reference length. The compressed length is corrected by
    subtracting the fixed zlib wrapper overhead, then bounded by the target code
    length. This prevents the reference description from exceeding the amount of
    information available in the target representation.

    Use ``surfeit_string()`` when a canonical model description is already
    available. Use ``surfeit_model()`` for a fitted estimator supported by the
    library serializer layer. ``description_analysis()`` and ``model_analysis()``
    explain the computation using code lengths in bits.

    Parameters
    ----------
    y_type : {"auto", "numeric", "categorical"}, default="auto"
        Encoding strategy for the target variable.

    n_bins : int, "auto", or "adaptive", default="auto"
        Number of uniform bins used for numeric targets. ``"auto"`` uses
        ``max(2, floor(2 * n_samples**(1/3)))``. ``"adaptive"`` is equivalent
        for target-only quantities.
        Integer counts must be at least two; bin settings are validated during fit.

    zlib_level : int, default=9
        Compression level passed to ``zlib.compress``. Must be between 0 and 9.

    zlib_overhead : int, default=6
        Estimated zlib wrapper overhead, in bytes, subtracted from the raw
        compressed length.
    """

    _VALID_Y_TYPES = get_args(YType)

    def __init__(
        self,
        y_type: YType = "auto",
        n_bins: BinSpec = "auto",
        zlib_level: int = 9,
        zlib_overhead: int = 6,
    ):
        """Initialize the estimator configuration."""
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

        The feature matrix is stored to preserve familiar estimator attributes
        and to support fitted-model serialization when no explicit evaluation
        matrix is provided.

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
        feature_names = self._feature_names_from_input(X)
        y = _validate_vector(y, name="y")
        self.X_, self.y_ = check_X_y(X, y, dtype=None, ensure_2d=True)
        self._fit_target(self.y_)
        self.n_features_in_ = self.X_.shape[1]
        self._model_X_ = X if feature_names is not None else self.X_
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(self.n_features_in_)]
        self.feature_names_in_ = np.asarray(feature_names, dtype=object)

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
        self._clear_feature_metadata()
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
        return float(self.description_analysis(model_string)["surfeit"])

    def surfeit_model(
        self,
        model,
        *,
        X=None,
        feature_names=None,
        feature_indices=None,
    ) -> float:
        """
        Compute surfeit for a fitted estimator.

        The estimator is serialized through the library model adapter layer.
        The resulting canonical model description is evaluated by
        ``surfeit_string()``.

        Parameters
        ----------
        model : fitted estimator
            Fitted model supported by the serializer layer.

        X : array-like of shape (n_samples, n_features), optional
            Evaluation matrix used by serializers that require model inputs.
            If omitted, the matrix supplied to ``fit(X, y)`` is used when
            available.

        feature_names : sequence of str, optional
            Names for the model input columns. If omitted, names are resolved
            from ``fit(X, y)``, from ``X``, or from the fitted estimator input
            dimension.

        feature_indices : sequence of int, optional
            Mapping from local estimator input columns to original feature
            indices.

        Returns
        -------
        float
            Surfeit value in the interval [0, 1].
        """
        check_is_fitted(self)
        model_string = self._model_string_from_model(
            model,
            X=X,
            feature_names=feature_names,
            feature_indices=feature_indices,
        )
        return self.surfeit_string(model_string)

    def model_analysis(
        self,
        model,
        *,
        X=None,
        feature_names=None,
        feature_indices=None,
    ) -> dict[str, object]:
        """
        Return canonical model-description diagnostics for a fitted estimator.

        Parameters
        ----------
        model : fitted estimator
            Fitted model supported by the serializer layer.

        X : array-like of shape (n_samples, n_features), optional
            Evaluation matrix used by serializers that require model inputs.

        feature_names : sequence of str, optional
            Names for the model input columns.

        feature_indices : sequence of int, optional
            Mapping from local estimator input columns to original feature
            indices.

        Returns
        -------
        dict
            Canonical model string, model type, effective feature indices, and
            the diagnostics returned by ``description_analysis()``.
        """
        check_is_fitted(self)
        artifacts = self._model_artifacts_from_model(
            model,
            X=X,
            feature_names=feature_names,
            feature_indices=feature_indices,
        )
        return {
            "model_string": artifacts.model_string,
            **self.description_analysis(artifacts.model_string),
            "model_type": artifacts.model_type,
            "selected_features": list(artifacts.subset),
            "n_selected_features": len(artifacts.subset),
        }

    def description_analysis(self, model_string: str) -> dict[str, object]:
        """Explain surfeit for an explicit description using lengths in bits.

        Return raw and compressed model code lengths, the overhead-corrected
        compressed length clipped to the raw length, target code length, and
        the reference length used in surfeit. ``reference_source`` is "target",
        "compression", or "both" when the two limits coincide.

        ``compression_ratio`` is raw compressed size divided by raw model size,
        before overhead correction or clipping, and can exceed one. The report
        describes compression arithmetic, not a statistical test of overfitting.
        """
        check_is_fitted(self)
        lengths = self.description_lengths(model_string)
        return self._description_measures(
            model_length=lengths["model_length"],
            compressed_length=lengths["model_compressed_length"],
        )

    def description_lengths(self, model_string: str) -> dict[str, int]:
        """
        Return byte-length diagnostics for a model description string.

        Parameters
        ----------
        model_string : str
            String representation of a model or description.

        Returns
        -------
        dict
            Raw UTF-8 byte length and zlib-compressed byte length.
        """
        model_bytes = self._validate_model_string(model_string)

        return {
            "model_length": len(model_bytes),
            "model_compressed_length": len(self._compress_bytes(model_bytes)),
        }

    def _model_string_from_model(
        self,
        model,
        *,
        X=None,
        feature_names=None,
        feature_indices=None,
    ) -> str:
        """Return a canonical model description for a fitted estimator."""
        return self._model_artifacts_from_model(
            model,
            X=X,
            feature_names=feature_names,
            feature_indices=feature_indices,
        ).model_string

    def _model_artifacts_from_model(
        self,
        model,
        *,
        X=None,
        feature_names=None,
        feature_indices=None,
    ):
        """Return serializer artifacts for a fitted estimator."""
        return model_artifacts(self, model, X=X, feature_names=feature_names,
                               feature_indices=feature_indices, allow_dummy=True)


    @staticmethod
    def _feature_names_from_input(X):
        """Return column names from a tabular input when available."""
        if hasattr(X, "columns"):
            return list(X.columns)
        return None


    def _clear_feature_metadata(self) -> None:
        """Remove feature metadata when fitting with only a target vector."""
        for name in ("n_features_in_", "feature_names_in_", "_model_X_"):
            if hasattr(self, name):
                delattr(self, name)

    def _fit_target(self, y) -> None:
        """Fit target-dependent attributes."""
        self.y_ = _validate_vector(y, name="y")
        self.y_isnumeric_ = self._infer_y_isnumeric(self.y_)
        self.len_y_ = self._target_code_length()
        self.n_samples_in_ = self.y_.shape[0]
        self.is_fitted_ = True

    def _target_code_length(self) -> float:
        """Return the empirical code length of the fitted target in bits."""
        return float(
            empirical_distribution_vector(
                self.y_,
                numeric=self.y_isnumeric_,
                n_bins=self.n_bins,
            ).code_length
        )

    def _description_measures(self, model_length: int,
                              compressed_length: int) -> dict[str, object]:
        """Convert byte counts to bits before comparing description lengths."""
        effective_length = self._effective_compressed_length(
            compressed_length=compressed_length,
            model_length=model_length,
        )

        model_bits = 8 * model_length
        effective_bits = 8 * effective_length
        target_bits = float(self.len_y_)
        reference_bits = min(target_bits, effective_bits)
        if target_bits < effective_bits:
            reference_source = "target"
        elif target_bits > effective_bits:
            reference_source = "compression"
        else:
            reference_source = "both"

        return {
            "surfeit": float(np.clip(1.0 - reference_bits / model_bits, 0.0, 1.0)),
            "model_code_length_bits": model_bits,
            "compressed_code_length_bits": 8 * compressed_length,
            "effective_compressed_code_length_bits": effective_bits,
            "target_code_length_bits": target_bits,
            "reference_code_length_bits": reference_bits,
            "compression_ratio": float(compressed_length / model_length),
            "reference_source": reference_source,
        }

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


def surfeit_string(
    model_string: str,
    *,
    y,
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

    n_bins : int, "auto", or "adaptive", default="auto"
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


def surfeit_model(
    model,
    *,
    X,
    y,
    feature_names=None,
    feature_indices=None,
    y_type: YType = "auto",
    n_bins: BinSpec = "auto",
    zlib_level: int = 9,
    zlib_overhead: int = 6,
) -> float:
    """
    Compute surfeit for a fitted estimator using a functional interface.

    Parameters
    ----------
    model : fitted estimator
        Fitted model supported by the serializer layer.

    X : array-like of shape (n_samples, n_features)
        Evaluation matrix.

    y : array-like of shape (n_samples,)
        Target values used to fit the surfeit metric.

    feature_names : sequence of str, optional
        Names for the model input columns.

    feature_indices : sequence of int, optional
        Mapping from local estimator input columns to original feature indices.

    y_type : {"auto", "numeric", "categorical"}, default="auto"
        Encoding strategy for the target variable.

    n_bins : int, "auto", or "adaptive", default="auto"
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
    metric.fit(X, y)
    return metric.surfeit_model(
        model,
        feature_names=feature_names,
        feature_indices=feature_indices,
    )


def model_analysis(
    model,
    *,
    X,
    y,
    feature_names=None,
    feature_indices=None,
    y_type: YType = "auto",
    n_bins: BinSpec = "auto",
    zlib_level: int = 9,
    zlib_overhead: int = 6,
) -> dict[str, object]:
    """
    Return model-description diagnostics using a functional interface.

    Parameters
    ----------
    model : fitted estimator
        Fitted model supported by the serializer layer.

    X : array-like of shape (n_samples, n_features)
        Evaluation matrix.

    y : array-like of shape (n_samples,)
        Target values used to fit the surfeit metric.

    feature_names : sequence of str, optional
        Names for the model input columns.

    feature_indices : sequence of int, optional
        Mapping from local estimator input columns to original feature indices.

    y_type : {"auto", "numeric", "categorical"}, default="auto"
        Encoding strategy for the target variable.

    n_bins : int, "auto", or "adaptive", default="auto"
        Number of uniform bins used for numeric targets.

    zlib_level : int, default=9
        Compression level passed to ``zlib.compress``.

    zlib_overhead : int, default=6
        Estimated zlib wrapper overhead, in bytes.

    Returns
    -------
    dict
        Canonical model string, model type, feature indices, and surfeit
        analysis with code lengths in bits.
    """
    metric = Surfeit(
        y_type=y_type,
        n_bins=n_bins,
        zlib_level=zlib_level,
        zlib_overhead=zlib_overhead,
    )
    metric.fit(X, y)
    return metric.model_analysis(
        model,
        feature_names=feature_names,
        feature_indices=feature_indices,
    )


def description_analysis(
    model_string: str,
    *,
    y,
    y_type: YType = "auto",
    n_bins: BinSpec = "auto",
    zlib_level: int = 9,
    zlib_overhead: int = 6,
) -> dict[str, object]:
    """Analyze an explicit model description with all code lengths in bits."""
    metric = Surfeit(y_type=y_type, n_bins=n_bins, zlib_level=zlib_level,
                     zlib_overhead=zlib_overhead).fit_y(y)
    return metric.description_analysis(model_string)
