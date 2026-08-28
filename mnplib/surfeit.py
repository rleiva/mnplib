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

from typing import Literal

import zlib

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted

from .models import sklearn_model_artifacts
from .utils import empirical_distribution

YType = Literal["auto", "numeric", "categorical"]
BinSpec = int | Literal["auto", "adaptive"]

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
    library serializer layer.

    Parameters
    ----------
    y_type : {"auto", "numeric", "categorical"}, default="auto"
        Encoding strategy for the target variable.

    n_bins : int, "auto", or "adaptive", default="auto"
        Number of uniform bins used for numeric targets. ``"auto"`` uses
        ``max(2, floor(2 * n_samples**(1/3)))``. ``"adaptive"`` is equivalent
        for target-only quantities.

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
        self.X_, self.y_ = check_X_y(X, y, dtype=None, ensure_2d=True)
        self._fit_target(self.y_)
        self.n_features_in_ = self.X_.shape[1]
        self._serializer_X_ = X if feature_names is not None else self.X_
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
        check_is_fitted(self)

        model_bytes = self._validate_model_string(model_string)
        model_length = len(model_bytes)
        compressed_length = len(self._compress_bytes(model_bytes))

        return self._surfeit_from_lengths(
            model_length=model_length,
            compressed_length=compressed_length,
        )

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

    def model_description(
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
            Model string, length diagnostics, surfeit, and serializer metadata.
        """
        check_is_fitted(self)
        artifacts = self._model_artifacts_from_model(
            model,
            X=X,
            feature_names=feature_names,
            feature_indices=feature_indices,
        )
        lengths = self.description_lengths(artifacts.model_string)
        value = self.surfeit_string(artifacts.model_string)

        return {
            "model_string": artifacts.model_string,
            **lengths,
            "surfeit": value,
            "model_type": artifacts.model_type,
            "selected_features": list(artifacts.subset),
            "n_selected_features": len(artifacts.subset),
        }

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
        X_model = self._resolve_model_X(
            model,
            X=X,
            feature_names=feature_names,
            feature_indices=feature_indices,
        )
        resolved_feature_names = self._resolve_model_feature_names(
            model,
            X=X_model,
            feature_names=feature_names,
            feature_indices=feature_indices,
        )
        return sklearn_model_artifacts(
            model,
            X_model,
            feature_names=resolved_feature_names,
            feature_indices=feature_indices,
        )

    def _resolve_model_X(
        self,
        model,
        *,
        X=None,
        feature_names=None,
        feature_indices=None,
    ):
        """Resolve the model input matrix used by the serializer layer."""
        if X is not None:
            if feature_indices is not None:
                shape = getattr(X, "shape", None)
                if (
                    shape is not None
                    and len(shape) == 2
                    and int(shape[1]) != len(list(feature_indices))
                ):
                    return self._take_columns(X, feature_indices)
            return X

        if getattr(self, "X_", None) is not None:
            fitted_X = getattr(self, "_serializer_X_", self.X_)
            if feature_indices is None:
                return fitted_X
            return self._take_columns(fitted_X, feature_indices)

        n_features = self._infer_model_input_count(
            model,
            feature_names=feature_names,
            feature_indices=feature_indices,
        )
        return np.zeros((1, n_features), dtype=float)

    def _resolve_model_feature_names(
        self,
        model,
        *,
        X,
        feature_names=None,
        feature_indices=None,
    ) -> list[str]:
        """Resolve model input names for serializer validation."""
        n_features = self._infer_model_input_count(
            model,
            X=X,
            feature_names=feature_names,
            feature_indices=feature_indices,
        )

        if feature_names is not None:
            return self._align_feature_names(
                feature_names,
                n_features=n_features,
                feature_indices=feature_indices,
            )

        if hasattr(self, "feature_names_in_"):
            return self._align_feature_names(
                self.feature_names_in_,
                n_features=n_features,
                feature_indices=feature_indices,
            )

        names = self._feature_names_from_input(X)
        if names is not None:
            return self._align_feature_names(
                names,
                n_features=n_features,
                feature_indices=feature_indices,
            )

        return [f"x{i}" for i in range(n_features)]

    @staticmethod
    def _feature_names_from_input(X):
        """Return column names from a tabular input when available."""
        if hasattr(X, "columns"):
            return [str(name) for name in X.columns]
        return None

    @classmethod
    def _align_feature_names(
        cls,
        names,
        *,
        n_features: int,
        feature_indices=None,
    ) -> list[str]:
        """Return feature names matching the estimator input dimension."""
        names = [str(name) for name in names]
        if len(names) == int(n_features):
            return names

        if feature_indices is not None:
            indices = [int(index) for index in feature_indices]
            if indices and max(indices) < len(names):
                aligned = [names[index] for index in indices]
                if len(aligned) == int(n_features):
                    return aligned

        raise ValueError(
            "feature_names must have length {}. Got {} names instead."
            .format(int(n_features), len(names))
        )

    @staticmethod
    def _take_columns(X, feature_indices):
        """Return selected columns from an array-like or DataFrame input."""
        indices = [int(index) for index in feature_indices]
        if hasattr(X, "iloc"):
            return X.iloc[:, indices]
        return np.asarray(X)[:, indices]

    @staticmethod
    def _infer_model_input_count(
        model,
        *,
        X=None,
        feature_names=None,
        feature_indices=None,
    ) -> int:
        """Infer the number of columns expected by the fitted model."""
        if X is not None:
            shape = getattr(X, "shape", None)
            if shape is not None and len(shape) == 2:
                return int(shape[1])

        if feature_indices is not None:
            return len(list(feature_indices))

        if feature_names is not None:
            return len(list(feature_names))

        n_features = getattr(model, "n_features_in_", None)
        if n_features is not None:
            return int(n_features)

        raise ValueError(
            "Feature information is required to serialize this model. "
            "Provide X or feature_names, or fit Surfeit with X and y."
        )

    def _clear_feature_metadata(self) -> None:
        """Remove feature metadata when fitting with only a target vector."""
        for name in ("n_features_in_", "feature_names_in_", "_serializer_X_"):
            if hasattr(self, name):
                delattr(self, name)

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


def surfeit_model_score(
    model,
    X,
    y,
    *,
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
        X=X,
        feature_names=feature_names,
        feature_indices=feature_indices,
    )


def model_description(
    model,
    X,
    y,
    *,
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
        Model string, length diagnostics, surfeit, and serializer metadata.
    """
    metric = Surfeit(
        y_type=y_type,
        n_bins=n_bins,
        zlib_level=zlib_level,
        zlib_overhead=zlib_overhead,
    )
    metric.fit(X, y)
    return metric.model_description(
        model,
        X=X,
        feature_names=feature_names,
        feature_indices=feature_indices,
    )
