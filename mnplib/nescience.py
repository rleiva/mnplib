"""
Nescience aggregation for fitted models and explicit model descriptions.

This module implements the nescience component of the library as a small
coordinator around three independent metrics:

``Miscoding``
    Computes feature deficiency and feature surplus from a selected subset of
    input variables.

``Inaccuracy``
    Computes the mismatch between the target representation and a vector of
    model predictions.

``Surfeit``
    Computes the redundancy of an explicit model description string.

Fitted models are evaluated through canonical serializers. Explicit evaluation
uses three artifacts:

    * ``subset``: the features used by the model;
    * ``predictions``: the predictions produced by the model;
    * ``model_string``: a string description of the model.

Both interfaces share the same empirical metrics and aggregation policy.

@author:    Rafael Garcia Leiva
@mail:      rgarcialeiva@gmail.com
@copyright: GNU GPLv3
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import get_args

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted

from ._types import Aggregation, BinSpec, XType, YType
from .miscoding import Miscoding
from .inaccuracy import Inaccuracy
from .surfeit import Surfeit
from .mismodel import mismodel
from .models.inputs import model_artifacts
from ._diagnostics import warn_nan_model
from .utils import _validate_vector


class Nescience(BaseEstimator):
    """
    Compute nescience from deficiency, surplus, inaccuracy, and surfeit.

    The class is a coordinator. It fits the target and feature representation
    once, then evaluates model nescience from explicitly supplied model
    artifacts:

    ``subset``
        Boolean feature mask or list of feature indices used by the model.

    ``predictions``
        Prediction vector produced by the model.

    ``model_string``
        String representation of the model or explanation being evaluated.

    Parameters
    ----------
    X_type : {"auto", "numeric", "categorical"}, default="auto"
        Encoding strategy for the feature variables passed to ``Miscoding``.

    y_type : {"auto", "numeric", "categorical"}, default="auto"
        Encoding strategy for the target variable used by all components.

    aggregation : {"euclidean", "arithmetic", "geometric", "harmonic",
                   "maximum", "addition", "product"}, default="euclidean"
        Method used to aggregate the four component values.

    weights : mapping or sequence of 4 floats, optional
        Component weights in the order ``deficiency``, ``surplus``,
        ``inaccuracy``, and ``surfeit``. If a mapping is supplied, valid keys
        are those component names. Missing mapping keys default to 1.0.

    n_bins : int, "auto", or "adaptive", default="adaptive"
        Number of uniform bins used for numeric variables. ``"auto"`` uses
        ``max(2, floor(2 * n_samples**(1/3)))``. ``"adaptive"`` applies the
        subset-size rule inside ``Miscoding`` and matches ``"auto"`` for
        target-only quantities.
        Integer counts must be at least two; bin settings are validated during fit.

    zlib_level : int, default=9
        Compression level used by ``Surfeit``.

    zlib_overhead : int, default=6
        zlib wrapper overhead subtracted by ``Surfeit``.
    """

    component_names_ = ("deficiency", "surplus", "inaccuracy", "surfeit")

    _VALID_X_TYPES = get_args(XType)
    _VALID_Y_TYPES = get_args(YType)
    _VALID_AGGREGATIONS = get_args(Aggregation)

    def __init__(
        self,
        X_type: XType = "auto",
        y_type: YType = "auto",
        aggregation: Aggregation = "euclidean",
        weights: Mapping[str, float] | Sequence[float] | None = None,
        n_bins: BinSpec = "adaptive",
        zlib_level: int = 9,
        zlib_overhead: int = 6,
    ):
        """Initialize the estimator configuration."""
        self._validate_init(
            X_type=X_type,
            y_type=y_type,
            aggregation=aggregation,
            zlib_level=zlib_level,
            zlib_overhead=zlib_overhead,
        )

        self.X_type = X_type
        self.y_type = y_type
        self.aggregation = aggregation
        self.weights = weights
        self.n_bins = n_bins
        self.zlib_level = int(zlib_level)
        self.zlib_overhead = int(zlib_overhead)

    def fit(self, X, y):
        """
        Fit the internal component estimators.

        Parameters
        ----------
        X : array-like or pandas.DataFrame of shape (n_samples, n_features)
            Feature matrix. DataFrames are passed unchanged to ``Miscoding`` so
            that feature names and per-column type inference can be preserved.

        y : array-like of shape (n_samples,)
            Target vector.

        Returns
        -------
        self : Nescience
            Fitted estimator.
        """
        y = _validate_vector(y, name="y")
        X_checked, y_checked = check_X_y(X, y, dtype=None, ensure_2d=True)

        self.X_ = X_checked
        self._model_X_ = X
        self.y_ = y_checked
        self.n_samples_in_, self.n_features_in_ = X_checked.shape
        self.weights_ = self._resolve_weights()

        self.miscoding_ = Miscoding(
            X_type=self.X_type,
            y_type=self.y_type,
            n_bins=self.n_bins,
        )
        self.miscoding_.fit(X, y_checked)
        self.feature_names_in_ = self.miscoding_.feature_names_in_.copy()

        self.inaccuracy_ = Inaccuracy(
            y_type=self.y_type,
            n_bins=self.n_bins,
        )
        self.inaccuracy_.fit_y(y_checked)

        self.surfeit_ = Surfeit(
            y_type=self.y_type,
            n_bins=self.n_bins,
            zlib_level=self.zlib_level,
            zlib_overhead=self.zlib_overhead,
        )
        self.surfeit_.fit_y(y_checked)

        self.is_fitted_ = True
        return self

    def nescience_model(self, model, *, X=None, feature_names=None, feature_indices=None) -> float:
        """Compute nescience through canonical artifacts on fitted evaluation data.

        Explicit X contains estimator input columns in fitted-target row order.
        feature_indices maps local input columns to the original feature space.
        Unsupported or unfitted estimators raise a serializer validation error.
        Unreliable subset estimates return NaN with a RuntimeWarning describing
        the sparsity diagnostics. ``model_analysis`` reports diagnostics without
        issuing this warning.
        """
        artifacts = model_artifacts(self, model, X=X, feature_names=feature_names,
                                    feature_indices=feature_indices)
        value = self.nescience(**artifacts.to_nescience_kwargs())
        if np.isnan(value):
            diagnostics = self.miscoding_.subset_analysis(artifacts.subset)
            warn_nan_model("nescience_model", diagnostics)
        return value

    def model_analysis(self, model, *, X=None, feature_names=None, feature_indices=None) -> dict[str, object]:
        """Return flat metric fields, reliability diagnostics, and canonical text.

        The deficiency, surplus, inaccuracy, and surfeit fields accompany the
        aggregated nescience value. Unreliable subsets produce NaN subset
        quantities without issuing a warning.
        """
        artifacts = model_artifacts(self, model, X=X, feature_names=feature_names,
                                    feature_indices=feature_indices)
        return {**self.analysis(**artifacts.to_nescience_kwargs()),
                "model_type": artifacts.model_type,
                "model_string": artifacts.model_string}

    def components(
        self,
        *,
        subset,
        predictions,
        model_string: str,
    ) -> dict[str, float]:
        """
        Return the four scalar nescience components.

        Parameters
        ----------
        subset : array-like
            Boolean feature mask or list of selected feature indices.

        predictions : array-like of shape (n_samples,)
            Prediction vector produced by the model.

        model_string : str
            String description of the model.

        Returns
        -------
        dict
            Dictionary with the scalar keys ``deficiency``, ``surplus``,
            ``inaccuracy``, and ``surfeit``.
        """
        check_is_fitted(self)

        return {
            "deficiency": float(
                self.miscoding_.deficiency_subset(subset)
            ),
            "surplus": float(
                self.miscoding_.surplus_subset(subset)
            ),
            "inaccuracy": float(
                self.inaccuracy_.inaccuracy_predictions(predictions)
            ),
            "surfeit": float(
                self.surfeit_.surfeit_string(model_string)
            ),
        }

    def nescience(
        self,
        *,
        subset,
        predictions,
        model_string: str,
    ) -> float:
        """
        Return scalar nescience for supplied model artifacts.

        Parameters
        ----------
        subset : array-like
            Boolean feature mask or list of selected feature indices.

        predictions : array-like of shape (n_samples,)
            Prediction vector produced by the model.

        model_string : str
            String description of the model.

        Returns
        -------
        float
            Aggregated nescience value.
        """
        values = self.components(
            subset=subset,
            predictions=predictions,
            model_string=model_string,
        )
        return self.aggregate_components(**values)

    def analysis(
        self,
        *,
        subset,
        predictions,
        model_string: str,
    ) -> dict[str, object]:
        """
        Return numerical nescience and reliability diagnostics for model artifacts.

        Parameters
        ----------
        subset : array-like
            Boolean feature mask or list of selected feature indices.

        predictions : array-like of shape (n_samples,)
            Prediction vector produced by the model.

        model_string : str
            String description of the model.

        Returns
        -------
        dict
            Flat dictionary containing nescience, deficiency, surplus,
            inaccuracy, surfeit, mismodel, and subset diagnostics.
        """
        component_values = self.components(
            subset=subset,
            predictions=predictions,
            model_string=model_string,
        )
        nescience_value = self.aggregate_components(**component_values)
        diagnostics = self.miscoding_.subset_analysis(subset)

        return {
            **diagnostics,
            "nescience": float(nescience_value),
            "aggregation": self.aggregation,
            "weights": dict(zip(self.component_names_, self.weights_)),
            **component_values,
            "mismodel": mismodel(inaccuracy=component_values["inaccuracy"],
                                 surfeit=component_values["surfeit"]),
        }


    def aggregate_components(
        self,
        *,
        deficiency: float,
        surplus: float,
        inaccuracy: float,
        surfeit: float,
    ) -> float:
        """
        Aggregate the four component values according to ``self.aggregation``.

        All component values are clipped below at zero before aggregation.
        ``euclidean``, ``arithmetic``, ``geometric``, and ``harmonic`` use
        normalized weights. ``maximum`` ignores zero-weighted components.
        ``addition`` returns the weighted sum and may therefore exceed one.
        ``product`` returns the product of each active component raised to its
        corresponding weight.
        """
        values = np.asarray(
            [deficiency, surplus, inaccuracy, surfeit],
            dtype=float,
        )
        if not np.all(np.isfinite(values)):
            return float("nan")
        values = np.clip(values, 0.0, None)

        weights = getattr(self, "weights_", self._resolve_weights())
        weight_sum = float(np.sum(weights))

        if weight_sum <= 0:
            raise ValueError("At least one component weight must be positive.")

        active = weights > 0
        active_values = values[active]
        active_weights = weights[active]

        if self.aggregation == "euclidean":
            value = math.sqrt(float(np.sum(weights * values**2) / weight_sum))

        elif self.aggregation == "arithmetic":
            value = float(np.sum(weights * values) / weight_sum)

        elif self.aggregation == "geometric":
            if np.any(active_values == 0):
                value = 0.0
            else:
                value = math.exp(
                    float(np.sum(active_weights * np.log(active_values)) / weight_sum)
                )

        elif self.aggregation == "harmonic":
            if np.any(active_values == 0):
                value = 0.0
            else:
                value = float(weight_sum / np.sum(active_weights / active_values))

        elif self.aggregation == "maximum":
            value = float(np.max(active_values))

        elif self.aggregation == "addition":
            value = float(np.sum(weights * values))

        elif self.aggregation == "product":
            value = float(np.prod(active_values ** active_weights))

        else:  # Defensive guard; validation happens in __init__.
            raise RuntimeError(f"Unknown aggregation {self.aggregation!r}.")

        return float(value)


    def _resolve_weights(self) -> np.ndarray:
        """
        Return component weights in deficiency, surplus, inaccuracy, surfeit order.
        """
        if self.weights is None:
            return np.ones(4, dtype=float)

        if isinstance(self.weights, Mapping):
            unknown = set(self.weights) - set(self.component_names_)
            if unknown:
                raise ValueError(
                    "Unknown weight keys {}. Valid keys are {}."
                    .format(sorted(unknown), self.component_names_)
                )

            weights = np.asarray(
                [
                    float(self.weights.get(name, 1.0))
                    for name in self.component_names_
                ],
                dtype=float,
            )

        else:
            if isinstance(self.weights, (str, bytes)):
                raise ValueError(
                    "weights must be a mapping or a sequence of four numeric values."
                )

            weights = np.asarray(self.weights, dtype=float)
            if weights.shape != (4,):
                raise ValueError(
                    "weights must be a mapping or a sequence of four values: "
                    "deficiency, surplus, inaccuracy, surfeit."
                )

        if np.any(~np.isfinite(weights)):
            raise ValueError("weights must be finite.")

        if np.any(weights < 0):
            raise ValueError("weights must be non-negative.")

        if float(np.sum(weights)) <= 0:
            raise ValueError("At least one weight must be positive.")

        return weights

    @classmethod
    def _validate_init(
        cls,
        *,
        X_type,
        y_type,
        aggregation,
        zlib_level,
        zlib_overhead,
    ) -> None:
        """
        Validate constructor arguments before storing them on the estimator.
        """
        if X_type not in cls._VALID_X_TYPES:
            raise ValueError(
                "Valid options for 'X_type' are {}. Got X_type={!r} instead."
                .format(cls._VALID_X_TYPES, X_type)
            )

        if y_type not in cls._VALID_Y_TYPES:
            raise ValueError(
                "Valid options for 'y_type' are {}. Got y_type={!r} instead."
                .format(cls._VALID_Y_TYPES, y_type)
            )

        if aggregation not in cls._VALID_AGGREGATIONS:
            raise ValueError(
                "Valid options for 'aggregation' are {}. Got aggregation={!r} instead."
                .format(cls._VALID_AGGREGATIONS, aggregation)
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


def nescience(
    *,
    X,
    y,
    subset,
    predictions,
    model_string: str,
    X_type: XType = "auto",
    y_type: YType = "auto",
    aggregation: Aggregation = "euclidean",
    weights: Mapping[str, float] | Sequence[float] | None = None,
    n_bins: BinSpec = "adaptive",
    zlib_level: int = 9,
    zlib_overhead: int = 6,
) -> float:
    """
    Compute scalar nescience using a functional interface.

    Parameters
    ----------
    X : array-like or pandas.DataFrame of shape (n_samples, n_features)
        Feature matrix.

    y : array-like of shape (n_samples,)
        Target vector.

    subset : array-like
        Boolean feature mask or list of selected feature indices.

    predictions : array-like of shape (n_samples,)
        Prediction vector produced by the model.

    model_string : str
        String description of the model.

    X_type, y_type, aggregation, weights, n_bins, zlib_level, zlib_overhead
        Configuration with the same meaning as in ``Nescience``.

    Returns
    -------
    float
        Aggregated nescience value.
    """
    metric = Nescience(X_type=X_type, y_type=y_type, aggregation=aggregation,
                       weights=weights, n_bins=n_bins, zlib_level=zlib_level,
                       zlib_overhead=zlib_overhead).fit(X, y)
    return metric.nescience(
        subset=subset,
        predictions=predictions,
        model_string=model_string,
    )


def nescience_components(
    *,
    X,
    y,
    subset,
    predictions,
    model_string: str,
    X_type: XType = "auto",
    y_type: YType = "auto",
    aggregation: Aggregation = "euclidean",
    weights: Mapping[str, float] | Sequence[float] | None = None,
    n_bins: BinSpec = "adaptive",
    zlib_level: int = 9,
    zlib_overhead: int = 6,
) -> dict[str, float]:
    """
    Compute the four scalar nescience components using a functional interface.
    """
    metric = Nescience(X_type=X_type, y_type=y_type, aggregation=aggregation,
                       weights=weights, n_bins=n_bins, zlib_level=zlib_level,
                       zlib_overhead=zlib_overhead).fit(X, y)
    return metric.components(
        subset=subset,
        predictions=predictions,
        model_string=model_string,
    )


def nescience_model(
    model, *, X, y, feature_names=None, feature_indices=None,
    X_type: XType = "auto", y_type: YType = "auto",
    aggregation: Aggregation = "euclidean",
    weights: Mapping[str, float] | Sequence[float] | None = None,
    n_bins: BinSpec = "adaptive", zlib_level: int = 9, zlib_overhead: int = 6,
) -> float:
    """Compute fitted-model nescience on evaluation data."""
    metric = Nescience(X_type=X_type, y_type=y_type, aggregation=aggregation,
                       weights=weights, n_bins=n_bins, zlib_level=zlib_level,
                       zlib_overhead=zlib_overhead).fit(X, y)
    return metric.nescience_model(
        model, feature_names=feature_names, feature_indices=feature_indices)


def model_analysis(
    model, *, X, y, feature_names=None, feature_indices=None,
    X_type: XType = "auto", y_type: YType = "auto",
    aggregation: Aggregation = "euclidean",
    weights: Mapping[str, float] | Sequence[float] | None = None,
    n_bins: BinSpec = "adaptive", zlib_level: int = 9, zlib_overhead: int = 6,
) -> dict[str, object]:
    """Explain fitted-model nescience, including subset reliability."""
    metric = Nescience(X_type=X_type, y_type=y_type, aggregation=aggregation,
                       weights=weights, n_bins=n_bins, zlib_level=zlib_level,
                       zlib_overhead=zlib_overhead).fit(X, y)
    return metric.model_analysis(
        model, feature_names=feature_names, feature_indices=feature_indices)
