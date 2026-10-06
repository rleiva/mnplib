"""Hierarchical RMS nescience for fitted models and explicit model artifacts.

Miscoding combines deficiency and surplus. Mismodel combines inaccuracy and
surfeit. Both use equal-weight RMS; only their final combination is weighted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Real
from typing import get_args

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted

from ._diagnostics import warn_nan_model
from ._rms import _rms_pair
from ._types import XType, YType
from .miscoding import Miscoding
from .mismodel import Mismodel
from .models.inputs import model_artifacts
from .utils import _validate_vector, _validate_y_type


class Nescience(BaseEstimator):
    """Evaluate weighted RMS of miscoding and mismodel.

    Miscoding is RMS(deficiency, surplus), and mismodel is
    RMS(inaccuracy, surfeit). With equal top-level weights, nescience equals
    the RMS of the four primitive values. These are practical scalar estimates,
    not exact computations of non-computable theoretical quantities.

    Parameters
    ----------
    X_type : {"auto", "numeric", "categorical"}, default="auto"
        Feature encoding strategy used by Miscoding.
    y_type : {"auto", "numeric", "categorical"}, default="auto"
        Target encoding strategy shared by the metrics.
    weights : mapping or sequence of two floats, optional
        Nonnegative finite ratios in (miscoding, mismodel) order. None uses
        equal weights; missing mapping keys default to 1. At least one weight
        must be positive. Zero removes a dimension's numerical contribution,
        but a nonfinite dimension still makes nescience NaN.

    Notes
    -----
    Reports and weights_ expose the resolved ratios, without normalization.
    Scoring normalizes a copy internally. Parameter changes take effect on the
    next evaluation; primitive metric encoding changes require refitting.
    """

    component_names_ = ("deficiency", "surplus", "inaccuracy", "surfeit")
    weight_names_    = ("miscoding", "mismodel")
    _VALID_X_TYPES   = get_args(XType)

    def __init__(
        self,
        X_type: XType = "auto",
        y_type: YType = "auto",
        weights: Mapping[str, float] | Sequence[float] | None = None,
    ):
        self._validate_init(X_type=X_type, y_type=y_type)
        self.X_type = X_type
        self.y_type = y_type
        self.weights = weights

    def fit(self, X, y):
        """Fit the feature and model metrics on aligned evaluation data.

        Preserve DataFrame input for canonical model prediction and feature
        names. No predictive estimator is fitted.
        """
        self.is_fitted_ = False
        self._validate_init(X_type=self.X_type, y_type=self.y_type)
        self._resolve_weights()
        y = _validate_vector(y, name="y")
        X_checked, y_checked = check_X_y(X, y, dtype=None, ensure_2d=True)
        self.X_ = X_checked
        self._model_X_ = X
        self.y_ = y_checked
        self.n_samples_in_, self.n_features_in_ = X_checked.shape
        self.miscoding_ = Miscoding(X_type=self.X_type, y_type=self.y_type).fit(X, y_checked)
        self.feature_names_in_ = self.miscoding_.feature_names_in_.copy()
        self.mismodel_ = Mismodel(y_type=self.y_type).fit_y(y_checked)
        self.is_fitted_ = True
        return self

    def __sklearn_is_fitted__(self) -> bool:
        """Report whether component fitting completed successfully."""
        return getattr(self, "is_fitted_", False)

    @property
    def weights_(self) -> np.ndarray:
        """Return the current resolved weight ratios for a fitted metric."""
        check_is_fitted(self)
        return self._resolve_weights()

    def nescience_model(self, model, *, X=None, feature_names=None, feature_indices=None) -> float:
        """Evaluate a fitted model through canonical artifacts.

        Explicit X contains estimator input columns in fitted-target row order.
        feature_indices maps local input columns to the original feature space.
        Unsupported or unfitted estimators raise serializer validation errors.
        An unavailable score emits one RuntimeWarning; model_analysis is quiet.
        """
        report = self.model_analysis(
            model, X=X, feature_names=feature_names, feature_indices=feature_indices,
        )
        value = float(report["nescience"])
        if np.isnan(value):
            warn_nan_model("nescience_model", report)
        return value

    def model_analysis(self, model, *, X=None, feature_names=None, feature_indices=None) -> dict[str, object]:
        """Return primitives, derived metrics, reliability, and canonical text."""
        artifacts = model_artifacts(
            self, model, X=X, feature_names=feature_names, feature_indices=feature_indices,
        )
        return {
            **self.analysis(**artifacts.to_nescience_kwargs()),
            "model_type": artifacts.model_type,
            "model_string": artifacts.model_string,
        }

    def components(self, *, subset, predictions, model_string: str) -> dict[str, float]:
        """Return deficiency, surplus, inaccuracy, and surfeit.

        subset is a Boolean feature mask or a sequence of feature indices.
        predictions must match the fitted target's sample count and row order.
        model_string is a nonempty description of the evaluated model.
        """
        check_is_fitted(self)
        diagnostics = self.miscoding_.subset_analysis(subset)
        return {
            "deficiency": float(diagnostics["deficiency"]),
            "surplus": float(diagnostics["surplus"]),
            **self.mismodel_.components(predictions=predictions, model_string=model_string),
        }

    def nescience(self, *, subset, predictions, model_string: str) -> float:
        """Return scalar nescience for explicit model artifacts."""
        return float(self.analysis(
            subset=subset, predictions=predictions, model_string=model_string,
        )["nescience"])

    def analysis(self, *, subset, predictions, model_string: str) -> dict[str, object]:
        """Return consistent primitive and derived metrics with reliability.

        Each metric is evaluated once. An unavailable miscoding or mismodel
        yields NaN nescience, even with zero weight, without erasing the other
        diagnostic. Resolved weight ratios include both top-level dimensions.
        """
        check_is_fitted(self)
        diagnostics = self.miscoding_.subset_analysis(subset)
        model_values = self.mismodel_.analysis(
            predictions=predictions, model_string=model_string,
        )
        value = self.aggregate_components(
            miscoding=diagnostics["miscoding"], mismodel=model_values["mismodel"],
        )
        return {
            **diagnostics,
            **model_values,
            "nescience": value,
            "weights": dict(zip(self.weight_names_, self._resolve_weights().tolist())),
        }

    def aggregate_components(self, *, miscoding: float, mismodel: float) -> float:
        """Return weighted RMS of the two derived metrics without fitting.

        Nonfinite metrics return NaN, including zero-weighted dimensions.
        Finite inputs must be nonnegative. Exact zeros are preserved.
        """
        weights = self._resolve_weights()
        weights = weights / np.max(weights)
        weights = weights / np.sum(weights)
        return float(_rms_pair(float(miscoding), float(mismodel), weights=weights))

    def _resolve_weights(self) -> np.ndarray:
        """Validate and copy raw weight ratios in (miscoding, mismodel) order."""
        if self.weights is None:
            return np.ones(2, dtype=float)
        if isinstance(self.weights, Mapping):
            unknown = set(self.weights) - set(self.weight_names_)
            if unknown:
                raise ValueError(
                    f"Unknown weight keys {sorted(unknown, key=str)}. "
                    f"Valid keys are {self.weight_names_}."
                )
            values = [self.weights.get(name, 1.0) for name in self.weight_names_]
        else:
            if (isinstance(self.weights, (str, bytes))
                    or not isinstance(self.weights, (Sequence, np.ndarray))):
                raise ValueError("weights must be a mapping or a sequence of two numeric values.")
            try:
                values = list(self.weights)
            except TypeError as exc:
                raise ValueError("weights must be a mapping or a sequence of two numeric values.") from exc
            if len(values) != 2:
                raise ValueError("weights must contain exactly two values: miscoding, mismodel.")
        if any(not isinstance(value, Real) or isinstance(value, (bool, np.bool_)) for value in values):
            raise ValueError("weights must contain numeric real values.")
        try:
            weights = np.asarray(values, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("weights must contain finite numeric values.") from exc
        if not np.all(np.isfinite(weights)):
            raise ValueError("weights must be finite.")
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative.")
        if not np.any(weights > 0):
            raise ValueError("At least one weight must be positive.")
        return weights

    @classmethod
    def _validate_init(cls, *, X_type, y_type) -> None:
        """Validate feature and target encoding policies."""
        if X_type not in cls._VALID_X_TYPES:
            raise ValueError(
                "Valid options for 'X_type' are {}. Got X_type={!r} instead."
                .format(cls._VALID_X_TYPES, X_type)
            )
        _validate_y_type(y_type)


def nescience(
    *, X, y, subset, predictions, model_string: str,
    X_type: XType = "auto", y_type: YType = "auto",
    weights: Mapping[str, float] | Sequence[float] | None = None,
) -> float:
    """Evaluate explicit artifacts using weighted RMS of miscoding and mismodel."""
    metric = Nescience(X_type=X_type, y_type=y_type, weights=weights).fit(X, y)
    return metric.nescience(subset=subset, predictions=predictions, model_string=model_string)


def nescience_components(
    *, X, y, subset, predictions, model_string: str,
    X_type: XType = "auto", y_type: YType = "auto",
    weights: Mapping[str, float] | Sequence[float] | None = None,
) -> dict[str, float]:
    """Return the four primitive components from explicit artifacts."""
    metric = Nescience(X_type=X_type, y_type=y_type, weights=weights).fit(X, y)
    return metric.components(subset=subset, predictions=predictions, model_string=model_string)


def nescience_model(
    model, *, X, y, feature_names=None, feature_indices=None,
    X_type: XType = "auto", y_type: YType = "auto",
    weights: Mapping[str, float] | Sequence[float] | None = None,
) -> float:
    """Compute fitted-model nescience on evaluation data."""
    metric = Nescience(X_type=X_type, y_type=y_type, weights=weights).fit(X, y)
    return metric.nescience_model(
        model, feature_names=feature_names, feature_indices=feature_indices,
    )


def model_analysis(
    model, *, X, y, feature_names=None, feature_indices=None,
    X_type: XType = "auto", y_type: YType = "auto",
    weights: Mapping[str, float] | Sequence[float] | None = None,
) -> dict[str, object]:
    """Analyze fitted-model nescience, including subset reliability."""
    metric = Nescience(X_type=X_type, y_type=y_type, weights=weights).fit(X, y)
    return metric.model_analysis(
        model, feature_names=feature_names, feature_indices=feature_indices,
    )
