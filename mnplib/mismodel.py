"""Practical model diagnostics combining inaccuracy and description surfeit."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted

from ._types import YType
from .inaccuracy import Inaccuracy
from .models.inputs import model_artifacts
from .surfeit import Surfeit
from .utils import _resolve_feature_names, _validate_vector, _validate_y_type


def mismodel(*, inaccuracy: float, surfeit: float) -> float:
    """Return the equal-weight RMS practical estimate of mismodel.

    Nonfinite components produce NaN. Finite components must be nonnegative.
    Normalized inputs yield a value in [0, 1]. This estimate is not an exact
    computation of non-computable theoretical quantities.
    """
    values = (float(inaccuracy), float(surfeit))
    if not all(math.isfinite(value) for value in values):
        return float("nan")
    if any(value < 0 for value in values):
        raise ValueError("Mismodel components must be nonnegative.")
    return math.sqrt((values[0] ** 2 + values[1] ** 2) / 2.0)


class Mismodel(BaseEstimator):
    """Evaluate model quality through inaccuracy and surfeit.

    Mismodel is the practical scalar estimate
    ``sqrt((inaccuracy**2 + surfeit**2) / 2)``. Inaccuracy measures target and
    prediction mismatch; surfeit measures redundancy in a model description.
    The components always have equal weight. Their encoding and compression
    policies are managed by ``Inaccuracy`` and ``Surfeit``.

    ``fit_y(y)`` enables explicit prediction and description evaluation.
    ``fit(X, y)`` additionally stores evaluation inputs for fitted-model methods.
    No predictive model or feature-relevance metric is fitted. All evaluation
    methods require fitting; ``aggregate_components`` requires only scalars.
    Nonfinite component values yield NaN, while invalid arguments raise their
    component's validation errors.

    Parameters
    ----------
    y_type : {"auto", "numeric", "categorical"}, default="auto"
        Target encoding policy shared by the component estimators.
    """

    def __init__(self, y_type: YType = "auto"):
        _validate_y_type(y_type)
        self.y_type = y_type

    def fit_y(self, y):
        """Fit both components to a target vector and clear evaluation inputs."""
        self._clear_fitted_state()
        self._fit_target(y)
        self.is_fitted_ = True
        return self

    def fit(self, X, y):
        """Fit target components and retain evaluation inputs in target row order.

        Store copies of X and y, preserving DataFrame columns and mixed feature
        types. X is used only for canonical model prediction and serialization.
        """
        self._clear_fitted_state()
        y = _validate_vector(y, name="y")
        X_checked, y_checked = check_X_y(X, y, dtype=None, ensure_2d=True)
        names = np.fromiter(_resolve_feature_names(X), dtype=object)
        X_checked = X_checked.copy()
        model_X = X.copy(deep=True) if isinstance(X, pd.DataFrame) else X_checked
        self._fit_target(y_checked)
        self.X_ = X_checked
        self._model_X_ = model_X
        self.n_features_in_ = X_checked.shape[1]
        self.feature_names_in_ = names
        self.is_fitted_ = True
        return self

    def __sklearn_is_fitted__(self) -> bool:
        """Report whether all fitting steps completed successfully."""
        return getattr(self, "is_fitted_", False)

    def components(self, *, predictions, model_string: str) -> dict[str, float]:
        """Return exactly inaccuracy and surfeit for explicit model artifacts."""
        check_is_fitted(self)
        return {
            "inaccuracy": float(self.inaccuracy_.inaccuracy_predictions(predictions)),
            "surfeit": float(self.surfeit_.surfeit_string(model_string)),
        }

    def mismodel(self, *, predictions, model_string: str) -> float:
        """Return equal-weight RMS from explicit predictions and a description."""
        values = self.components(predictions=predictions, model_string=model_string)
        return self.aggregate_components(**values)

    def analysis(self, *, predictions, model_string: str) -> dict[str, float]:
        """Return mismodel and the two component values used to compute it."""
        values = self.components(predictions=predictions, model_string=model_string)
        return {"mismodel": self.aggregate_components(**values), **values}

    @staticmethod
    def aggregate_components(*, inaccuracy: float, surfeit: float) -> float:
        """Combine supplied scalar components without requiring fitting."""
        return mismodel(inaccuracy=inaccuracy, surfeit=surfeit)

    def mismodel_model(
        self, model, *, X=None, feature_names=None, feature_indices=None,
    ) -> float:
        """Evaluate a fitted model using its canonical predictions and description.

        X defaults to the inputs retained by fit(X, y). After fit_y(y), explicit
        X is required. Its rows must match the fitted target's sample count and
        order; row order cannot be inferred or checked automatically.
        Explicit X contains estimator input columns. feature_indices maps those
        columns to original coordinates, or selects them from stored inputs.
        feature_names supplies the model-input labels for artifact extraction.
        Unsupported and unfitted estimators raise serializer validation errors.
        """
        artifacts = model_artifacts(
            self, model, X=X, feature_names=feature_names,
            feature_indices=feature_indices,
        )
        return self.mismodel(
            predictions=artifacts.predictions, model_string=artifacts.model_string,
        )

    def model_analysis(
        self, model, *, X=None, feature_names=None, feature_indices=None,
    ) -> dict[str, object]:
        """Return component diagnostics, model type, and canonical description.

        Input resolution and target-row alignment follow ``mismodel_model``.
        Predictions and the description are extracted once per call.
        """
        artifacts = model_artifacts(
            self, model, X=X, feature_names=feature_names,
            feature_indices=feature_indices,
        )
        return {
            **self.analysis(
                predictions=artifacts.predictions, model_string=artifacts.model_string,
            ),
            "model_type": artifacts.model_type,
            "model_string": artifacts.model_string,
        }

    def _fit_target(self, y) -> None:
        """Publish target state only after both component fits succeed."""
        target = _validate_vector(y, name="y").copy()
        inaccuracy = Inaccuracy(y_type=self.y_type).fit_y(target)
        surfeit = Surfeit(y_type=self.y_type).fit_y(target)
        self.y_ = target
        self.n_samples_in_ = len(target)
        self.inaccuracy_ = inaccuracy
        self.surfeit_ = surfeit

    def _clear_fitted_state(self) -> None:
        """Invalidate fitted state before validating a new evaluation dataset."""
        self.is_fitted_ = False
        self.X_ = None
        for name in (
            "y_", "n_samples_in_", "inaccuracy_", "surfeit_",
            "_model_X_", "n_features_in_", "feature_names_in_",
        ):
            self.__dict__.pop(name, None)
