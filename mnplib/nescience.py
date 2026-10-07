"""
Hierarchical RMS nescience for fitted models and explicit model artifacts.

Miscoding combines deficiency and surplus. Mismodel combines inaccuracy and
surfeit. Both use equal-weight RMS; only their final combination is weighted.

@author:    Rafael Garcia Leiva
@mail:      rgarcialeiva@gmail.com
"""

from __future__ import annotations

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
    """
    Evaluate weighted RMS of miscoding and mismodel.

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
    weight : float, default=0.5
        Miscoding's weight, a finite real number in [0, 1]. Mismodel receives
        1 - weight. The score is sqrt(weight * miscoding**2
        + (1 - weight) * mismodel**2). A nonfinite dimension still makes
        nescience NaN, including when its weight is zero.

    Notes
    -----
    Reports expose weight, the coefficient applied to miscoding. Parameter
    changes take effect on the next evaluation; primitive metric encoding
    changes require refitting.
    """

    component_names_ = ("deficiency", "surplus", "inaccuracy", "surfeit")
    _VALID_X_TYPES   = get_args(XType)

    def __init__(
        self,
        X_type: XType = "auto",
        y_type: YType = "auto",
        weight: float = 0.5,
    ):
        self.X_type = X_type
        self.y_type = y_type
        self.weight = weight

    def fit(self, X, y):
        """
        Fit the feature and model metrics on aligned evaluation data.

        Preserve DataFrame input for canonical model prediction and feature
        names. No predictive estimator is fitted.
        """
        self.is_fitted_ = False

        self._validate_init(X_type=self.X_type, y_type=self.y_type)
        self._validate_weight()
        y  = _validate_vector(y, name="y")
 
        X_checked, y_checked   = check_X_y(X, y, dtype=None, ensure_2d=True)
        self.X_                = X_checked
        self._model_X_         = X
        self.y_                = y_checked
        self.n_samples_in_     = X_checked.shape[0]
        self.n_features_in_    = X_checked.shape[1]
        self.miscoding_        = Miscoding(X_type=self.X_type, y_type=self.y_type).fit(X, y_checked)
        self.feature_names_in_ = self.miscoding_.feature_names_in_.copy()
        self.mismodel_         = Mismodel(y_type=self.y_type).fit_y(y_checked)

        self.is_fitted_ = True

        return self

    def __sklearn_is_fitted__(self) -> bool:
        """Report whether component fitting completed successfully."""
        return getattr(self, "is_fitted_", False)

    def nescience_model(self, model, *, X=None, feature_names=None, feature_indices=None) -> float:
        """
        Compute the nescience of a fitted predictive model.

        The model is converted into the canonical artifacts required by the
        nescience metric: the effective feature subset, the prediction vector,
        and the canonical string representation of the model. These artifacts
        are obtained through ``model_analysis()`` so that model evaluation and
        diagnostic reporting use exactly the same computation path.

        Parameters
        ----------
        model : fitted estimator
            Predictive model to evaluate.

        X : array-like or pandas.DataFrame of shape (n_samples, n_model_features), optional
            Evaluation data passed to the fitted model. If omitted, the feature data stored
            when this ``Nescience`` instance was fitted are used.

        feature_names : sequence of str, optional
            Names of the columns supplied in ``X``. 

        feature_indices : sequence of int, optional
            Mapping from the columns used by ``model`` to the original feature
            indices of the dataset on which this ``Nescience`` instance was
            fitted.

        Returns
        -------
        float
            Scalar nescience of the fitted model.
        """
        report = self.model_analysis(
            model, X=X, feature_names=feature_names, feature_indices=feature_indices,
        )
        value = float(report["nescience"])
        if np.isnan(value):
            warn_nan_model("nescience_model", report)
        return value

    def model_analysis(self, model, *, X=None, feature_names=None, feature_indices=None) -> dict[str, object]:

        """
        Return a complete nescience analysis for a fitted predictive model.

        The model is first converted into the canonical artifacts required by
        the nescience metric, including the effective feature subset, prediction
        vector, model type, and canonical string representation. The primitive
        and derived nescience metrics are then computed from these artifacts
        using the same evaluation path as the explicit-artifact interface.

        Parameters
        ----------
        model : fitted estimator
            Predictive model to analyze.

        X : array-like or pandas.DataFrame of shape (n_samples, n_model_features), optional
            Evaluation data passed to the fitted model.  If omitted, the feature data stored
            when this ``Nescience`` instance was fitted are used.

        feature_names : sequence of str, optional
            Names associated with the columns supplied in ``X``.

        feature_indices : sequence of int, optional
            Mapping from the columns used by ``model`` to the corresponding
            feature indices in the original fitted feature space. 

        Returns
        -------
        dict[str, object]
            Analysis report containing the primitive nescience components,
            derived miscoding and mismodel values, the final nescience score,
            reliability diagnostics, and canonical model metadata.
        """
        artifacts = model_artifacts(
            self, model, X=X, feature_names=feature_names, feature_indices=feature_indices,
        )
        return {
            **self.analysis(**artifacts.to_nescience_kwargs()),
            "model_type"   : artifacts.model_type,
            "model_string" : artifacts.model_string,
        }

    def components(self, *, subset, predictions, model_string: str) -> dict[str, float]:
        """
        Return the four primitive components used to compute nescience.

        The method evaluates the representation-related components,
        ``deficiency`` and ``surplus``, for the supplied feature subset and
        combines them with the description-related components, ``inaccuracy``
        and ``surfeit``, obtained from the fitted ``Mismodel`` estimator.

        Parameters
        ----------
        subset : array-like
            Features effectively used by the evaluated model.

        predictions : array-like of shape (n_samples,)
            Predictions produced by the evaluated model.

        model_string : str
            Canonical, non-empty string representation of the evaluated model.

        Returns
        -------
        dict[str, float]
            Dictionary containing the four primitive nescience components:
        """
        check_is_fitted(self)
        diagnostics = self.miscoding_.subset_analysis(subset)
        return {
            "deficiency": float(diagnostics["deficiency"]),
            "surplus": float(diagnostics["surplus"]),
            **self.mismodel_.components(predictions=predictions, model_string=model_string),
        }

    def nescience(self, *, subset, predictions, model_string: str) -> float:
        """
        Compute the scalar nescience of an explicit model description.

        Parameters
        ----------
        subset : array-like
            Features effectively used by the evaluated model.

        predictions : array-like of shape (n_samples,)
            Predictions produced by the evaluated model.

        model_string : str
            Canonical, non-empty string representation of the evaluated model.

        Returns
        -------
        float
            Scalar nescience of the supplied model artifacts.
        """
        return float(self.analysis(
            subset=subset, predictions=predictions, model_string=model_string,
        )["nescience"])

    def analysis(self, *, subset, predictions, model_string: str) -> dict[str, object]:
        """
        Return a complete nescience analysis for explicit model artifacts.

        Parameters
        ----------
        subset : array-like
            Features effectively used by the evaluated model.

        predictions : array-like of shape (n_samples,)
            Predictions produced by the evaluated model.

        model_string : str
            Canonical, non-empty string representation of the evaluated model.

        Returns
        -------
        dict[str, object]
            Analysis report containing the primitive components, the derived
            miscoding and mismodel values, the final nescience score, subset
            reliability diagnostics, and the configured miscoding weight.
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
            "weight": self._validate_weight(),
        }

    def aggregate_components(self, *, miscoding: float, mismodel: float) -> float:
        """
        Combine miscoding and mismodel into the scalar nescience value.

        The two derived components are aggregated using a weighted root mean
        square. The configured ``weight`` parameter determines the contribution
        of miscoding, while mismodel receives the complementary weight
        ``1 - weight``.

        Parameters
        ----------
        miscoding : float
            Scalar miscoding value, representing the quality of the selected representation.

        mismodel : float
            Scalar mismodel value, representing the quality of the model description.

        Returns
        -------
        float
            Weighted RMS of miscoding and mismodel.
        """
        weight = self._validate_weight()
        return float(_rms_pair(float(miscoding), float(mismodel), weights=(weight, 1 - weight)))

    def _validate_weight(self) -> float:
        """Return the current miscoding coefficient after validating its range."""

        if (not isinstance(self.weight, Real)
            or isinstance(self.weight, (bool, np.bool_))): # Explicitly reject True and False
            raise ValueError("weight must be a finite real number between 0 and 1."
        )

        weight = float(self.weight)

        if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
            raise ValueError("weight must be a finite real number between 0 and 1."
        )

        return weight

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
    weight: float = 0.5,
) -> float:
    """Evaluate explicit artifacts using weighted RMS of miscoding and mismodel."""
    metric = Nescience(X_type=X_type, y_type=y_type, weight=weight).fit(X, y)
    return metric.nescience(subset=subset, predictions=predictions, model_string=model_string)


def nescience_components(
    *, X, y, subset, predictions, model_string: str,
    X_type: XType = "auto", y_type: YType = "auto",
    weight: float = 0.5,
) -> dict[str, float]:
    """Return the four primitive components from explicit artifacts."""
    metric = Nescience(X_type=X_type, y_type=y_type, weight=weight).fit(X, y)
    return metric.components(subset=subset, predictions=predictions, model_string=model_string)


def nescience_model(
    model, *, X, y, feature_names=None, feature_indices=None,
    X_type: XType = "auto", y_type: YType = "auto",
    weight: float = 0.5,
) -> float:
    """Compute fitted-model nescience on evaluation data."""
    metric = Nescience(X_type=X_type, y_type=y_type, weight=weight).fit(X, y)
    return metric.nescience_model(
        model, feature_names=feature_names, feature_indices=feature_indices,
    )


def model_analysis(
    model, *, X, y, feature_names=None, feature_indices=None,
    X_type: XType = "auto", y_type: YType = "auto",
    weight: float = 0.5,
) -> dict[str, object]:
    """Analyze fitted-model nescience, including subset reliability."""
    metric = Nescience(X_type=X_type, y_type=y_type, weight=weight).fit(X, y)
    return metric.model_analysis(
        model, feature_names=feature_names, feature_indices=feature_indices,
    )
