"""
Nescience aggregation for explicit model descriptions.

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

The class deliberately does not inspect fitted model objects. The caller must
provide the three practical objects needed to evaluate a model:

    * ``subset``: the features used by the model;
    * ``predictions``: the predictions produced by the model;
    * ``model_string``: a string description of the model.

This explicit design keeps model inspection and model serialization outside the
nescience metric, making the code easier to understand, test, and maintain.

@author:    Rafael Garcia Leiva
@mail:      rgarcialeiva@gmail.com
@copyright: GNU GPLv3
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted

from .miscoding import Miscoding
from .inaccuracy import Inaccuracy
from .surfeit import Surfeit


XType = Literal["auto", "numeric", "categorical"]
YType = Literal["auto", "numeric", "categorical"]
BinSpec = int | Literal["auto"]
Aggregation = Literal[
    "euclidean",
    "arithmetic",
    "geometric",
    "harmonic",
    "maximum",
    "addition",
    "product",
]


class Nescience(BaseEstimator):
    """
    Compute nescience from deficiency, surplus, inaccuracy, and surfeit.

    The class is a coordinator. It fits the target and feature representation
    once, then evaluates model nescience from explicitly supplied model
    artifacts:

    ``subset``
        Binary feature mask or list of feature indices used by the model.

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

    n_bins : int or "auto", default="auto"
        Number of uniform bins used for numeric variables. If ``"auto"``,
        the empirical-distribution utilities resolve the number of bins.

    threshold_fraction : float, default=0.01
        Minimum relative target-code-length reduction used by the internal
        ``Miscoding`` estimator when it performs greedy feature selection.

    surplus_penalty : float, default=1.0
        Penalty applied by the internal ``Miscoding`` estimator during greedy
        feature selection.

    zlib_level : int, default=9
        Compression level used by ``Surfeit``.

    zlib_overhead : int, default=6
        zlib wrapper overhead subtracted by ``Surfeit``.
    """

    component_names_ = ("deficiency", "surplus", "inaccuracy", "surfeit")

    _VALID_X_TYPES = ("auto", "numeric", "categorical")
    _VALID_Y_TYPES = ("auto", "numeric", "categorical")
    _VALID_AGGREGATIONS = (
        "euclidean",
        "arithmetic",
        "geometric",
        "harmonic",
        "maximum",
        "addition",
        "product",
    )

    def __init__(
        self,
        X_type: XType = "auto",
        y_type: YType = "auto",
        aggregation: Aggregation = "euclidean",
        weights: Mapping[str, float] | Sequence[float] | None = None,
        n_bins: BinSpec = "auto",
        threshold_fraction: float = 0.01,
        surplus_penalty: float = 1.0,
        zlib_level: int = 9,
        zlib_overhead: int = 6,
    ):
        """Initialize the estimator and validate configuration parameters."""
        self._validate_init(
            X_type=X_type,
            y_type=y_type,
            aggregation=aggregation,
            threshold_fraction=threshold_fraction,
            surplus_penalty=surplus_penalty,
            zlib_level=zlib_level,
            zlib_overhead=zlib_overhead,
        )

        self.X_type = X_type
        self.y_type = y_type
        self.aggregation = aggregation
        self.weights = weights
        self.n_bins = n_bins
        self.threshold_fraction = threshold_fraction
        self.surplus_penalty = surplus_penalty
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
        X_checked, y_checked = check_X_y(X, y, dtype=None, ensure_2d=True)

        self.X_ = X_checked
        self.y_ = y_checked
        self.n_samples_in_, self.n_features_in_ = X_checked.shape
        self.weights_ = self._resolve_weights()

        self.miscoding_ = Miscoding(
            X_type=self.X_type,
            y_type=self.y_type,
            n_bins=self.n_bins,
            threshold_fraction=self.threshold_fraction,
            surplus_penalty=self.surplus_penalty,
        )
        self.miscoding_.fit(X, y_checked)

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
            Binary feature mask or list of selected feature indices.

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
                self.miscoding_.miscoding_subset(subset, mode="deficiency")
            ),
            "surplus": float(
                self.miscoding_.miscoding_subset(subset, mode="surplus")
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
            Binary feature mask or list of selected feature indices.

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

    def explain(
        self,
        *,
        subset,
        predictions,
        model_string: str,
    ) -> dict[str, object]:
        """
        Explain the nescience of supplied model artifacts.

        The explanation identifies the dominant component, assigns a qualitative
        profile, and provides a practical recommendation for reducing nescience.

        Parameters
        ----------
        subset : array-like
            Binary feature mask or list of selected feature indices.

        predictions : array-like of shape (n_samples,)
            Prediction vector produced by the model.

        model_string : str
            String description of the model.

        Returns
        -------
        dict
            Explanation dictionary containing the scalar nescience value,
            component values, dominant component, qualitative profile, and
            recommendation.
        """
        component_values = self.components(
            subset=subset,
            predictions=predictions,
            model_string=model_string,
        )
        nescience_value = self.aggregate_components(**component_values)
        dominant_component = max(component_values, key=component_values.get)
        profile, profile_explanation = self._profile_from_components(component_values)

        return {
            "nescience": float(nescience_value),
            "aggregation": self.aggregation,
            "weights": dict(zip(self.component_names_, self.weights_)),
            "components": component_values,
            "dominant_component": dominant_component,
            "profile": profile,
            "profile_explanation": profile_explanation,
            "recommendation": self._recommendation_from_dominant_component(
                dominant_component,
                component_values,
            ),
        }

    def score(
        self,
        *,
        subset,
        predictions,
        model_string: str,
    ) -> float:
        """
        Return a higher-is-better score for supplied model artifacts.

        Since lower nescience is better, the score is defined as
        ``1 - nescience``. This method is not intended to inspect or evaluate a
        scikit-learn model object directly.
        """
        return 1.0 - self.nescience(
            subset=subset,
            predictions=predictions,
            model_string=model_string,
        )

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

    def _profile_from_components(
        self,
        components: Mapping[str, float],
    ) -> tuple[str, str]:
        """
        Return a qualitative profile from the four component values.
        """
        deficiency = float(components["deficiency"])
        surplus = float(components["surplus"])
        inaccuracy = float(components["inaccuracy"])
        surfeit = float(components["surfeit"])

        high_threshold = 0.50
        low_threshold = 0.25

        high_deficiency = deficiency >= high_threshold
        high_surplus = surplus >= high_threshold
        high_inaccuracy = inaccuracy >= high_threshold
        high_surfeit = surfeit >= high_threshold

        low_deficiency = deficiency <= low_threshold
        low_surplus = surplus <= low_threshold
        low_inaccuracy = inaccuracy <= low_threshold
        low_surfeit = surfeit <= low_threshold

        if low_deficiency and low_surplus and low_inaccuracy and low_surfeit:
            return (
                "low_nescience_model",
                "The model has low deficiency, low surplus, low inaccuracy, "
                "and low surfeit.",
            )

        if high_deficiency and not high_surplus:
            return (
                "under_informed_model",
                "The selected features do not contain enough information to "
                "describe the target.",
            )

        if high_surplus and not high_deficiency:
            return (
                "over_fed_model",
                "The selected features contain substantial information that is "
                "not explained by the target.",
            )

        if low_deficiency and low_surplus and high_inaccuracy:
            return (
                "bad_learner",
                "The selected features appear adequate, but the predictions do "
                "not match the target accurately.",
            )

        if low_deficiency and low_surplus and low_inaccuracy and high_surfeit:
            return (
                "over_complex_model",
                "The model predicts well using adequate features, but its "
                "description appears unnecessarily redundant.",
            )

        if high_deficiency and high_surplus:
            return (
                "poor_input_representation",
                "The selected input representation is both insufficient for "
                "the target and rich in target-irrelevant information.",
            )

        if high_inaccuracy and high_surfeit:
            return (
                "complex_but_inaccurate_model",
                "The model is both inaccurate and apparently more complex than "
                "its performance justifies.",
            )

        return (
            "mixed_nescience_profile",
            "No single qualitative profile dominates; inspect the four "
            "component values.",
        )

    def _recommendation_from_dominant_component(
        self,
        dominant_component: str,
        components: Mapping[str, float],
    ) -> str:
        """
        Return a practical recommendation from the dominant component.
        """
        value = float(components[dominant_component])

        if dominant_component == "deficiency":
            return (
                f"Dominant source of nescience: deficiency ({value:.4f}). "
                "Improve the input representation: add relevant features, "
                "collect additional data, engineer more informative variables, "
                "or reconsider whether the current observables contain enough "
                "information about the target."
            )

        if dominant_component == "surplus":
            return (
                f"Dominant source of nescience: surplus ({value:.4f}). "
                "Reduce target-irrelevant input information: apply feature "
                "selection, increase the surplus penalty, remove noisy variables, "
                "or simplify the representation used by the model."
            )

        if dominant_component == "inaccuracy":
            return (
                f"Dominant source of nescience: inaccuracy ({value:.4f}). "
                "Improve the learner: tune hyperparameters, change the model "
                "class, improve training, handle imbalance, or check whether "
                "the predictive task is well specified."
            )

        if dominant_component == "surfeit":
            return (
                f"Dominant source of nescience: surfeit ({value:.4f}). "
                "Simplify the model description: prune, regularize, reduce "
                "model size, choose a more compact model family, or remove "
                "unnecessary parameters and rules."
            )

        return (
            "Unable to determine a dominant source of nescience. Inspect the "
            "component values."
        )

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
        threshold_fraction,
        surplus_penalty,
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

        if threshold_fraction < 0:
            raise ValueError("threshold_fraction must be non-negative.")

        if surplus_penalty < 0:
            raise ValueError("surplus_penalty must be non-negative.")

        zlib_level = int(zlib_level)
        if zlib_level < 0 or zlib_level > 9:
            raise ValueError(
                "zlib_level must be an integer between 0 and 9. "
                f"Got zlib_level={zlib_level!r} instead."
            )

        zlib_overhead = int(zlib_overhead)
        if zlib_overhead < 0:
            raise ValueError("zlib_overhead must be non-negative.")


def nescience_score(
    X,
    y,
    *,
    subset,
    predictions,
    model_string: str,
    **kwargs,
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
        Binary feature mask or list of selected feature indices.

    predictions : array-like of shape (n_samples,)
        Prediction vector produced by the model.

    model_string : str
        String description of the model.

    **kwargs
        Additional keyword arguments passed to ``Nescience``.

    Returns
    -------
    float
        Aggregated nescience value.
    """
    metric = Nescience(**kwargs).fit(X, y)
    return metric.nescience(
        subset=subset,
        predictions=predictions,
        model_string=model_string,
    )


def nescience_components(
    X,
    y,
    *,
    subset,
    predictions,
    model_string: str,
    **kwargs,
) -> dict[str, float]:
    """
    Compute the four scalar nescience components using a functional interface.
    """
    metric = Nescience(**kwargs).fit(X, y)
    return metric.components(
        subset=subset,
        predictions=predictions,
        model_string=model_string,
    )
