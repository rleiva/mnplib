"""
Inaccuracy based on empirical code lengths.

This module implements the inaccuracy component of nescience. Inaccuracy
measures how far the predictions produced by a model are from the target
representation, using empirical code lengths as practical approximations.

@author:    Rafael Garcia Leiva
@mail:      rgarcialeiva@gmail.com
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted

from .utils import empirical_distribution


YType = Literal["auto", "numeric", "categorical"]
BinSpec = int | Literal["auto"]


class Inaccuracy(BaseEstimator):
    """
    Compute the inaccuracy of predictions.

    Inaccuracy is computed from empirical code lengths as

        I(y, y_hat) = (L(y, y_hat) - min(L(y), L(y_hat))) / max(L(y), L(y_hat)),

    where L(y), L(y_hat), and L(y, y_hat) are empirical code lengths.

    Parameters
    ----------
    y_type : {"auto", "numeric", "categorical"}, default="auto"
        Encoding strategy for the target variable.

    n_bins : int or "auto", default="auto"
        Number of uniform bins used for numeric targets. If ``"auto"``,
        Rice's rule is used by the empirical-distribution utilities.
    """

    _VALID_Y_TYPES = ("auto", "numeric", "categorical")

    def __init__(
        self,
        y_type: YType = "auto",
        n_bins: BinSpec = "auto",
    ):
        """Initialize the estimator and validate configuration parameters."""
        if y_type not in self._VALID_Y_TYPES:
            raise ValueError(
                "Valid options for 'y_type' are {}. Got y_type={!r} instead."
                .format(self._VALID_Y_TYPES, y_type)
            )

        self.y_type = y_type
        self.n_bins = n_bins

    def fit(self, X, y):
        """
        Fit the inaccuracy object with a dataset.

        The method stores the target values and computes their empirical code
        length. The feature matrix X is stored so that trained models can later
        be evaluated through ``inaccuracy_model(model)`` or ``score(model)``.

        Parameters
        * X : array-like of shape (n_samples, n_features)
              Sample vectors used to evaluate model predictions.
        * y : array-like of shape (n_samples,)
              Target values.

        Returns
        * self : Inaccuracy
              Fitted estimator.
        """
        self.X_, self.y_ = check_X_y(X, y, dtype=None, ensure_2d=True)
        self._fit_target(self.y_)
        self.n_features_in_ = self.X_.shape[1]

        return self

    def fit_y(self, y):
        """
        Fit the inaccuracy object with only a target vector.

        This method is useful when predictions are already available and no
        feature matrix or model object is needed.

        Parameters
        * y : array-like of shape (n_samples,)
              Target values.

        Returns
        * self : Inaccuracy
              Fitted estimator.
        """
        self.X_ = None
        self._fit_target(y)

        return self

    def inaccuracy_model(self, model) -> float:
        """
        Compute the inaccuracy of a trained model.

        Parameters
        * model : object
              Trained model implementing a ``predict(X)`` method.

        Returns
        * float
              Inaccuracy value in the interval [0, 1], up to empirical
              approximation effects.
        """
        check_is_fitted(self)

        if self.X_ is None:
            raise ValueError(
                "This Inaccuracy instance was fitted with fit_y(y), so no "
                "feature matrix is available. Use inaccuracy_predictions(...) "
                "instead, or call fit(X, y)."
            )

        if not hasattr(model, "predict"):
            raise TypeError("model must implement a predict(X) method.")

        return self.inaccuracy_predictions(model.predict(self.X_))

    def score(self, model, y=None) -> float:
        """
        Return a higher-is-better score for a trained model.

        The score is defined as ``1 - inaccuracy_model(model)``. The optional
        ``y`` parameter is accepted only for scikit-learn scoring compatibility
        and is ignored.
        """
        return 1.0 - self.inaccuracy_model(model)

    def inaccuracy_predictions(self, predictions) -> float:
        """
        Compute the inaccuracy of a prediction vector.

        Parameters
        * predictions : array-like of shape (n_samples,)
              Predicted values.

        Returns
        * float
              Inaccuracy value in the interval [0, 1], up to empirical
              approximation effects.
        """
        check_is_fitted(self)

        pred = self._validate_predictions(predictions)
        len_pred = self._code_length(pred)
        len_joint = self._code_length(pred, self.y_)

        return self._inaccuracy_from_lengths(
            len_pred=len_pred,
            len_joint=len_joint,
            pred=pred,
        )

    def _fit_target(self, y) -> None:
        """Fit target-dependent attributes."""
        self.y_ = self._validate_1d_vector(y, name="y")
        self.y_isnumeric_ = self._infer_y_isnumeric(self.y_)
        self.len_y_ = self._code_length(self.y_)
        self.n_samples_in_ = self.y_.shape[0]
        self.is_fitted_ = True

    def _code_length(self, *columns) -> float:
        """
        Compute the empirical joint code length of target-like variables.

        All variables passed to this method are interpreted with the same
        numeric/categorical type as the fitted target.
        """
        return float(
            empirical_distribution(
                columns=columns,
                numeric=[self.y_isnumeric_] * len(columns),
                n_bins=self.n_bins,
            ).code_length
        )

    def _inaccuracy_from_lengths(
        self,
        len_pred: float,
        len_joint: float,
        pred: np.ndarray,
    ) -> float:
        """
        Compute normalized inaccuracy from prediction and joint code lengths.
        """
        len_y = float(self.len_y_)
        len_pred = float(len_pred)
        len_joint = float(len_joint)

        denominator = max(len_y, len_pred)

        if denominator == 0.0:
            same_predictions = (
                np.allclose(self.y_, pred)
                if self.y_isnumeric_
                else np.array_equal(self.y_, pred)
            )
            return 0.0 if same_predictions else 1.0

        inacc = (len_joint - min(len_y, len_pred)) / denominator

        # Empirical estimates and floating-point arithmetic may occasionally
        # produce tiny negative values or tiny values above one.
        return float(np.clip(inacc, 0.0, 1.0))

    def _infer_y_isnumeric(self, y: np.ndarray) -> bool:
        """
        Infer whether the target should be treated as numeric.

        Returns
        * bool
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
        """
        Convert an input vector into a non-empty one-dimensional NumPy array.
        """
        arr = np.asarray(values)

        if arr.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional array.")

        if arr.shape[0] == 0:
            raise ValueError(f"{name} must not be empty.")

        return arr

    def _validate_predictions(self, predictions) -> np.ndarray:
        """
        Validate predictions against the fitted target vector.
        """
        pred = self._validate_1d_vector(predictions, name="predictions")

        if pred.shape[0] != self.y_.shape[0]:
            raise ValueError(
                "predictions and y must have the same number of samples. "
                f"Got {pred.shape[0]} predictions and {self.y_.shape[0]} targets."
            )

        return pred


def inaccuracy_score(
    y_true,
    y_pred,
    *,
    y_type: YType = "auto",
    n_bins: BinSpec = "auto",
) -> float:
    """
    Compute inaccuracy directly from true and predicted target vectors.

    This convenience function does not require a feature matrix or a model.
    """
    metric = Inaccuracy(
        y_type=y_type,
        n_bins=n_bins,
    )

    metric.fit_y(y_true)

    return metric.inaccuracy_predictions(y_pred)
