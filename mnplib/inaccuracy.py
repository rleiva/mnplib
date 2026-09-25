"""
Inaccuracy based on empirical code lengths.

This module implements the inaccuracy component of nescience. Inaccuracy
measures how far the predictions produced by a model are from the target
representation, using empirical code lengths as practical approximations.

@author:    Rafael Garcia Leiva
@mail:      rgarcialeiva@gmail.com
"""

from __future__ import annotations

from typing import get_args

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted

from ._types import YType
from .utils import (
    _resolve_bins,
    _validate_vector,
    empirical_distribution_array,
    empirical_distribution_vector,
)
from .models.inputs import model_input


class Inaccuracy(BaseEstimator):
    """
    Compute the inaccuracy of predictions.

    Inaccuracy is computed from empirical code lengths as

        I(y, y_hat) = (L(y, y_hat) - min(L(y), L(y_hat))) / max(L(y), L(y_hat)),

    where L(y), L(y_hat), and L(y, y_hat) are empirical code lengths.

    ``prediction_analysis()`` and ``model_analysis()`` return flat reports with
    code lengths, descriptive joint-state counts, and conventional error metrics.

    Numeric targets use ``max(2, floor(2 * n_samples**(1/3)))`` uniform bins.
    This target-based count is shared by the target, predictions, and their
    joint distribution.

    Parameters
    ----------
    y_type : {"auto", "numeric", "categorical"}, default="auto"
        Encoding strategy for the target variable.
    """

    _VALID_Y_TYPES = get_args(YType)

    def __init__(self, y_type: YType = "auto"):
        """Initialize the estimator configuration."""
        if y_type not in self._VALID_Y_TYPES:
            raise ValueError(
                "Valid options for 'y_type' are {}. Got y_type={!r} instead."
                .format(self._VALID_Y_TYPES, y_type)
            )

        self.y_type = y_type

    def fit(self, X, y):
        """
        Fit the inaccuracy object with a dataset.

        The method stores the target values and computes their empirical code
        length. The feature matrix X is stored so that trained models can later
        be evaluated through ``inaccuracy_model(model)``.

        Parameters
        * X : array-like of shape (n_samples, n_features)
              Sample vectors used to evaluate model predictions.
        * y : array-like of shape (n_samples,)
              Target values.

        Returns
        * self : Inaccuracy
              Fitted estimator.
        """
        y = _validate_vector(y, name="y")
        self.X_, self.y_ = check_X_y(X, y, dtype=None, ensure_2d=True)
        self._model_X_ = X
        self.feature_names_in_ = np.asarray(
            getattr(X, "columns", [f"x{i}" for i in range(self.X_.shape[1])]), dtype=object)
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
        for name in ("_model_X_", "feature_names_in_", "n_features_in_"):
            if hasattr(self, name):
                delattr(self, name)
        self._fit_target(y)

        return self

    def inaccuracy_model(self, model, *, X=None, feature_names=None, feature_indices=None) -> float:
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
        predictions = self._predictions_from_model(
            model, X=X, feature_names=feature_names, feature_indices=feature_indices)
        return self.inaccuracy_predictions(predictions)


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
        return float(self.prediction_analysis(predictions)["inaccuracy"])

    def prediction_analysis(self, predictions) -> dict[str, object]:
        """Analyze predictions against the fitted target.

        Return inaccuracy, sample count, resolved ``y_type`` and numeric bin
        count, and target, prediction, and joint code lengths in bits. Numeric
        targets include ``mae`` and ``rmse`` in target units; categorical targets
        include ``accuracy`` and use None for ``resolved_n_bins``.

        Joint-state counts, mean occupancy, and singleton fraction describe
        sparsity without imposing a reliability rejection threshold. Conventional
        prediction errors are complementary to information-based inaccuracy.
        """
        check_is_fitted(self)
        pred = self._validate_predictions(predictions)
        prediction_summary = self._empirical_summary(pred)
        joint_summary = self._empirical_summary(pred, self.y_)
        n_states = int(joint_summary.n_states)
        n_singletons = int(np.count_nonzero(joint_summary.counts == 1))
        report = {
            "inaccuracy": self._inaccuracy_from_lengths(
                len_pred=prediction_summary.code_length,
                len_joint=joint_summary.code_length,
                pred=pred,
            ),
            "n_samples": int(self.n_samples_in_),
            "y_type": "numeric" if self.y_isnumeric_ else "categorical",
            "resolved_n_bins": (
                _resolve_bins("auto", n_samples=self.n_samples_in_)
                if self.y_isnumeric_ else None
            ),
            "target_code_length_bits": float(self.len_y_),
            "prediction_code_length_bits": float(prediction_summary.code_length),
            "joint_code_length_bits": float(joint_summary.code_length),
            "n_observed_joint_states": n_states,
            "mean_joint_occupancy": float(self.n_samples_in_ / n_states),
            "n_singleton_joint_states": n_singletons,
            "singleton_fraction": float(n_singletons / n_states),
        }
        if self.y_isnumeric_:
            report["mae"] = float(mean_absolute_error(self.y_, pred))
            report["rmse"] = float(np.sqrt(mean_squared_error(self.y_, pred)))
        else:
            report["accuracy"] = float(np.mean(self.y_ == pred))
        return report

    def model_analysis(self, model, *, X=None, feature_names=None,
                       feature_indices=None) -> dict[str, object]:
        """Return prediction analysis for a fitted model implementing predict.

        X defaults to fitted evaluation data. Explicit X contains estimator
        input columns in fitted-target row order; feature_indices maps those
        columns into the original feature space. No model serializer is required.
        """
        predictions = self._predictions_from_model(
            model, X=X, feature_names=feature_names, feature_indices=feature_indices)
        return self.prediction_analysis(predictions)

    def _predictions_from_model(self, model, *, X, feature_names, feature_indices):
        """Resolve evaluation inputs and predict without fitting the model."""
        check_is_fitted(self)
        if not hasattr(model, "predict"):
            raise TypeError("model must implement a predict(X) method.")
        source = model_input(self, model, X=X, feature_indices=feature_indices)
        if feature_names is not None and len(feature_names) != np.shape(source)[1]:
            raise ValueError("feature_names must match the estimator input columns.")
        return model.predict(source)

    def _fit_target(self, y) -> None:
        """Fit target-dependent attributes."""
        self.y_ = _validate_vector(y, name="y")
        self.y_isnumeric_ = self._infer_y_isnumeric(self.y_)
        self.len_y_ = float(self._empirical_summary(self.y_).code_length)
        self.n_samples_in_ = self.y_.shape[0]
        self.is_fitted_ = True

    def _empirical_summary(self, *columns):
        """
        Summarize the empirical joint distribution of target-like variables.

        All variables passed to this method are interpreted with the same
        numeric/categorical type as the fitted target. Marginal and joint
        distributions share the target-only bin count.
        """
        bins = _resolve_bins("auto", n_samples=self.y_.size)
        if len(columns) == 1:
            return empirical_distribution_vector(
                columns[0], numeric=self.y_isnumeric_, n_bins=bins,
            )
        return empirical_distribution_array(
            np.asarray(columns, dtype=object).T,
            numeric=self.y_isnumeric_,
            n_bins=bins,
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

    def _validate_predictions(self, predictions) -> np.ndarray:
        """
        Validate predictions against the fitted target vector.
        """
        pred = _validate_vector(predictions, name="predictions")

        if pred.shape[0] != self.y_.shape[0]:
            raise ValueError(
                "predictions and y must have the same number of samples. "
                f"Got {pred.shape[0]} predictions and {self.y_.shape[0]} targets."
            )

        return pred


def inaccuracy_predictions(
    predictions,
    *,
    y,
    y_type: YType = "auto",
) -> float:
    """
    Compute inaccuracy directly from true and predicted target vectors.

    This convenience function does not require a feature matrix or a model.
    """
    return Inaccuracy(y_type=y_type).fit_y(y).inaccuracy_predictions(predictions)


def inaccuracy_model(model, *, X, y, feature_names=None, feature_indices=None,
                      y_type: YType = "auto") -> float:
    """Compute a fitted model's inaccuracy on evaluation data."""
    return Inaccuracy(y_type=y_type).fit(X, y).inaccuracy_model(
        model, feature_names=feature_names, feature_indices=feature_indices)


def prediction_analysis(predictions, *, y, y_type: YType = "auto") -> dict[str, object]:
    """Analyze predictions using code lengths and conventional error metrics."""
    return Inaccuracy(y_type=y_type).fit_y(y).prediction_analysis(predictions)


def model_analysis(model, *, X, y, feature_names=None, feature_indices=None,
                   y_type: YType = "auto") -> dict[str, object]:
    """Return prediction diagnostics for a fitted model on evaluation data."""
    return Inaccuracy(y_type=y_type).fit(X, y).model_analysis(
        model, feature_names=feature_names, feature_indices=feature_indices)
