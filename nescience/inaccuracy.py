"""
inaccuracy.py

Machine learning with the Minimum Nescience Principle.

This module implements the inaccuracy component of nescience. Inaccuracy
measures how far the predictions produced by a model are from the target
representation, using empirical code lengths as practical approximations.

rather than the old optimal_code_length(x1=..., numeric1=...) interface.

@author:    Rafael Garcia Leiva
@mail:      rgarcialeiva@gmail.com
@copyright: GNU GPLv3
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted

from .utils import empirical_code_length

YType = Literal["auto", "numeric", "categorical"]
BinSpec = int | Literal["auto"]
EntropyCorrection = Literal["none", "miller_madow", "dirichlet"]

class Inaccuracy(BaseEstimator):
    """
    Compute the inaccuracy of predictions according to the Minimum Nescience
    Principle.

    Inaccuracy is computed from empirical code lengths as

        I(y, y_hat) = (L(y, y_hat) - min(L(y), L(y_hat))) / max(L(y), L(y_hat)),

    where L(y), L(y_hat), and L(y, y_hat) are empirical code lengths.

    Parameters
    ----------
    y_type : {"auto", "numeric", "categorical"}, default="auto"
        Type of target variable.

        - "auto": infer the target type from y.
        - "numeric": treat y as a numeric/regression target.
        - "categorical": treat y as a categorical/classification target.

    n_bins : int or "auto", default="auto"
        Number of bins used for numeric targets. If "auto", the automatic
        occupancy-based discretization policy from utils.empirical_code_length
        is used.

    min_samples_per_cell : int, default=5
        Desired average number of samples per empirical joint cell when
        n_bins="auto".

    min_bins : int, default=2
        Minimum number of bins per numeric variable when n_bins="auto".

    max_bins : int, optional
        Maximum number of bins for numeric discretization when n_bins="auto".

    missing : {"raise"}, default="raise"
        Missing-value handling policy. Currently only "raise" is supported by
        the corresponding utilities.

    base : float, default=2.0
        Logarithm base used for empirical code lengths. base=2 gives bits.

    correction : {"none", "miller_madow", "dirichlet"}, default="none"
        Entropy-bias correction used by the empirical code-length estimator.

    alpha : float, default=0.5
        Dirichlet smoothing parameter when correction="dirichlet".

    alphabet_size : int, optional
        Number of possible states used by entropy corrections. In the default
        plug-in computation, only observed states contribute.

    Examples
    --------
    Classification example:

    >>> from sklearn.datasets import load_digits
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from mnplib.inaccuracy import Inaccuracy
    >>>
    >>> X, y = load_digits(return_X_y=True)
    >>> tree = DecisionTreeClassifier(min_samples_leaf=5, random_state=42)
    >>> tree.fit(X, y)
    >>>
    >>> inacc = Inaccuracy(y_type="auto")
    >>> inacc.fit(X, y)
    >>> inacc.inaccuracy_model(tree)

    Regression example:

    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.tree import DecisionTreeRegressor
    >>> from mnplib.inaccuracy import Inaccuracy
    >>>
    >>> X, y = load_diabetes(return_X_y=True)
    >>> tree = DecisionTreeRegressor(min_samples_leaf=5, random_state=42)
    >>> tree.fit(X, y)
    >>>
    >>> inacc = Inaccuracy(y_type="auto")
    >>> inacc.fit(X, y)
    >>> inacc.inaccuracy_model(tree)
    """

    def __init__(
        self,
        y_type: YType = "auto",
        n_bins: BinSpec = "auto",
        min_samples_per_cell: int = 5,
        min_bins: int = 2,
        max_bins: int | None = None,
        missing: str = "raise",
        base: float = 2.0,
        correction: EntropyCorrection = "none",
        alpha: float = 0.5,
        alphabet_size: int | None = None,
    ):
        valid_y_types     = ("auto", "numeric", "categorical")
        valid_corrections = ("none", "miller_madow", "dirichlet")

        if y_type not in valid_y_types:
            raise ValueError(
                "Valid options for 'y_type' are {}. Got y_type={!r} instead."
                .format(valid_y_types, y_type)
            )

        if correction not in valid_corrections:
            raise ValueError(
                "Valid options for 'correction' are {}. Got correction={!r} instead."
                .format(valid_corrections, correction)
            )

        self.y_type = y_type
        self.n_bins = n_bins
        self.min_samples_per_cell = min_samples_per_cell
        self.min_bins = min_bins
        self.max_bins = max_bins
        self.missing = missing
        self.base = base
        self.correction = correction
        self.alpha = alpha
        self.alphabet_size = alphabet_size

    def fit(self, X, y):
        """
        Fit the inaccuracy object with a dataset.

        The method stores the target values and computes their empirical code
        length. The feature matrix X is stored so that trained models can later
        be evaluated through inaccuracy_model(model) or score(model).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Sample vectors used to evaluate model predictions.

        y : array-like of shape (n_samples,)
            Target values. Numeric and categorical targets are supported.

        Returns
        -------
        self
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
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self
            Fitted estimator.
        """
        self.X_ = None
        self._fit_target(y)

        return self

    def inaccuracy_model(self, model) -> float:
        """
        Compute the inaccuracy of a trained model.

        Parameters
        ----------
        model : object
            A trained model implementing a predict(X) method.

        Returns
        -------
        float
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

        predictions = model.predict(self.X_)

        return self.inaccuracy_predictions(predictions)

    def score(self, model, y=None) -> float:
        """
        Return a higher-is-better score for a trained model.

        The score is defined as 1 - inaccuracy_model(model). The optional y
        parameter is accepted only for compatibility with scikit-learn-style
        scoring conventions and is ignored.
        """
        return 1.0 - self.inaccuracy_model(model)

    def inaccuracy_predictions(self, predictions) -> float:
        """
        Compute the inaccuracy of a list of predicted values.

        Parameters
        ----------
        predictions : array-like of shape (n_samples,)
            Predicted values.

        Returns
        -------
        float
            Inaccuracy value in the interval [0, 1], up to empirical
            approximation effects.
        """
        return self.inaccuracy_predictions_detailed(predictions)["inaccuracy"]

    def inaccuracy_predictions_detailed(self, predictions) -> dict[str, float]:
        """
        Compute inaccuracy and return the code-length components.

        Parameters
        ----------
        predictions : array-like of shape (n_samples,)
            Predicted values.

        Returns
        -------
        dict
            Dictionary containing inaccuracy, score, len_y, len_pred,
            len_joint, excess_length, and denominator.
        """
        check_is_fitted(self)

        pred = self._validate_predictions(predictions)
        len_pred = self._prediction_code_length(pred)
        len_joint = self._joint_code_length(pred)

        inacc = self._compute_inaccuracy(
            len_y=self.len_y_,
            len_pred=len_pred,
            len_joint=len_joint,
            y_true=self.y_,
            y_pred=pred,
            y_isnumeric=self.y_isnumeric_,
        )

        denominator = max(float(self.len_y_), float(len_pred))
        excess_length = float(len_joint) - min(float(self.len_y_), float(len_pred))

        return {
            "inaccuracy": float(inacc),
            "score": float(1.0 - inacc),
            "len_y": float(self.len_y_),
            "len_pred": float(len_pred),
            "len_joint": float(len_joint),
            "excess_length": float(excess_length),
            "denominator": float(denominator),
        }

    def score_predictions(self, predictions) -> float:
        """
        Return a higher-is-better score for a prediction vector.

        This is defined as 1 - inaccuracy.
        """
        return 1.0 - self.inaccuracy_predictions(predictions)

    def _fit_target(self, y) -> None:
        """
        Fit target-dependent attributes.
        """
        self.y_ = self._validate_1d_vector(y, name="y")
        self.y_isnumeric_ = self._infer_y_isnumeric(self.y_)
        self.len_y_ = self._target_code_length()
        self.n_samples_in_ = self.y_.shape[0]
        self.is_fitted_ = True

    def _target_code_length(self) -> float:
        return self._code_length(
            columns=[self.y_],
            numeric=[self.y_isnumeric_],
        )

    def _prediction_code_length(self, pred: np.ndarray) -> float:
        return self._code_length(
            columns=[pred],
            numeric=[self.y_isnumeric_],
        )

    def _joint_code_length(self, pred: np.ndarray) -> float:
        return self._code_length(
            columns=[pred, self.y_],
            numeric=[self.y_isnumeric_, self.y_isnumeric_],
        )

    def _code_length(self, *, columns, numeric) -> float:
        """
        Compute empirical code length using the utils API.
        """
        return empirical_code_length(
            columns=columns,
            numeric=numeric,
            n_bins=self.n_bins,
            missing=self.missing,
            min_samples_per_cell=self.min_samples_per_cell,
            min_bins=self.min_bins,
            max_bins=self.max_bins,
            base=self.base,
            correction=self.correction,
            alpha=self.alpha,
            alphabet_size=self.alphabet_size,
            per_sample=False,
            normalized=False,
        )

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

    @staticmethod
    def _compute_inaccuracy(
        *,
        len_y: float,
        len_pred: float,
        len_joint: float,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_isnumeric: bool,
    ) -> float:
        """
        Compute the normalized inaccuracy from empirical code lengths.

        This method also handles degenerate zero-code-length cases. Such cases
        arise, for example, when y and/or predictions are constant.
        """
        len_y = float(len_y)
        len_pred = float(len_pred)
        len_joint = float(len_joint)

        denominator = max(len_y, len_pred)

        if denominator == 0.0:
            if y_isnumeric:
                same_predictions = np.allclose(y_true, y_pred)
            else:
                same_predictions = np.array_equal(y_true, y_pred)

            return 0.0 if same_predictions else 1.0

        inacc = (len_joint - min(len_y, len_pred)) / denominator

        # Empirical estimates and floating-point arithmetic may occasionally
        # produce tiny negative values or tiny values above one.
        return float(np.clip(inacc, 0.0, 1.0))


def inaccuracy_score(
    y_true,
    y_pred,
    *,
    y_type: YType = "auto",
    n_bins: BinSpec = "auto",
    min_samples_per_cell: int = 5,
    min_bins: int = 2,
    max_bins: int | None = None,
    missing: str = "raise",
    base: float = 2.0,
    correction: EntropyCorrection = "none",
    alpha: float = 0.5,
    alphabet_size: int | None = None,
) -> float:
    """
    Compute inaccuracy directly from true and predicted target vectors.

    This convenience function does not require a feature matrix or a model.
    """
    metric = Inaccuracy(
        y_type=y_type,
        n_bins=n_bins,
        min_samples_per_cell=min_samples_per_cell,
        min_bins=min_bins,
        max_bins=max_bins,
        missing=missing,
        base=base,
        correction=correction,
        alpha=alpha,
        alphabet_size=alphabet_size,
    )

    metric.fit_y(y_true)

    return metric.inaccuracy_predictions(y_pred)


def inaccuracy_score_detailed(
    y_true,
    y_pred,
    *,
    y_type: YType = "auto",
    n_bins: BinSpec = "auto",
    min_samples_per_cell: int = 5,
    min_bins: int = 2,
    max_bins: int | None = None,
    missing: str = "raise",
    base: float = 2.0,
    correction: EntropyCorrection = "none",
    alpha: float = 0.5,
    alphabet_size: int | None = None,
) -> dict[str, float]:
    """
    Compute detailed inaccuracy information directly from true and predicted
    target vectors.
    """
    metric = Inaccuracy(
        y_type=y_type,
        n_bins=n_bins,
        min_samples_per_cell=min_samples_per_cell,
        min_bins=min_bins,
        max_bins=max_bins,
        missing=missing,
        base=base,
        correction=correction,
        alpha=alpha,
        alphabet_size=alphabet_size,
    )

    metric.fit_y(y_true)

    return metric.inaccuracy_predictions_detailed(y_pred)
