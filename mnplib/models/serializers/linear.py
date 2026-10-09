"""Arithmetic Abstract Syntax Tree for fitted linear prediction rules."""

import numpy as np
from ..._types import ResolvedTask
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.utils.validation import check_is_fitted
from .base import SklearnSerializer
from ..language import Constant, Label, Binary, Conditional, Vector, Classify


def linear_expression(coefficients, intercept, features):
    """Preserve estimator column ordering and exact zero sparsity."""
    result = Constant(float(intercept))
    for coefficient, feature in zip(coefficients, features):
        coefficient = float(coefficient)
        if coefficient:
            term   = Binary("*", Constant(abs(coefficient)), feature)
            result = Binary("+" if coefficient > 0 else "-", result, term)
    return result


def classification_rule(model, features):
    scores = tuple(linear_expression(coef, intercept, features)
                   for coef, intercept in zip(model.coef_, model.intercept_))
    labels = tuple(Label(value) for value in model.classes_)
    if len(labels) == 2 and len(scores) == 1:
        return Conditional(Binary(">", scores[0], Constant(0)), labels[1], labels[0])
    return Classify(Vector(scores), labels)


class LinearModelSerializer(SklearnSerializer):
    name = "linear_model"
    supported_types = (LinearRegression,)

    def task(self, model) -> ResolvedTask:
        return "regression"

    def subset(self, model):
        check_is_fitted(model)
        coefficients = np.atleast_2d(model.coef_)
        return np.flatnonzero(np.any(coefficients != 0, axis=0)).tolist()

    def serialize(self, model, *, features):
        coefficients = np.asarray(model.coef_)
        if coefficients.ndim != 1:
            raise ValueError("Linear regression serialization requires a one-dimensional target.")
        return linear_expression(coefficients, np.asarray(model.intercept_).item(), features)

    def metadata(self, model):
        return {"n_terms": int(np.count_nonzero(model.coef_))}


class LogisticRegressionSerializer(LinearModelSerializer):
    name = "logistic_regression"
    supported_types = (LogisticRegression,)

    def task(self, model) -> ResolvedTask:
        return "classification"

    def serialize(self, model, *, features):
        return classification_rule(model, features)
