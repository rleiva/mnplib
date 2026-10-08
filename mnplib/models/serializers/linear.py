"""Arithmetic ASTs for fitted linear prediction rules."""

import numpy as np
from ..._types import ResolvedTask
from sklearn.linear_model import LinearRegression, LogisticRegression
from .base import SklearnSerializer, require_fitted
from ..language import Constant, Label, Binary, Conditional, Vector, Classify


def linear_expression(coefficients, intercept, features):
    """Preserve estimator column ordering and exact zero sparsity."""
    result = Constant(float(intercept))
    for coefficient, feature in zip(coefficients, features):
        coefficient = float(coefficient)
        if coefficient:
            term = Binary("*", Constant(abs(coefficient)), feature)
            result = Binary("+" if coefficient > 0 else "-", result, term)
    return result


def linear_outputs(model, features):
    coefficients = np.asarray(model.coef_)
    intercepts = np.broadcast_to(np.asarray(model.intercept_).reshape(-1),
                                 (1 if coefficients.ndim == 1 else len(coefficients),))
    if coefficients.ndim == 1:
        return linear_expression(coefficients, intercepts[0], features)
    return Vector(tuple(linear_expression(coef, intercept, features)
                        for coef, intercept in zip(coefficients, intercepts)))


def classification_rule(model, features):
    scores = linear_outputs(model, features)
    labels = tuple(Label(value) for value in model.classes_)
    if len(labels) == 2 and len(scores.items) == 1:
        return Conditional(Binary(">", scores.items[0], Constant(0)), labels[1], labels[0])
    return Classify(scores, labels)


class LinearModelSerializer(SklearnSerializer):
    name = "linear_model"
    supported_types = (LinearRegression,)

    def task(self, model) -> ResolvedTask:
        return "regression"

    def subset(self, model):
        require_fitted(model)
        coefficients = np.atleast_2d(model.coef_)
        return np.flatnonzero(np.any(coefficients != 0, axis=0)).tolist()

    def serialize(self, model, *, features):
        return linear_outputs(model, features)

    def metadata(self, model):
        return {"n_terms": int(np.count_nonzero(model.coef_))}


class LogisticRegressionSerializer(LinearModelSerializer):
    name = "logistic_regression"
    supported_types = (LogisticRegression,)

    def task(self, model) -> ResolvedTask:
        return "classification"

    def serialize(self, model, *, features):
        return classification_rule(model, features)
