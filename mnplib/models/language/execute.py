"""Restricted, vectorized interpretation of canonical model semantics."""

import operator
import numpy as np

from .nodes import (
    ModelNode, Constant, Label, Feature, Negate, Binary, Conditional,
    Vector, Element, Classify, Dense,
)
from .normalize import normalize
from .parse import parse

_OPERATIONS = {
    "+": operator.add, "-": operator.sub, "*": operator.mul,
    "/": operator.truediv, "**": operator.pow, "<": operator.lt,
    "<=": operator.le, ">": operator.gt, ">=": operator.ge, "==": operator.eq,
}


def execute(description, X):
    """Predict from an AST or grammar-validated text using quantized parameters.

    Rows of X are observations; feature indices refer to its original columns.
    Conditional branches are evaluated only on rows that take that branch.
    """
    node = parse(description) if isinstance(description, str) else normalize(description)
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional feature matrix.")

    def visit(item, rows):
        kind = type(item)
        if kind in (Constant, Label):
            return np.full(len(rows), item.value)
        if kind is Feature:
            if item.index >= X.shape[1]:
                raise ValueError(f"Feature index {item.index} is out of bounds.")
            return X[rows, item.index]
        if kind is Negate:
            return -visit(item.operand, rows)
        if kind is Binary:
            return _OPERATIONS[item.operator](visit(item.left, rows), visit(item.right, rows))
        if kind is Vector:
            if not item.items:
                return np.empty((len(rows), 0))
            return np.stack([visit(child, rows) for child in item.items], axis=1)
        if kind is Conditional:
            condition = np.asarray(visit(item.condition, rows), dtype=bool)
            if condition.ndim != 1:
                raise ValueError("A condition must provide one value per observation.")
            yes, no = visit(item.if_true, rows[condition]), visit(item.if_false, rows[~condition])
            if yes.shape[1:] != no.shape[1:]:
                raise ValueError("Conditional branches must have matching output dimensions.")
            output = np.empty((len(rows), *yes.shape[1:]), dtype=np.result_type(yes, no))
            output[condition], output[~condition] = yes, no
            return output
        if kind is Element:
            value = visit(item.vector, rows)
            if value.ndim != 2 or item.index >= value.shape[1]:
                raise ValueError("Element index is out of bounds.")
            return value[:, item.index]
        if kind is Classify:
            scores = visit(item.scores, rows)
            if scores.ndim != 2 or scores.shape[1] != len(item.labels):
                raise ValueError("Class scores must match the number of labels.")
            labels = np.asarray([label.value for label in item.labels])
            return labels[np.argmax(scores, axis=1)]
        if kind is Dense:
            inputs = visit(item.inputs, rows)
            weights = np.asarray([[x.value for x in row] for row in item.weights])
            bias = np.asarray([x.value for x in item.bias])
            if inputs.ndim != 2 or inputs.shape[1] != weights.shape[0]:
                raise ValueError("Dense input dimension does not match its weights.")
            z = inputs @ weights + bias
            if item.activation == "relu":
                return np.maximum(z, 0)
            if item.activation == "tanh":
                return np.tanh(z)
            if item.activation == "logistic":
                # Both branches keep the exponential argument non-positive.
                positive = z >= 0
                result = np.empty_like(z)
                result[positive] = 1 / (1 + np.exp(-z[positive]))
                e = np.exp(z[~positive])
                result[~positive] = e / (1 + e)
                return result
            return z
        raise TypeError(f"Unsupported model node: {kind.__name__}")

    if not isinstance(node, ModelNode):
        raise TypeError("Expected a model expression.")
    return visit(node, np.arange(X.shape[0]))
