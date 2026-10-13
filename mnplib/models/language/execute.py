"""Restricted, vectorized interpretation of canonical model semantics."""

import operator
import numpy as np

from .nodes import (
    ModelNode, Constant, Label, Feature, Negate, Binary, And, Conditional,
    Vector, Element, Index, Sum, Comprehension, Exp,
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

    def visit(item, rows, indices):
        kind = type(item)
        if kind in (Constant, Label):
            return np.full(len(rows), item.value)
        if kind is Feature:
            if item.index >= X.shape[1]:
                raise ValueError(f"Feature index {item.index} is out of bounds.")
            return X[rows, item.index]
        if kind is Index:
            return np.full(len(rows), indices[item.name], dtype=int)
        if kind is Sum:
            result = 0
            for index in range(item.size):
                result = result + visit(item.expression, rows, {**indices, item.index: index})
            return result
        if kind is Comprehension:
            return np.stack([
                visit(item.expression, rows, {**indices, item.index: index})
                for index in range(item.size)
            ], axis=1)
        if kind is Exp:
            return np.exp(visit(item.operand, rows, indices))
        if kind is Negate:
            return -visit(item.operand, rows, indices)
        if kind is Binary:
            return _OPERATIONS[item.operator](visit(item.left, rows, indices),
                                              visit(item.right, rows, indices))
        if kind is And:
            result = np.ones(len(rows), dtype=bool)
            for child in item.items:
                condition = np.asarray(visit(child, rows, indices), dtype=bool)
                if condition.ndim != 1:
                    raise ValueError("A condition must provide one value per observation.")
                result = np.logical_and(result, condition)
            return result
        if kind is Vector:
            if not item.items:
                return np.empty((len(rows), 0))
            return np.stack([visit(child, rows, indices) for child in item.items], axis=1)
        if kind is Conditional:
            condition = np.asarray(visit(item.condition, rows, indices), dtype=bool)
            if condition.ndim != 1:
                raise ValueError("A condition must provide one value per observation.")
            yes = visit(item.if_true, rows[condition], indices)
            no = visit(item.if_false, rows[~condition], indices)
            if yes.shape[1:] != no.shape[1:]:
                raise ValueError("Conditional branches must have matching output dimensions.")
            output = np.empty((len(rows), *yes.shape[1:]), dtype=np.result_type(yes, no))
            output[condition], output[~condition] = yes, no
            return output
        if kind is Element:
            index = indices[item.index.name] if type(item.index) is Index else item.index
            # Indexed construction evaluates only the requested element.
            if type(item.vector) is Vector:
                if index >= len(item.vector.items):
                    raise ValueError("Element index is out of bounds.")
                return visit(item.vector.items[index], rows, indices)
            if type(item.vector) is Comprehension:
                if index >= item.vector.size:
                    raise ValueError("Element index is out of bounds.")
                return visit(item.vector.expression, rows, {**indices, item.vector.index: index})
            value = visit(item.vector, rows, indices)
            if value.ndim < 2 or index >= value.shape[1]:
                raise ValueError("Element index is out of bounds.")
            return value[:, index]
        raise TypeError(f"Unsupported model node: {kind.__name__}")

    if not isinstance(node, ModelNode):
        raise TypeError("Expected a model expression.")
    return visit(node, np.arange(X.shape[0]), {})
