"""Quantization and bounded, bottom-up local expression simplification."""

from decimal import Decimal, localcontext
import math
from numbers import Integral, Real

from .nodes import (
    ModelNode, Constant, Label, Feature, Negate, Binary, Conditional,
    Vector, Element, Classify, Dense,
)

OPERATORS = frozenset(("+", "-", "*", "/", "**", "<", "<=", ">", ">=", "=="))
ACTIVATIONS = frozenset(("identity", "relu", "logistic", "tanh"))


def quantize(value):
    """Round a finite parameter to three significant decimal digits."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("Model parameters must be real numbers.")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Model parameters must be finite.")
    if value == 0:
        return 0.0
    with localcontext() as context:
        context.prec = 3
        rounded = float(+Decimal.from_float(value))
    if not math.isfinite(rounded):
        raise ValueError("Rounded model parameter exceeds the finite numeric range.")
    return rounded


def _index(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError("Feature and element indices must be non-negative integers.")
    return int(value)


def normalize(node: ModelNode) -> ModelNode:
    """Normalize children once, then apply one immediate local rewrite."""
    kind = type(node)
    if kind is Constant:
        return Constant(quantize(node.value))
    if kind is Label:
        value = node.value.item() if hasattr(node.value, "item") else node.value
        if type(value) not in (str, int, float, bool):
            raise TypeError("Class labels must be JSON scalar values.")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Class labels must be finite.")
        return Label(value)
    if kind is Feature:
        if node.name is not None and not isinstance(node.name, str):
            raise TypeError("Feature names must be strings.")
        return Feature(_index(node.index), node.name)
    if kind is Negate:
        operand = normalize(node.operand)
        if type(operand) is Negate:
            return operand.operand
        if type(operand) is Constant:
            return Constant(quantize(-operand.value))
        return Negate(operand)
    if kind is Binary:
        if node.operator not in OPERATORS:
            raise ValueError(f"Unsupported operator: {node.operator}")
        left, right = normalize(node.left), normalize(node.right)
        lv = left.value if type(left) is Constant else None
        rv = right.value if type(right) is Constant else None
        if node.operator == "*":
            if lv == 0 or rv == 0:
                return Constant(0.0)
            if lv == 1:
                return right
            if rv == 1:
                return left
        if node.operator == "+":
            if lv == 0:
                return right
            if rv == 0:
                return left
            if rv is not None and rv < 0:
                return Binary("-", left, Constant(-rv))
        if node.operator == "-" and rv == 0:
            return left
        return Binary(node.operator, left, right)
    if kind is Conditional:
        return Conditional(*(normalize(x) for x in
                             (node.condition, node.if_true, node.if_false)))
    if kind is Vector:
        return Vector(tuple(normalize(x) for x in node.items))
    if kind is Element:
        return Element(normalize(node.vector), _index(node.index))
    if kind is Classify:
        labels = tuple(normalize(x) for x in node.labels)
        if not labels or any(type(x) is not Label for x in labels):
            raise ValueError("Classify requires categorical labels.")
        return Classify(normalize(node.scores), labels)
    if kind is Dense:
        if node.activation not in ACTIVATIONS:
            raise ValueError(f"Unsupported activation: {node.activation}")
        weights = tuple(tuple(normalize(x) for x in row) for row in node.weights)
        bias = tuple(normalize(x) for x in node.bias)
        if not bias or not weights or any(len(row) != len(bias) for row in weights):
            raise ValueError("Dense weights must be rectangular and match the bias.")
        if any(type(x) is not Constant for row in weights for x in row):
            raise TypeError("Dense weights must be constants.")
        if any(type(x) is not Constant for x in bias):
            raise TypeError("Dense biases must be constants.")
        return Dense(normalize(node.inputs), weights, bias, node.activation)
    raise TypeError(f"Unsupported model node: {kind.__name__}")
