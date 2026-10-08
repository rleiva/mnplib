"""The sole canonical text renderer for model descriptions."""

import json
from .nodes import (
    Constant, Label, Feature, Negate, Binary, Conditional,
    Vector, Element, Classify, Dense,
)
from .normalize import normalize

_PRECEDENCE = {"<": 2, "<=": 2, ">": 2, ">=": 2, "==": 2,
               "+": 3, "-": 3, "*": 4, "/": 4, "**": 6}


def format_number(value: float) -> str:
    """Format a canonical finite parameter with three significant digits."""
    return format(0.0 if value == 0 else value, ".2e")


def _precedence(node):
    if type(node) is Conditional:
        return 1
    if type(node) is Binary:
        return _PRECEDENCE[node.operator]
    if type(node) is Negate or (type(node) is Constant and node.value < 0):
        return 5
    return 7


def _child(node, minimum):
    text = _render(node)
    return "(" + text + ")" if _precedence(node) < minimum else text


def _render(node):
    kind = type(node)
    if kind is Constant:
        return format_number(node.value)
    if kind is Label:
        encoded = json.dumps(node.value, ensure_ascii=True, allow_nan=False)
        return "label(" + json.dumps(encoded, ensure_ascii=True) + ")"
    if kind is Feature:
        return f"x{node.index}"
    if kind is Negate:
        return "-" + _child(node.operand, 5)
    if kind is Binary:
        p = _PRECEDENCE[node.operator]
        if node.operator == "**":
            left, right = _child(node.left, p + 1), _child(node.right, 5)
        elif p == 2:
            left, right = _child(node.left, p + 1), _child(node.right, p + 1)
        else:
            left, right = _child(node.left, p), _child(node.right, p + 1)
        return left + node.operator + right
    if kind is Conditional:
        return (_child(node.if_true, 2) + " if " + _child(node.condition, 2)
                + " else " + _child(node.if_false, 1))
    if kind is Vector:
        return "[" + ",".join(_render(x) for x in node.items) + "]"
    if kind is Element:
        return "at(" + _render(node.vector) + "," + str(node.index) + ")"
    if kind is Classify:
        return "classify(" + _render(node.scores) + "," + _render(Vector(node.labels)) + ")"
    if kind is Dense:
        weights = Vector(tuple(Vector(row) for row in node.weights))
        return ("dense(" + _render(node.inputs) + "," + _render(weights) + ","
                + _render(Vector(node.bias)) + "," + json.dumps(node.activation) + ")")
    raise TypeError(f"Unsupported model node: {kind.__name__}")


def render(node) -> str:
    """Normalize and render an expression, verifying its grammar round trip."""
    from .parse import parse
    canonical = normalize(node)
    text = _render(canonical)
    if parse(text) != canonical:
        raise ValueError("Canonical model expression failed its grammar round trip.")
    return text
