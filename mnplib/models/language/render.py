"""Render model AST nodes into the canonical text representation."""

import json

from .nodes import (
    Constant, Label, Feature, Negate, Binary, And,
    Conditional, Vector, Element, Index, Sum, Comprehension, Exp
)
from .normalize import normalize

# Higher values bind more tightly.
_PRECEDENCE = {
    "<": 3, "<=": 3, ">": 3, ">=": 3, "==": 3,
    "+": 4, "-": 4,
    "*": 5, "/": 5,
    "**": 7,
}


def format_number(value: float) -> str:
    """Return the canonical scientific-notation form of a finite number."""
    return format(0.0 if value == 0 else value, ".2e")


def _precedence(node):
    """Return the precedence level of an AST node."""
    if type(node) is Conditional:
        return 1

    if type(node) is And:
        return 2

    if type(node) is Binary:
        return _PRECEDENCE[node.operator]

    if type(node) is Negate or (
        type(node) is Constant and node.value < 0
    ):
        return 6

    return 8


def _child(node, minimum):
    """Render a child node, adding parentheses when required."""
    text = _render(node)

    if _precedence(node) < minimum:
        return "(" + text + ")"

    return text


def _render(node):
    """Render a normalized AST node recursively."""
    kind = type(node)

    if kind is Constant:
        return format_number(node.value)

    if kind is Label:
        encoded = json.dumps(
            node.value,
            ensure_ascii=True,
            allow_nan=False,
        )
        return "label(" + json.dumps(encoded, ensure_ascii=True) + ")"

    if kind is Feature:
        return f"x{node.index}"

    if kind is Index:
        return node.name

    if kind is Exp:
        return "exp(" + _render(node.operand) + ")"

    if kind in (Sum, Comprehension):
        body = _render(node.expression) + f" for {node.index} in range({node.size})"
        return "sum(" + body + ")" if kind is Sum else "[" + body + "]"

    if kind is Negate:
        return "-" + _child(node.operand, 6)

    if kind is Binary:
        precedence = _PRECEDENCE[node.operator]

        # Exponentiation is right-associative.
        if node.operator == "**":
            left = _child(node.left, precedence + 1)
            right = _child(node.right, 6)

        # Comparisons require both operands to bind more tightly.
        elif precedence == 3:
            left = _child(node.left, precedence + 1)
            right = _child(node.right, precedence + 1)

        else:
            left = _child(node.left, precedence)
            right = _child(node.right, precedence + 1)

        return left + node.operator + right

    if kind is And:
        return " and ".join(_child(item, 3) for item in node.items)

    if kind is Conditional:
        return (
            _child(node.if_true, 2)
            + " if "
            + _child(node.condition, 2)
            + " else "
            + _child(node.if_false, 1)
        )

    if kind is Vector:
        return "[" + ",".join(
            _render(item)
            for item in node.items
        ) + "]"

    if kind is Element:
        return (
            "at("
            + _render(node.vector)
            + ","
            + (_render(node.index) if type(node.index) is Index else str(node.index))
            + ")"
        )

    raise TypeError(
        f"Unsupported model node: {kind.__name__}"
    )


def render(node) -> str:
    """Normalize and render an AST, verifying the grammar round trip."""
    from .parse import parse

    canonical = normalize(node)
    text = _render(canonical)

    if parse(text) != canonical:
        raise ValueError(
            "Canonical model expression failed its grammar round trip."
        )

    return text
