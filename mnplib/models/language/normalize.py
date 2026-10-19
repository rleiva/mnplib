"""Quantization and bounded, bottom-up local expression simplification."""

from decimal import Decimal, localcontext
from keyword import iskeyword
import math
from numbers import Integral, Real
import re

from .nodes import (
    ModelNode, Constant, Label, Feature, Negate, Binary, And, Conditional,
    Vector, Element, Index, Sum, Comprehension,
)

OPERATORS = frozenset(("+", "-", "*", "/", "**", "<", "<=", ">", ">=", "=="))


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


def _index_name(name):
    if (not isinstance(name, str) or re.fullmatch(r"[a-z][a-z0-9_]*", name) is None
            or iskeyword(name) or name in {"sum", "range", "at", "label"}
            or re.fullmatch(r"x[0-9]+", name)):
        raise ValueError("Iteration indices require a non-reserved lowercase identifier.")
    return name


def normalize(node: ModelNode) -> ModelNode:
    """Validate lexical scope and apply bounded, bottom-up local rewrites."""
    return _normalize(node, frozenset())


def _normalize(node, scope):
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
    if kind is Index:
        name = _index_name(node.name)
        if name not in scope:
            raise ValueError(f"Unbound iteration index: {name}.")
        return Index(name)
    if kind in (Sum, Comprehension):
        name = _index_name(node.index)
        if name in scope:
            raise ValueError(f"Iteration index shadowing is not allowed: {name}.")
        if isinstance(node.size, bool) or not isinstance(node.size, Integral) or node.size < 0:
            raise ValueError("Iteration size must be a non-negative integer.")
        expression = _normalize(node.expression, scope | {name})
        if node.size == 0:
            return Constant(0.0) if kind is Sum else Vector(())
        return kind(name, int(node.size), expression)
    if kind is Negate:
        operand = _normalize(node.operand, scope)
        if type(operand) is Negate:
            return operand.operand
        if type(operand) is Constant:
            return Constant(quantize(-operand.value))
        return Negate(operand)
    if kind is Binary:
        if node.operator not in OPERATORS:
            raise ValueError(f"Unsupported operator: {node.operator}")
        left, right = _normalize(node.left, scope), _normalize(node.right, scope)
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
    if kind is And:
        items = []
        for child in node.items:
            child = _normalize(child, scope)
            items.extend(child.items if type(child) is And else (child,))
        if not items:
            raise ValueError("And requires at least one condition.")
        return items[0] if len(items) == 1 else And(tuple(items))
    if kind is Conditional:
        return Conditional(*(_normalize(x, scope) for x in
                             (node.condition, node.if_true, node.if_false)))
    if kind is Vector:
        return Vector(tuple(_normalize(x, scope) for x in node.items))
    if kind is Element:
        index = _normalize(node.index, scope) if type(node.index) is Index else _index(node.index)
        return Element(_normalize(node.vector, scope), index)
    raise TypeError(f"Unsupported model node: {kind.__name__}")
