"""Immutable semantic nodes for the restricted prediction language."""

from dataclasses import dataclass, field
from typing import Literal

Arithmetic = Literal["+", "-", "*", "/", "**", "<", "<=", ">", ">=", "=="]


class ModelNode:
    """Base type for computational expressions."""


@dataclass(frozen=True)
class Constant(ModelNode):
    value: float


@dataclass(frozen=True)
class Label(ModelNode):
    """A categorical value, preserved exactly rather than numerically rounded."""
    value: str | int | float | bool


@dataclass(frozen=True)
class Feature(ModelNode):
    index: int
    name: str | None = field(default=None, compare=False)


@dataclass(frozen=True)
class Negate(ModelNode):
    operand: ModelNode


@dataclass(frozen=True)
class Exp(ModelNode):
    operand: ModelNode


@dataclass(frozen=True)
class Binary(ModelNode):
    operator: Arithmetic
    left: ModelNode
    right: ModelNode


@dataclass(frozen=True)
class And(ModelNode):
    items: tuple[ModelNode, ...]


@dataclass(frozen=True)
class Conditional(ModelNode):
    condition: ModelNode
    if_true: ModelNode
    if_false: ModelNode


@dataclass(frozen=True)
class Vector(ModelNode):
    items: tuple[ModelNode, ...]


@dataclass(frozen=True)
class Index(ModelNode):
    name: str


@dataclass(frozen=True)
class Element(ModelNode):
    vector: ModelNode
    index: int | Index


@dataclass(frozen=True)
class Sum(ModelNode):
    index: str
    size: int
    expression: ModelNode


@dataclass(frozen=True)
class Comprehension(ModelNode):
    index: str
    size: int
    expression: ModelNode
