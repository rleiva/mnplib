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
class Binary(ModelNode):
    operator: Arithmetic
    left: ModelNode
    right: ModelNode


@dataclass(frozen=True)
class Conditional(ModelNode):
    condition: ModelNode
    if_true: ModelNode
    if_false: ModelNode


@dataclass(frozen=True)
class Vector(ModelNode):
    items: tuple[ModelNode, ...]


@dataclass(frozen=True)
class Element(ModelNode):
    vector: ModelNode
    index: int


@dataclass(frozen=True)
class Classify(ModelNode):
    """Choose the first maximal score and return its corresponding label."""
    scores: ModelNode
    labels: tuple[Label, ...]


@dataclass(frozen=True)
class Dense(ModelNode):
    """A feed-forward layer with input-by-output weights and fixed activation."""
    inputs: ModelNode
    weights: tuple[tuple[Constant, ...], ...]
    bias: tuple[Constant, ...]
    activation: str
