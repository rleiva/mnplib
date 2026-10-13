"""Parse only the declared expression grammar; no Python execution is used."""

from functools import lru_cache
from importlib.resources import files
import json
from lark import Lark, Transformer, UnexpectedInput
from lark.exceptions import VisitError

from .nodes import (
    Constant, Label, Feature, Negate, Binary, And, Conditional,
    Vector, Element, Index, Sum, Comprehension, Exp,
)
from .normalize import normalize


@lru_cache(maxsize=1)
def _grammar():
    return Lark(files(__package__).joinpath("model.lark").read_text(encoding="utf-8"),
                parser="lalr", maybe_placeholders=False)


class _Build(Transformer):
    def number(self, values):
        return Constant(float(values[0]))

    def feature(self, values):
        return Feature(int(str(values[0])[1:]))

    def index(self, values):
        return Index(str(values[0]))

    def literal_index(self, values):
        return int(values[0])

    def bounded_sum(self, values):
        return Sum(str(values[1]), int(values[2]), values[0])

    def comprehension(self, values):
        return Comprehension(str(values[1]), int(values[2]), values[0])

    def exponential(self, values):
        return Exp(values[0])

    def negate(self, values):
        return Negate(values[0])

    def binary(self, values):
        return Binary(str(values[1]), values[0], values[2])

    def conditional(self, values):
        return Conditional(values[1], values[0], values[2])

    def conjunction(self, values):
        return And(tuple(values))

    def vector(self, values):
        return Vector(tuple(values))

    def label(self, values):
        return Label(json.loads(json.loads(values[0])))

    def element(self, values):
        return Element(values[0], values[1])


def parse(text: str):
    """Return a normalized AST from a restricted model expression."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("A non-empty model expression is required.")
    try:
        return normalize(_Build().transform(_grammar().parse(text)))
    except (UnexpectedInput, VisitError, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid model expression: {exc}") from exc
