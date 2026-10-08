"""Parse only the declared expression grammar; no Python execution is used."""

from functools import lru_cache
from importlib.resources import files
import json
from lark import Lark, Transformer, UnexpectedInput
from lark.exceptions import VisitError

from .nodes import (
    Constant, Label, Feature, Negate, Binary, Conditional,
    Vector, Element, Classify, Dense,
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

    def negate(self, values):
        return Negate(values[0])

    def binary(self, values):
        return Binary(str(values[1]), values[0], values[2])

    def conditional(self, values):
        return Conditional(values[1], values[0], values[2])

    def vector(self, values):
        return Vector(tuple(values))

    def label(self, values):
        return Label(json.loads(json.loads(values[0])))

    def element(self, values):
        index = float(values[1])
        if not index.is_integer():
            raise ValueError("Element index must be an integer.")
        return Element(values[0], int(index))

    def classify(self, values):
        return Classify(values[0], values[1].items)

    def dense(self, values):
        if any(type(row) is not Vector for row in values[1].items):
            raise ValueError("Dense weights must be a matrix.")
        return Dense(values[0], tuple(row.items for row in values[1].items),
                     values[2].items, json.loads(values[3]))


def parse(text: str):
    """Return a normalized AST from a restricted model expression."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("A non-empty model expression is required.")
    try:
        return normalize(_Build().transform(_grammar().parse(text)))
    except (UnexpectedInput, VisitError, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid model expression: {exc}") from exc
