"""Structural input substitution for fitted preprocessing."""

from dataclasses import fields, replace
from .nodes import ModelNode, Feature


def substitute_features(node, replacements):
    """Replace feature leaves without altering arithmetic evaluation order."""
    if type(node) is Feature:
        return replacements.get(node.index, node)
    def transform(value):
        if isinstance(value, ModelNode):
            return substitute_features(value, replacements)
        if isinstance(value, tuple):
            return tuple(transform(item) for item in value)
        return value
    return replace(node, **{field.name: transform(getattr(node, field.name))
                            for field in fields(node)})
