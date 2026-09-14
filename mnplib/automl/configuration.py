"""Validate per-family search options without changing constructor inputs."""

from collections.abc import Mapping
from inspect import signature


def validated_search_options(options, factories):
    """Return independent option dictionaries for known model families."""
    if options is None:
        options = {}
    if not isinstance(options, Mapping):
        raise ValueError("search_options must be a mapping keyed by model family.")
    unknown = set(options) - set(factories)
    if unknown:
        raise ValueError(f"Unknown search_options model families: {sorted(unknown)}.")
    result = {}
    for family, factory in factories.items():
        values = options.get(family, {})
        if not isinstance(values, Mapping):
            raise ValueError(f"search_options[{family!r}] must be a mapping.")
        allowed = (set(factory) if isinstance(factory, (set, frozenset))
                   else set(signature(factory).parameters) - {"estimator_cls"})
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unknown search options for {family}: {sorted(unknown)}.")
        result[family] = dict(values)
    return result
