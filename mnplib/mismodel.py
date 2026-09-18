"""Numerical aggregation of prediction inaccuracy and model surfeit."""

import math


def mismodel(*, inaccuracy: float, surfeit: float) -> float:
    """Return the root-mean-square of inaccuracy and surfeit.

    Nonfinite components produce NaN. Finite components must be nonnegative.
    This two-component score does not affect minimum-nescience selection.
    """
    values = (float(inaccuracy), float(surfeit))
    if not all(math.isfinite(value) for value in values):
        return float("nan")
    if any(value < 0 for value in values):
        raise ValueError("Mismodel components must be nonnegative.")
    return math.sqrt((values[0] ** 2 + values[1] ** 2) / 2.0)
