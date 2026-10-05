"""Numerically stable RMS for two nonnegative metric components."""

import math


def _rms_pair(first: float, second: float, *, weights=(0.5, 0.5)) -> float:
    """Combine two scalars using normalized, nonnegative weights.

    Both components must be finite, including zero-weighted
    dimensions. Scaling avoids overflow when squaring finite components.
    """
    if not (math.isfinite(first) and math.isfinite(second)):
        return float("nan")
    if first < 0 or second < 0:
        raise ValueError("RMS components must be nonnegative.")
    scale = max(first, second)
    if scale == 0:
        return 0.0
    return scale * math.sqrt(
        weights[0] * (first / scale)**2 + weights[1] * (second / scale)**2,
    )
