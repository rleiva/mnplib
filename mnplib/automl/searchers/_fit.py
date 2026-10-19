"""Fit candidates while preserving warnings unrelated to convergence."""

import warnings

from sklearn.exceptions import ConvergenceWarning


def fit_with_convergence_flag(model, X, y) -> bool:
    """Record convergence status and forward all other fitting warnings."""
    caught = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X, y)
    finally:
        for warning in caught:
            if not issubclass(warning.category, ConvergenceWarning):
                warnings.warn_explicit(
                    warning.message, warning.category, warning.filename, warning.lineno,
                )
    return not any(issubclass(warning.category, ConvergenceWarning) for warning in caught)
