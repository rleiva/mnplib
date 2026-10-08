"""Warnings at public boundaries for statistically weak empirical estimates."""

import sys
import warnings


def warn_unreliable_estimate(operation, diagnostics):
    """Warn once about an unreliable estimate, pointing to the external caller."""
    if diagnostics.get("is_reliable", True):
        return

    details = ", ".join(
        f"{name}={diagnostics[name]:.3g}"
        for name in (
            "n_samples", "n_observed_joint_states", "mean_joint_occupancy",
            "singleton_fraction", "resolved_n_bins",
        )
        if diagnostics.get(name) is not None
    )
    stacklevel = 2
    frame = sys._getframe(1)
    while frame is not None:
        module = frame.f_globals.get("__name__", "")
        if not module.startswith("mnplib."):
            break
        stacklevel += 1
        frame = frame.f_back
    del frame
    warnings.warn(
        f"{operation}(): the empirical estimate may be unreliable because the "
        f"joint empirical state space is sparsely populated ({details}). "
        "Inspect the analysis report for reliability diagnostics.",
        RuntimeWarning,
        stacklevel=stacklevel,
    )
