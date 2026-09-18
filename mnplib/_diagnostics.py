"""Warnings for direct model evaluations with unavailable empirical metrics."""

import warnings


def warn_nan_model(method, diagnostics):
    """Explain a NaN model metric and direct callers to model_analysis()."""
    if not diagnostics["is_reliable"]:
        reason = (
            f"{diagnostics['failure_reason']} "
            f"(n_samples={diagnostics['n_samples']}, "
            f"n_selected_features={diagnostics['n_selected_features']}, "
            f"resolved_n_bins={diagnostics['resolved_n_bins']}, "
            f"mean_joint_occupancy={diagnostics['mean_joint_occupancy']:.3f}, "
            f"singleton_fraction={diagnostics['singleton_fraction']:.3f})."
        )
    else:
        reason = "One or more metric components or their aggregate are not finite."
    warnings.warn(
        f"{method}() returned NaN. {reason} "
        "Use model_analysis(model) to inspect the diagnostics.",
        RuntimeWarning,
        stacklevel=3,
    )
