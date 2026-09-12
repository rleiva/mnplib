"""
Canonical serializers for time-series forecasting candidates.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from sklearn.utils.validation import check_is_fitted

from ..artifacts import ModelArtifacts


TIME_SERIES_SCHEMA = "canonical_nescience_time_series_model_v1"


def time_series_model_artifacts(
    *,
    model,
    subset,
    predictions,
    model_string: str,
    model_type: str | None = None,
) -> ModelArtifacts:
    """
    Build explicit nescience artifacts for a forecasting candidate.
    """
    return ModelArtifacts(
        subset=resolve_subset_indices(subset),
        predictions=np.asarray(predictions, dtype=float).ravel(),
        model_string=str(model_string),
        model_type=type(model).__name__ if model_type is None else str(model_type),
    )


def resolve_subset_indices(subset) -> list[int]:
    """
    Return sorted selected feature indices from a mask or index sequence.
    """
    values = np.asarray(subset)
    if values.dtype == bool:
        return [int(index) for index in np.flatnonzero(values)]
    return [int(index) for index in values.ravel()]


def canonical_linear_model_string(
    *,
    model,
    model_name: str,
    feature_names: Sequence[str],
    precision: int = 6,
) -> str:
    """
    Serialize a fitted linear autoregressive model.
    """
    check_is_fitted(model)
    coefficients = np.asarray(model.coef_, dtype=float).ravel()
    intercept = float(np.asarray(model.intercept_).ravel()[0])
    return canonical_weighted_model_string(
        model_type="autoregressive_linear",
        model_name=model_name,
        feature_names=feature_names,
        weights=coefficients,
        intercept=intercept,
        precision=precision,
        learned=True,
    )


def canonical_fixed_model_string(
    *,
    model_type: str,
    model_name: str,
    feature_names: Sequence[str],
    weights,
    intercept: float = 0.0,
    precision: int = 6,
) -> str:
    """
    Serialize a fixed-coefficient forecasting model.
    """
    return canonical_weighted_model_string(
        model_type=model_type,
        model_name=model_name,
        feature_names=feature_names,
        weights=np.asarray(weights, dtype=float),
        intercept=float(intercept),
        precision=precision,
        learned=False,
    )


def canonical_weighted_model_string(
    *,
    model_type: str,
    model_name: str,
    feature_names: Sequence[str],
    weights,
    intercept: float,
    precision: int,
    learned: bool,
) -> str:
    """
    Serialize a weighted one-step forecasting rule.
    """
    names = [str(name) for name in feature_names]
    weights = np.asarray(weights, dtype=float).ravel()
    if len(names) != len(weights):
        raise ValueError("feature_names and weights must have the same length.")

    lines = [
        f"SCHEMA {TIME_SERIES_SCHEMA}",
        f"MODEL {model_type}",
        "TASK forecasting",
        f"NAME {model_name}",
        f"INPUTS {', '.join(names) if names else '<none>'}",
        "PARAMETERS",
        f"    n_features = {len(names)}",
        f"    learned_coefficients = {str(bool(learned)).lower()}",
        "RULE",
        f"    y_hat = {format_number(intercept, precision)}",
    ]

    for weight, name in zip(weights, names):
        lines.append(f"    y_hat += {format_number(float(weight), precision)} * {name}")

    lines.append("    return y_hat")
    return "\n".join(lines) + "\n"


def canonical_arima_model_string(
    *,
    result,
    model_name: str,
    order: tuple[int, int, int],
    trend: str,
    precision: int = 6,
) -> str:
    """
    Serialize a fitted statsmodels ARIMA/SARIMAX result.
    """
    model = result.model
    lines = [
        f"SCHEMA {TIME_SERIES_SCHEMA}",
        "MODEL arima",
        "TASK forecasting",
        f"NAME {model_name}",
        "PARAMETERS",
        f"    order = {format_tuple(order)}",
        f"    trend = {str(trend)}",
        f"    k_states = {int(getattr(model, 'k_states', 0))}",
    ]
    _append_information_criteria(lines, result, precision)
    _append_parameters(lines, result, precision)
    lines.extend(
        [
            "RULE",
            "    estimate state-space representation with maximum likelihood",
            "    return one_step_ahead_prediction",
        ]
    )
    return "\n".join(lines) + "\n"


def canonical_state_space_model_string(
    *,
    result,
    model_name: str,
    specification: str,
    precision: int = 6,
) -> str:
    """
    Serialize a fitted statsmodels state-space result.
    """
    model = result.model
    state_names = [str(name) for name in getattr(model, "state_names", [])]
    lines = [
        f"SCHEMA {TIME_SERIES_SCHEMA}",
        "MODEL state_space",
        "TASK forecasting",
        f"NAME {model_name}",
        "PARAMETERS",
        f"    specification = {str(specification)}",
        f"    k_states = {int(getattr(model, 'k_states', 0))}",
        f"    states = {', '.join(state_names) if state_names else '<unknown>'}",
    ]
    _append_information_criteria(lines, result, precision)
    _append_parameters(lines, result, precision)
    lines.extend(
        [
            "RULE",
            "    filter latent state with Kalman recursion",
            "    return one_step_ahead_prediction",
        ]
    )
    return "\n".join(lines) + "\n"


def _append_information_criteria(lines: list[str], result, precision: int) -> None:
    for name in ("llf", "aic", "bic"):
        value = getattr(result, name, None)
        if value is not None and np.isfinite(float(value)):
            lines.append(f"    {name} = {format_number(float(value), precision)}")


def _append_parameters(lines: list[str], result, precision: int) -> None:
    pairs = result_parameter_pairs(result)
    lines.append("COEFFICIENTS")
    if not pairs:
        lines.append("    <none>")
        return
    for name, value in pairs:
        lines.append(f"    {name} = {format_number(value, precision)}")


def result_parameter_pairs(result) -> list[tuple[str, float]]:
    """
    Return named finite parameters from a statsmodels result object.
    """
    params = getattr(result, "params", [])
    if hasattr(params, "items"):
        raw_pairs = list(params.items())
    else:
        values = np.asarray(params, dtype=float).ravel()
        names = getattr(result, "param_names", None)
        if names is None:
            names = getattr(getattr(result, "model", None), "param_names", None)
        if names is None or len(names) != len(values):
            names = [f"theta_{index}" for index in range(len(values))]
        raw_pairs = list(zip(names, values))

    pairs: list[tuple[str, float]] = []
    for name, value in raw_pairs:
        value = float(value)
        if np.isfinite(value):
            pairs.append((str(name), value))
    return pairs


def format_tuple(values) -> str:
    """
    Format integer tuples used in statistical model specifications.
    """
    return "(" + ", ".join(str(int(value)) for value in values) + ")"


def format_number(value: float, precision: int) -> str:
    """
    Format finite numbers in canonical model descriptions.
    """
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("Model descriptions require finite numeric values.")
    rounded = f"{value:.{int(precision)}f}"
    if "." in rounded:
        rounded = rounded.rstrip("0").rstrip(".")
    return rounded if "." in rounded else f"{rounded}.0"
