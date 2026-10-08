"""Compact, plain-text presentation of structured analysis reports."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import zip_longest
from math import isfinite, isnan
from numbers import Real
from textwrap import wrap

import pandas as pd


__all__ = ["format_analysis"]

_METRIC_FIELDS = {
    "nescience": (
        ("nescience", "Nescience", ".4f"),
        ("deficiency", "Deficiency", ".4f"),
        ("surplus", "Surplus", ".4f"),
        ("inaccuracy", "Inaccuracy", ".4f"),
        ("surfeit", "Surfeit", ".4f"),
        ("miscoding", "Miscoding", ".4f"),
        ("mismodel", "Mismodel", ".4f"),
    ),
    "mismodel": (
        ("mismodel", "Mismodel", ".4f"),
        ("inaccuracy", "Inaccuracy", ".4f"),
        ("surfeit", "Surfeit", ".4f"),
    ),
    "miscoding": (
        ("deficiency", "Deficiency", ".4f"),
        ("surplus", "Surplus", ".4f"),
        ("miscoding", "Miscoding", ".4f"),
    ),
    "inaccuracy": (
        ("inaccuracy", "Inaccuracy", ".4f"),
    ),
    "surfeit": (
        ("surfeit", "Surfeit", ".4f"),
        ("compression_ratio", "Compression ratio", ".4f"),
        ("reference_source", "Reference source", None),
    ),
}

_CONTEXT_FIELDS = (
    ("n_samples", "Samples", "d"),
    ("y_type", "Target type", None),
    ("model_type", "Model type", None),
    ("resolved_n_bins", "Numeric bins", "d"),
    ("n_selected_features", "Selected count", "d"),
)

_JOINT_FIELDS = (
    ("n_observed_joint_states", "Observed states", "d"),
    ("mean_joint_occupancy", "Mean occupancy", ".4f"),
    ("n_singleton_joint_states", "Singleton states", "d"),
    ("singleton_fraction", "Singleton fraction", ".4f"),
)

_AUTOML_TASKS = {
    "classification": ("Auto-Classification Analysis", "Accuracy", ".2%"),
    "regression": ("Auto-Regression Analysis", "R-squared", ".4f"),
    "forecasting": ("Auto-Time-Series Analysis", "R-squared", ".4f"),
}

_CODE_LENGTH_FIELDS = {
    "miscoding": (
        ("code_length_bits", "K(X)", ".3f"),
        ("target_code_length_bits", "K(Y)", ".3f"),
        ("joint_code_length_bits", "K(X, Y)", ".3f"),
        ("target_conditional_code_length_bits", "K(Y | X)", ".3f"),
        ("feature_conditional_code_length_bits", "K(X | Y)", ".3f"),
    ),
    "inaccuracy": (
        ("target_code_length_bits", "Target", ".3f"),
        ("prediction_code_length_bits", "Predictions", ".3f"),
        ("joint_code_length_bits", "Joint", ".3f"),
    ),
    "surfeit": (
        ("model_code_length_bits", "Model", ".3f"),
        ("compressed_code_length_bits", "Compressed", ".3f"),
        ("effective_compressed_code_length_bits", "Effective compressed", ".3f"),
        ("target_code_length_bits", "Target", ".3f"),
        ("reference_code_length_bits", "Reference", ".3f"),
    ),
}

_MISCODING_DEBUG_CONTEXT = (
    ("target_n_bins", "Target bins", "d"),
    ("n_observed_feature_states", "Feature states", "d"),
    ("n_observed_target_states", "Target states", "d"),
)


def format_analysis(report: Mapping[str, object] | pd.DataFrame) -> str:
    """Return an aligned text summary without modifying or evaluating its input.

    Accept analysis dictionaries from Miscoding, Inaccuracy, Surfeit, Mismodel, Nescience,
    NescienceClassifier, NescienceRegressor, TimeSeries, and ResidualAnalysis.
    This includes subset, prediction, and description analysis dictionaries.
    Feature-analysis and lag-analysis DataFrames are also supported. Functional
    analysis helpers return the same supported report structures.

    Scores use four decimal places and code lengths use three decimal places
    in bits. Nonfinite values and subset failures are explicit. AutoML scores
    are labeled with their recorded evaluation context; classification scores
    use percentages.
    Only supplied hyperparameters are displayed; no model defaults are inferred.

    Dictionary summaries wrap at 78 characters and feature lists show at most
    eight entries. Tables show feature or lag identifiers and metric values
    for at most 20 rows, with long cells abbreviated.
    Model strings, prediction arrays, and unrecognized fields are omitted.
    Miscoding debug reports include empirical code lengths and bin/state counts.
    In these reports, X denotes the feature or selected feature subset.
    No models are fitted or scored, and nothing is printed. For display, use
    ``print(format_analysis(report))``.

    Raises
    ------
    TypeError
        If report is not a mapping or DataFrame.
    ValueError
        If the report structure is unsupported or a displayed numeric field
        is invalid. The primary metric score must be a real scalar, not None.
    """
    if isinstance(report, pd.DataFrame):
        return _format_table(report)
    if not isinstance(report, Mapping):
        raise TypeError("report must be an analysis mapping or DataFrame.")
    if report.get("report_kind") == "residuals":
        return _format_residuals(report)
    metric = next((key for key in _METRIC_FIELDS if key in report), None)
    if metric is None:
        raise ValueError("report must contain a recognized metric or residual analysis.")
    _value(report[metric], ".4f", key=metric)

    task = report.get("task")
    automl = metric == "nescience" and "candidate" in report and task in _AUTOML_TASKS
    title = _AUTOML_TASKS[task][0] if automl else f"{metric.title()} Analysis"
    context = _rows(report, (
        ("candidate", "Selected candidate", None),
        ("family", "Model family", None),
        ("window_size", "Lag window", "d"),
    )) if automl else []
    context.extend(_rows(report, _CONTEXT_FIELDS))
    if metric == "miscoding":
        context.extend(_rows(report, _MISCODING_DEBUG_CONTEXT))
    context.extend(_selected_features(report))
    context.extend(_reliability(report))
    sections = [(None, context), (None, _rows(report, _METRIC_FIELDS[metric]))]

    if automl:
        evaluation = []
        if report.get("native_estimator_score") is not None:
            source = report.get("evaluation_context")
            label = {"training": "Training data", "lagged_training": "Lagged training data"}.get(
                source, source or "Not specified",
            )
            _, score_label, spec = _AUTOML_TASKS[task]
            evaluation = [("Data", _text(label)), (score_label, _value(
                report["native_estimator_score"], spec, key="native_estimator_score",
            ))]
        sections.append(("Candidate evaluation", evaluation))
        parameters = report.get("hyperparameters")
        if parameters is not None:
            if not isinstance(parameters, Mapping):
                raise ValueError("hyperparameters must be a mapping.")
            sections.append(("Hyperparameters", [
                (_text(key), _text(value))
                for key, value in sorted(parameters.items(), key=lambda item: str(item[0]))
            ]))

    sections.append(("Joint distribution", _rows(report, _JOINT_FIELDS)))
    sections.append(("Code lengths (bits)", _rows(report, _CODE_LENGTH_FIELDS.get(metric, ()))))
    if metric == "nescience":
        sections.append(("RMS weighting", _rows(report, (("weight", "Miscoding weight", ".4g"),))))
    return _render(title, sections)


def _format_residuals(report):
    return _render("Residual Analysis", [
        (None, _rows(report, _CONTEXT_FIELDS + (
            ("task", "Task", None), ("scope", "Scope", None),
            ("inaccuracy", "Inaccuracy", ".4f"),
            ("inaccuracy_encoding", "Inaccuracy encoding", None),
            ("correction_encoding", "Correction encoding", None),
            ("n_anomalies", "Anomalies", "d"), ("anomaly_rate", "Anomaly rate", ".2%"),
            ("n_correction_patterns", "Correction patterns", "d"),
        ))),
        ("Empirical common-state code lengths (bits)", _rows(report, (
            ("target_code_length_bits", "Target", ".3f"),
            ("prediction_code_length_bits", "Predictions", ".3f"),
            ("joint_code_length_bits", "Joint", ".3f"),
            ("target_conditional_code_length_bits", "Target given predictions", ".3f"),
            ("prediction_conditional_code_length_bits", "Predictions given target", ".3f"),
        ))), ("Joint distribution", _rows(report, _JOINT_FIELDS)),
    ])


def _format_table(report):
    if {"lag", "feature_name", "miscoding"}.issubset(report.columns):
        title = "Lag Analysis"
        fields = (("lag", "Lag"), ("feature_name", "Feature"))
    elif {"feature_index", "feature_name", "miscoding"}.issubset(report.columns):
        title = "Feature Analysis"
        fields = (("feature_index", "Index"), ("feature_name", "Feature"),
                  ("code_length_bits", "Code (bits)"))
    else:
        raise ValueError("report must be a feature-analysis or lag-analysis DataFrame.")
    labels = {key: label for key, label in fields + (
        ("deficiency", "Deficiency"), ("surplus", "Surplus"), ("miscoding", "Miscoding"),
        ("is_reliable", "Reliable"), ("failure_reason", "Failure reason"),
    ) if key in report.columns}
    if "target_conditional_code_length_bits" in report.columns:
        debug_fields = _CODE_LENGTH_FIELDS["miscoding"] + _MISCODING_DEBUG_CONTEXT + (
            ("resolved_n_bins", "Numeric bins", "d"),
            ("n_samples", "Samples", "d"),
            ("n_observed_joint_states", "Joint states", "d"),
            ("mean_joint_occupancy", "Joint occupancy", ".4f"),
            ("n_singleton_joint_states", "Joint singletons", "d"),
            ("singleton_fraction", "Singleton fraction", ".4f"),
            ("is_reliable", "Reliable", None),
            ("failure_reason", "Failure reason", None),
        )
        labels.update({key: label for key, label, _ in debug_fields if key in report.columns})
    table = report.loc[:, list(labels)].rename(columns=labels)
    formatters = {
        label: lambda value, key=key: _value(value, ".3f", key=key)
        for key, label in labels.items() if key.endswith("code_length_bits")
    }
    body = table.to_string(index=False, max_rows=20, max_colwidth=24, line_width=78,
                          formatters=formatters,
                          float_format=lambda value: _value(value, ".4f", key="table value"))
    units = "Code lengths in bits.\n" if "target_conditional_code_length_bits" in report.columns else ""
    return f"{title}\n{'=' * 78}\n{units}{body}\n{'=' * 78}"


def _selected_features(report):
    if report.get("selected_lags") is not None:
        return [("Selected lags", _feature_list(lag["feature_name"] for lag in report["selected_lags"]))]
    features = report.get("selected_feature_names")
    if features is None:
        features = report.get("selected_features")
    return [] if features is None else [("Selected features", _feature_list(features))]


def _reliability(report):
    rows = []
    reliable = report.get("is_reliable")
    if reliable is not None:
        rows.append(("Subset reliability", "Reliable" if reliable else "Unreliable"))
    if report.get("failure_reason") is not None:
        rows.append(("Failure reason", _text(report["failure_reason"])))
    elif reliable is not None and not reliable:
        rows.append(("Failure reason", "Not provided"))
    return rows


def _rows(report, fields):
    return [(label, _value(report[key], spec, key=key)) for key, label, spec in fields
            if key in report and report[key] is not None]


def _value(value, spec, *, key):
    if spec is None:
        return _text(value)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"report field {key!r} must be a real scalar.")
    if not isfinite(value):
        return "NaN" if isnan(value) else ("inf" if value > 0 else "-inf")
    if spec == "d":
        if value != int(value):
            raise ValueError(f"report field {key!r} must be an integer count.")
        return str(int(value))
    return format(value, spec)


def _text(value):
    return " ".join(str(value).split())


def _feature_list(features):
    values = list(features)
    if not values:
        return "(none)"
    text = ", ".join(_text(value) for value in values[:8])
    if len(values) > 8:
        text += f", ... (+{len(values) - 8} more)"
    return text


def _render(title, sections):
    sections = [(heading, rows) for heading, rows in sections if rows]
    all_rows = [row for _, rows in sections for row in rows]
    label_width = min(30, max(len(label) for label, _ in all_rows))
    width = max(40, min(78, label_width + 2 + max(len(value) for _, value in all_rows)))
    value_width = width - label_width - 2
    lines = [title, "=" * width]
    for index, (heading, rows) in enumerate(sections):
        if index:
            lines.append("")
        if heading:
            lines.extend((heading, "-" * width))
        for label, value in rows:
            labels = wrap(label, width=label_width, break_on_hyphens=False) or [""]
            parts = wrap(value, width=value_width, break_on_hyphens=False) or [""]
            for name, part in zip_longest(labels, parts, fillvalue=""):
                lines.append(f"{name:<{label_width}}  {part:>{value_width}}")
    lines.append("=" * width)
    return "\n".join(lines)
