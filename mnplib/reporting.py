"""Compact text summaries for metric and AutoML analysis dictionaries."""

from collections.abc import Mapping
from itertools import zip_longest
from math import isfinite, isnan
from textwrap import wrap


__all__ = ["format_analysis"]

_METRIC_FIELDS = {
    "nescience": (
        ("nescience", "Nescience", ".4f"),
        ("deficiency", "Deficiency", ".4f"),
        ("surplus", "Surplus", ".4f"),
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
        ("accuracy", "Accuracy", ".2%"),
        ("mae", "MAE", ".4f"),
        ("rmse", "RMSE", ".4f"),
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

_AUTOML_TASKS = {
    "classification": ("Auto-Classification Analysis", "Accuracy", ".2%"),
    "regression": ("Auto-Regression Analysis", "R-squared", ".4f"),
    "forecasting": ("Auto-Time-Series Analysis", "R-squared", ".4f"),
}

_AUTOML_CONTEXT_FIELDS = (
    ("candidate", "Selected candidate", None),
    ("family", "Model family", None),
    ("model_type", "Model type", None),
    ("n_samples", "Evaluated samples", "d"),
    ("resolved_n_bins", "Numeric bins", "d"),
    ("n_selected_features", "Selected count", "d"),
    ("window_size", "Lag window", "d"),
)

_EVALUATION_CONTEXTS = {
    "training": "Training data",
    "lagged_training": "Lagged training data",
}

_CODE_LENGTH_FIELDS = {
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


def format_analysis(report: Mapping[str, object]) -> str:
    """Return an aligned, plain-text summary of a metric or AutoML report.

    Accept a dictionary from ``model_analysis()`` on Miscoding, Inaccuracy,
    Surfeit, or Nescience, or from ``subset_analysis()``,
    ``prediction_analysis()``, or ``description_analysis()``. The metric is
    identified by its scalar keys. Also accept ``explain()`` dictionaries from
    NescienceClassifier, NescienceRegressor, and TimeSeries. Their task and
    candidate fields identify AutoML reports. Other fields are ignored.

    Scores use four decimal places, accuracy uses percentages, and code lengths
    use three decimal places in bits. Nonfinite values remain explicit. Subset
    reliability and any failure reason are always included when available.
    Feature lists show at most eight entries and text wraps at 78 characters.

    AutoML summaries include the selected candidate, supplied hyperparameters,
    and recorded estimator score with its evaluation context. Time-series
    summaries show the lag window and selected lag features. No model is fitted,
    scored, or inspected by the formatter, and no parameter defaults are added.

    The input is not modified, and nothing is printed. Use
    ``print(format_analysis(report))`` in a notebook, terminal, or log.

    Raises
    ------
    TypeError
        If report is not a mapping.
    ValueError
        If report has no recognized metric key or its primary score is None.
    """
    if not isinstance(report, Mapping):
        raise TypeError("report must be a metric analysis mapping.")
    metric = next((key for key in _METRIC_FIELDS if key in report), None)
    if metric is None:
        raise ValueError("report must contain nescience, miscoding, inaccuracy, or surfeit.")
    if report[metric] is None:
        raise ValueError(f"report must contain a numeric {metric} score, not None.")

    task = report.get("task")
    automl = metric == "nescience" and "candidate" in report and task in _AUTOML_TASKS
    context = _rows(report, _AUTOML_CONTEXT_FIELDS if automl else _CONTEXT_FIELDS)
    features = report.get("selected_feature_names")
    if features is None:
        features = report.get("selected_features")
    if automl and task == "forecasting" and report.get("selected_lags") is not None:
        context.append(("Selected lags", _feature_list(
            lag["feature_name"] for lag in report["selected_lags"])))
    elif features is not None:
        context.append(("Selected features", _feature_list(features)))
    if report.get("is_reliable") is not None:
        context.append(("Subset reliability", "Reliable" if report["is_reliable"] else "Unreliable"))
    reason = report.get("failure_reason")
    if reason is not None:
        context.append(("Failure reason", _text(reason)))
    elif report.get("is_reliable") is not None and not report["is_reliable"]:
        context.append(("Failure reason", "Not provided"))

    sections = [(None, context), (None, _rows(report, _METRIC_FIELDS[metric]))]
    if automl:
        evaluation = []
        if report.get("native_estimator_score") is not None:
            _, label, spec = _AUTOML_TASKS[task]
            source = report.get("evaluation_context")
            evaluation = [
                ("Data", _text(_EVALUATION_CONTEXTS.get(source, source or "Not specified"))),
                (label, _value(report["native_estimator_score"], spec)),
            ]
        sections.append(("Candidate evaluation", evaluation))
        parameters = report.get("hyperparameters") or {}
        sections.append(("Hyperparameters", [
            (_text(key), _text(value)) for key, value in sorted(parameters.items())
        ]))
    if metric in _CODE_LENGTH_FIELDS:
        sections.append(("Code lengths (bits)", _rows(report, _CODE_LENGTH_FIELDS[metric])))
    if metric == "nescience":
        aggregation = _rows(report, (("aggregation", "Method", None),))
        if report.get("weights") is not None:
            weights = report["weights"]
            value = ", ".join(
                f"{key}={_value(weights[key], '.4g')}"
                for key in ("deficiency", "surplus", "inaccuracy", "surfeit")
                if key in weights
            )
            aggregation.append(("Weights", value))
        sections.append(("Aggregation", aggregation))

    sections = [(heading, rows) for heading, rows in sections if rows]
    all_rows = [row for _, rows in sections for row in rows]
    label_width = min(30, max(len(label) for label, _ in all_rows))
    width = max(40, min(78, label_width + 2 + max(len(value) for _, value in all_rows)))
    value_width = width - label_width - 2
    title = _AUTOML_TASKS[task][0] if automl else f"{metric.title()} Analysis"
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


def _rows(report, fields):
    return [(label, _value(report[key], spec)) for key, label, spec in fields
            if key in report and report[key] is not None]


def _value(value, spec):
    if spec is None:
        return _text(value)
    if not isfinite(value):
        return "NaN" if isnan(value) else ("inf" if value > 0 else "-inf")
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
