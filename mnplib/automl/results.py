"""
Structured results produced by AutoML model-family searchers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


def candidate_result_row(result, feature_names=None):
    """Build the common candidate report with effective feature diagnostics."""
    features = list(result.artifacts.subset)
    metadata = dict(getattr(result, "metadata", {}))
    names = (list(metadata.get("selected_feature_names", [])) if feature_names is None
             else [str(feature_names[index]) for index in features])
    row = {
        "candidate": result.name,
        "family": result.family,
        "model_type": result.artifacts.model_type,
        "hyperparameters": dict(result.hyperparameters),
        "nescience": float(result.nescience),
        **result.components,
        "native_estimator_score": result.estimator_score,
        "selected_features": features,
        "selected_feature_names": names,
        "n_selected_features": len(features),
        "description_length": len(result.artifacts.model_string.encode("utf-8")),
        "is_reliable": result.is_reliable,
    }
    for key in ("failure_reason", "resolved_n_bins", "n_samples", "n_observed_joint_states",
                "mean_joint_occupancy", "n_singleton_joint_states", "singleton_fraction"):
        row[key] = result.subset_diagnostics.get(key)
    extra = {key: value for key, value in metadata.items()
             if key not in row and key not in {"selected_features", "family", "model_name"}}
    if extra:
        row["metadata"] = extra
    return row


def candidate_results_dataframe(results, feature_names=None):
    """Return a stable comparison table, with unreliable candidates last."""
    return pd.DataFrame([candidate_result_row(result, feature_names) for result in results]).sort_values(
        ["is_reliable", "nescience"], ascending=[False, True], na_position="last",
        kind="stable", ignore_index=True)


@dataclass(frozen=True)
class CandidateResult:
    """
    Result from evaluating one fitted candidate through explicit artifacts.
    """

    name       : str
    family     : str
    model      : object
    nescience  : float
    components : dict[str, float]
    artifacts  : object
    estimator_score : float = float("nan")
    n_selected_features : int | None = None
    hyperparameters : dict[str, Any] = field(default_factory=dict)
    subset_diagnostics : dict[str, Any] = field(default_factory=dict)

    @property
    def is_reliable(self) -> bool:
        """
        Return whether the candidate subset diagnostics are reliable.
        """
        return bool(self.subset_diagnostics.get("is_reliable", True))

    @property
    def estimator(self):
        """
        Convenience alias.
        """
        return self.model

@dataclass(frozen=True)
class SearchReport:
    """
    Results and diagnostics returned by one model-family searcher.
    """

    family      : str
    results     : list[CandidateResult]
    diagnostics : list[dict[str, Any]] = field(default_factory=list)
