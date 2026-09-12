"""
Result-table helpers for time-series forecasting candidates.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from mnplib.automl.results import CandidateResult


@dataclass(frozen=True)
class TimeSeriesCandidateResult(CandidateResult):
    """
    Shared candidate result with forecasting metadata.
    """

    metadata: dict[str, Any] = field(default_factory=dict)


def candidate_results_dataframe(results: Sequence[CandidateResult]) -> pd.DataFrame:
    """
    Convert candidate results into a stable, sorted DataFrame.
    """
    rows = [candidate_result_row(result) for result in results]
    return pd.DataFrame(rows).sort_values(
        ["is_reliable", "nescience"],
        ascending=[False, True],
        ignore_index=True,
        na_position="last",
    )


def candidate_result_row(result: CandidateResult) -> dict[str, object]:
    """
    Convert one shared candidate result into a time-series diagnostics row.
    """
    metadata = dict(result.metadata)
    selected_indices = tuple(int(index) for index in result.artifacts.subset)
    selected_names = tuple(
        metadata.get(
            "selected_feature_names",
            tuple(f"X{index}" for index in selected_indices),
        )
    )
    components = dict(result.components)

    row = {
        "model_name": result.name,
        "model_family": result.family,
        "model_type": result.artifacts.model_type,
        "window_size": metadata.get("window_size"),
        "nescience": float(result.nescience),
        "estimator_score": result.estimator_score,
        "n_selected_features": result.n_selected_features,
        "description_length": len(result.artifacts.model_string.encode("utf-8")),
        "selected_feature_indices": selected_indices,
        "selected_feature_names": selected_names,
        "miscoding": subset_miscoding(components),
        "is_reliable": result.is_reliable,
        "failure_reason": result.subset_diagnostics.get("failure_reason"),
        "n_samples": result.subset_diagnostics.get("n_samples"),
        "n_observed_joint_states": result.subset_diagnostics.get("n_observed_joint_states"),
        "mean_joint_occupancy": result.subset_diagnostics.get("mean_joint_occupancy"),
        "n_singleton_joint_states": result.subset_diagnostics.get("n_singleton_joint_states"),
        "singleton_fraction": result.subset_diagnostics.get("singleton_fraction"),
    }
    row.update(components)
    return row


def subset_miscoding(components: dict[str, float]) -> float:
    """
    Return subset miscoding as the maximum of deficiency and surplus.
    """
    deficiency = float(components["deficiency"])
    surplus = float(components["surplus"])
    if not (np.isfinite(deficiency) and np.isfinite(surplus)):
        return float("nan")
    return max(deficiency, surplus)
