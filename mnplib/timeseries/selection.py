"""
Candidate evaluation for time-series forecasting.

This module converts fitted forecasting candidates into the explicit artifacts
required by the latest mnplib metric API:

    * ``subset``: selected lagged features;
    * ``predictions``: one-step predictions on the lagged representation;
    * ``model_string``: canonical description of the forecasting rule.

The evaluator deliberately does not inspect the internal structure of the
nescience estimator. It computes the four components through the dedicated
metric objects and uses the configured aggregator only to combine those values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .models import TimeSeriesCandidateSpec


@dataclass(frozen=True)
class TimeSeriesCandidateResult:
    """
    Nescience evaluation result for one fitted forecasting candidate.

    Result construction is explicit and typed. Values displayed in result
    tables are derived from this object in one place.
    """

    model_name: str
    model_family: str
    model: object
    subset: np.ndarray
    window_size: int
    nescience: float
    components: dict[str, float]
    estimator_score: float
    selected_feature_indices: tuple[int, ...]
    selected_feature_names: tuple[str, ...]
    model_string: str
    predictions: np.ndarray
    metadata: dict[str, Any]
    subset_diagnostics: dict[str, Any]

    @property
    def is_reliable(self) -> bool:
        """Return whether the subset diagnostics are reliable."""
        return bool(self.subset_diagnostics.get("is_reliable", True))

    @property
    def miscoding(self) -> float:
        """Return subset miscoding as ``max(deficiency, surplus)``."""
        deficiency = float(self.components["deficiency"])
        surplus = float(self.components["surplus"])
        if not (np.isfinite(deficiency) and np.isfinite(surplus)):
            return float("nan")
        return max(deficiency, surplus)

    @property
    def n_selected_features(self) -> int:
        """Return the number of selected lagged features."""
        return len(self.selected_feature_indices)

    @property
    def description_length(self) -> int:
        """Return the UTF-8 byte length of the canonical model string."""
        return len(self.model_string.encode("utf-8"))


class TimeSeriesCandidateEvaluator:
    """
    Evaluate fitted time-series candidates with the latest explicit metric API.

    Parameters
    ----------
    miscoding, inaccuracy, surfeit : fitted metric objects
        Component estimators from the current mnplib API.

    aggregator : object
        Object exposing ``aggregate_components(**components)``. In practice this
        is a configured ``Nescience`` instance. The evaluator does not call
        ``fit`` or ``components`` on it.

    X, y : numpy.ndarray
        Full lagged representation used for candidate fitting and evaluation.
    """

    def __init__(
        self,
        *,
        miscoding,
        inaccuracy,
        surfeit,
        aggregator,
        X: np.ndarray,
        y: np.ndarray,
        feature_names,
    ):
        self.miscoding = miscoding
        self.inaccuracy = inaccuracy
        self.surfeit = surfeit
        self.aggregator = aggregator
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.feature_names = tuple(str(name) for name in feature_names)

    def evaluate(self, spec: TimeSeriesCandidateSpec) -> TimeSeriesCandidateResult:
        """Return a complete nescience result for one candidate specification."""
        subset = np.asarray(spec.subset, dtype=bool)
        selected = np.flatnonzero(subset)
        if selected.size == 0:
            raise ValueError("Candidate subset must select at least one feature.")

        predictions = np.asarray(spec.model.predict(self.X[:, selected]), dtype=float).ravel()
        subset_diagnostics = self.miscoding.subset_analysis(subset)
        components = self._components(
            subset_diagnostics=subset_diagnostics,
            predictions=predictions,
            model_string=spec.model_string,
        )
        nescience_value = self.aggregator.aggregate_components(**components)
        estimator_score = float(spec.model.score(self.X[:, selected], self.y))

        selected_names = tuple(self.feature_names[index] for index in selected)
        metadata = {
            "schema": "canonical_nescience_time_series_model_v1",
            "model_family": spec.model_family,
            "window_size": int(spec.window_size),
            "n_selected_features": int(selected.size),
            "selected_feature_names": selected_names,
        }

        return TimeSeriesCandidateResult(
            model_name=spec.model_name,
            model_family=spec.model_family,
            model=spec.model,
            subset=subset,
            window_size=int(spec.window_size),
            nescience=float(nescience_value),
            components={key: float(value) for key, value in components.items()},
            estimator_score=estimator_score,
            selected_feature_indices=tuple(int(index) for index in selected),
            selected_feature_names=selected_names,
            model_string=spec.model_string,
            predictions=predictions,
            metadata=metadata,
            subset_diagnostics=subset_diagnostics,
        )

    def _components(
        self,
        *,
        subset_diagnostics,
        predictions,
        model_string: str,
    ) -> dict[str, float]:
        """Compute the four nescience components from explicit artifacts."""
        return {
            "deficiency": float(subset_diagnostics["deficiency"]),
            "surplus": float(subset_diagnostics["surplus"]),
            "inaccuracy": float(self.inaccuracy.inaccuracy_predictions(predictions)),
            "surfeit": float(self.surfeit.surfeit_string(model_string)),
        }


def candidate_results_dataframe(results: list[TimeSeriesCandidateResult]) -> pd.DataFrame:
    """Convert candidate results into a stable, sorted DataFrame."""
    rows = []
    for result in results:
        row = {
            "model_name": result.model_name,
            "model_family": result.model_family,
            "window_size": result.window_size,
            "nescience": result.nescience,
            "estimator_score": result.estimator_score,
            "n_selected_features": result.n_selected_features,
            "description_length": result.description_length,
            "selected_feature_indices": result.selected_feature_indices,
            "selected_feature_names": result.selected_feature_names,
            "miscoding": result.miscoding,
            "is_reliable": result.is_reliable,
            "failure_reason": result.subset_diagnostics.get("failure_reason"),
            "n_samples": result.subset_diagnostics.get("n_samples"),
            "n_observed_joint_states": result.subset_diagnostics.get("n_observed_joint_states"),
            "mean_joint_occupancy": result.subset_diagnostics.get("mean_joint_occupancy"),
            "n_singleton_joint_states": result.subset_diagnostics.get("n_singleton_joint_states"),
            "singleton_fraction": result.subset_diagnostics.get("singleton_fraction"),
        }
        row.update(result.components)
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["is_reliable", "nescience"],
        ascending=[False, True],
        ignore_index=True,
        na_position="last",
    )
