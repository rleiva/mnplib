"""
Reusable candidate evaluator for minimum-nescience AutoML search.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mnplib.models import ModelArtifacts, sklearn_model_artifacts

from .results import CandidateResult


class CandidateEvaluator:
    """
    Evaluate fitted models through the explicit artifact workflow.
    """

    def __init__(
        self,
        *,
        X,
        y,
        nescience,
        feature_names: Sequence[str],
    ):
        self.X = X
        self.y = y
        self.nescience = nescience
        self.feature_names = [str(name) for name in feature_names]

    def evaluate(
        self,
        *,
        name: str,
        family: str,
        model,
        hyperparameters: Mapping[str, Any] | None = None,
        feature_indices: Sequence[int] | None = None,
        X_adapter=None,
        result_model=None,
        model_string_prefix: str | None = None,
    ) -> CandidateResult:
        """
        Return a structured result for a fitted candidate model.
        """
        if feature_indices is None:
            X_for_adapter = self.X if X_adapter is None else X_adapter
            adapter_feature_names = self.feature_names
            subset_mapping = None
        else:
            feature_indices = tuple(int(index) for index in feature_indices)
            X_for_adapter = (
                self.X[:, feature_indices]
                if X_adapter is None
                else X_adapter
            )
            adapter_feature_names = [
                self.feature_names[index]
                for index in feature_indices
            ]
            subset_mapping = feature_indices

        adapter_artifacts = sklearn_model_artifacts(
            model,
            X_for_adapter,
            feature_names=adapter_feature_names,
            feature_indices=subset_mapping,
        )
        artifacts = self._remap_artifacts(
            adapter_artifacts,
            subset_mapping=subset_mapping,
            model_string_prefix=model_string_prefix,
        )
        public_model = model if result_model is None else result_model

        return self.evaluate_artifacts(
            name=name,
            family=family,
            model=public_model,
            artifacts=artifacts,
            hyperparameters=hyperparameters,
            score_X=X_for_adapter,
        )

    def evaluate_artifacts(
        self,
        *,
        name: str,
        family: str,
        model,
        artifacts: ModelArtifacts,
        hyperparameters: Mapping[str, Any] | None = None,
        estimator_score: float | None = None,
        score_X=None,
        metadata: Mapping[str, Any] | None = None,
        result_factory=None,
    ) -> CandidateResult:
        """
        Return a structured result from explicit model artifacts.
        """
        subset_diagnostics = self.nescience.miscoding_.subset_analysis(
            artifacts.subset
        )
        components = {
            "deficiency": float(subset_diagnostics["deficiency"]),
            "surplus": float(subset_diagnostics["surplus"]),
            "inaccuracy": float(
                self.nescience.inaccuracy_.inaccuracy_predictions(
                    artifacts.predictions
                )
            ),
            "surfeit": float(
                self.nescience.surfeit_.surfeit_string(
                    artifacts.model_string
                )
            ),
        }
        value = self.nescience.aggregate_components(**components)
        native_score = (
            float(estimator_score)
            if estimator_score is not None
            else self._native_score(model, self.X if score_X is None else score_X)
        )

        result_kwargs = {
            "name": str(name),
            "family": str(family),
            "model": model,
            "nescience": float(value),
            "components": dict(components),
            "artifacts": artifacts,
            "estimator_score": native_score,
            "n_selected_features": int(len(artifacts.subset)),
            "hyperparameters": dict(hyperparameters or {}),
            "subset_diagnostics": dict(subset_diagnostics),
        }

        if result_factory is None:
            return CandidateResult(**result_kwargs)

        return result_factory(
            **result_kwargs,
            metadata=dict(metadata or {}),
        )

    def _remap_artifacts(
        self,
        artifacts: ModelArtifacts,
        *,
        subset_mapping: Sequence[int] | None,
        model_string_prefix: str | None,
    ) -> ModelArtifacts:
        """
        Map adapter-local feature indices back to the original representation.
        """
        subset = list(artifacts.subset)

        if subset_mapping is not None:
            subset = [int(subset_mapping[index]) for index in subset]

        model_string = artifacts.model_string
        if model_string_prefix:
            model_string = model_string_prefix.rstrip() + "\n" + model_string

        return ModelArtifacts(
            subset=subset,
            predictions=artifacts.predictions,
            model_string=model_string,
            model_type=artifacts.model_type,
        )

    def _native_score(self, public_model, X_for_adapter) -> float:
        """
        Return the estimator's native score without letting failures stop search.
        """
        try:
            if hasattr(public_model, "selected_features"):
                return float(public_model.score(self.X, self.y))
            return float(public_model.score(X_for_adapter, self.y))
        except Exception:
            return float("nan")
