"""
Structured results produced by AutoML model-family searchers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CandidateResult:
    """
    Result from evaluating one fitted candidate through explicit artifacts.
    """

    name: str
    family: str
    model: object
    nescience: float
    components: dict[str, float]
    artifacts: object
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def estimator(self):
        """
        Backward-compatible alias for older public tests and user code.
        """
        return self.model

    @property
    def estimator_score(self) -> float:
        """
        Backward-compatible alias for the native estimator score.
        """
        return float(self.metadata.get("native_score", float("nan")))


@dataclass(frozen=True)
class SearchReport:
    """
    Results and diagnostics returned by one model-family searcher.
    """

    family: str
    results: list[CandidateResult]
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
