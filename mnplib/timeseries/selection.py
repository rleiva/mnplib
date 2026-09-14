"""
Structured results for time-series forecasting candidates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mnplib.automl.results import CandidateResult


@dataclass(frozen=True)
class TimeSeriesCandidateResult(CandidateResult):
    """
    Shared candidate result with forecasting metadata.
    """

    metadata: dict[str, Any] = field(default_factory=dict)
