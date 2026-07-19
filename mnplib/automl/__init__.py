"""
AutoML helpers for minimum-nescience model selection.
"""

from .evaluator import CandidateEvaluator
from .results import CandidateResult, SearchReport

__all__ = [
    "CandidateEvaluator",
    "CandidateResult",
    "SearchReport",
]
