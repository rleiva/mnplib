"""
Base interfaces for model-family searchers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from mnplib.automl.evaluator import CandidateEvaluator
from mnplib.automl.results import SearchReport

Task = Literal["classification", "regression"]

@dataclass(frozen=True)
class SearchContext:
    """
    Shared context supplied to each model-family searcher.
    """
    X             : np.ndarray
    y             : np.ndarray
    feature_names : list[str]
    evaluator     : CandidateEvaluator
    task          : Task
    random_state  : Any = None
    verbose       : int = 0


class ModelFamilySearcher:
    """
    Base class for searchers dedicated to one estimator family.
    """

    family: str = "base"

    def search(self, context: SearchContext) -> SearchReport:
        raise NotImplementedError


def search_report(family: str, results, diagnostics=None) -> SearchReport:
    """
    Build a normalized report from a searcher implementation.
    """
    return SearchReport(
        family      = family,
        results     = list(results),
        diagnostics = [] if diagnostics is None else list(diagnostics),
    )
