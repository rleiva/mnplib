"""
Candidate model-description helpers for AutoML estimators.
"""

from __future__ import annotations

from collections.abc import Sequence

from mnplib.surfeit import Surfeit

from .results import CandidateResult


def describe_candidate_model(
    results: Sequence[CandidateResult],
    candidate: str | None = None,
    *,
    best_result: CandidateResult | None = None,
    surfeit: Surfeit,
) -> dict[str, object]:
    """
    Return serialized model text and length diagnostics for one candidate.
    """
    result = _find_result_by_candidate_name(
        results,
        candidate,
        best_result=best_result,
    )
    model_string = result.artifacts.model_string
    lengths = surfeit.description_lengths(model_string)

    return {
        "candidate": result.name,
        "family": result.family,
        "model_type": result.artifacts.model_type,
        "model_string": model_string,
        "model_length": lengths["model_length"],
        "model_compressed_length": lengths["model_compressed_length"],
        "surfeit": float(result.components["surfeit"]),
    }


def _find_result_by_candidate_name(
    results: Sequence[CandidateResult],
    candidate: str | None = None,
    *,
    best_result: CandidateResult | None = None,
) -> CandidateResult:
    """
    Return the matching candidate result by public candidate name.
    """
    if candidate is None:
        if best_result is not None:
            return best_result

        if results:
            return min(results, key=lambda result: result.nescience)

        raise KeyError("No candidate results are available.")

    candidate_name = str(candidate)
    for result in results:
        if result.name == candidate_name:
            return result

    available = ", ".join(result.name for result in results[:8])
    if len(results) > 8:
        available += ", ..."

    raise KeyError(
        f"Unknown candidate {candidate_name!r}. "
        f"Available candidates are: {available}."
    )
