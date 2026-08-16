"""
Feature-ranking helpers based on the fitted Miscoding estimator.
"""

from __future__ import annotations

import numpy as np


def miscoding_feature_order(
    miscoding,
    n_features: int,
    *,
    criterion: str = "deficiency",
    max_features: int | None = None,
) -> tuple[list[int], dict]:
    """
    Return a miscoding-guided feature order for AutoML prefix search.
    """
    limit = n_features if max_features is None else min(int(max_features), n_features)
    details = miscoding.rank_features(
        max_features=limit,
        criterion=criterion,
        return_details=True,
    )

    order = [int(index) for index in details["feature_order"]]
    details = dict(details)

    return order, details


def feature_mask(indices, n_features: int) -> list[int]:
    """
    Return a binary mask for selected features.
    """
    mask = np.zeros(int(n_features), dtype=int)
    mask[list(indices)] = 1
    return mask.astype(int).tolist()
