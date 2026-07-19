"""
Feature-ranking helpers based on the fitted Miscoding estimator.
"""

from __future__ import annotations

import numpy as np


def miscoding_feature_order(miscoding, n_features: int) -> tuple[list[int], dict]:
    """
    Return a full feature order seeded by ``Miscoding.select_features()``.
    """
    details = miscoding.select_features(
        max_features=n_features,
        min_improvement=0.0,
        return_details=True,
    )

    selected = [int(index) for index in details["selected_feature_indices"]]
    selected_set = set(selected)

    analysis = details["features"]
    remaining = [
        int(row.feature_index)
        for row in analysis.itertuples(index=False)
        if int(row.feature_index) not in selected_set
    ]

    order = selected + remaining
    if len(order) != n_features:
        present = set(order)
        order.extend(index for index in range(n_features) if index not in present)

    details = dict(details)
    details["feature_order"] = order

    return order, details


def feature_mask(indices, n_features: int) -> list[int]:
    """
    Return a binary mask for reporting selected features.
    """
    mask = np.zeros(int(n_features), dtype=int)
    mask[list(indices)] = 1
    return mask.astype(int).tolist()
