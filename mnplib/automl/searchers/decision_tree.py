"""
Decision-tree search through cost-complexity pruning.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .base import ModelFamilySearcher, SearchContext, search_report
from mnplib.utils import discretize_vector

class DecisionTreePruningSearcher(ModelFamilySearcher):
    """
    Search a decision-tree family by evaluating pruning-path trees.
    """

    def __init__(self, estimator_cls, *, n_jobs: int | None = None, random_state: Any = None,):
        
        if estimator_cls not in (DecisionTreeClassifier, DecisionTreeRegressor):
            raise TypeError(
                "estimator_cls must be DecisionTreeClassifier or "
                "DecisionTreeRegressor."
            )

        self.estimator_cls = estimator_cls
        self.n_jobs        = n_jobs
        self.random_state  = random_state
        self.family        = (
            "decision_tree_classifier"
            if estimator_cls is DecisionTreeClassifier
            else "decision_tree_regressor"
        )

    def search(self, context: SearchContext):

        initial = self.estimator_cls(random_state=self.random_state, min_samples_leaf=5)
        initial.fit(context.X, context.y)
        pruning_path = initial.cost_complexity_pruning_path(context.X, context.y)
        alphas = self._unique_alphas(pruning_path.ccp_alphas)

        results         = []
        diagnostics     = []
        seen_structures = set()

        for index, alpha in enumerate(alphas):
            model = self.estimator_cls(
                ccp_alpha        = float(alpha),
                random_state     = self.random_state,
            )
            model.fit(context.X, context.y)

            signature = self._tree_structure_signature(model)
            if signature in seen_structures:
                diagnostics.append(
                    {
                        "family"    : self.family,
                        "candidate" : self._candidate_name(index, alpha),
                        "reason"    : "duplicate_tree_structure",
                        "ccp_alpha" : float(alpha),
                    }
                )
                continue

            seen_structures.add(signature)
            results.append(
                context.evaluator.evaluate(
                    name     = self._candidate_name(index, alpha),
                    family   = self.family,
                    model    = model,
                    hyperparameters = {"ccp_alpha": float(alpha)},
                )
            )

        return search_report(self.family, results, diagnostics)


    def _unique_alphas(self, alphas) -> list[float]:
        """
        Return representative pruning alphas.

        The pruning path may contain thousands of effective ccp_alpha values.
        This method reduces them by discretizing a scaled log-transformation of
        the alpha distribution and selecting one original alpha per occupied bin.
        """
        values = np.asarray(alphas, dtype=float)
        values = values[np.isfinite(values)]
        values = values[values >= 0.0]
        values = np.unique(values)

        if values.size == 0:
            return []

        positive = values[values > 0.0]

        positive.sort()

        if positive.size == 0:
            return [0.0]

        scale = float(positive.min())
        transformed = np.log1p(values / scale)

        bins = discretize_vector(transformed)

        representatives: list[float] = []

        for bin_id in np.unique(bins):
            indices = np.flatnonzero(bins == bin_id)

            bin_values = transformed[indices]
            center = 0.5 * (bin_values.min() + bin_values.max())

            representative_index = indices[
                np.argmin(np.abs(transformed[indices] - center))
            ]

            representatives.append(float(values[representative_index]))

        return sorted(set(representatives))

    @staticmethod
    def _tree_structure_signature(model) -> tuple:
        tree = model.tree_
        thresholds = np.round(np.asarray(tree.threshold, dtype=float), 12)
        return (
            tuple(np.asarray(tree.children_left, dtype=int).tolist()),
            tuple(np.asarray(tree.children_right, dtype=int).tolist()),
            tuple(np.asarray(tree.feature, dtype=int).tolist()),
            tuple(thresholds.tolist()),
        )

    def _candidate_name(self, index: int, alpha: float) -> str:
        return f"{self.family}_ccp_{index}_alpha_{float(alpha):.6g}"
