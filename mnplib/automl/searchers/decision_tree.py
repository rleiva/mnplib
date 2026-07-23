"""
Decision-tree search through cost-complexity pruning.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .base import ModelFamilySearcher, SearchContext, search_report

class DecisionTreePruningSearcher(ModelFamilySearcher):
    """
    Search a decision-tree family by evaluating pruning-path trees.
    """

    def __init__(
        self,
        estimator_cls,
        *,
        min_samples_leaf: int = 1,
        alpha_tol: float = 1e-12,
        n_jobs: int | None = None,
        random_state: Any = None,
    ):
        if estimator_cls not in (DecisionTreeClassifier, DecisionTreeRegressor):
            raise TypeError(
                "estimator_cls must be DecisionTreeClassifier or "
                "DecisionTreeRegressor."
            )

        self.estimator_cls = estimator_cls
        self.min_samples_leaf = int(min_samples_leaf)
        self.alpha_tol     = float(alpha_tol)
        self.n_jobs        = n_jobs
        self.random_state  = random_state
        self.family        = (
            "decision_tree_classifier"
            if estimator_cls is DecisionTreeClassifier
            else "decision_tree_regressor"
        )

    def search(self, context: SearchContext):

        initial = self.estimator_cls(
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
        initial.fit(context.X, context.y)
        pruning_path = initial.cost_complexity_pruning_path(context.X, context.y)
        alphas = self._unique_alphas(pruning_path.ccp_alphas)

        results         = []
        diagnostics     = []
        seen_structures = set()

        for index, alpha in enumerate(alphas):
            model = self.estimator_cls(
                ccp_alpha        = float(alpha),
                min_samples_leaf = self.min_samples_leaf,
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
            metadata = {
                "ccp_alpha" : float(alpha),
                "alpha_tol" : self.alpha_tol,
                "min_samples_leaf": self.min_samples_leaf,
                "n_jobs"    : self.n_jobs,
                "n_nodes"   : int(model.tree_.node_count),
                "n_leaves"  : int(model.get_n_leaves()),
                "max_depth" : int(model.get_depth()),
            }
            results.append(
                context.evaluator.evaluate(
                    name     = self._candidate_name(index, alpha),
                    family   = self.family,
                    model    = model,
                    metadata = metadata,
                )
            )

        return search_report(self.family, results, diagnostics)

    def _unique_alphas(self, alphas) -> list[float]:

        values = sorted(float(alpha) for alpha in np.asarray(alphas, dtype=float))
        unique: list[float] = []

        for alpha in values:
            if not np.isfinite(alpha):
                continue
            if alpha < 0.0 and abs(alpha) <= self.alpha_tol:
                alpha = 0.0
            if not unique:
                unique.append(alpha)
                continue

            scale = max(1.0, abs(unique[-1]), abs(alpha))
            if abs(alpha - unique[-1]) > self.alpha_tol * scale:
                unique.append(alpha)

        return unique

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
