"""
Compact linear SVM model-family searchers.
"""

from __future__ import annotations

import warnings

from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import LinearSVC, LinearSVR

from mnplib.automl.wrappers import SelectedFeaturesEstimator

from ._feature_order import miscoding_feature_order
from .base import ModelFamilySearcher, SearchContext, search_report


class LinearSVCSearcher(ModelFamilySearcher):
    """
    Search a small meaningful set of C values for LinearSVC.
    """

    family = "linear_svc"

    def __init__(
        self,
        *,
        C_values=(0.1, 1.0, 10.0),
        max_iter: int = 5000,
        tol: float = 1e-4,
        random_state=None,
    ):
        self.C_values = tuple(float(value) for value in C_values)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state

    def search(self, context: SearchContext):
        results = []
        diagnostics = []
        seen = set()

        for C in self.C_values:
            if C in seen:
                continue
            seen.add(C)

            model = LinearSVC(
                C=float(C),
                dual=False,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state,
            )
            try:
                _fit_with_convergence_flag(model, context.X, context.y)
            except Exception as exc:
                diagnostics.append(
                    {
                        "family": self.family,
                        "reason": "fit_failed",
                        "C": float(C),
                        "error": str(exc),
                    }
                )
                continue

            results.append(
                context.evaluator.evaluate(
                    name=f"linear_svc_C_{C:.6g}",
                    family=self.family,
                    model=model,
                    hyperparameters={"C": float(C)},
                )
            )

        return search_report(self.family, results, diagnostics)


class LinearSVRSearcher(ModelFamilySearcher):
    """
    Evaluate nested miscoding-ranked feature prefixes for LinearSVR.
    """

    family = "linear_svr"

    def __init__(
        self,
        *,
        C: float = 1.0,
        epsilon: float = 0.0,
        max_iter: int = 5000,
        tol: float = 1e-4,
        random_state=None,
    ):
        self.C = float(C)
        self.epsilon = float(epsilon)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state

    def search(self, context: SearchContext):
        order, _ = miscoding_feature_order(
            context.evaluator.nescience.miscoding_,
            context.X.shape[1],
        )

        order = tuple(int(index) for index in order)
        results = []
        diagnostics = []

        if not order:
            diagnostics.append(
                {
                    "family": self.family,
                    "reason": "empty_feature_order",
                }
            )
            return search_report(self.family, results, diagnostics)

        for n_selected_features in range(1, len(order) + 1):
            selected = tuple(order[:n_selected_features])
            model = LinearSVR(
                C=self.C,
                epsilon=self.epsilon,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state,
            )
            try:
                converged = _fit_with_convergence_flag(
                    model,
                    context.X[:, selected],
                    context.y,
                )
            except Exception as exc:
                diagnostics.append(
                    {
                        "family": self.family,
                        "reason": "fit_failed",
                        "n_selected_features": int(n_selected_features),
                        "selected_feature_indices": list(selected),
                        "error": str(exc),
                    }
                )
                continue

            if not converged:
                diagnostics.append(
                    {
                        "family": self.family,
                        "reason": "linear_svr_convergence_warning",
                        "n_selected_features": int(n_selected_features),
                        "selected_feature_indices": list(selected),
                    }
                )

            public_model = SelectedFeaturesEstimator(
                model,
                selected,
                n_features_in=context.X.shape[1],
                feature_names=context.feature_names,
            )

            results.append(
                context.evaluator.evaluate(
                    name=f"linear_svr_prefix_{n_selected_features}",
                    family=self.family,
                    model=model,
                    feature_indices=selected,
                    result_model=public_model,
                    hyperparameters={
                        "C": self.C,
                        "epsilon": self.epsilon,
                        "max_iter": self.max_iter,
                        "tol": self.tol,
                    },
                )
            )

        return search_report(self.family, results, diagnostics)


def _fit_with_convergence_flag(model, X, y) -> bool:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X, y)

    return not any(
        issubclass(warning.category, ConvergenceWarning)
        for warning in caught
    )
