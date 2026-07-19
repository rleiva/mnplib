"""
Compact linear SVM model-family searchers.
"""

from __future__ import annotations

import warnings

from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import LinearSVC, LinearSVR

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
                converged = _fit_with_convergence_flag(model, context.X, context.y)
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
                    metadata={
                        "C": float(C),
                        "max_iter": self.max_iter,
                        "tol": self.tol,
                        "dual": False,
                        "converged": bool(converged),
                    },
                )
            )

        return search_report(self.family, results, diagnostics)


class LinearSVRSearcher(ModelFamilySearcher):
    """
    Search a small set of C and epsilon values for LinearSVR.
    """

    family = "linear_svr"

    def __init__(
        self,
        *,
        C_values=(0.1, 1.0, 10.0),
        epsilon_values=(0.0, 0.1),
        max_iter: int = 5000,
        tol: float = 1e-4,
        random_state=None,
    ):
        self.C_values = tuple(float(value) for value in C_values)
        self.epsilon_values = tuple(float(value) for value in epsilon_values)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state

    def search(self, context: SearchContext):
        results = []
        diagnostics = []
        seen = set()

        for C in self.C_values:
            for epsilon in self.epsilon_values:
                key = (float(C), float(epsilon))
                if key in seen:
                    continue
                seen.add(key)

                model = LinearSVR(
                    C=float(C),
                    epsilon=float(epsilon),
                    max_iter=self.max_iter,
                    tol=self.tol,
                    random_state=self.random_state,
                )
                try:
                    converged = _fit_with_convergence_flag(
                        model,
                        context.X,
                        context.y,
                    )
                except Exception as exc:
                    diagnostics.append(
                        {
                            "family": self.family,
                            "reason": "fit_failed",
                            "C": float(C),
                            "epsilon": float(epsilon),
                            "error": str(exc),
                        }
                    )
                    continue

                results.append(
                    context.evaluator.evaluate(
                        name=f"linear_svr_C_{C:.6g}_epsilon_{epsilon:.6g}",
                        family=self.family,
                        model=model,
                        metadata={
                            "C": float(C),
                            "epsilon": float(epsilon),
                            "max_iter": self.max_iter,
                            "tol": self.tol,
                            "converged": bool(converged),
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
