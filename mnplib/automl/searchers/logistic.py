"""
Nescience-based feature-prefix search for logistic regression.
"""

from __future__ import annotations

import warnings

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from mnplib.automl.wrappers import SelectedFeaturesEstimator

from ._feature_order import feature_mask, miscoding_feature_order
from .base import ModelFamilySearcher, SearchContext, search_report


class LogisticRegressionPrefixSearcher(ModelFamilySearcher):
    """
    Evaluate miscoding-ranked feature prefixes using mostly unregularized fits.
    """

    family = "logistic_regression"

    def __init__(
        self,
        *,
        solver: str = "lbfgs",
        max_iter: int = 1000,
        tol: float = 1e-4,
        fallback_C_values=(1.0, 10.0, 100.0),
        patience: int | None = None,
        random_state=None,
    ):
        self.solver = solver
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.fallback_C_values = tuple(float(value) for value in fallback_C_values)
        self.patience = patience
        self.random_state = random_state

    def search(self, context: SearchContext):
        order, details = miscoding_feature_order(
            context.evaluator.nescience.miscoding_,
            context.X.shape[1],
        )

        results = []
        diagnostics = []
        best_nescience = float("inf")
        failures_without_improvement = 0

        for n_features_used in range(1, len(order) + 1):
            selected = tuple(order[:n_features_used])
            fitted = self._fit_prefix_candidates(context, selected)

            if not fitted:
                diagnostics.append(
                    {
                        "family": self.family,
                        "reason": "all_logistic_fits_failed",
                        "n_features_used": int(n_features_used),
                    }
                )
                continue

            for model, fit_metadata in fitted:
                public_model = SelectedFeaturesEstimator(
                    model,
                    selected,
                    n_features_in=context.X.shape[1],
                    feature_names=context.feature_names,
                )
                metadata = {
                    "feature_order": list(order),
                    "n_features_used": int(n_features_used),
                    "selected_features": feature_mask(selected, context.X.shape[1]),
                    "selected_feature_indices": list(selected),
                    "feature_names": [
                        context.feature_names[index]
                        for index in selected
                    ],
                    "selection_path_length": int(len(details["path"])),
                }
                metadata.update(fit_metadata)
                name = self._candidate_name(n_features_used, fit_metadata)
                result = context.evaluator.evaluate(
                    name=name,
                    family=self.family,
                    model=model,
                    feature_indices=selected,
                    result_model=public_model,
                    metadata=metadata,
                )
                results.append(result)

                if result.nescience < best_nescience:
                    best_nescience = result.nescience
                    failures_without_improvement = 0
                else:
                    failures_without_improvement += 1

            if (
                self.patience is not None
                and failures_without_improvement >= int(self.patience)
            ):
                diagnostics.append(
                    {
                        "family": self.family,
                        "reason": "early_stopping_patience",
                        "patience": int(self.patience),
                        "n_features_used": int(n_features_used),
                    }
                )
                break

        return search_report(self.family, results, diagnostics)

    def _fit_prefix_candidates(self, context: SearchContext, selected):
        X_selected = context.X[:, selected]
        unregularized, metadata, failed_or_unstable = self._fit_unregularized(
            X_selected,
            context.y,
        )

        if unregularized is not None and not failed_or_unstable:
            return [(unregularized, metadata)]

        fitted = []
        for C in self.fallback_C_values:
            try:
                model, converged = self._fit_model(
                    X_selected,
                    context.y,
                    penalty="l2",
                    C=float(C),
                )
            except Exception:
                continue

            fitted.append(
                (
                    model,
                    {
                        "penalty": "l2",
                        "C": float(C),
                        "solver": self.solver,
                        "used_stability_fallback": True,
                        "converged": bool(converged),
                    },
                )
            )

        if fitted:
            return fitted

        if unregularized is not None:
            return [(unregularized, metadata)]

        return []

    def _fit_unregularized(self, X, y):
        try:
            model, converged = self._fit_model(X, y, penalty=None, C=None)
            return (
                model,
                {
                    "penalty": None,
                    "C": None,
                    "solver": self.solver,
                    "used_stability_fallback": False,
                    "converged": bool(converged),
                },
                not converged,
            )
        except Exception:
            pass

        try:
            model, converged = self._fit_model(X, y, penalty="none", C=None)
            return (
                model,
                {
                    "penalty": "none",
                    "C": None,
                    "solver": self.solver,
                    "used_stability_fallback": False,
                    "converged": bool(converged),
                },
                not converged,
            )
        except Exception:
            return None, {}, True

    def _fit_model(self, X, y, *, penalty, C):
        kwargs = {
            "penalty": penalty,
            "solver": self.solver,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "random_state": self.random_state,
        }
        if C is not None:
            kwargs["C"] = float(C)

        model = LogisticRegression(**kwargs)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X, y)

        converged = not any(
            issubclass(warning.category, ConvergenceWarning)
            for warning in caught
        )
        return model, converged

    def _candidate_name(self, n_features_used: int, metadata: dict) -> str:
        penalty = metadata.get("penalty")
        if metadata.get("used_stability_fallback"):
            return (
                f"logistic_regression_prefix_{n_features_used}_"
                f"l2_C_{metadata.get('C'):.6g}"
            )
        return f"logistic_regression_prefix_{n_features_used}_penalty_{penalty}"
