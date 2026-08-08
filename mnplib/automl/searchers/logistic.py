"""
Nescience-based feature-prefix search for logistic regression.
"""

from __future__ import annotations

import warnings

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from mnplib.automl.wrappers import SelectedFeaturesEstimator

from ._feature_order import miscoding_feature_order
from .base import ModelFamilySearcher, SearchContext, search_report


class LogisticRegressionPrefixSearcher(ModelFamilySearcher):
    """
    Evaluate miscoding-ranked feature prefixes with unregularized logistic regression.

    The search space is formed by prefixes of the feature order induced by
    miscoding. For each prefix, exactly one logistic-regression model is fitted.

    Regularization is intentionally not used here. Model complexity is controlled
    by nescience through the feature subset, prediction inaccuracy, and serialized
    model description.
    """

    family = "logistic_regression"

    def __init__(self, *, random_state=None, max_iter: int = 1000):
        """
        Initialize the logistic-regression prefix searcher.

        Parameters
        ----------
        random_state:
            Random-state value forwarded to scikit-learn when supported by the
            chosen solver.

        max_iter:
            Maximum number of optimization iterations. This is an optimization
            safeguard, not a model-selection parameter.
        """
        if int(max_iter) <= 0:
            raise ValueError("max_iter must be a positive integer.")

        self.solver       = "lbfgs"
        self.penalty      = None
        self.max_iter     = int(max_iter)
        self.random_state = random_state

    def search(self, context: SearchContext):
        """
        Search logistic-regression candidates along the miscoding feature order.

        For a feature order (f_1, ..., f_p), the candidates are fitted on:

            (f_1),
            (f_1, f_2),
            ...
            (f_1, ..., f_p).

        Each fitted candidate is evaluated by the shared CandidateEvaluator.
        """
        order, _ = miscoding_feature_order(
            context.evaluator.nescience.miscoding_,
            context.X.shape[1],
        )

        order = tuple(int(index) for index in order)
        results     = []
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

            model, fit_details = self._fit_unregularized(context, selected)

            if model is None:
                diagnostics.append(
                    {
                        "family"                   : self.family,
                        "reason"                   : "logistic_fit_failed",
                        "n_selected_features"      : int(n_selected_features),
                        "selected_feature_indices" : list(selected),
                        **fit_details,
                    }
                )
                continue

            if not fit_details["converged"]:
                diagnostics.append(
                    {
                        "family"                   : self.family,
                        "reason"                   : "logistic_convergence_warning",
                        "n_selected_features"      : int(n_selected_features),
                        "selected_feature_indices" : list(selected),
                    }
                )

            public_model = SelectedFeaturesEstimator(
                model,
                selected,
                n_features_in = context.X.shape[1],
                feature_names = context.feature_names,
            )

            result = context.evaluator.evaluate(
                name            = self._candidate_name(n_selected_features),
                family          = self.family,
                model           = model,
                feature_indices = selected,
                result_model    = public_model,
                hyperparameters = {
                    "penalty": None,
                    "solver": self.solver,
                    "max_iter": self.max_iter,
                },
            )

            results.append(result)

        return search_report(self.family, results, diagnostics)

    def _fit_unregularized(self, context: SearchContext, selected: tuple[int, ...]
                          ) -> tuple[LogisticRegression | None, dict[str, object]]:
        """
        Fit one canonical unregularized logistic-regression candidate.

        The fitted model receives only the selected feature columns. The wrapper
        created in search() exposes the candidate as a model over the original
        full input space.
        """
        X_selected = context.X[:, selected]
        y          = context.y

        try:
            model, converged = self._fit_model(X_selected, y)
        except Exception as exc:
            return (
                None,
                {
                    "penalty"       : None,
                    "solver"        : self.solver,
                    "max_iter"      : self.max_iter,
                    "converged"     : False,
                    "error_type"    : type(exc).__name__,
                    "error_message" : str(exc),
                },
            )

        fit_details: dict[str, object] = {
            "penalty"   : None,
            "solver"    : self.solver,
            "max_iter"  : self.max_iter,
            "converged" : bool(converged),
        }

        return model, fit_details

    def _fit_model(self, X, y) -> tuple[LogisticRegression, bool]:
        """
        Fit the sklearn logistic-regression estimator.

        No regularization parameter is supplied. In particular, C is not passed,
        because C only has meaning for regularized logistic regression.
        """
        model = LogisticRegression(
            penalty      = None,
            solver       = self.solver,
            max_iter     = self.max_iter,
            random_state = self.random_state,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X, y)

        converged = not any(
            issubclass(warning.category, ConvergenceWarning)
            for warning in caught
        )

        return model, converged

    def _candidate_name(self, n_selected_features: int) -> str:
        """
        Return the stable candidate name for a feature-prefix model.
        """
        return f"logistic_regression_prefix_{int(n_selected_features)}"
