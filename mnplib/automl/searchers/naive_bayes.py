"""
Nescience-based feature-prefix search for Gaussian Naive Bayes.
"""

from __future__ import annotations

from sklearn.naive_bayes import GaussianNB

from mnplib.automl.wrappers import SelectedFeaturesEstimator

from ._feature_order import feature_mask, miscoding_feature_order
from .base import ModelFamilySearcher, SearchContext, search_report


class NaiveBayesSearcher(ModelFamilySearcher):
    """
    Evaluate Gaussian Naive Bayes over miscoding-ranked feature prefixes.

    The Naive Bayes family is represented by GaussianNB only. The search is not
    performed over Naive Bayes variants or smoothing grids. Instead, the search
    dimension is the representation itself: increasingly large prefixes of the
    feature order induced by miscoding.

    For a feature order (f_1, ..., f_p), the evaluated candidates are:

        (f_1),
        (f_1, f_2),
        ...
        (f_1, ..., f_p).

    Each candidate is fitted once and then evaluated by nescience through the
    standard artifact workflow.
    """

    family = "naive_bayes"

    def __init__(self, *, var_smoothing: float = 1e-9):
        """
        Initialize the Gaussian Naive Bayes prefix searcher.

        Parameters
        ----------
        var_smoothing:
            Numerical smoothing parameter passed to GaussianNB. This is treated
            as a fixed estimator-stability setting, not as a model-selection
            dimension.
        """
        if float(var_smoothing) < 0.0:
            raise ValueError("var_smoothing must be non-negative.")

        self.var_smoothing = float(var_smoothing)

    def search(self, context: SearchContext):
        """
        Fit and evaluate one GaussianNB candidate for each feature prefix.

        The feature order is obtained from the fitted miscoding component. Each
        candidate is trained on the selected feature matrix, while the public
        result model is wrapped so that it can receive the original full input
        representation.
        """
        order, details = miscoding_feature_order(
            context.evaluator.nescience.miscoding_,
            context.X.shape[1],
        )

        order = tuple(int(index) for index in order)
        path = details.get("path", ())

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

        for n_features_used in range(1, len(order) + 1):
            selected = tuple(order[:n_features_used])
            X_selected = context.X[:, selected]

            model = GaussianNB(var_smoothing=self.var_smoothing)

            try:
                model.fit(X_selected, context.y)
            except Exception as exc:
                diagnostics.append(
                    {
                        "family": self.family,
                        "reason": "fit_failed",
                        "n_features_used": int(n_features_used),
                        "selected_feature_indices": list(selected),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
                continue

            public_model = SelectedFeaturesEstimator(
                model,
                selected,
                n_features_in=context.X.shape[1],
                feature_names=context.feature_names,
            )

            metadata = {
                "variant": "GaussianNB",
                "var_smoothing": self.var_smoothing,
                "feature_order": list(order),
                "n_features_used": int(n_features_used),
                "selected_features": feature_mask(selected, context.X.shape[1]),
                "selected_feature_indices": list(selected),
                "feature_names": [
                    context.feature_names[index]
                    for index in selected
                ],
                "selection_path_length": int(len(path)),
            }

            result = context.evaluator.evaluate(
                name=self._candidate_name(n_features_used),
                family=self.family,
                model=model,
                feature_indices=selected,
                result_model=public_model,
                metadata=metadata,
            )

            results.append(result)

        return search_report(self.family, results, diagnostics)

    def _candidate_name(self, n_features_used: int) -> str:
        """
        Return a stable name for a GaussianNB feature-prefix candidate.
        """
        return f"gaussian_nb_prefix_{int(n_features_used)}"