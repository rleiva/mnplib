"""
Nescience-based feature-prefix search for linear regression.
"""

from __future__ import annotations

from sklearn.linear_model import LinearRegression

from mnplib.automl.wrappers import SelectedFeaturesEstimator

from ._feature_order import feature_mask, miscoding_feature_order
from .base import ModelFamilySearcher, SearchContext, search_report


class LinearRegressionPrefixSearcher(ModelFamilySearcher):
    """
    Evaluate nested miscoding-ranked feature prefixes for LinearRegression.
    """

    family = "linear_regression"

    def __init__(self, *, patience: int | None = None):
        self.patience = patience

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
            model = LinearRegression()
            model.fit(context.X[:, selected], context.y)
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
            result = context.evaluator.evaluate(
                name=f"linear_regression_prefix_{n_features_used}",
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
