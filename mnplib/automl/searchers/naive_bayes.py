"""
Model-appropriate Naive Bayes searches.
"""

from __future__ import annotations

import numpy as np

from sklearn.naive_bayes import BernoulliNB, CategoricalNB, GaussianNB, MultinomialNB

from .base import ModelFamilySearcher, SearchContext, search_report


class NaiveBayesSearcher(ModelFamilySearcher):
    """
    Search compatible Naive Bayes variants with family-specific parameters.
    """

    family = "naive_bayes"

    def __init__(
        self,
        *,
        gaussian_var_smoothing=(1e-12, 1e-9, 1e-6, 1e-3),
        alpha_values=(0.1, 1.0, 10.0),
    ):
        self.gaussian_var_smoothing = tuple(
            float(value)
            for value in gaussian_var_smoothing
        )
        self.alpha_values = tuple(float(value) for value in alpha_values)

    def search(self, context: SearchContext):
        results = []
        diagnostics = []

        results.extend(self._search_gaussian(context, diagnostics))
        results.extend(self._search_multinomial(context, diagnostics))
        results.extend(self._search_bernoulli(context, diagnostics))
        results.extend(self._search_categorical(context, diagnostics))

        return search_report(self.family, results, diagnostics)

    def _search_gaussian(self, context: SearchContext, diagnostics):
        results = []
        for var_smoothing in self._dedupe(self.gaussian_var_smoothing):
            model = GaussianNB(var_smoothing=float(var_smoothing))
            try:
                model.fit(context.X, context.y)
            except Exception as exc:
                diagnostics.append(
                    self._diagnostic(
                        "gaussian_nb",
                        "fit_failed",
                        var_smoothing=float(var_smoothing),
                        error=str(exc),
                    )
                )
                continue

            results.append(
                context.evaluator.evaluate(
                    name=f"gaussian_nb_var_smoothing_{var_smoothing:.6g}",
                    family="gaussian_nb",
                    model=model,
                    metadata={
                        "variant": "GaussianNB",
                        "var_smoothing": float(var_smoothing),
                    },
                )
            )
        return results

    def _search_multinomial(self, context: SearchContext, diagnostics):
        if not self._is_non_negative(context.X):
            diagnostics.append(
                self._diagnostic(
                    "multinomial_nb",
                    "incompatible_negative_features",
                )
            )
            return []

        return self._search_alpha_variant(
            context,
            diagnostics,
            family="multinomial_nb",
            estimator_cls=MultinomialNB,
        )

    def _search_bernoulli(self, context: SearchContext, diagnostics):
        if not self._is_binary(context.X):
            diagnostics.append(
                self._diagnostic(
                    "bernoulli_nb",
                    "incompatible_non_binary_features",
                )
            )
            return []

        return self._search_alpha_variant(
            context,
            diagnostics,
            family="bernoulli_nb",
            estimator_cls=BernoulliNB,
        )

    def _search_categorical(self, context: SearchContext, diagnostics):
        if not self._is_categorical_integer_encoded(context.X):
            diagnostics.append(
                self._diagnostic(
                    "categorical_nb",
                    "incompatible_non_integer_or_negative_features",
                )
            )
            return []

        return self._search_alpha_variant(
            context,
            diagnostics,
            family="categorical_nb",
            estimator_cls=CategoricalNB,
        )

    def _search_alpha_variant(
        self,
        context: SearchContext,
        diagnostics,
        *,
        family: str,
        estimator_cls,
    ):
        results = []
        for alpha in self._dedupe(self.alpha_values):
            model = estimator_cls(alpha=float(alpha))
            try:
                model.fit(context.X, context.y)
            except Exception as exc:
                diagnostics.append(
                    self._diagnostic(
                        family,
                        "fit_failed",
                        alpha=float(alpha),
                        error=str(exc),
                    )
                )
                continue

            results.append(
                context.evaluator.evaluate(
                    name=f"{family}_alpha_{alpha:.6g}",
                    family=family,
                    model=model,
                    metadata={
                        "variant": type(model).__name__,
                        "alpha": float(alpha),
                    },
                )
            )

        return results

    @staticmethod
    def _is_non_negative(X) -> bool:
        try:
            values = np.asarray(X, dtype=float)
        except Exception:
            return False
        return bool(np.all(np.isfinite(values)) and np.min(values) >= 0.0)

    @staticmethod
    def _is_binary(X) -> bool:
        try:
            values = np.asarray(X, dtype=float)
        except Exception:
            return False
        finite = np.isfinite(values)
        return bool(np.all(finite) and np.all(np.isin(values, [0.0, 1.0])))

    @staticmethod
    def _is_categorical_integer_encoded(X) -> bool:
        try:
            values = np.asarray(X, dtype=float)
        except Exception:
            return False
        return bool(
            np.all(np.isfinite(values))
            and np.min(values) >= 0.0
            and np.all(np.isclose(values, np.round(values)))
        )

    @staticmethod
    def _dedupe(values):
        seen = set()
        result = []
        for value in values:
            value = float(value)
            if value in seen:
                continue
            seen.add(value)
            result.append(value)
        return result

    def _diagnostic(self, family: str, reason: str, **extra):
        diagnostic = {
            "family": family,
            "searcher_family": self.family,
            "reason": reason,
        }
        diagnostic.update(extra)
        return diagnostic
