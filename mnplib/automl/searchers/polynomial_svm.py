"""Bounded polynomial-SVM search with canonical prediction fidelity checks."""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from mnplib.automl.wrappers import SelectedFeaturesEstimator
from mnplib.models.language import execute

from ._feature_order import miscoding_feature_order
from ._fit import fit_with_convergence_flag
from .base import ModelFamilySearcher, SearchContext, search_report


class _PolynomialSVMSearcher(ModelFamilySearcher):
    """Vary C and degree on one fixed, standardized miscoding-ranked prefix."""

    estimator_cls = None

    def __init__(self, *, C_values=(0.1, 1.0, 10.0), degree_values=(2, 3, 4),
                 max_iter: int = 5000, tol: float = 1e-3):
        self.C_values = tuple(C_values)
        self.degree_values = tuple(degree_values)
        self.max_iter = max_iter
        self.tol = tol

    def search(self, context: SearchContext):
        if self.estimator_cls is SVC and len(np.unique(context.y)) != 2:
            return search_report(self.family, [], [{
                "family": self.family, "reason": "unsupported_multiclass",
                "error": "Polynomial SVC serialization supports binary classification only.",
            }])

        order, _ = miscoding_feature_order(
            context.evaluator.nescience.miscoding_, context.X.shape[1],
            criterion=context.feature_ranking_criterion,
            max_features=context.max_feature_prefixes,
        )
        if not order:
            return search_report(self.family, [], [{
                "family": self.family, "reason": "empty_feature_order",
            }])
        selected = tuple(order)
        scaler = StandardScaler().fit(context.X[:, selected])
        X_scaled = scaler.transform(context.X[:, selected])
        results, diagnostics = [], []
        for C in dict.fromkeys(self.C_values):
            for degree in dict.fromkeys(self.degree_values):
                parameters = {"C": C, "degree": degree,
                              "max_iter": self.max_iter, "tol": self.tol}
                model = self.estimator_cls(kernel="poly", gamma="scale", coef0=1.0, **parameters)
                detail = {"family": self.family, "C": C, "degree": degree}
                try:
                    converged = fit_with_convergence_flag(model, X_scaled, context.y)
                except Exception as exc:
                    diagnostics.append({**detail, "reason": "fit_failed", "error": str(exc)})
                    continue
                if not converged:
                    diagnostics.append({**detail, "reason": "polynomial_svm_convergence_warning"})

                public_model = SelectedFeaturesEstimator(
                    model, selected, n_features_in=context.X.shape[1],
                    feature_names=context.feature_names, transformer=scaler,
                )
                try:
                    result = context.evaluator.evaluate(
                        name=f"{self.family}_C_{C:.6g}_degree_{degree}", family=self.family,
                        model=model, feature_indices=selected, X_adapter=X_scaled,
                        result_model=public_model, input_transformer=scaler,
                        hyperparameters=parameters,
                    )
                    with np.errstate(over="raise", invalid="raise", divide="raise"):
                        canonical = execute(result.artifacts.description.ast, context.X)
                except (ValueError, TypeError, ArithmeticError) as exc:
                    diagnostics.append({**detail, "reason": "description_failed", "error": str(exc)})
                    continue

                predictions = result.artifacts.predictions
                if self.estimator_cls is SVC:
                    matches = canonical == predictions
                else:
                    matches = np.isfinite(canonical) & np.isclose(
                        canonical, predictions, rtol=0.02, atol=0.03,
                    )
                if not np.all(matches):
                    diagnostics.append({
                        **detail, "reason": "description_fidelity_failed",
                        "n_mismatched_predictions": int(np.count_nonzero(~matches)),
                        "error": "Canonical predictions exceed the model-description fidelity tolerance.",
                    })
                    continue
                results.append(result)
        return search_report(self.family, results, diagnostics)


class PolynomialSVCSearcher(_PolynomialSVMSearcher):
    """Search binary polynomial classifiers by nescience."""

    estimator_cls = SVC
    family = "polynomial_svc"


class PolynomialSVRSearcher(_PolynomialSVMSearcher):
    """Search polynomial regressors by nescience."""

    estimator_cls = SVR
    family = "polynomial_svr"
