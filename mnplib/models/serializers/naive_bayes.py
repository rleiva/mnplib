"""
Canonical serializer for scikit-learn Gaussian Naive Bayes classifiers.

Gaussian Naive Bayes provides a compact probabilistic description of a
classification rule. The fitted model consists of class priors, class-conditional
means, and class-conditional variances. The serializer below exposes those
quantities as an executable simplified-Python prediction function.
"""

from __future__ import annotations

import numpy as np

from sklearn.naive_bayes import GaussianNB

from .base import (
    SklearnSerializer,
    Task,
    format_label,
    format_number,
    require_fitted
)

class NaiveBayesSerializer(SklearnSerializer):
    """
    Canonical serializer for Gaussian Naive Bayes classifiers.

    The serializer supports only ``GaussianNB``. This keeps the Auto
    Classification module focused on a single Naive Bayes representation:
    continuous numerical attributes modeled with class-conditional Gaussian
    likelihoods.

    The serialized description is an executable predictor of the form:

        def predict(x):
            ...
            return class_index

    The returned class index is a compact internal token. The mapping from this
    token to the original class label is stored in metadata.
    """

    name            = "naive_bayes"
    supported_types = (GaussianNB,)

    def task(self, model) -> Task:
        """
        Return the task type of the supported Naive Bayes estimator.
        """
        if not isinstance(model, GaussianNB):
            raise TypeError(
                "Expected GaussianNB. "
                f"Got {type(model).__name__} instead."
            )

        return "classification"

    def subset(self, model) -> list[int]:
        """
        Return features with class-dependent Gaussian parameters.

        A feature is considered part of the effective representation when its
        fitted mean or variance differs across classes. If both the mean and
        variance are identical for all classes, the feature contributes the same
        likelihood term to every class score and can be omitted from the
        executable decision rule.
        """
        require_fitted(model)
        self.task(model)

        theta = np.asarray(model.theta_, dtype=float)
        var   = np.asarray(model.var_, dtype=float)

        mean_differs = np.ptp(theta, axis=0) != 0.0
        var_differs  = np.ptp(var, axis=0) != 0.0
        used         = mean_differs | var_differs

        return [int(index) for index in np.flatnonzero(used)]

    def serialize(self, model, *, feature_names: list[str]) -> str:
        """
        Return an executable simplified-Python description of the classifier.

        The generated program computes one Gaussian log-joint score per class
        and returns the class with the largest score. The exponential function is
        not needed because classification depends only on score comparisons.
        """
        require_fitted(model)
        self.task(model)

        # Feature names are accepted to satisfy the common serializer interface.
        # The executable model string uses compact positional references x[i].
        del feature_names

        subset = self.subset(model)
        lines  = self._prediction_function_lines(model, subset)

        return "\n".join(lines) + "\n"

    def metadata(self, model, *, feature_names: list[str], subset: list[int]) -> dict:
        """
        Return diagnostic metadata for the fitted GaussianNB model.

        Metadata is intentionally kept outside the serialized model string. It
        supports interpretation and diagnostics without inflating the model
        description used to compute surfeit.
        """
        require_fitted(model)
        self.task(model)

        return {
            "likelihood"             : "gaussian",
            "n_classes"              : int(len(model.classes_)),
            "classes"                : [format_label(label) for label in model.classes_],
            "n_features_in_use"      : int(len(subset)),
            "selected_feature_names" : [
                feature_names[index]
                for index in subset
            ],
            "var_smoothing"          : float(getattr(model, "var_smoothing", 0.0)),
            "epsilon"                : float(getattr(model, "epsilon_", 0.0))
        }

    def _prediction_function_lines(self, model, subset: list[int]) -> list[str]:
        """
        Build the executable prediction function.

        GaussianNB predicts the class with the largest log-joint score:

            log P(C_k) + sum_j log p(x_j | C_k)

        For each selected feature, the Gaussian log-likelihood contribution is

            -0.5 * log(2*pi*var) - ((x_j - mean)^2) / (2*var)

        The logarithmic constant is precomputed during serialization, so the
        generated function only uses arithmetic operations and comparisons.
        """
        indent = " "

        lines: list[str] = ["def predict(x):"]

        lines.append(
            f"{indent}s0={self._class_score_expression(model, 0, subset)}"
        )
        lines.append(f"{indent}best=0")
        lines.append(f"{indent}best_s=s0")

        for class_index in range(1, len(model.classes_)):
            score_name = f"s{class_index}"
            score_expression = self._class_score_expression(
                model,
                class_index,
                subset,
            )

            lines.append(f"{indent}{score_name}={score_expression}")
            lines.append(f"{indent}if {score_name}>best_s:")
            lines.append(f"{indent}{indent}best={class_index}")
            lines.append(f"{indent}{indent}best_s={score_name}")

        lines.append(f"{indent}return best")

        return lines

    def _class_score_expression(self, model, class_index: int, subset: list[int]) -> str:
        """
        Return the executable score expression for one class.

        The score is written as a sum of precomputed constants and quadratic
        terms. Coefficients and constants are formatted by ``format_number`` so
        that the model description remains canonical and compact.
        """
        theta = np.asarray(model.theta_, dtype=float)
        var = np.asarray(model.var_, dtype=float)

        terms: list[str] = []

        prior_term = self._class_log_prior(model)[class_index]
        if prior_term != 0.0:
            terms.append(format_number(float(prior_term)))

        for feature_index in subset:
            mean = float(theta[class_index, feature_index])
            variance = self._positive_variance(
                float(var[class_index, feature_index])
            )

            constant = -0.5 * np.log(2.0 * np.pi * variance)
            quadratic_denominator = 2.0 * variance

            if constant != 0.0:
                terms.append(format_number(float(constant)))

            mean_text = format_number(mean)
            denominator_text = format_number(quadratic_denominator)

            terms.append(
                f"-((x[{feature_index}]-{mean_text})**2)/{denominator_text}"
            )

        if not terms:
            return format_number(0.0)

        return self._join_terms(terms)

    @staticmethod
    def _class_log_prior(model) -> np.ndarray:
        """
        Return log-priors for the fitted classes.

        ``GaussianNB`` exposes class priors as probabilities. The serializer
        converts them to log-priors because prediction is performed by comparing
        Gaussian log-joint scores.
        """
        priors = np.asarray(model.class_prior_, dtype=float)
        priors = np.maximum(priors, np.finfo(float).tiny)

        return np.log(priors)

    @staticmethod
    def _positive_variance(value: float) -> float:
        """
        Return a strictly positive variance for executable score generation.

        A fitted ``GaussianNB`` normally stores positive variances. The guard
        below prevents invalid generated expressions if an externally modified
        estimator contains a degenerate value.
        """
        return max(float(value), np.finfo(float).tiny)

    @staticmethod
    def _join_terms(terms: list[str]) -> str:
        """
        Join signed arithmetic terms into one valid Python expression.
        """
        expression = terms[0]

        for term in terms[1:]:
            if term.startswith("-"):
                expression += term
            else:
                expression += "+" + term

        return expression
