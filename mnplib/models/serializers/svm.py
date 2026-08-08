"""
Canonical serializer for compact linear support-vector models.

The serializer supports explicitly linear support-vector estimators only:
LinearSVC for classification and LinearSVR for regression. The generated model
description is an executable simplified-Python procedure of the form:

    def predict(x):
        ...
        return y

The function is intentionally compact and uses positional input references
x[i].
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from sklearn.svm import LinearSVC, LinearSVR

from .base import (
    SklearnSerializer,
    Task,
    format_number,
    nonzero_mask,
    require_fitted,
)


class LinearSVMSerializer(SklearnSerializer):
    """
    Canonical serializer for LinearSVC and LinearSVR.

    The fitted primal coefficients and intercepts are serialized directly as an
    executable predictor. For classification, the procedure computes linear
    decision scores and returns the winning class token. For regression, it
    returns the fitted linear response.
    """

    name = "linear_svm"
    supported_types = (LinearSVC, LinearSVR)

    def task(self, model) -> Task:
        """
        Return the task type of the fitted linear SVM estimator.
        """
        if isinstance(model, LinearSVC):
            return "classification"

        if isinstance(model, LinearSVR):
            return "regression"

        raise TypeError(
            "Expected LinearSVC or LinearSVR. "
            f"Got {type(model).__name__} instead."
        )

    def subset(self, model) -> list[int]:
        """
        Return local feature indices with non-zero fitted coefficients.

        The returned indices are expressed in the coordinate system used by the
        fitted estimator. If the estimator was trained on a selected feature
        matrix, the adapter layer may later map these local indices back to the
        original representation.
        """
        require_fitted(model)

        coefficients = np.asarray(model.coef_, dtype=float)

        if coefficients.ndim == 1:
            used = nonzero_mask(coefficients)
        else:
            used = np.any(nonzero_mask(coefficients), axis=0)

        return [int(index) for index in np.flatnonzero(used)]

    def serialize(
        self,
        model,
        *,
        feature_names: list[str],
        feature_indices: Sequence[int] | None = None,
    ) -> str:
        """
        Return an executable simplified-Python description of the fitted model.

        Feature names are intentionally not used in the model string. The model
        description uses compact references of the form x[i]. If feature_indices
        is provided, local estimator coordinates are mapped back to the original
        input representation.
        """
        require_fitted(model)

        del feature_names

        original_indices = self._original_feature_indices(
            model,
            feature_indices=feature_indices,
        )

        if self.task(model) == "classification":
            lines = self._classification_rule_lines(
                model,
                original_indices=original_indices,
            )
        else:
            lines = self._regression_rule_lines(
                model,
                original_indices=original_indices,
            )

        return "\n".join(lines) + "\n"

    def _classification_rule_lines(
        self,
        model,
        *,
        original_indices: tuple[int, ...],
    ) -> list[str]:
        """
        Serialize LinearSVC as an executable classification procedure.

        For binary LinearSVC, scikit-learn stores one separating hyperplane.
        Positive scores correspond to class token 1; non-positive scores
        correspond to class token 0.

        For multiclass LinearSVC, scikit-learn stores one linear score per
        class. Prediction is the class with the largest decision score.
        """
        coefficients = np.asarray(model.coef_, dtype=float)
        intercept = np.asarray(model.intercept_, dtype=float).reshape(-1)
        n_classes = int(len(model.classes_))
        indent = " "

        lines: list[str] = ["def predict(x):"]

        if n_classes == 2 and coefficients.shape[0] == 1:
            score = self._linear_expression(
                intercept=float(intercept[0]),
                coefficients=coefficients[0],
                original_indices=original_indices,
            )

            lines.append(f"{indent}z={score}")
            lines.append(f"{indent}if z>0:")
            lines.append(f"{indent}{indent}return 1")
            lines.append(f"{indent}return 0")

            return lines

        first_score = self._linear_expression(
            intercept=float(intercept[0]),
            coefficients=coefficients[0],
            original_indices=original_indices,
        )

        lines.append(f"{indent}s0={first_score}")
        lines.append(f"{indent}best=0")
        lines.append(f"{indent}best_s=s0")

        for class_index in range(1, n_classes):
            score_name = f"s{class_index}"
            score = self._linear_expression(
                intercept=float(intercept[class_index]),
                coefficients=coefficients[class_index],
                original_indices=original_indices,
            )

            lines.append(f"{indent}{score_name}={score}")
            lines.append(f"{indent}if {score_name}>best_s:")
            lines.append(f"{indent}{indent}best={class_index}")
            lines.append(f"{indent}{indent}best_s={score_name}")

        lines.append(f"{indent}return best")

        return lines

    def _regression_rule_lines(
        self,
        model,
        *,
        original_indices: tuple[int, ...],
    ) -> list[str]:
        """
        Serialize LinearSVR as an executable regression procedure.
        """
        coefficients = np.asarray(model.coef_, dtype=float).reshape(-1)
        intercept = float(np.asarray(model.intercept_, dtype=float).reshape(-1)[0])
        indent = " "

        expression = self._linear_expression(
            intercept=intercept,
            coefficients=coefficients,
            original_indices=original_indices,
        )

        return [
            "def predict(x):",
            f"{indent}y={expression}",
            f"{indent}return y",
        ]

    @staticmethod
    def _linear_expression(
        *,
        intercept: float,
        coefficients: np.ndarray,
        original_indices: tuple[int, ...],
    ) -> str:
        """
        Return a compact executable expression for a linear score.

        The expression has the form:

            b+w0*x[i0]+w1*x[i1]+...

        Zero coefficients are omitted before formatting. Coefficients equal to
        one or minus one are not special-cased; the serializer uses one uniform
        representation for all non-zero coefficients.
        """
        terms: list[str] = []

        if nonzero_mask(np.asarray([intercept], dtype=float))[0]:
            terms.append(format_number(float(intercept)))

        for local_index, coefficient in enumerate(coefficients):
            if not nonzero_mask(np.asarray([coefficient], dtype=float))[0]:
                continue

            original_index = int(original_indices[local_index])
            coefficient_text = format_number(float(coefficient))

            terms.append(f"{coefficient_text}*x[{original_index}]")

        if not terms:
            return format_number(0.0)

        return LinearSVMSerializer._join_terms(terms)

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

    @staticmethod
    def _original_feature_indices(
        model,
        *,
        feature_indices: Sequence[int] | None,
    ) -> tuple[int, ...]:
        """
        Return the feature indices used by the executable predictor.

        If the estimator was fitted on a selected feature matrix, feature_indices
        maps local estimator columns back to the original representation. If no
        mapping is provided, the local estimator coordinates are already assumed
        to be original coordinates.
        """
        n_features = int(model.n_features_in_)

        if feature_indices is None:
            return tuple(range(n_features))

        original_indices = tuple(int(index) for index in feature_indices)

        if len(original_indices) != n_features:
            raise ValueError(
                "feature_indices must contain one original feature index for "
                "each column used by the fitted Linear SVM estimator."
            )

        return original_indices
