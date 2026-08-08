"""
Canonical serializers for scikit-learn linear models.
"""

from __future__ import annotations

import numpy as np

from sklearn.linear_model import LinearRegression, LogisticRegression

from .base import (
    SklearnSerializer,
    Task,
    format_number,
    nonzero_mask,
    require_fitted,
)


class LinearModelSerializer(SklearnSerializer):
    """
    Canonical serializer for linear regression estimators.
    """

    name = "linear_model"
    supported_types = (LinearRegression,)

    def task(self, model) -> Task:
        """
        Return the task type of the linear model.
        """
        return "regression"

    def subset(self, model) -> list[int]:
        """
        Return feature indices with non-zero coefficients.
        """
        require_fitted(model)

        coef = np.asarray(model.coef_, dtype=float)

        if coef.ndim == 1:
            used = nonzero_mask(coef)
        else:
            used = np.any(nonzero_mask(coef), axis=0)

        return [int(j) for j in np.flatnonzero(used)]

    def serialize(
        self,
        model,
        *,
        feature_names: list[str]
    ) -> str:
        """
        Return a canonical string description of the linear model.
        """
        require_fitted(model)

        lines = linear_regression_rule_lines(
                model,
                feature_names=feature_names
            )

        return "\n".join(lines) + "\n"

class LogisticRegressionSerializer(SklearnSerializer):
    """
    Canonical serializer for logistic regression classifiers.
    """

    name            = "logistic_regression"
    supported_types = (LogisticRegression,)

    def task(self, model) -> Task:
        """
        Return the task type of logistic regression.
        """
        return "classification"

    def subset(self, model) -> list[int]:
        """
        Return feature indices with non-zero logistic-regression coefficients.
        """
        require_fitted(model)

        coef = np.asarray(model.coef_, dtype=float)
        used = np.any(nonzero_mask(coef), axis=0)

        return [int(j) for j in np.flatnonzero(used)]

    def serialize(self, model, *, feature_names: list[str]) -> str:
        """
        Return a canonical string description of logistic regression.
        """
        require_fitted(model)

        lines = logistic_regression_rule_lines(
                model,
                feature_names=feature_names
            )

        return "\n".join(lines) + "\n"

def linear_regression_rule_lines(
    model,
    *,
    feature_names: list[str]
) -> list[str]:
    """
    Serialize a linear regression rule.
    """
    coef = np.asarray(model.coef_, dtype=float)
    intercept = np.asarray(model.intercept_, dtype=float)
    indent = " "

    if coef.ndim == 1:
        return single_output_linear_rule(
            output_name="y",
            intercept=float(intercept.reshape(-1)[0]),
            coefficients=coef,
            feature_names=feature_names,
        ) + [f"{indent}return y"]

    lines: list[str] = []
    intercept_values = intercept.reshape(-1)

    for output_index, coefficients in enumerate(coef):
        output_name = f"y_{output_index}"
        lines.extend(
            single_output_linear_rule(
                output_name=output_name,
                intercept=float(intercept_values[output_index]),
                coefficients=coefficients,
                feature_names=feature_names
            )
        )

    outputs = " ".join(f"y_{i}" for i in range(coef.shape[0]))
    lines.append(f"{indent}return [{outputs}]")

    return lines


def single_output_linear_rule(
    *,
    output_name: str,
    intercept: float,
    coefficients: np.ndarray,
    feature_names: list[str]
) -> list[str]:
    """
    Serialize one linear output equation.
    """
    indent         = " "
    zero_tolerance = 0
    lines          = [f"{indent}{output_name} = {format_number(intercept)}"]

    for feature_index, coefficient in enumerate(coefficients):
        coefficient = float(coefficient)
        if abs(coefficient) <= zero_tolerance:
            continue

        sign = "+=" if coefficient >= 0 else "-="
        magnitude = format_number(abs(coefficient))
        lines.append(
            f"{indent}{output_name} {sign} {magnitude}*{feature_names[feature_index]}"
        )

    return lines


def logistic_regression_rule_lines(
    model,
    *,
    feature_names: list[str],
) -> list[str]:
    """
    Serialize logistic regression as an executable simplified-Python predictor.

    The generated description defines a function:

        def predict(x):
            ...
            return class_index

    The returned value is the zero-based class token.
    """
    del feature_names  # Feature names are intentionally not used in model strings.

    coef = np.asarray(model.coef_, dtype=float)
    intercept = np.asarray(model.intercept_, dtype=float).reshape(-1)
    classes = list(model.classes_)

    lines: list[str] = ["def predict(x):"]
    indent = " "

    if len(classes) == 2 and coef.shape[0] == 1:
        score = _linear_expression(
            intercept=float(intercept[0]),
            coefficients=coef[0],
        )

        lines.append(f"{indent}z={score}")
        lines.append(f"{indent}if z>0:")
        lines.append(f"{indent}{indent}return 1")
        lines.append(f"{indent}return 0")

        return lines

    first_score = _linear_expression(
        intercept=float(intercept[0]),
        coefficients=coef[0],
    )

    lines.append(f"{indent}s0={first_score}")
    lines.append(f"{indent}best=0")
    lines.append(f"{indent}best_s=s0")

    for class_index in range(1, len(classes)):
        score = _linear_expression(
            intercept=float(intercept[class_index]),
            coefficients=coef[class_index],
        )

        lines.append(f"{indent}s{class_index}={score}")
        lines.append(f"{indent}if s{class_index}>best_s:")
        lines.append(f"{indent}{indent}best={class_index}")
        lines.append(f"{indent}{indent}best_s=s{class_index}")

    lines.append(f"{indent}return best")

    return lines


def _linear_expression(*, intercept: float, coefficients: np.ndarray) -> str:
    """
    Return a compact executable Python expression for a linear score.

    The expression has the form:

        b+w0*x[0]+w1*x[1]+...

    Coefficients that are effectively zero are omitted before formatting.
    Numerical formatting is delegated to format_number().
    """
    terms: list[str] = []

    if nonzero_mask(np.asarray([intercept], dtype=float))[0]:
        terms.append(format_number(float(intercept)))

    for feature_index, coefficient in enumerate(coefficients):
        if not nonzero_mask(np.asarray([coefficient], dtype=float))[0]:
            continue

        coef_text = format_number(float(coefficient))
        terms.append(f"{coef_text}*x[{feature_index}]")

    if not terms:
        return format_number(0.0)

    expression = terms[0]

    for term in terms[1:]:
        if term.startswith("-"):
            expression += term
        else:
            expression += "+" + term

    return expression
