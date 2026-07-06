"""
Canonical serializers for scikit-learn linear models.
"""

from __future__ import annotations

import numpy as np

from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge

from ..artifacts import SerializationConfig
from .base import (
    SklearnSerializer,
    Task,
    canonical_header,
    format_label,
    format_number,
    require_fitted,
)


class LinearModelSerializer(SklearnSerializer):
    """
    Canonical serializer for linear regression-family estimators.
    """

    name = "linear_model"
    support_level = "stable"
    supported_types = (LinearRegression, Ridge, Lasso, ElasticNet)

    def task(self, model) -> Task:
        """
        Return the task type of the linear model.
        """
        return "regression"

    def subset(self, model, *, config: SerializationConfig) -> list[int]:
        """
        Return feature indices with non-zero coefficients.
        """
        require_fitted(model)

        coef = np.asarray(model.coef_, dtype=float)

        if coef.ndim == 1:
            used = np.abs(coef) > config.zero_tolerance
        else:
            used = np.any(np.abs(coef) > config.zero_tolerance, axis=0)

        return [int(j) for j in np.flatnonzero(used)]

    def serialize(
        self,
        model,
        *,
        feature_names: list[str],
        config: SerializationConfig,
    ) -> str:
        """
        Return a canonical string description of the linear model.
        """
        require_fitted(model)

        subset = self.subset(model, config=config)

        lines = canonical_header(
            model_type=type(model).__name__,
            task="regression",
            feature_names=[feature_names[j] for j in subset],
            config=config,
        )

        if config.include_metadata:
            lines.extend(
                [
                    "PARAMETERS",
                    f"{config.indent}n_outputs = {self._n_outputs(model)}",
                    f"{config.indent}n_nonzero_coefficients = {len(subset)}",
                ]
            )
            self._regularization_metadata_lines(model, lines, config)

        lines.append("RULE")
        lines.extend(
            linear_regression_rule_lines(
                model,
                feature_names=feature_names,
                config=config,
            )
        )

        return "\n".join(lines) + "\n"

    def metadata(
        self,
        model,
        *,
        feature_names: list[str],
        subset: list[int],
        config: SerializationConfig,
    ) -> dict:
        """
        Return linear-model metadata.
        """
        metadata = {
            "n_outputs": self._n_outputs(model),
            "n_nonzero_coefficients": int(len(subset)),
        }

        for name in ("alpha", "l1_ratio"):
            if hasattr(model, name):
                metadata[name] = float(getattr(model, name))

        return metadata

    @staticmethod
    def _n_outputs(model) -> int:
        """
        Return the number of model outputs.
        """
        coef = np.asarray(model.coef_)

        if coef.ndim == 1:
            return 1

        return int(coef.shape[0])

    @staticmethod
    def _regularization_metadata_lines(model, lines: list[str], config: SerializationConfig) -> None:
        """
        Append regularization parameters when present.
        """
        if hasattr(model, "alpha"):
            lines.append(
                f"{config.indent}alpha = {format_number(float(model.alpha), config)}"
            )

        if hasattr(model, "l1_ratio"):
            lines.append(
                f"{config.indent}l1_ratio = {format_number(float(model.l1_ratio), config)}"
            )


class LogisticRegressionSerializer(SklearnSerializer):
    """
    Canonical serializer for logistic regression classifiers.
    """

    name = "logistic_regression"
    support_level = "stable"
    supported_types = (LogisticRegression,)

    def task(self, model) -> Task:
        """
        Return the task type of logistic regression.
        """
        return "classification"

    def subset(self, model, *, config: SerializationConfig) -> list[int]:
        """
        Return feature indices with non-zero logistic-regression coefficients.
        """
        require_fitted(model)

        coef = np.asarray(model.coef_, dtype=float)
        used = np.any(np.abs(coef) > config.zero_tolerance, axis=0)

        return [int(j) for j in np.flatnonzero(used)]

    def serialize(
        self,
        model,
        *,
        feature_names: list[str],
        config: SerializationConfig,
    ) -> str:
        """
        Return a canonical string description of logistic regression.
        """
        require_fitted(model)

        subset = self.subset(model, config=config)

        lines = canonical_header(
            model_type=type(model).__name__,
            task="classification",
            feature_names=[feature_names[j] for j in subset],
            config=config,
        )

        if config.include_metadata:
            lines.extend(
                [
                    "PARAMETERS",
                    f"{config.indent}classes = {[format_label(label) for label in model.classes_]}",
                    f"{config.indent}n_nonzero_coefficients = {len(subset)}",
                ]
            )
            if hasattr(model, "C"):
                lines.append(
                    f"{config.indent}C = {format_number(float(model.C), config)}"
                )
            if hasattr(model, "penalty"):
                lines.append(f"{config.indent}penalty = {repr(model.penalty)}")

        lines.append("RULE")
        lines.extend(
            logistic_regression_rule_lines(
                model,
                feature_names=feature_names,
                config=config,
            )
        )

        return "\n".join(lines) + "\n"

    def metadata(
        self,
        model,
        *,
        feature_names: list[str],
        subset: list[int],
        config: SerializationConfig,
    ) -> dict:
        """
        Return logistic-regression metadata.
        """
        metadata = {
            "n_classes": int(len(model.classes_)),
            "classes": [format_label(label) for label in model.classes_],
            "n_nonzero_coefficients": int(len(subset)),
        }

        if hasattr(model, "C"):
            metadata["C"] = float(model.C)
        if hasattr(model, "penalty"):
            metadata["penalty"] = model.penalty

        return metadata


def linear_regression_rule_lines(
    model,
    *,
    feature_names: list[str],
    config: SerializationConfig,
) -> list[str]:
    """
    Serialize a linear regression rule.
    """
    coef = np.asarray(model.coef_, dtype=float)
    intercept = np.asarray(model.intercept_, dtype=float)

    if coef.ndim == 1:
        return single_output_linear_rule(
            output_name="y",
            intercept=float(intercept.reshape(-1)[0]),
            coefficients=coef,
            feature_names=feature_names,
            config=config,
        ) + [f"{config.indent}return y"]

    lines: list[str] = []
    intercept_values = intercept.reshape(-1)

    for output_index, coefficients in enumerate(coef):
        output_name = f"y_{output_index}"
        lines.extend(
            single_output_linear_rule(
                output_name=output_name,
                intercept=float(intercept_values[output_index]),
                coefficients=coefficients,
                feature_names=feature_names,
                config=config,
            )
        )

    outputs = ", ".join(f"y_{i}" for i in range(coef.shape[0]))
    lines.append(f"{config.indent}return [{outputs}]")

    return lines


def single_output_linear_rule(
    *,
    output_name: str,
    intercept: float,
    coefficients: np.ndarray,
    feature_names: list[str],
    config: SerializationConfig,
) -> list[str]:
    """
    Serialize one linear output equation.
    """
    lines = [f"{config.indent}{output_name} = {format_number(intercept, config)}"]

    for feature_index, coefficient in enumerate(coefficients):
        coefficient = float(coefficient)
        if abs(coefficient) <= config.zero_tolerance:
            continue

        sign = "+=" if coefficient >= 0 else "-="
        magnitude = format_number(abs(coefficient), config)
        lines.append(
            f"{config.indent}{output_name} {sign} {magnitude} * {feature_names[feature_index]}"
        )

    return lines


def logistic_regression_rule_lines(
    model,
    *,
    feature_names: list[str],
    config: SerializationConfig,
) -> list[str]:
    """
    Serialize logistic regression as linear class scores and an argmax rule.
    """
    coef = np.asarray(model.coef_, dtype=float)
    intercept = np.asarray(model.intercept_, dtype=float).reshape(-1)
    classes = list(model.classes_)

    lines: list[str] = []

    if len(classes) == 2 and coef.shape[0] == 1:
        class0 = format_label(classes[0])
        class1 = format_label(classes[1])
        lines.append(f"{config.indent}score[{class0}] = {format_number(0.0, config)}")
        lines.extend(
            single_output_linear_rule(
                output_name=f"score[{class1}]",
                intercept=float(intercept[0]),
                coefficients=coef[0],
                feature_names=feature_names,
                config=config,
            )
        )
        lines.append(f"{config.indent}return argmax(score)")
        return lines

    for class_index, label in enumerate(classes):
        label_text = format_label(label)
        lines.extend(
            single_output_linear_rule(
                output_name=f"score[{label_text}]",
                intercept=float(intercept[class_index]),
                coefficients=coef[class_index],
                feature_names=feature_names,
                config=config,
            )
        )

    lines.append(f"{config.indent}return argmax(score)")

    return lines
