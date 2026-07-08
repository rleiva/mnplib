"""
Canonical serializers for compact linear support-vector models.

Only the explicitly linear estimators ``LinearSVC`` and ``LinearSVR`` are
supported here. Kernel SVMs are intentionally excluded from this serializer,
even when configured with a linear kernel, because their fitted libsvm
representation is naturally support-vector based and can become strongly
training-set dependent.

The canonical descriptions expose the fitted primal coefficients and intercepts,
which makes these estimators comparable to other compact parametric models in
the nescience model-adapter layer.
"""

from __future__ import annotations

import numpy as np

from sklearn.svm import LinearSVC, LinearSVR

from ..artifacts import SerializationConfig
from .base import (
    SklearnSerializer,
    Task,
    canonical_header,
    format_label,
    format_number,
    require_fitted,
)
from .linear import single_output_linear_rule


class LinearSVMSerializer(SklearnSerializer):
    """
    Canonical serializer for ``LinearSVC`` and ``LinearSVR``.
    """

    name = "linear_svm"
    support_level = "stable"
    supported_types = (LinearSVC, LinearSVR)

    def task(self, model) -> Task:
        """
        Return the task type of the fitted estimator.
        """
        if isinstance(model, LinearSVC):
            return "classification"
        if isinstance(model, LinearSVR):
            return "regression"

        raise TypeError(f"Unsupported linear SVM type {type(model).__name__}.")

    def subset(self, model, *, config: SerializationConfig) -> list[int]:
        """
        Return feature indices with non-zero fitted coefficients.
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
        Return a canonical string description of the fitted linear SVM.
        """
        require_fitted(model)

        task = self.task(model)
        subset = self.subset(model, config=config)

        lines = canonical_header(
            model_type=type(model).__name__,
            task=task,
            feature_names=[feature_names[j] for j in subset],
            config=config,
        )

        if config.include_metadata:
            lines.extend(
                [
                    "PARAMETERS",
                    f"{config.indent}n_nonzero_coefficients = {len(subset)}",
                    f"{config.indent}C = {format_number(float(model.C), config)}",
                ]
            )
            if hasattr(model, "epsilon"):
                lines.append(
                    f"{config.indent}epsilon = {format_number(float(model.epsilon), config)}"
                )
            if hasattr(model, "loss"):
                lines.append(f"{config.indent}loss = {repr(model.loss)}")
            if hasattr(model, "penalty"):
                lines.append(f"{config.indent}penalty = {repr(model.penalty)}")
            if hasattr(model, "dual"):
                lines.append(f"{config.indent}dual = {repr(model.dual)}")
            if task == "classification":
                lines.append(
                    f"{config.indent}classes = {[format_label(label) for label in model.classes_]}"
                )

        lines.append("RULE")
        if task == "classification":
            lines.extend(
                self._classification_rule_lines(
                    model,
                    feature_names=feature_names,
                    config=config,
                )
            )
        else:
            lines.extend(
                self._regression_rule_lines(
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
        Return compact linear-SVM metadata.
        """
        metadata = {
            "n_nonzero_coefficients": int(len(subset)),
            "C": float(model.C),
        }

        if isinstance(model, LinearSVC):
            metadata["n_classes"] = int(len(model.classes_))
            metadata["classes"] = [format_label(label) for label in model.classes_]
            metadata["loss"] = model.loss
            metadata["penalty"] = model.penalty
            metadata["dual"] = model.dual

        if isinstance(model, LinearSVR):
            metadata["epsilon"] = float(model.epsilon)
            metadata["loss"] = model.loss
            metadata["dual"] = model.dual

        return metadata

    @staticmethod
    def _regression_rule_lines(
        model,
        *,
        feature_names: list[str],
        config: SerializationConfig,
    ) -> list[str]:
        """
        Serialize ``LinearSVR`` as a single linear prediction rule.
        """
        coef = np.asarray(model.coef_, dtype=float).reshape(-1)
        intercept = float(np.asarray(model.intercept_, dtype=float).reshape(-1)[0])

        return (
            single_output_linear_rule(
                output_name="y",
                intercept=intercept,
                coefficients=coef,
                feature_names=feature_names,
                config=config,
            )
            + [f"{config.indent}return y"]
        )

    @staticmethod
    def _classification_rule_lines(
        model,
        *,
        feature_names: list[str],
        config: SerializationConfig,
    ) -> list[str]:
        """
        Serialize ``LinearSVC`` as linear class scores and an argmax rule.
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
