"""
Canonical serializers for scikit-learn MLP neural networks.

Neural networks are supported as explicit, high-surfeit model descriptions. The
serializer exposes the fitted architecture, all non-zero weights, all biases, and
the activation conventions used by scikit-learn. This is intentionally verbose:
surfeit should reflect the descriptive cost of the fitted network.
"""

from __future__ import annotations

import numpy as np

from sklearn.neural_network import MLPClassifier, MLPRegressor

from ..artifacts import SerializationConfig
from .base import (
    SklearnSerializer,
    Task,
    canonical_header,
    format_label,
    format_number,
    require_fitted,
)


class MLPSerializer(SklearnSerializer):
    """
    Canonical serializer for ``MLPClassifier`` and ``MLPRegressor``.
    """

    name = "mlp_neural_network"
    support_level = "experimental"
    supported_types = (MLPClassifier, MLPRegressor)

    def task(self, model) -> Task:
        """
        Return the task type of the fitted MLP.
        """
        if isinstance(model, MLPClassifier):
            return "classification"
        if isinstance(model, MLPRegressor):
            return "regression"

        raise TypeError(f"Unsupported MLP type {type(model).__name__}.")

    def subset(self, model, *, config: SerializationConfig) -> list[int]:
        """
        Return input features connected to at least one first-layer unit.
        """
        require_fitted(model)

        first_layer = np.asarray(model.coefs_[0], dtype=float)
        used = np.any(np.abs(first_layer) > config.zero_tolerance, axis=1)

        return [int(j) for j in np.flatnonzero(used)]

    def serialize(
        self,
        model,
        *,
        feature_names: list[str],
        config: SerializationConfig,
    ) -> str:
        """
        Return a canonical string description of the fitted MLP.
        """
        require_fitted(model)

        task = self.task(model)
        subset = self.subset(model, config=config)
        n_parameters = self._n_nonzero_parameters(model, config)

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
                    f"{config.indent}hidden_layer_sizes = {self._hidden_layer_sizes(model)}",
                    f"{config.indent}activation = {repr(model.activation)}",
                    f"{config.indent}output_activation = {repr(model.out_activation_)}",
                    f"{config.indent}n_layers = {int(model.n_layers_)}",
                    f"{config.indent}n_outputs = {int(model.n_outputs_)}",
                    f"{config.indent}n_nonzero_parameters = {int(n_parameters)}",
                    f"{config.indent}n_iter = {int(getattr(model, 'n_iter_', 0))}",
                ]
            )
            if task == "classification":
                lines.append(
                    f"{config.indent}classes = {[format_label(label) for label in model.classes_]}"
                )

        lines.append("RULE")
        lines.extend(
            self._network_rule_lines(
                model,
                feature_names=feature_names,
                config=config,
            )
        )

        if task == "classification":
            lines.extend(self._classification_return_lines(model, config))
        else:
            lines.extend(self._regression_return_lines(model, config))

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
        Return neural-network metadata.
        """
        metadata = {
            "hidden_layer_sizes": self._hidden_layer_sizes(model),
            "activation": model.activation,
            "output_activation": model.out_activation_,
            "n_layers": int(model.n_layers_),
            "n_outputs": int(model.n_outputs_),
            "n_nonzero_parameters": int(self._n_nonzero_parameters(model, config)),
            "n_iter": int(getattr(model, "n_iter_", 0)),
        }

        if isinstance(model, MLPClassifier):
            metadata["n_classes"] = int(len(model.classes_))
            metadata["classes"] = [format_label(label) for label in model.classes_]

        return metadata


    @staticmethod
    def _hidden_layer_sizes(model) -> tuple[int, ...]:
        """
        Return hidden-layer sizes as a normalized tuple.
        """
        sizes = model.hidden_layer_sizes
        if isinstance(sizes, int):
            return (int(sizes),)
        return tuple(int(size) for size in sizes)

    @staticmethod
    def _n_nonzero_parameters(model, config: SerializationConfig) -> int:
        """
        Count non-zero weights and biases after applying zero tolerance.
        """
        total = 0

        for matrix in model.coefs_:
            total += int(np.sum(np.abs(np.asarray(matrix, dtype=float)) > config.zero_tolerance))

        for vector in model.intercepts_:
            total += int(np.sum(np.abs(np.asarray(vector, dtype=float)) > config.zero_tolerance))

        return total

    def _network_rule_lines(
        self,
        model,
        *,
        feature_names: list[str],
        config: SerializationConfig,
    ) -> list[str]:
        """
        Serialize all fitted layers as explicit weighted sums and activations.
        """
        source_names = list(feature_names)
        lines: list[str] = []

        last_layer_index = len(model.coefs_) - 1

        for layer_index, (weights, biases) in enumerate(zip(model.coefs_, model.intercepts_)):
            weights = np.asarray(weights, dtype=float)
            biases = np.asarray(biases, dtype=float).reshape(-1)

            is_output_layer = layer_index == last_layer_index
            target_prefix = "output" if is_output_layer else f"hidden_{layer_index}"

            next_names: list[str] = []

            for unit_index in range(weights.shape[1]):
                pre_name = f"pre_{target_prefix}_{unit_index}"
                unit_name = f"{target_prefix}_{unit_index}"
                next_names.append(unit_name)

                lines.append(
                    f"{config.indent}{pre_name} = {format_number(biases[unit_index], config)}"
                )

                for source_index, source_name in enumerate(source_names):
                    coefficient = float(weights[source_index, unit_index])
                    if abs(coefficient) <= config.zero_tolerance:
                        continue

                    sign = "+=" if coefficient >= 0 else "-="
                    magnitude = format_number(abs(coefficient), config)
                    lines.append(
                        f"{config.indent}{pre_name} {sign} {magnitude} * {source_name}"
                    )

                activation = model.out_activation_ if is_output_layer else model.activation
                lines.append(
                    f"{config.indent}{unit_name} = {activation}({pre_name})"
                )

            source_names = next_names

        return lines

    @staticmethod
    def _classification_return_lines(model, config: SerializationConfig) -> list[str]:
        """
        Serialize the MLPClassifier prediction convention.
        """
        classes = list(model.classes_)

        if len(classes) == 2 and int(model.n_outputs_) == 1:
            class0 = format_label(classes[0])
            class1 = format_label(classes[1])
            return [
                f"{config.indent}probability[{class1}] = output_0",
                f"{config.indent}probability[{class0}] = 1 - output_0",
                f"{config.indent}return {class1} if output_0 >= 0.5 else {class0}",
            ]

        score_items = ", ".join(
            f"{format_label(label)}: output_{i}"
            for i, label in enumerate(classes)
        )
        return [
            f"{config.indent}score = {{{score_items}}}",
            f"{config.indent}return argmax(score)",
        ]

    @staticmethod
    def _regression_return_lines(model, config: SerializationConfig) -> list[str]:
        """
        Serialize the MLPRegressor prediction convention.
        """
        n_outputs = int(model.n_outputs_)

        if n_outputs == 1:
            return [f"{config.indent}return output_0"]

        outputs = ", ".join(f"output_{i}" for i in range(n_outputs))
        return [f"{config.indent}return [{outputs}]"]
