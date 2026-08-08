"""
Canonical serializer for scikit-learn MLP neural networks.

MLP models are represented as explicit executable predictors. The generated
model string stores the fitted weight matrices and bias vectors as numeric
literals, then evaluates the network through a compact loop-based forward pass.

The serialized predictor intentionally uses a restricted Python style:

    def predict(x):
        ...
        return y

The model string uses positional input references x[i].
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from sklearn.neural_network import MLPClassifier, MLPRegressor

from .base import (
    SklearnSerializer,
    Task,
    format_number,
    nonzero_mask,
    require_fitted,
)


class MLPSerializer(SklearnSerializer):
    """
    Canonical serializer for MLPClassifier and MLPRegressor.

    The serializer exposes the fitted architecture, weights, and biases directly.
    Loops are used to represent the repeated layer computation compactly, while
    all learned numerical parameters remain explicit in the generated predictor.
    """

    name = "mlp_neural_network"
    supported_types = (MLPClassifier, MLPRegressor)

    def task(self, model) -> Task:
        """
        Return the task type of the fitted MLP estimator.
        """
        if isinstance(model, MLPClassifier):
            return "classification"

        if isinstance(model, MLPRegressor):
            return "regression"

        raise TypeError(
            "Expected MLPClassifier or MLPRegressor. "
            f"Got {type(model).__name__} instead."
        )

    def subset(self, model) -> list[int]:
        """
        Return local input features connected to the first hidden or output layer.

        The returned indices are expressed in the coordinate system used by the
        fitted estimator. When the estimator has been trained on a selected
        feature matrix, the adapter layer can map these local indices back to
        the original representation through feature_indices.
        """
        require_fitted(model)

        first_layer = np.asarray(model.coefs_[0], dtype=float)
        used = np.any(nonzero_mask(first_layer), axis=1)

        return [int(index) for index in np.flatnonzero(used)]

    def serialize(
        self,
        model,
        *,
        feature_names: list[str],
        feature_indices: Sequence[int] | None = None,
    ) -> str:
        """
        Return an executable simplified-Python description of the fitted MLP.

        Feature names are intentionally not included in the model string. The
        generated predictor operates on the original input vector through x[i]
        references. If feature_indices is provided, local estimator inputs are
        mapped back to their original coordinates.
        """
        require_fitted(model)

        del feature_names

        self._validate_supported_prediction_case(model)

        original_indices = self._original_feature_indices(
            model,
            feature_indices=feature_indices,
        )

        lines = self._prediction_function_lines(
            model,
            original_indices=original_indices,
        )

        return "\n".join(lines) + "\n"

    def _prediction_function_lines(
        self,
        model,
        *,
        original_indices: tuple[int, ...],
    ) -> list[str]:
        """
        Build the executable MLP predictor.

        The predictor stores all fitted layers as W and B, initializes the input
        activation vector a from x, and then evaluates each layer using nested
        loops. Hidden layers apply the fitted hidden activation. The output layer
        keeps raw output scores whenever that is sufficient for prediction.
        """
        indent = " "
        lines: list[str] = ["def predict(x):"]

        if self._needs_exp_constant(model):
            lines.append(f"{indent}E=2.718281828459045")

        lines.append(f"{indent}W={self._format_weight_layers(model)}")
        lines.append(f"{indent}B={self._format_bias_layers(model)}")
        lines.append(
            f"{indent}a={self._input_reference_vector(original_indices)}"
        )

        lines.append(f"{indent}for l in range(len(W)):")
        lines.append(f"{indent}{indent}h=[]")
        lines.append(f"{indent}{indent}for j in range(len(B[l])):")
        lines.append(f"{indent}{indent}{indent}z=B[l][j]")
        lines.append(f"{indent}{indent}{indent}for i in range(len(a)):")
        lines.append(f"{indent}{indent}{indent}{indent}z+=a[i]*W[l][i][j]")
        lines.append(f"{indent}{indent}{indent}if l<len(W)-1:")

        lines.extend(
            self._hidden_activation_lines(
                model.activation,
                base_indent=indent * 4,
            )
        )

        lines.append(f"{indent}{indent}{indent}else:")
        lines.append(f"{indent}{indent}{indent}{indent}h.append(z)")
        lines.append(f"{indent}{indent}a=h")

        if isinstance(model, MLPClassifier):
            lines.extend(self._classification_return_lines(model))
        else:
            lines.extend(self._regression_return_lines(model))

        return lines

    @staticmethod
    def _hidden_activation_lines(
        activation: str,
        *,
        base_indent: str,
    ) -> list[str]:
        """
        Return executable lines for the hidden-layer activation function.

        The generated predictor avoids external imports. Logistic and tanh are
        written with the Euler constant E defined inside predict(x) when needed.
        """
        if activation == "identity":
            return [f"{base_indent}h.append(z)"]

        if activation == "relu":
            return [
                f"{base_indent}if z>0:",
                f"{base_indent} h.append(z)",
                f"{base_indent}else:",
                f"{base_indent} h.append(0)",
            ]

        if activation == "logistic":
            return [
                f"{base_indent}if z>=0:",
                f"{base_indent} h.append(1/(1+E**(-z)))",
                f"{base_indent}else:",
                f"{base_indent} e=E**z",
                f"{base_indent} h.append(e/(1+e))",
            ]

        if activation == "tanh":
            return [
                f"{base_indent}if z>20:",
                f"{base_indent} h.append(1)",
                f"{base_indent}elif z<-20:",
                f"{base_indent} h.append(-1)",
                f"{base_indent}else:",
                f"{base_indent} e=E**(2*z)",
                f"{base_indent} h.append((e-1)/(e+1))",
            ]

        raise ValueError(
            "Unsupported MLP hidden activation "
            f"{activation!r}."
        )

    @staticmethod
    def _classification_return_lines(model) -> list[str]:
        """
        Return the executable classification decision rule.

        Binary MLP classifiers use a logistic output internally. Since the
        logistic function is monotone and the prediction threshold is 0.5, the
        class decision can be made from the raw output score: class 1 is selected
        exactly when the raw score is positive.

        Multiclass MLP classifiers use softmax internally. Since softmax
        preserves the order of logits, the predicted class is the index with the
        largest raw output score. Ties are resolved by keeping the first maximum,
        matching the usual argmax convention.
        """
        indent = " "

        classes = list(model.classes_)
        n_outputs = int(model.n_outputs_)

        if len(classes) == 2 and n_outputs == 1:
            return [
                f"{indent}if a[0]>0:",
                f"{indent}{indent}return 1",
                f"{indent}return 0",
            ]

        if n_outputs != len(classes):
            raise ValueError(
                "MLPClassifier serialization expects one output score per "
                "class for multiclass prediction."
            )

        lines = [
            f"{indent}best=0",
            f"{indent}best_s=a[0]",
            f"{indent}for j in range(1,len(a)):",
            f"{indent}{indent}if a[j]>best_s:",
            f"{indent}{indent}{indent}best=j",
            f"{indent}{indent}{indent}best_s=a[j]",
            f"{indent}return best",
        ]

        return lines

    @staticmethod
    def _regression_return_lines(model) -> list[str]:
        """
        Return the executable regression prediction rule.
        """
        indent = " "
        n_outputs = int(model.n_outputs_)

        if n_outputs == 1:
            return [f"{indent}return a[0]"]

        return [f"{indent}return a"]

    @staticmethod
    def _format_weight_layers(model) -> str:
        """
        Return all fitted weight matrices as compact nested Python lists.

        Each matrix keeps scikit-learn's orientation: rows are source units and
        columns are target units. The generated predictor therefore accesses a
        weight as W[l][i][j].
        """
        layers = []

        for matrix in model.coefs_:
            matrix = np.asarray(matrix, dtype=float)
            layers.append(MLPSerializer._format_matrix(matrix))

        return "[" + ",".join(layers) + "]"

    @staticmethod
    def _format_bias_layers(model) -> str:
        """
        Return all fitted bias vectors as compact nested Python lists.
        """
        vectors = []

        for vector in model.intercepts_:
            vector = np.asarray(vector, dtype=float).reshape(-1)
            vectors.append(MLPSerializer._format_vector(vector))

        return "[" + ",".join(vectors) + "]"

    @staticmethod
    def _format_matrix(matrix: np.ndarray) -> str:
        """
        Format one two-dimensional numeric array as a Python list literal.
        """
        rows = [
            MLPSerializer._format_vector(row)
            for row in matrix
        ]

        return "[" + ",".join(rows) + "]"

    @staticmethod
    def _format_vector(vector: np.ndarray) -> str:
        """
        Format one numeric vector as a Python list literal.
        """
        values = [
            MLPSerializer._format_parameter(float(value))
            for value in vector
        ]

        return "[" + ",".join(values) + "]"

    @staticmethod
    def _format_parameter(value: float) -> str:
        """
        Format one fitted numeric parameter canonically.

        Values considered zero by the library tolerance are serialized as the
        exact literal zero. This avoids model strings containing negative zero
        or insignificant numerical noise.
        """
        if not nonzero_mask(np.asarray([value], dtype=float))[0]:
            return format_number(0.0)

        return format_number(float(value))

    @staticmethod
    def _input_reference_vector(original_indices: tuple[int, ...]) -> str:
        """
        Return the input activation vector using original-coordinate references.
        """
        references = [
            f"x[{int(index)}]"
            for index in original_indices
        ]

        return "[" + ",".join(references) + "]"

    @staticmethod
    def _original_feature_indices(
        model,
        *,
        feature_indices: Sequence[int] | None,
    ) -> tuple[int, ...]:
        """
        Return original input coordinates for the fitted estimator.

        If the estimator was trained on a selected feature matrix,
        feature_indices maps local estimator columns back to the original input
        representation. If no mapping is provided, estimator coordinates are
        assumed to already be original coordinates.
        """
        n_features = int(model.n_features_in_)

        if feature_indices is None:
            return tuple(range(n_features))

        original_indices = tuple(int(index) for index in feature_indices)

        if len(original_indices) != n_features:
            raise ValueError(
                "feature_indices must contain one original feature index for "
                "each input column used by the fitted MLP estimator."
            )

        return original_indices

    @staticmethod
    def _needs_exp_constant(model) -> bool:
        """
        Return whether the generated predictor needs the Euler constant E.
        """
        return str(model.activation) in {"logistic", "tanh"}

    @staticmethod
    def _validate_supported_prediction_case(model) -> None:
        """
        Validate that the fitted estimator has a supported prediction structure.

        The serializer supports standard single-target classification and
        standard regression. Multilabel and multioutput classification require a
        different return convention and are intentionally rejected.
        """
        if isinstance(model, MLPClassifier):
            if isinstance(model.classes_, list):
                raise ValueError(
                    "Multilabel MLPClassifier serialization is not supported."
                )

            if model.out_activation_ not in {"logistic", "softmax"}:
                raise ValueError(
                    "Unsupported MLPClassifier output activation "
                    f"{model.out_activation_!r}."
                )

        if isinstance(model, MLPRegressor):
            if model.out_activation_ != "identity":
                raise ValueError(
                    "Unsupported MLPRegressor output activation "
                    f"{model.out_activation_!r}."
                )
