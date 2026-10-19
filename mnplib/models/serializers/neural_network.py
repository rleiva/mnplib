"""Feed-forward prediction rules using bounded sums and vector construction."""

import numpy as np
from ..._types import ResolvedTask
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.utils.validation import check_is_fitted
from .base import SklearnSerializer, score_conditionals
from ..language import (
    Constant, Label, Binary, Conditional, Vector, Element,
    Index, Sum, Comprehension,
)


def _relu_expression(score):
    """Express ReLU through a comparison and conditional arithmetic."""
    zero = Constant(0)
    return Conditional(Binary(">", score, zero), score, zero)


class MLPSerializer(SklearnSerializer):
    name = "mlp_neural_network"
    supported_types = (MLPClassifier, MLPRegressor)

    def task(self, model) -> ResolvedTask:
        return "classification" if isinstance(model, MLPClassifier) else "regression"

    def subset(self, model):
        check_is_fitted(model)
        if isinstance(model, MLPClassifier) and len(model.classes_) == 1:
            return []
        return np.flatnonzero(np.any(model.coefs_[0] != 0, axis=1)).tolist()

    def metadata(self, model):
        hidden = [int(weights.shape[1]) for weights in model.coefs_[:-1]]
        metadata = {
            "hidden_layer_sizes": hidden,
            "activations": [str(model.activation)] * len(hidden) + [str(model.out_activation_)],
        }
        if isinstance(model, MLPClassifier):
            metadata["n_classes"] = int(len(model.classes_))
        return metadata

    def serialize(self, model, *, features):
        if model.activation != "relu":
            raise ValueError(
                f"Unsupported MLP hidden activation: {model.activation!r}. "
                "mnplib supports only ReLU hidden activation for MLP models."
            )
        classification = isinstance(model, MLPClassifier)
        if classification and getattr(model._label_binarizer, "y_type_", "") == "multilabel-indicator":
            raise ValueError("Multilabel MLP classification is not supported.")
        if classification and len(model.classes_) == 1:
            return Label(model.classes_[0])
        output = Vector(tuple(features))
        for index, (weights, bias) in enumerate(zip(model.coefs_, model.intercepts_)):
            last = index == len(model.coefs_) - 1
            input_index, output_index = Index(f"i{index}"), Index(f"j{index}")
            parameters = Vector(tuple(
                Vector(tuple(Constant(float(value)) for value in row))
                for row in weights.T
            ))
            biases = Vector(tuple(Constant(float(value)) for value in bias))
            product = Binary("*", Element(Element(parameters, output_index), input_index),
                             Element(output, input_index))
            score = Binary("+", Element(biases, output_index),
                           Sum(input_index.name, weights.shape[0], product))
            output = Comprehension(output_index.name, weights.shape[1],
                                   score if last else _relu_expression(score))
        if classification:
            labels = tuple(Label(value) for value in model.classes_)
            if model.out_activation_ == "logistic" and model.n_outputs_ == 1:
                return Conditional(Binary(">", Element(output, 0), Constant(0)), labels[1], labels[0])
            if model.out_activation_ != "softmax":
                raise ValueError("Unsupported MLP classification output activation.")
            scores = tuple(Element(output, index) for index in range(len(labels)))
            return score_conditionals(scores, labels)
        if model.out_activation_ != "identity":
            raise ValueError("Unsupported MLP regression output activation.")
        return Element(output, 0) if model.n_outputs_ == 1 else output
