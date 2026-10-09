"""Feed-forward network semantics with explicitly bounded layer operations."""

import numpy as np
from ..._types import ResolvedTask
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.utils.validation import check_is_fitted
from .base import SklearnSerializer
from ..language import Constant, Label, Binary, Conditional, Vector, Element, Dense, Classify


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
        return {"layer_sizes": [int(model.coefs_[0].shape[0])]
                + [int(weights.shape[1]) for weights in model.coefs_]}

    def serialize(self, model, *, features):
        classification = isinstance(model, MLPClassifier)
        if classification and getattr(model._label_binarizer, "y_type_", "") == "multilabel-indicator":
            raise ValueError("Multilabel MLP classification is not supported.")
        if classification and len(model.classes_) == 1:
            return Label(model.classes_[0])
        output = Vector(tuple(features))
        for index, (weights, bias) in enumerate(zip(model.coefs_, model.intercepts_)):
            last = index == len(model.coefs_) - 1
            activation = "identity" if last else model.activation
            output = Dense(
                output, tuple(tuple(Constant(float(x)) for x in row) for row in weights),
                tuple(Constant(float(x)) for x in bias), activation,
            )
        if classification:
            labels = tuple(Label(value) for value in model.classes_)
            if model.out_activation_ == "logistic" and model.n_outputs_ == 1:
                return Conditional(Binary(">", Element(output, 0), Constant(0)), labels[1], labels[0])
            if model.out_activation_ != "softmax":
                raise ValueError("Unsupported MLP classification output activation.")
            return Classify(output, labels)
        if model.out_activation_ != "identity":
            raise ValueError("Unsupported MLP regression output activation.")
        return Element(output, 0) if model.n_outputs_ == 1 else output
