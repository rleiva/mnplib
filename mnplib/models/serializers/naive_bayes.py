"""Gaussian class scores using precomputed logarithmic constants."""

import numpy as np
from ..._types import ResolvedTask
from sklearn.naive_bayes import GaussianNB
from .base import SklearnSerializer, require_fitted
from ..language import Constant, Label, Binary, Vector, Classify


class NaiveBayesSerializer(SklearnSerializer):
    name = "naive_bayes"
    supported_types = (GaussianNB,)

    def task(self, model) -> ResolvedTask:
        return "classification"

    def subset(self, model):
        require_fitted(model)
        used = (np.ptp(model.theta_, axis=0) != 0) | (np.ptp(model.var_, axis=0) != 0)
        return np.flatnonzero(used).tolist()

    def serialize(self, model, *, features):
        scores = []
        for index, prior in enumerate(model.class_prior_):
            score = Constant(float(np.log(prior)))
            for column in self.subset(model):
                variance = float(model.var_[index, column])
                if variance <= 0:
                    raise ValueError("Gaussian variances must be positive.")
                score = Binary("+", score, Constant(float(-0.5 * np.log(2 * np.pi * variance))))
                residual = Binary("-", features[column], Constant(float(model.theta_[index, column])))
                penalty = Binary("/", Binary("**", residual, Constant(2)), Constant(2 * variance))
                score = Binary("-", score, penalty)
            scores.append(score)
        return Classify(Vector(tuple(scores)), tuple(Label(value) for value in model.classes_))
