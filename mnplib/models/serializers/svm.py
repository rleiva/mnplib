"""Linear SVM rules represented by arithmetic and class selection."""

from sklearn.svm import LinearSVC, LinearSVR
from ..._types import ResolvedTask
from .linear import LinearModelSerializer, linear_outputs, classification_rule


class LinearSVMSerializer(LinearModelSerializer):
    name = "linear_svm"
    supported_types = (LinearSVC, LinearSVR)

    def task(self, model) -> ResolvedTask:
        return "classification" if isinstance(model, LinearSVC) else "regression"

    def serialize(self, model, *, features):
        if isinstance(model, LinearSVC):
            return classification_rule(model, features)
        return linear_outputs(model, features)
