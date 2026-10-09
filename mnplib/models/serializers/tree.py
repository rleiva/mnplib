"""Decision trees expressed as nested conditional predictions."""

import numpy as np
from ..._types import ResolvedTask
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted
from .base import SklearnSerializer
from ..language import Constant, Label, Binary, Conditional, Vector


class DecisionTreeSerializer(SklearnSerializer):
    name = "decision_tree"
    supported_types = (DecisionTreeClassifier, DecisionTreeRegressor)

    def task(self, model) -> ResolvedTask:
        return "classification" if isinstance(model, DecisionTreeClassifier) else "regression"

    def subset(self, model):
        check_is_fitted(model)
        return sorted(int(index) for index in np.unique(model.tree_.feature) if index >= 0)

    def metadata(self, model):
        return {"depth": model.get_depth(), "n_nodes": int(model.tree_.node_count),
                "n_leaves": model.get_n_leaves()}

    def serialize(self, model, *, features):
        tree = model.tree_

        def build(index):
            left, right = int(tree.children_left[index]), int(tree.children_right[index])
            if left == right:
                values = tree.value[index]
                if self.task(model) == "classification":
                    if model.n_outputs_ == 1:
                        return Label(model.classes_[np.argmax(values[0])])
                    return Vector(tuple(Label(labels[np.argmax(values[output])])
                                        for output, labels in enumerate(model.classes_)))
                values = values.reshape(-1)
                return (Constant(float(values[0])) if len(values) == 1 else
                        Vector(tuple(Constant(float(value)) for value in values)))
            feature = features[int(tree.feature[index])]
            condition = Binary("<=", feature, Constant(float(tree.threshold[index])))
            if tree.missing_go_to_left[index]:
                # Self-equality distinguishes observed values from NaN.
                condition = Conditional(Binary("==", feature, feature), condition, Constant(1))
            return Conditional(condition, build(left), build(right))

        return build(0)
