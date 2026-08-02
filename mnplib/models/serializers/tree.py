"""
Canonical serializers for scikit-learn decision trees.
"""

from __future__ import annotations

import numpy as np

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .base import (
    SklearnSerializer,
    Task,
    class_token,
    format_number,
    require_fitted,
)

class DecisionTreeSerializer(SklearnSerializer):
    """
    Canonical serializer for decision-tree classifiers and regressors.
    """

    name = "decision_tree"
    supported_types = (DecisionTreeClassifier, DecisionTreeRegressor)

    def task(self, model) -> Task:
        """
        Return the task type of the decision tree.
        """
        if isinstance(model, DecisionTreeClassifier):
            return "classification"
        if isinstance(model, DecisionTreeRegressor):
            return "regression"

        raise TypeError(
            "Expected DecisionTreeClassifier or DecisionTreeRegressor. "
            f"Got {type(model).__name__} instead."
        )

    def subset(self, model) -> list[int]:
        """
        Return the feature indices used by internal split nodes.
        """
        require_fitted(model)

        used = np.asarray(model.tree_.feature, dtype=int)
        used = used[used >= 0]

        return sorted(int(j) for j in np.unique(used))

    def serialize(self, model, *, feature_names: list[str]) -> str:
        """
        Return a canonical string description of the decision tree.
        """
        require_fitted(model)

        task = self.task(model)

        lines = self._tree_rule_lines(
            model,
            node_id       = 0,
            depth         = 0,
            feature_names = feature_names,
            task          = task,
        )

        return "\n".join(lines) + "\n"

    def metadata(self, model, *, feature_names: list[str], subset: list[int]) -> dict:
        """
        Return structural tree metadata.
        """
        return {
            "n_nodes"   : int(model.tree_.node_count),
            "n_leaves"  : int(model.get_n_leaves()),
            "max_depth" : int(model.get_depth()),
        }

    def _tree_rule_lines(self, model, *, node_id: int, depth: int,
                         feature_names: list[str], task: Task) -> list[str]:
        """
        Recursively serialize one decision-tree node using indentation.

        Internal nodes are represented as nested if/else blocks.
        Leaf nodes are represented as return statements.
        """
        tree   = model.tree_
        indent = " "
        prefix = indent * depth
        left   = int(tree.children_left[node_id])
        right  = int(tree.children_right[node_id])

        # For a leaf node, scikit-learn stores both child references as the same
        # sentinel value, normally -1.
        if left == right:
            return [
                f"{prefix}return {self._leaf_value(model, node_id, task)}"
            ]

        feature_index = int(tree.feature[node_id])
        threshold     = format_number(float(tree.threshold[node_id]))
        feature_name  = feature_names[feature_index]

        lines = [
            f"{prefix}if {feature_name}<={threshold}:"
        ]

        lines.extend(
            self._tree_rule_lines(
                model,
                node_id=left,
                depth=depth + 1,
                feature_names=feature_names,
                task=task,
            )
        )

        lines.append(f"{prefix}else:")

        lines.extend(
            self._tree_rule_lines(
                model,
                node_id=right,
                depth=depth + 1,
                feature_names=feature_names,
                task=task,
            )
        )

        return lines

    def _leaf_value(self, model, node_id: int, task: Task) -> str:
        """
        Return the canonical prediction at a decision-tree leaf.
        """
        value = np.asarray(model.tree_.value[node_id])

        if task == "classification":
            if int(model.n_outputs_) == 1:
                class_index = int(np.argmax(value[0]))
                return class_token(class_index)

            labels = []
            for output_index in range(int(model.n_outputs_)):
                class_index = int(np.argmax(value[output_index]))
                labels.append(class_token(class_index))

            return "[" + ", ".join(labels) + "]"

        values = value.reshape(-1)

        if values.size == 1:
            return format_number(float(values[0]))

        return "[" + " ".join(format_number(float(v)) for v in values) + "]"
