"""
Canonical serializers for scikit-learn decision trees.
"""

from __future__ import annotations

import numpy as np

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ..artifacts import SerializationConfig
from .base import (
    SklearnSerializer,
    Task,
    canonical_header,
    format_label,
    format_number,
    require_fitted,
)

class DecisionTreeSerializer(SklearnSerializer):
    """
    Canonical serializer for decision-tree classifiers and regressors.
    """

    name = "decision_tree"
    support_level = "stable"
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

    def subset(self, model, *, config: SerializationConfig) -> list[int]:
        """
        Return the feature indices used by internal split nodes.
        """
        require_fitted(model)

        used = np.asarray(model.tree_.feature, dtype=int)
        used = used[used >= 0]

        return sorted(int(j) for j in np.unique(used))

    def serialize(
        self,
        model,
        *,
        feature_names: list[str],
        config: SerializationConfig,
    ) -> str:
        """
        Return a canonical string description of the decision tree.
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
                    f"{config.indent}n_nodes = {int(model.tree_.node_count)}",
                    f"{config.indent}n_leaves = {int(model.get_n_leaves())}",
                    f"{config.indent}max_depth = {int(model.get_depth())}",
                ]
            )

        lines.append("RULE")
        lines.extend(
            self._tree_rule_lines(
                model,
                node_id=0,
                depth=1,
                feature_names=feature_names,
                task=task,
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
        Return structural tree metadata.
        """
        return {
            "n_nodes": int(model.tree_.node_count),
            "n_leaves": int(model.get_n_leaves()),
            "max_depth": int(model.get_depth()),
        }

    def _tree_rule_lines(
        self,
        model,
        *,
        node_id: int,
        depth: int,
        feature_names: list[str],
        task: Task,
        config: SerializationConfig,
    ) -> list[str]:
        """
        Recursively serialize one decision-tree node.
        """
        tree = model.tree_
        indent = config.indent * depth

        left = int(tree.children_left[node_id])
        right = int(tree.children_right[node_id])

        if left == right:
            return [f"{indent}return {self._leaf_value(model, node_id, task, config)}"]

        feature_index = int(tree.feature[node_id])
        threshold = format_number(float(tree.threshold[node_id]), config)
        feature_name = feature_names[feature_index]

        lines = [f"{indent}if {feature_name} <= {threshold}:"]
        lines.extend(
            self._tree_rule_lines(
                model,
                node_id=left,
                depth=depth + 1,
                feature_names=feature_names,
                task=task,
                config=config,
            )
        )
        lines.append(f"{indent}else:")
        lines.extend(
            self._tree_rule_lines(
                model,
                node_id=right,
                depth=depth + 1,
                feature_names=feature_names,
                task=task,
                config=config,
            )
        )

        return lines

    def _leaf_value(
        self,
        model,
        node_id: int,
        task: Task,
        config: SerializationConfig,
    ) -> str:
        """
        Return the canonical prediction at a decision-tree leaf.
        """
        value = np.asarray(model.tree_.value[node_id])

        if task == "classification":
            if int(model.n_outputs_) == 1:
                class_index = int(np.argmax(value[0]))
                return format_label(model.classes_[class_index])

            labels = []
            for output_index in range(int(model.n_outputs_)):
                class_index = int(np.argmax(value[output_index]))
                labels.append(format_label(model.classes_[output_index][class_index]))

            return "[" + ", ".join(labels) + "]"

        values = value.reshape(-1)

        if values.size == 1:
            return format_number(float(values[0]), config)

        return "[" + ", ".join(format_number(float(v), config) for v in values) + "]"
