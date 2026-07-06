"""
Canonical serializers for scikit-learn tree ensembles.

The simplified nescience metrics require explicit artifacts:

    subset, predictions, model_string

This module provides canonical model descriptions for the main tree-ensemble
families in scikit-learn. The descriptions intentionally expose the fitted
structure rather than relying on ``repr(model)``, because surfeit should be
sensitive to the descriptive complexity of the trained model.

Supported estimators
--------------------
- RandomForestRegressor
- RandomForestClassifier
- ExtraTreesRegressor
- ExtraTreesClassifier
- GradientBoostingRegressor
- GradientBoostingClassifier
- HistGradientBoostingRegressor
- HistGradientBoostingClassifier
"""

from __future__ import annotations

import numpy as np

from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)

from ..artifacts import SerializationConfig
from .base import (
    SklearnSerializer,
    Task,
    canonical_header,
    format_label,
    format_number,
    require_fitted,
)


_RANDOM_FOREST_TYPES = (
    RandomForestRegressor,
    RandomForestClassifier,
    ExtraTreesRegressor,
    ExtraTreesClassifier,
)

_GRADIENT_BOOSTING_TYPES = (
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)

_HIST_GRADIENT_BOOSTING_TYPES = (
    HistGradientBoostingRegressor,
    HistGradientBoostingClassifier,
)


class TreeEnsembleSerializer(SklearnSerializer):
    """
    Canonical serializer for scikit-learn tree ensembles.

    The serializer uses the union of features used by all constituent trees as
    the selected feature subset. This is the representation consumed by the
    nescience metrics.
    """

    name = "tree_ensemble"
    support_level = "beta"
    supported_types = (
        RandomForestRegressor,
        RandomForestClassifier,
        ExtraTreesRegressor,
        ExtraTreesClassifier,
        GradientBoostingRegressor,
        GradientBoostingClassifier,
        HistGradientBoostingRegressor,
        HistGradientBoostingClassifier,
    )

    def task(self, model) -> Task:
        """
        Return the task type of the ensemble.
        """
        if isinstance(
            model,
            (
                RandomForestClassifier,
                ExtraTreesClassifier,
                GradientBoostingClassifier,
                HistGradientBoostingClassifier,
            ),
        ):
            return "classification"

        if isinstance(
            model,
            (
                RandomForestRegressor,
                ExtraTreesRegressor,
                GradientBoostingRegressor,
                HistGradientBoostingRegressor,
            ),
        ):
            return "regression"

        raise TypeError(f"Unsupported ensemble type {type(model).__name__}.")

    def subset(self, model, *, config: SerializationConfig) -> list[int]:
        """
        Return the union of features used by all fitted trees.
        """
        require_fitted(model)

        used: set[int] = set()

        if isinstance(model, _RANDOM_FOREST_TYPES):
            for estimator in model.estimators_:
                used.update(self._sklearn_tree_features(estimator))

        elif isinstance(model, _GRADIENT_BOOSTING_TYPES):
            for estimator in np.asarray(model.estimators_, dtype=object).ravel():
                used.update(self._sklearn_tree_features(estimator))

        elif isinstance(model, _HIST_GRADIENT_BOOSTING_TYPES):
            for predictor_group in model._predictors:
                for predictor in predictor_group:
                    used.update(self._hist_tree_features(predictor))

        return sorted(used)

    def serialize(
        self,
        model,
        *,
        feature_names: list[str],
        config: SerializationConfig,
    ) -> str:
        """
        Return a canonical string description of the fitted ensemble.
        """
        require_fitted(model)

        if isinstance(model, _RANDOM_FOREST_TYPES):
            return self._serialize_forest(model, feature_names=feature_names, config=config)

        if isinstance(model, _GRADIENT_BOOSTING_TYPES):
            return self._serialize_gradient_boosting(
                model,
                feature_names=feature_names,
                config=config,
            )

        if isinstance(model, _HIST_GRADIENT_BOOSTING_TYPES):
            return self._serialize_hist_gradient_boosting(
                model,
                feature_names=feature_names,
                config=config,
            )

        raise TypeError(f"Unsupported ensemble type {type(model).__name__}.")

    def metadata(
        self,
        model,
        *,
        feature_names: list[str],
        subset: list[int],
        config: SerializationConfig,
    ) -> dict:
        """
        Return compact structural metadata for reporting.
        """
        if isinstance(model, _RANDOM_FOREST_TYPES):
            trees = list(model.estimators_)
            return {
                "ensemble_family": self._family_name(model),
                "n_estimators": int(len(trees)),
                "total_nodes": int(sum(tree.tree_.node_count for tree in trees)),
                "total_leaves": int(sum(tree.get_n_leaves() for tree in trees)),
                "max_tree_depth": int(max((tree.get_depth() for tree in trees), default=0)),
            }

        if isinstance(model, _GRADIENT_BOOSTING_TYPES):
            trees = list(np.asarray(model.estimators_, dtype=object).ravel())
            return {
                "ensemble_family": self._family_name(model),
                "n_estimators": int(getattr(model, "n_estimators_", len(trees))),
                "n_boosting_trees": int(len(trees)),
                "learning_rate": float(model.learning_rate),
                "total_nodes": int(sum(tree.tree_.node_count for tree in trees)),
                "total_leaves": int(sum(tree.get_n_leaves() for tree in trees)),
                "max_tree_depth": int(max((tree.get_depth() for tree in trees), default=0)),
            }

        if isinstance(model, _HIST_GRADIENT_BOOSTING_TYPES):
            predictors = [p for group in model._predictors for p in group]
            return {
                "ensemble_family": self._family_name(model),
                "n_iter": int(getattr(model, "n_iter_", len(model._predictors))),
                "n_trees_per_iteration": int(getattr(model, "n_trees_per_iteration_", 1)),
                "n_boosting_trees": int(len(predictors)),
                "learning_rate": float(model.learning_rate),
                "total_nodes": int(sum(len(p.nodes) for p in predictors)),
                "total_leaves": int(sum(self._hist_leaf_count(p) for p in predictors)),
                "max_tree_depth": int(max((p.get_max_depth() for p in predictors), default=0)),
            }

        return {}

    # ------------------------------------------------------------------
    # Forests and extra-trees
    # ------------------------------------------------------------------

    def _serialize_forest(
        self,
        model,
        *,
        feature_names: list[str],
        config: SerializationConfig,
    ) -> str:
        """
        Serialize a random forest or extra-trees ensemble.
        """
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
                    f"{config.indent}ensemble_family = {self._family_name(model)}",
                    f"{config.indent}n_estimators = {int(len(model.estimators_))}",
                    f"{config.indent}criterion = {getattr(model, 'criterion', None)!r}",
                    f"{config.indent}bootstrap = {getattr(model, 'bootstrap', None)!r}",
                ]
            )

        lines.append("RULE")
        lines.append(f"{config.indent}predictions = []")

        for index, tree in enumerate(model.estimators_):
            lines.append(f"{config.indent}TREE {index}")
            lines.extend(
                self._sklearn_tree_rule_lines(
                    tree,
                    node_id=0,
                    depth=2,
                    feature_names=feature_names,
                    task=task,
                    config=config,
                    classifier_labels=getattr(model, "classes_", None),
                    probability_leaf=task == "classification",
                )
            )

        if task == "classification":
            lines.append(
                f"{config.indent}return class_with_highest_mean_tree_probability"
            )
        else:
            lines.append(f"{config.indent}return mean_tree_prediction")

        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Classic gradient boosting
    # ------------------------------------------------------------------

    def _serialize_gradient_boosting(
        self,
        model,
        *,
        feature_names: list[str],
        config: SerializationConfig,
    ) -> str:
        """
        Serialize a classic gradient-boosting ensemble.
        """
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
                    f"{config.indent}ensemble_family = {self._family_name(model)}",
                    f"{config.indent}n_estimators = {int(getattr(model, 'n_estimators_', model.n_estimators))}",
                    f"{config.indent}learning_rate = {format_number(float(model.learning_rate), config)}",
                    f"{config.indent}loss = {getattr(model, 'loss', None)!r}",
                ]
            )

        lines.append("RULE")
        lines.append(f"{config.indent}raw_prediction = initial_prediction")

        estimators = np.asarray(model.estimators_, dtype=object)

        for stage_index in range(estimators.shape[0]):
            for output_index in range(estimators.shape[1]):
                tree = estimators[stage_index, output_index]
                if estimators.shape[1] == 1:
                    lines.append(f"{config.indent}BOOSTING_STAGE {stage_index}")
                else:
                    label = self._output_label(model, output_index)
                    lines.append(
                        f"{config.indent}BOOSTING_STAGE {stage_index} OUTPUT {label}"
                    )

                lines.extend(
                    self._sklearn_tree_rule_lines(
                        tree,
                        node_id=0,
                        depth=2,
                        feature_names=feature_names,
                        task="regression",
                        config=config,
                        classifier_labels=None,
                        probability_leaf=False,
                    )
                )

        if task == "classification":
            lines.append(f"{config.indent}return inverse_link_highest_score_class")
        else:
            lines.append(f"{config.indent}return raw_prediction")

        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Histogram gradient boosting
    # ------------------------------------------------------------------

    def _serialize_hist_gradient_boosting(
        self,
        model,
        *,
        feature_names: list[str],
        config: SerializationConfig,
    ) -> str:
        """
        Serialize a histogram-gradient-boosting ensemble.

        Scikit-learn stores histogram-gradient-boosting trees as
        ``TreePredictor`` objects. Their node arrays are used here to expose the
        fitted split structure in a canonical string.
        """
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
                    f"{config.indent}ensemble_family = {self._family_name(model)}",
                    f"{config.indent}n_iter = {int(getattr(model, 'n_iter_', len(model._predictors)))}",
                    f"{config.indent}n_trees_per_iteration = {int(getattr(model, 'n_trees_per_iteration_', 1))}",
                    f"{config.indent}learning_rate = {format_number(float(model.learning_rate), config)}",
                    f"{config.indent}loss = {getattr(model, 'loss', None)!r}",
                ]
            )

        lines.append("RULE")
        lines.append(f"{config.indent}raw_prediction = baseline_prediction")

        for stage_index, predictor_group in enumerate(model._predictors):
            for output_index, predictor in enumerate(predictor_group):
                if len(predictor_group) == 1:
                    lines.append(f"{config.indent}HIST_BOOSTING_STAGE {stage_index}")
                else:
                    label = self._output_label(model, output_index)
                    lines.append(
                        f"{config.indent}HIST_BOOSTING_STAGE {stage_index} OUTPUT {label}"
                    )

                lines.extend(
                    self._hist_tree_rule_lines(
                        predictor,
                        node_id=0,
                        depth=2,
                        feature_names=feature_names,
                        config=config,
                    )
                )

        if task == "classification":
            lines.append(f"{config.indent}return inverse_link_highest_score_class")
        else:
            lines.append(f"{config.indent}return raw_prediction")

        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Tree formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sklearn_tree_features(tree) -> set[int]:
        """
        Return split features used by a fitted scikit-learn tree estimator.
        """
        features = np.asarray(tree.tree_.feature, dtype=int)
        return {int(j) for j in features[features >= 0]}

    @staticmethod
    def _hist_tree_features(predictor) -> set[int]:
        """
        Return split features used by a fitted histogram tree predictor.
        """
        nodes = predictor.nodes
        result = set()
        for node in nodes:
            if not bool(node["is_leaf"]):
                result.add(int(node["feature_idx"]))
        return result

    @staticmethod
    def _hist_leaf_count(predictor) -> int:
        """
        Return the number of leaves in a histogram tree predictor.
        """
        return int(np.sum(np.asarray(predictor.nodes["is_leaf"], dtype=bool)))

    def _sklearn_tree_rule_lines(
        self,
        tree,
        *,
        node_id: int,
        depth: int,
        feature_names: list[str],
        task: Task,
        config: SerializationConfig,
        classifier_labels,
        probability_leaf: bool,
    ) -> list[str]:
        """
        Recursively serialize one fitted scikit-learn tree.
        """
        sklearn_tree = tree.tree_
        indent = config.indent * depth

        left = int(sklearn_tree.children_left[node_id])
        right = int(sklearn_tree.children_right[node_id])

        if left == right:
            leaf = self._sklearn_leaf_value(
                tree,
                node_id=node_id,
                task=task,
                config=config,
                classifier_labels=classifier_labels,
                probability_leaf=probability_leaf,
            )
            return [f"{indent}return {leaf}"]

        feature_index = int(sklearn_tree.feature[node_id])
        threshold = format_number(float(sklearn_tree.threshold[node_id]), config)
        feature_name = feature_names[feature_index]

        lines = [f"{indent}if {feature_name} <= {threshold}:"]
        lines.extend(
            self._sklearn_tree_rule_lines(
                tree,
                node_id=left,
                depth=depth + 1,
                feature_names=feature_names,
                task=task,
                config=config,
                classifier_labels=classifier_labels,
                probability_leaf=probability_leaf,
            )
        )
        lines.append(f"{indent}else:")
        lines.extend(
            self._sklearn_tree_rule_lines(
                tree,
                node_id=right,
                depth=depth + 1,
                feature_names=feature_names,
                task=task,
                config=config,
                classifier_labels=classifier_labels,
                probability_leaf=probability_leaf,
            )
        )

        return lines

    def _sklearn_leaf_value(
        self,
        tree,
        *,
        node_id: int,
        task: Task,
        config: SerializationConfig,
        classifier_labels,
        probability_leaf: bool,
    ) -> str:
        """
        Return the canonical value of a scikit-learn tree leaf.
        """
        value = np.asarray(tree.tree_.value[node_id])

        if task == "classification":
            labels = np.asarray(classifier_labels if classifier_labels is not None else tree.classes_)

            if labels.ndim == 1:
                counts = np.asarray(value[0], dtype=float)
                total = float(np.sum(counts))
                probabilities = counts / total if total > 0 else np.zeros_like(counts)

                if probability_leaf:
                    pairs = [
                        f"{format_label(labels[i])}: {format_number(float(probabilities[i]), config)}"
                        for i in range(len(labels))
                    ]
                    return "{" + ", ".join(pairs) + "}"

                return format_label(labels[int(np.argmax(counts))])

            outputs = []
            for output_index in range(labels.shape[0]):
                counts = np.asarray(value[output_index], dtype=float)
                outputs.append(format_label(labels[output_index][int(np.argmax(counts))]))

            return "[" + ", ".join(outputs) + "]"

        values = value.reshape(-1)

        if values.size == 1:
            return format_number(float(values[0]), config)

        return "[" + ", ".join(format_number(float(v), config) for v in values) + "]"

    def _hist_tree_rule_lines(
        self,
        predictor,
        *,
        node_id: int,
        depth: int,
        feature_names: list[str],
        config: SerializationConfig,
    ) -> list[str]:
        """
        Recursively serialize one histogram-gradient-boosting tree predictor.
        """
        nodes = predictor.nodes
        node = nodes[node_id]
        indent = config.indent * depth

        if bool(node["is_leaf"]):
            return [f"{indent}return {format_number(float(node['value']), config)}"]

        feature_index = int(node["feature_idx"])
        threshold = format_number(float(node["num_threshold"]), config)
        feature_name = feature_names[feature_index]
        missing_direction = "left" if bool(node["missing_go_to_left"]) else "right"

        lines = [
            f"{indent}if {feature_name} <= {threshold}:",
        ]
        lines.extend(
            self._hist_tree_rule_lines(
                predictor,
                node_id=int(node["left"]),
                depth=depth + 1,
                feature_names=feature_names,
                config=config,
            )
        )
        lines.append(f"{indent}else:")
        lines.extend(
            self._hist_tree_rule_lines(
                predictor,
                node_id=int(node["right"]),
                depth=depth + 1,
                feature_names=feature_names,
                config=config,
            )
        )
        lines.append(f"{indent}missing_values_go_{missing_direction}")

        return lines

    @staticmethod
    def _family_name(model) -> str:
        """
        Return the broad ensemble family name.
        """
        if isinstance(model, (RandomForestRegressor, RandomForestClassifier)):
            return "random_forest"
        if isinstance(model, (ExtraTreesRegressor, ExtraTreesClassifier)):
            return "extra_trees"
        if isinstance(model, (GradientBoostingRegressor, GradientBoostingClassifier)):
            return "gradient_boosting"
        if isinstance(model, (HistGradientBoostingRegressor, HistGradientBoostingClassifier)):
            return "hist_gradient_boosting"
        return "unknown_ensemble"

    @staticmethod
    def _output_label(model, output_index: int) -> str:
        """
        Return the canonical output label for a boosting tree.
        """
        if hasattr(model, "classes_"):
            classes = np.asarray(model.classes_)
            if output_index < len(classes):
                return format_label(classes[output_index])
        return str(output_index)
