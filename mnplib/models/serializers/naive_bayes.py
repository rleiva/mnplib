"""
Canonical serializers for scikit-learn Naive Bayes classifiers.

Naive Bayes models are generally compact descriptions: they consist of class
priors and feature-conditional likelihood parameters. The serializers below
expose those fitted quantities explicitly so that surfeit reflects the actual size of
the probabilistic description.
"""

from __future__ import annotations

import numpy as np

from sklearn.naive_bayes import BernoulliNB, CategoricalNB, GaussianNB, MultinomialNB

from ..artifacts import SerializationConfig
from .base import (
    SklearnSerializer,
    Task,
    canonical_header,
    format_label,
    format_number,
    require_fitted,
)


class NaiveBayesSerializer(SklearnSerializer):
    """
    Canonical serializer for scikit-learn Naive Bayes classifiers.
    """

    name = "naive_bayes"
    support_level = "stable"
    supported_types = (GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB)

    def task(self, model) -> Task:
        """
        Return the task type of Naive Bayes classifiers.
        """
        return "classification"

    def subset(self, model, *, config: SerializationConfig) -> list[int]:
        """
        Return features whose likelihood parameters differ across classes.
        """
        require_fitted(model)

        if isinstance(model, GaussianNB):
            theta = np.asarray(model.theta_, dtype=float)
            var = np.asarray(model.var_, dtype=float)
            used = (
                np.ptp(theta, axis=0) > config.zero_tolerance
            ) | (
                np.ptp(var, axis=0) > config.zero_tolerance
            )
            return [int(j) for j in np.flatnonzero(used)]

        if isinstance(model, (MultinomialNB, BernoulliNB)):
            log_prob = np.asarray(model.feature_log_prob_, dtype=float)
            used = np.ptp(log_prob, axis=0) > config.zero_tolerance
            return [int(j) for j in np.flatnonzero(used)]

        if isinstance(model, CategoricalNB):
            used = []
            for feature_index, log_prob in enumerate(model.feature_log_prob_):
                if np.max(np.ptp(np.asarray(log_prob, dtype=float), axis=0)) > config.zero_tolerance:
                    used.append(int(feature_index))
            return used

        raise TypeError(f"Unsupported Naive Bayes type {type(model).__name__}.")

    def serialize(
        self,
        model,
        *,
        feature_names: list[str],
        config: SerializationConfig,
    ) -> str:
        """
        Return a canonical string description of the fitted Naive Bayes model.
        """
        require_fitted(model)

        subset = self.subset(model, config=config)

        lines = canonical_header(
            model_type=type(model).__name__,
            task="classification",
            feature_names=[feature_names[j] for j in subset],
            config=config,
        )

        if config.include_metadata:
            lines.extend(
                [
                    "PARAMETERS",
                    f"{config.indent}classes = {[format_label(label) for label in model.classes_]}",
                    f"{config.indent}n_features_in_use = {len(subset)}",
                ]
            )
            if hasattr(model, "alpha"):
                lines.append(
                    f"{config.indent}alpha = {format_number(float(model.alpha), config)}"
                )
            if hasattr(model, "var_smoothing"):
                lines.append(
                    f"{config.indent}var_smoothing = {format_number(float(model.var_smoothing), config)}"
                )

        lines.append("RULE")

        if isinstance(model, GaussianNB):
            lines.extend(self._gaussian_rule_lines(model, feature_names, subset, config))
        elif isinstance(model, MultinomialNB):
            lines.extend(self._multinomial_rule_lines(model, feature_names, subset, config))
        elif isinstance(model, BernoulliNB):
            lines.extend(self._bernoulli_rule_lines(model, feature_names, subset, config))
        elif isinstance(model, CategoricalNB):
            lines.extend(self._categorical_rule_lines(model, feature_names, subset, config))
        else:
            raise TypeError(f"Unsupported Naive Bayes type {type(model).__name__}.")

        lines.append(f"{config.indent}return argmax(score)")

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
        Return Naive Bayes metadata.
        """
        metadata = {
            "n_classes": int(len(model.classes_)),
            "classes": [format_label(label) for label in model.classes_],
            "n_features_in_use": int(len(subset)),
        }

        if isinstance(model, GaussianNB):
            metadata["likelihood"] = "gaussian"
            metadata["epsilon"] = float(getattr(model, "epsilon_", 0.0))
        elif isinstance(model, MultinomialNB):
            metadata["likelihood"] = "multinomial"
        elif isinstance(model, BernoulliNB):
            metadata["likelihood"] = "bernoulli"
        elif isinstance(model, CategoricalNB):
            metadata["likelihood"] = "categorical"
            metadata["n_categories_per_feature"] = [
                int(np.asarray(log_prob).shape[1])
                for log_prob in model.feature_log_prob_
            ]

        if hasattr(model, "alpha"):
            metadata["alpha"] = float(model.alpha)
        if hasattr(model, "var_smoothing"):
            metadata["var_smoothing"] = float(model.var_smoothing)

        return metadata

    @staticmethod
    def _class_log_prior(model) -> np.ndarray:
        """
        Return class log-priors for all supported Naive Bayes variants.
        """
        if hasattr(model, "class_log_prior_"):
            return np.asarray(model.class_log_prior_, dtype=float)

        if hasattr(model, "class_prior_"):
            priors = np.asarray(model.class_prior_, dtype=float)
            return np.log(np.maximum(priors, np.finfo(float).tiny))

        raise AttributeError("Naive Bayes model does not expose class priors.")

    def _initial_score_lines(self, model, config: SerializationConfig) -> list[str]:
        """
        Serialize class-prior initialization.
        """
        class_log_prior = self._class_log_prior(model)
        lines = []

        for class_index, label in enumerate(model.classes_):
            label_text = format_label(label)
            lines.append(
                f"{config.indent}score[{label_text}] = {format_number(class_log_prior[class_index], config)}"
            )

        return lines

    def _gaussian_rule_lines(
        self,
        model,
        feature_names: list[str],
        subset: list[int],
        config: SerializationConfig,
    ) -> list[str]:
        """
        Serialize Gaussian class-conditional likelihoods.
        """
        theta = np.asarray(model.theta_, dtype=float)
        var = np.asarray(model.var_, dtype=float)

        lines = self._initial_score_lines(model, config)

        for class_index, label in enumerate(model.classes_):
            label_text = format_label(label)
            for feature_index in subset:
                lines.append(
                    "{}score[{}] += gaussian_log_pdf({}, mean={}, variance={})".format(
                        config.indent,
                        label_text,
                        feature_names[feature_index],
                        format_number(theta[class_index, feature_index], config),
                        format_number(var[class_index, feature_index], config),
                    )
                )

        return lines

    def _multinomial_rule_lines(
        self,
        model,
        feature_names: list[str],
        subset: list[int],
        config: SerializationConfig,
    ) -> list[str]:
        """
        Serialize MultinomialNB likelihood contributions.
        """
        feature_log_prob = np.asarray(model.feature_log_prob_, dtype=float)
        lines = self._initial_score_lines(model, config)

        for class_index, label in enumerate(model.classes_):
            label_text = format_label(label)
            for feature_index in subset:
                lines.append(
                    "{}score[{}] += {} * {}".format(
                        config.indent,
                        label_text,
                        feature_names[feature_index],
                        format_number(feature_log_prob[class_index, feature_index], config),
                    )
                )

        return lines

    def _bernoulli_rule_lines(
        self,
        model,
        feature_names: list[str],
        subset: list[int],
        config: SerializationConfig,
    ) -> list[str]:
        """
        Serialize BernoulliNB likelihood contributions.
        """
        feature_log_prob = np.asarray(model.feature_log_prob_, dtype=float)
        feature_prob = np.exp(feature_log_prob)
        complement_log_prob = np.log(
            np.maximum(1.0 - feature_prob, np.finfo(float).tiny)
        )

        lines = self._initial_score_lines(model, config)

        for class_index, label in enumerate(model.classes_):
            label_text = format_label(label)
            for feature_index in subset:
                lines.append(
                    "{}score[{}] += {} * {} + (1 - {}) * {}".format(
                        config.indent,
                        label_text,
                        feature_names[feature_index],
                        format_number(feature_log_prob[class_index, feature_index], config),
                        feature_names[feature_index],
                        format_number(complement_log_prob[class_index, feature_index], config),
                    )
                )

        return lines

    def _categorical_rule_lines(
        self,
        model,
        feature_names: list[str],
        subset: list[int],
        config: SerializationConfig,
    ) -> list[str]:
        """
        Serialize CategoricalNB likelihood tables.
        """
        lines = self._initial_score_lines(model, config)

        for class_index, label in enumerate(model.classes_):
            label_text = format_label(label)
            for feature_index in subset:
                log_prob = np.asarray(model.feature_log_prob_[feature_index], dtype=float)
                n_categories = int(log_prob.shape[1])
                for category in range(n_categories):
                    lines.append(
                        "{}if {} == {}: score[{}] += {}".format(
                            config.indent,
                            feature_names[feature_index],
                            category,
                            label_text,
                            format_number(log_prob[class_index, category], config),
                        )
                    )

        return lines
