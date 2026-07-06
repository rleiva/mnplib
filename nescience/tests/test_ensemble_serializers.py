"""
Tests for tree-ensemble scikit-learn serializers.
"""

import numpy as np
import pytest

from sklearn.datasets import make_classification, make_regression
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

from mnplib.nescience import Nescience
from mnplib.models import (
    ModelArtifacts,
    SerializationConfig,
    components_model,
    explain_model,
    nescience_model,
    score_model,
    sklearn_model_artifacts,
)
from mnplib.models.sklearn import registry


REGRESSION_MODELS = [
    RandomForestRegressor(n_estimators=3, max_depth=3, random_state=42),
    ExtraTreesRegressor(n_estimators=3, max_depth=3, random_state=42),
    GradientBoostingRegressor(n_estimators=3, max_depth=2, random_state=42),
    HistGradientBoostingRegressor(max_iter=3, max_leaf_nodes=4, random_state=42),
]


CLASSIFICATION_MODELS = [
    RandomForestClassifier(n_estimators=3, max_depth=3, random_state=42),
    ExtraTreesClassifier(n_estimators=3, max_depth=3, random_state=42),
    GradientBoostingClassifier(n_estimators=3, max_depth=2, random_state=42),
    HistGradientBoostingClassifier(max_iter=3, max_leaf_nodes=4, random_state=42),
]


def test_tree_ensemble_regressors_produce_artifacts_and_nescience():
    X, y = make_regression(
        n_samples=90,
        n_features=5,
        n_informative=3,
        noise=0.5,
        random_state=42,
    )
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    config = SerializationConfig(precision=4)
    metric = Nescience(X_type="numeric", y_type="numeric", n_bins=3).fit(X, y)

    for model in REGRESSION_MODELS:
        model.fit(X, y)

        artifacts = sklearn_model_artifacts(
            model,
            X,
            feature_names=feature_names,
            config=config,
        )

        assert isinstance(artifacts, ModelArtifacts)
        assert artifacts.model_string.startswith("SCHEMA canonical_nescience_model_v1")
        assert f"MODEL {type(model).__name__}" in artifacts.model_string
        assert "TASK regression" in artifacts.model_string
        assert "RULE" in artifacts.model_string
        assert len(artifacts.predictions) == len(y)
        assert len(artifacts.subset) > 0
        assert artifacts.metadata["serializer"] == "tree_ensemble"
        assert artifacts.metadata["support_level"] == "beta"
        assert "ensemble_family" in artifacts.metadata

        value = nescience_model(metric, model, X, feature_names=feature_names, config=config)
        assert isinstance(value, float)
        assert value >= 0.0

        components = components_model(metric, model, X, feature_names=feature_names, config=config)
        assert set(components) == {"deficiency", "surplus", "inaccuracy", "surfeit"}

        explanation = explain_model(metric, model, X, feature_names=feature_names, config=config)
        assert explanation["model_type"] == type(model).__name__
        assert "model_metadata" in explanation

        assert score_model(metric, model, X, feature_names=feature_names, config=config) == pytest.approx(1.0 - value)


def test_tree_ensemble_classifiers_produce_artifacts_and_nescience():
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )
    config = SerializationConfig(precision=4)
    metric = Nescience(X_type="numeric", y_type="categorical", n_bins=3).fit(X, y)

    for model in CLASSIFICATION_MODELS:
        model.fit(X, y)

        artifacts = sklearn_model_artifacts(model, X, config=config)

        assert isinstance(artifacts, ModelArtifacts)
        assert f"MODEL {type(model).__name__}" in artifacts.model_string
        assert "TASK classification" in artifacts.model_string
        assert "RULE" in artifacts.model_string
        assert len(artifacts.predictions) == len(y)
        assert len(artifacts.subset) > 0
        assert artifacts.metadata["serializer"] == "tree_ensemble"
        assert "ensemble_family" in artifacts.metadata

        value = nescience_model(metric, model, X, config=config)
        assert isinstance(value, float)
        assert value >= 0.0


def test_ensemble_model_types_are_registered():
    supported = registry.supported_model_types()

    for expected in [
        RandomForestRegressor,
        RandomForestClassifier,
        ExtraTreesRegressor,
        ExtraTreesClassifier,
        GradientBoostingRegressor,
        GradientBoostingClassifier,
        HistGradientBoostingRegressor,
        HistGradientBoostingClassifier,
    ]:
        assert expected in supported

    assert "tree_ensemble" in registry.serializer_names()


def test_string_labels_are_serialized_for_ensemble_classifier():
    X, y_numeric = make_classification(
        n_samples=80,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=42,
    )
    y = np.array([f"class_{label}" for label in y_numeric])

    model = RandomForestClassifier(n_estimators=3, max_depth=2, random_state=42)
    model.fit(X, y)

    artifacts = sklearn_model_artifacts(
        model,
        X,
        config=SerializationConfig(precision=4),
    )

    assert "class_" in artifacts.model_string
    assert set(artifacts.predictions).issubset(set(y))


def test_hist_gradient_boosting_serialization_contains_hist_stage_marker():
    X, y = make_regression(n_samples=80, n_features=4, random_state=42)
    model = HistGradientBoostingRegressor(
        max_iter=2,
        max_leaf_nodes=3,
        random_state=42,
    ).fit(X, y)

    artifacts = sklearn_model_artifacts(model, X, config=SerializationConfig(precision=4))

    assert "HIST_BOOSTING_STAGE" in artifacts.model_string
    assert artifacts.metadata["ensemble_family"] == "hist_gradient_boosting"
