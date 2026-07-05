"""
Tests for the modular scikit-learn adapter package.
"""

import numpy as np
import pytest

from sklearn.datasets import make_classification, make_regression
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from nescience.nescience import Nescience
from nescience.models import (
    ModelArtifacts,
    SerializationConfig,
    components_model,
    explain_model,
    nescience_model,
    score_model,
    sklearn_model_artifacts,
)
from nescience.models.sklearn import registry
from nescience.models.serializers.tree import DecisionTreeSerializer
from nescience.models.serializers.linear import LinearModelSerializer, LogisticRegressionSerializer


def test_supported_regression_models_produce_artifacts_and_nescience():
    X, y = make_regression(
        n_samples=80,
        n_features=4,
        n_informative=2,
        noise=0.1,
        random_state=42,
    )
    feature_names = ["a", "b", "c", "d"]
    config = SerializationConfig(precision=4)

    models = [
        DecisionTreeRegressor(max_depth=3, random_state=42),
        LinearRegression(),
        Ridge(alpha=1.0),
        Lasso(alpha=0.01, max_iter=10000),
        ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000),
    ]

    metric = Nescience(X_type="numeric", y_type="numeric", n_bins=3).fit(X, y)

    for model in models:
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

        value = nescience_model(
            metric,
            model,
            X,
            feature_names=feature_names,
            config=config,
        )
        assert isinstance(value, float)
        assert value >= 0.0

        components = components_model(
            metric,
            model,
            X,
            feature_names=feature_names,
            config=config,
        )
        assert set(components) == {"deficiency", "surplus", "inaccuracy", "surfeit"}

        explanation = explain_model(
            metric,
            model,
            X,
            feature_names=feature_names,
            config=config,
        )
        assert explanation["model_type"] == type(model).__name__
        assert "model_metadata" in explanation

        assert score_model(
            metric,
            model,
            X,
            feature_names=feature_names,
            config=config,
        ) == pytest.approx(1.0 - value)


def test_supported_classification_models_produce_artifacts_and_nescience():
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )
    config = SerializationConfig(precision=4)

    models = [
        DecisionTreeClassifier(max_depth=3, random_state=42),
        LogisticRegression(max_iter=1000),
    ]

    metric = Nescience(X_type="numeric", y_type="categorical").fit(X, y)

    for model in models:
        model.fit(X, y)

        artifacts = sklearn_model_artifacts(model, X, config=config)

        assert isinstance(artifacts, ModelArtifacts)
        assert "TASK classification" in artifacts.model_string
        assert len(artifacts.predictions) == len(y)

        value = nescience_model(metric, model, X, config=config)
        assert isinstance(value, float)
        assert value >= 0.0


def test_serializer_registry_supports_expected_model_types():
    supported = registry.supported_model_types()

    for expected in [
        DecisionTreeClassifier,
        DecisionTreeRegressor,
        LinearRegression,
        Ridge,
        Lasso,
        ElasticNet,
        LogisticRegression,
    ]:
        assert expected in supported

    assert set(registry.serializer_names()) == {
        "decision_tree",
        "linear_model",
        "logistic_regression",
    }


def test_unfitted_model_raises_not_fitted_error():
    with pytest.raises(NotFittedError):
        sklearn_model_artifacts(DecisionTreeRegressor(), [[0.0, 1.0]])


def test_unsupported_model_raises_not_implemented_error():
    X, y = make_regression(n_samples=30, n_features=3, random_state=42)
    model = DummyRegressor().fit(X, y)

    with pytest.raises(NotImplementedError):
        sklearn_model_artifacts(model, X)


def test_feature_name_length_validation():
    X, y = make_regression(n_samples=30, n_features=3, random_state=42)
    model = LinearRegression().fit(X, y)

    with pytest.raises(ValueError, match="feature_names"):
        sklearn_model_artifacts(model, X, feature_names=["too_short"])


def test_serialization_config_validation():
    with pytest.raises(ValueError):
        SerializationConfig(precision=-1)

    with pytest.raises(ValueError):
        SerializationConfig(zero_tolerance=-1)

    with pytest.raises(ValueError):
        SerializationConfig(schema_name="")

    with pytest.raises(TypeError):
        SerializationConfig(indent=123)


def test_individual_serializers_report_subsets():
    Xr, yr = make_regression(n_samples=50, n_features=4, random_state=42)
    tree = DecisionTreeRegressor(max_depth=2, random_state=42).fit(Xr, yr)
    linear = LinearRegression().fit(Xr, yr)

    tree_serializer = DecisionTreeSerializer()
    linear_serializer = LinearModelSerializer()

    assert tree_serializer.subset(tree, config=SerializationConfig())
    assert linear_serializer.subset(linear, config=SerializationConfig())

    Xc, yc = make_classification(
        n_samples=50,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=42,
    )
    logistic = LogisticRegression(max_iter=1000).fit(Xc, yc)
    logistic_serializer = LogisticRegressionSerializer()

    assert logistic_serializer.subset(logistic, config=SerializationConfig())


def test_artifacts_to_nescience_kwargs():
    artifacts = ModelArtifacts(
        subset=[0, 2],
        predictions=np.array([1, 0, 1]),
        model_string="MODEL test",
        model_type="TestModel",
        metadata={"a": 1},
    )

    kwargs = artifacts.to_nescience_kwargs()

    assert set(kwargs) == {"subset", "predictions", "model_string"}
    assert kwargs["subset"] == [0, 2]
