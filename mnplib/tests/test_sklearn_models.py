"""
Tests for the static scikit-learn adapter package.
"""

from __future__ import annotations

import numpy as np
import pytest

from sklearn.datasets import make_classification, make_regression
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from mnplib.automl import CandidateEvaluator
from mnplib.automl.results import CandidateResult
from mnplib.models import (
    ModelArtifacts,
    SerializationConfig,
    components_model,
    explain_model,
    nescience_model,
    score_model,
    sklearn_model_artifacts,
)
from mnplib.models.serializers.linear import (
    LinearModelSerializer,
    LogisticRegressionSerializer,
)
from mnplib.models.serializers.tree import DecisionTreeSerializer
from mnplib.models.sklearn import _find_serializer
from mnplib.nescience import Nescience


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
        LinearSVR(max_iter=10000, random_state=42),
        MLPRegressor(
            hidden_layer_sizes=(2,),
            max_iter=25,
            random_state=42,
        ),
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


@pytest.mark.parametrize(
    "model,X_transform",
    [
        (DecisionTreeClassifier(max_depth=3, random_state=42), lambda X: X),
        (LogisticRegression(max_iter=1000), lambda X: X),
        (LinearSVC(max_iter=10000, random_state=42), lambda X: X),
        (GaussianNB(), lambda X: X),
        (MultinomialNB(), np.abs),
        (BernoulliNB(), lambda X: (X > 0).astype(int)),
        (CategoricalNB(), lambda X: np.digitize(X, bins=[-1.0, 0.0, 1.0])),
        (
            MLPClassifier(
                hidden_layer_sizes=(3,),
                max_iter=25,
                random_state=42,
            ),
            lambda X: X,
        ),
    ],
)
def test_supported_classification_models_produce_artifacts_and_nescience(
    model,
    X_transform,
):
    X_raw, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )
    X = X_transform(X_raw)
    config = SerializationConfig(precision=4)
    metric = Nescience(X_type="numeric", y_type="categorical").fit(X, y)

    model.fit(X, y)
    artifacts = sklearn_model_artifacts(model, X, config=config)

    assert isinstance(artifacts, ModelArtifacts)
    assert "TASK classification" in artifacts.model_string
    assert len(artifacts.predictions) == len(y)

    value = nescience_model(metric, model, X, config=config)
    assert isinstance(value, float)
    assert value >= 0.0


@pytest.mark.parametrize(
    "model,expected_serializer",
    [
        (DecisionTreeClassifier(max_depth=2, random_state=42), "decision_tree"),
        (LogisticRegression(max_iter=1000), "logistic_regression"),
        (LinearSVC(max_iter=10000, random_state=42), "linear_svm"),
        (GaussianNB(), "naive_bayes"),
        (MLPClassifier(hidden_layer_sizes=(2,), max_iter=25, random_state=42), "mlp_neural_network"),
    ],
)
def test_static_dispatch_selects_expected_classifier_serializers(
    model,
    expected_serializer,
):
    X, y = make_classification(
        n_samples=60,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=42,
    )

    model.fit(X, y)

    assert _find_serializer(model).name == expected_serializer
    assert sklearn_model_artifacts(model, X).metadata["serializer"] == expected_serializer


def test_static_dispatch_selects_expected_regressor_serializers():
    X, y = make_regression(n_samples=60, n_features=4, random_state=42)
    expectations = [
        (DecisionTreeRegressor(max_depth=2, random_state=42), "decision_tree"),
        (LinearRegression(), "linear_model"),
        (LinearSVR(max_iter=10000, random_state=42), "linear_svm"),
        (
            MLPRegressor(hidden_layer_sizes=(2,), max_iter=25, random_state=42),
            "mlp_neural_network",
        ),
    ]

    for model, expected_serializer in expectations:
        model.fit(X, y)
        assert _find_serializer(model).name == expected_serializer


def test_no_public_dynamic_registration_api_remains():
    import mnplib.models as models
    import mnplib.models.sklearn as sklearn_adapter

    for name in (
        "SklearnModelRegistry",
        "create_default_registry",
        "register_sklearn_serializer",
        "registry",
    ):
        assert not hasattr(models, name)
        assert not hasattr(sklearn_adapter, name)


def test_unfitted_model_raises_not_fitted_error():
    with pytest.raises(NotFittedError):
        sklearn_model_artifacts(DecisionTreeRegressor(), [[0.0, 1.0]])


def test_unsupported_model_raises_clear_value_error():
    X, y = make_regression(n_samples=30, n_features=3, random_state=42)
    model = DummyRegressor().fit(X, y)

    with pytest.raises(
        ValueError,
        match="Unsupported scikit-learn model type DummyRegressor",
    ) as exc_info:
        sklearn_model_artifacts(model, X)

    message = str(exc_info.value)
    assert "DecisionTreeClassifier" in message
    assert "LogisticRegression" in message
    assert "No generic or repr-based model serialization is available" in message


def test_no_repr_or_generic_fallback_for_unsupported_model():
    class ReprOnlyDummy(DummyRegressor):
        def __repr__(self):
            return "GENERIC_MODEL_STRING_SHOULD_NOT_APPEAR"

    X, y = make_regression(n_samples=30, n_features=3, random_state=42)
    model = ReprOnlyDummy().fit(X, y)

    with pytest.raises(ValueError) as exc_info:
        sklearn_model_artifacts(model, X)

    assert "GENERIC_MODEL_STRING_SHOULD_NOT_APPEAR" not in str(exc_info.value)


def test_candidate_evaluator_uses_static_adapter():
    X, y = make_classification(
        n_samples=60,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=42,
    )
    metric = Nescience(X_type="numeric", y_type="categorical", n_bins=3).fit(X, y)
    evaluator = CandidateEvaluator(
        X=X,
        y=y,
        nescience=metric,
        feature_names=[f"x{i}" for i in range(X.shape[1])],
        serialization_config=SerializationConfig(precision=4),
    )
    model = DecisionTreeClassifier(max_depth=2, random_state=42).fit(X, y)

    result = evaluator.evaluate(
        name="tree",
        family="decision_tree_classifier",
        model=model,
    )

    assert isinstance(result, CandidateResult)
    assert isinstance(result.artifacts, ModelArtifacts)
    assert result.artifacts.to_nescience_kwargs()["model_string"]
    assert result.metadata["serializer"] == "decision_tree"


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
