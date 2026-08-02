"""
Tests for the static, canonical scikit-learn adapter package.
"""

from __future__ import annotations

import inspect
import re

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
    components_model,
    explain_model,
    nescience_model,
    score_model,
    sklearn_model_artifacts,
)
from mnplib.models.serializers.base import format_number
from mnplib.models.serializers.linear import (
    LinearModelSerializer,
    LogisticRegressionSerializer,
)
from mnplib.models.serializers.tree import DecisionTreeSerializer
from mnplib.models.sklearn import _find_serializer
from mnplib.nescience import Nescience


def _assert_explicit_model_string(model_string: str, original_feature_names=()):
    assert model_string.strip()
    assert "SCHEMA" not in model_string
    assert "schema_name" not in model_string
    assert "PARAMETERS" not in model_string
    assert "include_metadata" not in model_string
    assert "RULE" not in model_string
    assert "\t" not in model_string
    assert re.search(r"\bX\d+\b|x\[\d+\]", model_string)
    assert re.search(
        r"def predict\(x\):|^if\s+|^\s*y\s*=|^P StandardScaler$",
        model_string,
        re.MULTILINE,
    )
    assert re.search(r"\d\.\d{2}e[+-]\d{2}", model_string)

    for name in original_feature_names:
        assert name not in model_string


def test_supported_regression_models_produce_artifacts_and_nescience():
    X, y = make_regression(
        n_samples=80,
        n_features=4,
        n_informative=2,
        noise=0.1,
        random_state=42,
    )
    feature_names = ["long_feature_a", "long_feature_b", "long_feature_c", "long_feature_d"]

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
        )

        assert isinstance(artifacts, ModelArtifacts)
        assert artifacts.model_type == type(model).__name__
        assert len(artifacts.predictions) == len(y)
        _assert_explicit_model_string(artifacts.model_string, feature_names)

        value = nescience_model(
            metric,
            model,
            X,
            feature_names=feature_names,
        )
        assert isinstance(value, float)
        assert value >= 0.0

        components = components_model(
            metric,
            model,
            X,
            feature_names=feature_names,
        )
        assert set(components) == {"deficiency", "surplus", "inaccuracy", "surfeit"}

        explanation = explain_model(
            metric,
            model,
            X,
            feature_names=feature_names,
        )
        assert explanation["model_type"] == type(model).__name__
        assert "model_metadata" in explanation

        assert score_model(
            metric,
            model,
            X,
            feature_names=feature_names,
        ) == pytest.approx(1.0 - value)


@pytest.mark.parametrize(
    "model,X_transform",
    [
        (DecisionTreeClassifier(max_depth=3, random_state=42), lambda X: X),
        (LogisticRegression(max_iter=1000), lambda X: X),
        (LinearSVC(max_iter=10000, random_state=42), lambda X: X),
        (GaussianNB(), lambda X: X),
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
    metric = Nescience(X_type="numeric", y_type="categorical").fit(X, y)

    model.fit(X, y)
    artifacts = sklearn_model_artifacts(model, X)

    assert isinstance(artifacts, ModelArtifacts)
    assert artifacts.metadata["task"] == "classification"
    assert len(artifacts.predictions) == len(y)
    _assert_explicit_model_string(artifacts.model_string)

    value = nescience_model(metric, model, X)
    assert isinstance(value, float)
    assert value >= 0.0


@pytest.mark.parametrize(
    "model,X_transform",
    [
        (MultinomialNB(), np.abs),
        (BernoulliNB(), lambda X: (X > 0).astype(int)),
        (CategoricalNB(), lambda X: np.digitize(X, bins=[-1.0, 0.0, 1.0])),
    ],
)
def test_unsupported_naive_bayes_variants_raise_clear_value_error(
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
    model.fit(X, y)

    with pytest.raises(
        ValueError,
        match=f"Unsupported scikit-learn model type {type(model).__name__}",
    ):
        sklearn_model_artifacts(model, X)


@pytest.mark.parametrize(
    "model,expected_serializer",
    [
        (DecisionTreeClassifier(max_depth=2, random_state=42), "decision_tree"),
        (LogisticRegression(max_iter=1000), "logistic_regression"),
        (LinearSVC(max_iter=10000, random_state=42), "linear_svm"),
        (GaussianNB(), "naive_bayes"),
        (
            MLPClassifier(hidden_layer_sizes=(2,), max_iter=25, random_state=42),
            "mlp_neural_network",
        ),
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


def test_no_public_dynamic_registration_or_serialization_config_api_remains():
    import mnplib.models as models
    import mnplib.models.sklearn as sklearn_adapter

    for name in (
        "SklearnModelRegistry",
        "create_default_registry",
        "register_sklearn_serializer",
        "registry",
        "SerializationConfig",
    ):
        assert not hasattr(models, name)
        assert not hasattr(sklearn_adapter, name)

    for function in (
        sklearn_model_artifacts,
        nescience_model,
        components_model,
        explain_model,
        score_model,
    ):
        assert "config" not in inspect.signature(function).parameters
        assert "serialization_config" not in inspect.signature(function).parameters


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


def test_candidate_evaluator_uses_fixed_adapter_policy():
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


def test_original_feature_names_are_metadata_not_model_string():
    X, y = make_regression(n_samples=30, n_features=3, random_state=42)
    names = ["sepal_length_cm", "petal_width_cm", "fragile_human_name"]
    model = LinearRegression().fit(X, y)

    artifacts = sklearn_model_artifacts(model, X, feature_names=names)

    assert "sepal" not in artifacts.model_string
    assert "petal" not in artifacts.model_string
    assert artifacts.metadata["feature_names"] == names
    assert artifacts.metadata["feature_reference_map"] == {
        "X0": "sepal_length_cm",
        "X1": "petal_width_cm",
        "X2": "fragile_human_name",
    }


def test_feature_indices_control_compact_tokens_for_selected_adapter_data():
    X, y = make_regression(n_samples=40, n_features=4, random_state=42)
    selected = (2, 0)
    model = LinearRegression().fit(X[:, selected], y)

    artifacts = sklearn_model_artifacts(
        model,
        X[:, selected],
        feature_names=["third", "first"],
        feature_indices=selected,
    )

    assert "X2" in artifacts.model_string
    assert "X0" in artifacts.model_string
    assert "third" not in artifacts.model_string
    assert artifacts.metadata["feature_reference_map"] == {
        "X2": "third",
        "X0": "first",
    }


def test_real_values_are_formatted_canonically_in_model_strings():
    X, y = make_regression(n_samples=40, n_features=2, random_state=42)
    model = LinearRegression().fit(X, y)
    artifacts = sklearn_model_artifacts(model, X)
    expected_intercept = format_number(
        float(np.asarray(model.intercept_).reshape(-1)[0])
    )

    assert expected_intercept in artifacts.model_string
    assert re.search(r"\bX\d+\b", artifacts.model_string)
    assert not re.search(r"\d+\.\d{4,}", artifacts.model_string)


def test_feature_name_length_validation():
    X, y = make_regression(n_samples=30, n_features=3, random_state=42)
    model = LinearRegression().fit(X, y)

    with pytest.raises(ValueError, match="feature_names"):
        sklearn_model_artifacts(model, X, feature_names=["too_short"])


def test_individual_serializers_report_subsets():
    Xr, yr = make_regression(n_samples=50, n_features=4, random_state=42)
    tree = DecisionTreeRegressor(max_depth=2, random_state=42).fit(Xr, yr)
    linear = LinearRegression().fit(Xr, yr)

    tree_serializer = DecisionTreeSerializer()
    linear_serializer = LinearModelSerializer()

    assert tree_serializer.subset(tree)
    assert linear_serializer.subset(linear)

    Xc, yc = make_classification(
        n_samples=50,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=42,
    )
    logistic = LogisticRegression(max_iter=1000).fit(Xc, yc)
    logistic_serializer = LogisticRegressionSerializer()

    assert logistic_serializer.subset(logistic)


def test_artifacts_to_nescience_kwargs():
    artifacts = ModelArtifacts(
        subset=[0, 2],
        predictions=np.array([1, 0, 1]),
        model_string="M Test\nT classification\nI X0 X2\nR\n return C1\n",
        model_type="TestModel",
        metadata={"a": 1},
    )

    kwargs = artifacts.to_nescience_kwargs()

    assert set(kwargs) == {"subset", "predictions", "model_string"}
    assert kwargs["subset"] == [0, 2]
