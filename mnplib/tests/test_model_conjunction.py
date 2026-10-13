"""Conjunction semantics and explicit class-score decision rules."""

import json

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from mnplib import describe_model, ModelDescription, Surfeit, Mismodel, Nescience
from mnplib.models import sklearn_model_artifacts
from mnplib.models.language import (
    And, Binary, Conditional, Constant, Feature, Label, Vector,
    execute, normalize, parse, render, node_from_dict, node_to_dict,
)
from mnplib.models.language.transform import substitute_features
from mnplib.models.serializers.base import score_conditionals


@pytest.mark.parametrize("n_conditions", [2, 3])
def test_conjunction_ast_text_json_and_execution(n_conditions):
    comparisons = tuple(Binary(">=", Feature(0), Feature(index))
                        for index in range(1, n_conditions + 1))
    node = And(comparisons)
    text = " and ".join(f"x0>=x{index}" for index in range(1, n_conditions + 1))
    assert render(node) == text
    assert parse(text) == node
    assert normalize(parse(render(normalize(node)))) == normalize(node)
    payload = node_to_dict(node)
    assert payload["type"] == "and"
    assert len(payload["items"]) == n_conditions
    assert node_from_dict(json.loads(json.dumps(payload))) == node

    X = np.array([[3, 2, 1, 0], [2, 3, 1, 0], [2, 1, 3, 0],
                  [2, 1, 0, 3], [2, 2, 2, 2]])
    expected = np.all(X[:, :1] >= X[:, 1:n_conditions + 1], axis=1)
    for description in (node, text):
        actual = execute(description, X)
        assert actual.dtype == np.bool_
        np.testing.assert_array_equal(actual, expected)
        assert execute(description, np.empty((0, 4))).shape == (0,)


@pytest.mark.parametrize("text", [
    "x0+x1>=x2 and x0*x1<x2",
    "x0 if x1>=x2 and x1>=x3 else x2 if x2>=x3 else x3",
    "x0>=x1 and x1<x2 if x2>x3 else x3<x0 and x1==x2",
    "(x0 if x1<x2 else x3) and x1>=x2",
    "(x0>=x1 and x1>=x2)==(x2>=x3)",
    "(x0>=x1 and x1>=x2)+x3",
    "-(x0>=x1 and x1>=x2)",
    "x0**(x1>=x2 and x2>=x3)",
])
def test_conjunction_precedence_roundtrip(text):
    node = parse(text)
    assert render(node) == text
    assert render(parse(render(node))) == text
    assert node_from_dict(node_to_dict(node)) == node


def test_conjunction_normalization_preserves_order():
    a, b, c = (Binary(">", Feature(index), Constant(0)) for index in range(3))
    assert normalize(And((a,))) == a
    node = And((And((b, a)), And((c,))))
    assert normalize(node) == And((b, a, c))
    assert normalize(normalize(node)) == normalize(node)
    assert render(node) == "x1>0.00e+00 and x0>0.00e+00 and x2>0.00e+00"


def test_conjunction_validates_nonempty_scalar_conditions():
    with pytest.raises(ValueError, match="at least one condition"):
        normalize(And(()))
    with pytest.raises(ValueError, match="at least one condition"):
        node_from_dict({"type": "and", "items": []})
    with pytest.raises(ValueError, match="Invalid 'and'"):
        node_from_dict({"type": "and"})
    with pytest.raises(ValueError, match="one value per observation"):
        execute(And((Vector((Feature(0),)), Constant(1))), [[0], [1]])


@pytest.mark.parametrize("text", [
    'classify([1,2],[label("0"),label("1")])',
    "x0 or x1", "not x0", "x0 and", "and x0", "x0 and and x1",
])
def test_undeclared_prediction_syntax_is_rejected(text):
    with pytest.raises(ValueError, match="Invalid model expression"):
        parse(text)


def test_undeclared_semantic_node_is_rejected():
    with pytest.raises(ValueError, match="Unsupported semantic node"):
        node_from_dict({"type": "classify", "scores": {}, "labels": []})


@pytest.mark.parametrize("scores,expected", [
    ((3, 3), "first"),
    ((3, 3, 1), "first"),
    ((3, 3, 3), "first"),
    ((3, 1, 3), "first"),
    ((1, 3, 3), "second"),
    ((1, 2, 3), "third"),
])
def test_explicit_score_conditions_choose_first_maximum(scores, expected):
    labels = tuple(Label(value) for value in ("first", "second", "third")[:len(scores)])
    node = score_conditionals(tuple(Constant(value) for value in scores), labels)
    assert isinstance(node, Conditional)
    for description in (node, render(node)):
        np.testing.assert_array_equal(execute(description, [[0], [1]]), [expected] * 2)


@pytest.mark.parametrize("model_type", [LogisticRegression, LinearSVC])
def test_fitted_multiclass_score_ties_match_estimator(model_type):
    model = model_type().fit(np.eye(3), ["alpha", "beta", "gamma"])
    model.coef_ = np.eye(3)
    model.intercept_ = np.zeros(3)
    X = np.array([[3., 3., 1.], [3., 3., 3.], [3., 1., 3.],
                  [1., 3., 3.], [1., 2., 3.]])
    expected = ["alpha", "alpha", "alpha", "beta", "gamma"]
    np.testing.assert_array_equal(model.predict(X), expected)
    description = describe_model(model)
    for representation in (description.ast, description.canonical):
        np.testing.assert_array_equal(execute(representation, X), model.predict(X))


@pytest.mark.parametrize("n_classes", [3, 4, 6])
def test_multiclass_logistic_description_roundtrips(n_classes):
    X, y = make_blobs(n_samples=120, n_features=3, centers=n_classes,
                      cluster_std=0.1, random_state=21)
    model = LogisticRegression(max_iter=1000).fit(X, y)
    description = describe_model(model)
    for token in (" if ", " else ", " and ", ">="):
        assert token in description.canonical
    assert "classify" not in description.canonical
    assert description.canonical.count(" if ") == n_classes - 1
    assert render(parse(description.canonical)) == description.canonical
    payload = json.loads(json.dumps(description.to_dict()))
    assert payload["schema_version"] == 3
    assert '"type": "and"' in json.dumps(payload["ast"])
    assert '"type": "classify"' not in json.dumps(payload["ast"])
    restored = ModelDescription.from_dict(payload)
    assert restored.to_dict() == payload
    for representation in (description.ast, description.canonical, restored.ast):
        np.testing.assert_array_equal(execute(representation, X), model.predict(X))


@pytest.mark.parametrize("model_type", [LogisticRegression, LinearSVC])
def test_binary_score_threshold_matches_estimator(model_type):
    model = model_type().fit([[-1.], [1.]], ["negative", "positive"])
    model.coef_ = np.array([[1.]])
    model.intercept_ = np.zeros(1)
    X = np.array([[-1.], [0.], [1.]])
    description = describe_model(model)
    assert description.ast.condition == Binary(">", Feature(0, "x0"), Constant(0))
    assert " and " not in description.canonical
    np.testing.assert_array_equal(model.predict(X), ["negative", "negative", "positive"])
    np.testing.assert_array_equal(execute(description.canonical, X), model.predict(X))


def test_conjunction_feature_substitution():
    node = And((Binary(">=", Feature(0), Feature(1)),
                Binary(">=", Feature(0), Feature(2))))
    transformed = substitute_features(node, {0: Binary("-", Feature(3), Constant(1))})
    assert render(transformed) == "x3-1.00e+00>=x1 and x3-1.00e+00>=x2"
    np.testing.assert_array_equal(execute(transformed, [[0, 2, 2, 3], [0, 2, 3, 3]]),
                                  [True, False])


def test_multiclass_metric_integration_uses_explicit_description():
    X = np.tile(np.eye(3), (30, 1))
    y = np.tile(np.arange(3), 30)
    model = LogisticRegression().fit(X, y)
    description = describe_model(model)
    assert " and " in description.canonical
    surfeit = Surfeit().fit(X, y)
    assert surfeit.surfeit_model(model) == surfeit.surfeit_string(description.canonical)
    mismodel = Mismodel().fit(X, y)
    assert mismodel.mismodel_model(model) == pytest.approx(mismodel.mismodel(
        predictions=model.predict(X), model_string=description.canonical,
    ))
    nescience = Nescience().fit(X, y)
    assert nescience.nescience_model(model) == pytest.approx(nescience.nescience(
        subset=sklearn_model_artifacts(model, X).subset,
        predictions=model.predict(X), model_string=description.canonical,
    ))
