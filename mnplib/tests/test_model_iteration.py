"""Bounded iteration, lexical scope, and explicit network computations."""

from dataclasses import fields
from copy import deepcopy
import json

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.neural_network import MLPClassifier, MLPRegressor

from mnplib import ModelDescription, describe_model, Surfeit, Mismodel, Nescience
from mnplib.models import sklearn_model_artifacts
from mnplib.models.language import (
    ModelNode, Constant, Feature, Binary, Conditional, Vector, Element,
    Index, Sum, Comprehension, Exp, normalize, parse, render, execute,
    node_to_dict, node_from_dict,
)
from mnplib.models.language.transform import substitute_features
from mnplib.models.language.normalize import quantize
from mnplib.models.serializers.neural_network import _activation_expression


def _nodes(value):
    if isinstance(value, ModelNode):
        yield value
        for field in fields(value):
            yield from _nodes(getattr(value, field.name))
    elif isinstance(value, tuple):
        for item in value:
            yield from _nodes(item)


@pytest.mark.parametrize("text,expected", [
    ("sum(i for i in range(4))", [6, 6]),
    ("[i for i in range(3)]", [[0, 1, 2], [0, 1, 2]]),
    ("sum(at([x0,x1],i) for i in range(2))", [5, 9]),
    ("[x0*i for i in range(3)]", [[0, 2, 4], [0, 4, 8]]),
    ("[sum(i+j for i in range(3)) for j in range(2)]", [[3, 6], [3, 6]]),
])
def test_iteration_execution_and_roundtrips(text, expected):
    node = parse(text)
    assert render(node) == text
    assert normalize(parse(render(normalize(node)))) == normalize(node)
    assert render(parse(render(node))) == render(node)
    payload = json.loads(json.dumps(node_to_dict(node)))
    assert node_from_dict(payload) == node
    for representation in (node, text, node_from_dict(payload)):
        np.testing.assert_array_equal(execute(representation, [[2, 3], [4, 5]]), expected)


def test_iteration_matrix_access_and_json_schema():
    weights = Vector((Vector((Constant(2), Constant(3))),
                      Vector((Constant(4), Constant(5)))))
    inputs = Vector((Feature(0), Feature(1)))
    product = Binary("*", Element(Element(weights, Index("j")), Index("i")),
                     Element(inputs, Index("i")))
    node = Comprehension("j", 2, Sum("i", 2, product))
    description = ModelDescription("bounded", node, (0, 1), ("a", "b"))
    assert description.canonical == (
        "[sum(at(at([[2.00e+00,3.00e+00],[4.00e+00,5.00e+00]],j),i)"
        "*at([x0,x1],i) for i in range(2)) for j in range(2)]"
    )
    payload = json.loads(json.dumps(description.to_dict()))
    assert payload["schema_version"] == 3
    assert payload["ast"]["type"] == "comprehension"
    assert payload["ast"]["expression"]["type"] == "sum"
    assert payload["ast"]["expression"]["expression"]["left"]["index"] == {
        "type": "index", "name": "i",
    }
    restored = ModelDescription.from_dict(payload)
    assert restored.to_dict() == payload
    np.testing.assert_array_equal(execute(restored.ast, [[1, 2], [3, 4]]),
                                  [[8, 14], [18, 32]])


@pytest.mark.parametrize("node,text,shape", [
    (Sum("i", 0, Index("i")), "0.00e+00", (2,)),
    (Comprehension("i", 0, Index("i")), "[]", (2, 0)),
    (Comprehension("j", 2, Comprehension("i", 0, Index("i"))),
     "[[] for j in range(2)]", (2, 2, 0)),
])
def test_zero_size_iteration(node, text, shape):
    assert render(node) == text
    assert execute(node, [[0], [1]]).shape == shape
    assert node_from_dict(node_to_dict(node)) == normalize(node)
    assert normalize(normalize(node)) == normalize(node)


def test_sum_supports_vector_values_and_empty_observation_batches():
    node = Sum("i", 3, Vector((Index("i"), Feature(0))))
    np.testing.assert_array_equal(execute(node, [[2], [4]]), [[3, 6], [3, 12]])
    assert execute(node, np.empty((0, 1))).shape == (0, 2)
    vector = Comprehension("i", 3, Feature(0))
    assert execute(vector, np.empty((0, 1))).shape == (0, 3)


@pytest.mark.parametrize("kind", [Sum, Comprehension])
@pytest.mark.parametrize("size", [-1, 1.5, True, "3", np.inf, Constant(2)])
def test_iteration_rejects_nonliteral_bounds(kind, size):
    with pytest.raises(ValueError, match="size must be a non-negative integer"):
        normalize(kind("i", size, Index("i")))


@pytest.mark.parametrize("name", ["", "I", "_i", "2i", "i-j", "i.x", "x0",
                                   "for", "sum", "range", "exp", "at", "label", None])
def test_iteration_rejects_invalid_names(name):
    with pytest.raises(ValueError, match="identifier"):
        normalize(Sum(name, 2, Constant(1)))


@pytest.mark.parametrize("node", [
    Index("i"),
    Sum("i", 2, Index("j")),
    Sum("i", 0, Index("j")),
    Binary("+", Sum("i", 2, Index("i")), Index("i")),
    Sum("i", 2, Sum("j", 2, Element(Vector((Constant(1),)), Index("k")))),
])
def test_unbound_indices_are_rejected_at_public_boundaries(node):
    for operation in (normalize, render, node_to_dict):
        with pytest.raises(ValueError, match="Unbound iteration index"):
            operation(node)
    with pytest.raises(ValueError, match="Unbound iteration index"):
        execute(node, [[0]])


@pytest.mark.parametrize("outer", [Sum, Comprehension])
@pytest.mark.parametrize("inner", [Sum, Comprehension])
def test_iteration_rejects_shadowing(outer, inner):
    with pytest.raises(ValueError, match="shadowing"):
        normalize(outer("i", 2, inner("i", 3, Index("i"))))


def test_sibling_scopes_are_independent():
    node = Binary("+", Sum("i", 2, Index("i")), Sum("i", 3, Index("i")))
    np.testing.assert_array_equal(execute(node, [[0]]), [4])


@pytest.mark.parametrize("text", [
    "sum(i for i in range(x0))", "sum(i for i in range(1,3))",
    "sum(i for i in range(0,3,1))", "sum(i for i in range(-1))",
    "sum(i for i in range(2.0))", "sum(i for i in range(2e+00))",
    "sum(i for i in range(True))", "sum(i for i in [0,1])",
    "sum(i for i in arbitrary(x0))", "[i for i in range(2) if i>0]",
    "[i+j for i in range(2) for j in range(2)]", "for i in range(2): x0",
    "[i.real for i in range(2)]", "[i:=x0 for i in range(2)]",
    "sum(j for i in range(2))", "sum(sum(i for i in range(2)) for i in range(2))",
    "[at([x0],i+1) for i in range(1)]", "at([x0],1.0)", "at([x0],-1)",
    "[x0][0]", "[x0][:]", "dense([x0],[[1]],[0],\"relu\")",
    "relu(x0)", "sigmoid(x0)", "tanh(x0)", "activation(x0)",
])
def test_iteration_grammar_is_restricted(text):
    with pytest.raises(ValueError, match="Invalid model expression"):
        parse(text)


@pytest.mark.parametrize("payload", [
    {"type": "index", "name": "i"},
    {"type": "sum", "index": "i", "size": -1, "expression": {"type": "constant", "value": 1}},
    {"type": "sum", "index": "i", "size": 0, "expression": {"type": "index", "name": "j"}},
    {"type": "comprehension", "index": "i", "size": 2, "expression": {
        "type": "sum", "index": "i", "size": 2, "expression": {"type": "index", "name": "i"}}},
    {"type": "element", "vector": {"type": "vector", "items": []},
     "index": {"type": "constant", "value": 0}},
    {"type": "dense"}, {"type": "activation"},
    {"type": "sum", "index": "i", "size": 2},
])
def test_iteration_json_validates_scope_and_vocabulary(payload):
    with pytest.raises(ValueError):
        node_from_dict(payload)


@pytest.mark.parametrize("node", [
    Element(Vector((Constant(1),)), 1),
    Element(Comprehension("i", 2, Index("i")), 2),
    Sum("i", 2, Element(Vector((Constant(1),)), Index("i"))),
    Element(Constant(1), 0),
])
def test_indexed_access_validates_bounds(node):
    with pytest.raises(ValueError, match="out of bounds"):
        execute(node, [[0]])


def test_bound_indices_are_exact_and_not_quantized():
    node = Element(Comprehension("i", 12346, Index("i")), 12345)
    np.testing.assert_array_equal(execute(node, [[0]]), [12345])
    assert render(node) == "at([i for i in range(12346)],12345)"


def test_indexed_comprehension_evaluates_only_requested_item():
    node = Element(Comprehension("i", 2, Binary("/", Constant(1), Index("i"))), 1)
    with np.errstate(divide="raise"):
        np.testing.assert_array_equal(execute(node, [[0]]), [1])


def test_generic_exponential_and_iteration_precedence():
    node = Sum("i", 2, Exp(Binary("*", Index("i"), Feature(0))))
    assert render(node) == "sum(exp(i*x0) for i in range(2))"
    np.testing.assert_allclose(execute(node, [[0], [1]]), [2, 1 + np.e])
    for text in ("[x0 if i==0.00e+00 else x1 for i in range(2)]",
                 "-sum(i for i in range(2))**2.00e+00",
                 "sum(x0 if i==0.00e+00 else x1 for i in range(2))"):
        assert render(parse(text)) == text


def test_iteration_preserves_feature_substitution():
    node = Sum("i", 2, Element(Vector((Feature(0), Feature(1))), Index("i")))
    transformed = substitute_features(node, {0: Binary("-", Feature(2), Constant(1))})
    np.testing.assert_array_equal(execute(transformed, [[99, 3, 2]]), [4])
    assert node_from_dict(node_to_dict(transformed)) == normalize(transformed)


@pytest.mark.parametrize("activation", ["identity", "relu", "logistic", "tanh"])
def test_activation_arithmetic_is_stable(activation):
    X = np.array([[-1000.], [-10.], [-0.001], [0.], [0.001], [10.], [1000.]])
    node = _activation_expression(Feature(0), activation)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        actual = execute(node, X)
        np.testing.assert_array_equal(actual, execute(render(node), X))
    z = X[:, 0]
    if activation == "identity":
        expected = z
    elif activation == "relu":
        expected = np.maximum(0, z)
        assert isinstance(node, Conditional)
    elif activation == "logistic":
        expected = np.exp(-np.logaddexp(0, -z))
    else:
        expected = np.tanh(z)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-15)


def test_network_neuron_and_layer_structure():
    X = np.array([[-1., 0.], [0., -1.], [0., 1.], [1., 0.]])
    model = MLPRegressor(hidden_layer_sizes=(), solver="lbfgs", random_state=2).fit(X, X[:, 0])
    model.coefs_ = [np.array([[2.], [3.]])]
    model.intercepts_ = [np.array([1.])]
    description = describe_model(model)
    assert isinstance(description.ast, Element)
    layer = description.ast.vector
    assert isinstance(layer, Comprehension)
    assert layer.size == 1
    assert isinstance(layer.expression, Binary)
    assert layer.expression.operator == "+"
    assert isinstance(layer.expression.right, Sum)
    assert layer.expression.right.size == 2
    np.testing.assert_allclose(execute(description.ast, X), model.predict(X))
    constants = [node.value for node in _nodes(description.ast) if isinstance(node, Constant)]
    assert constants == [1., 2., 3.]


@pytest.mark.parametrize("activation", ["identity", "relu", "logistic", "tanh"])
@pytest.mark.parametrize("hidden_layers", [(3,), (3, 2)])
def test_network_regression_iteration_contract(activation, hidden_layers):
    X = np.random.default_rng(4).integers(-1, 2, size=(40, 2)).astype(float)
    y = 1 + X[:, 0] - 2 * X[:, 1]
    model = MLPRegressor(hidden_layer_sizes=hidden_layers, activation=activation,
                         solver="lbfgs", alpha=0.1, random_state=3, max_iter=2000).fit(X, y)
    description = describe_model(model)
    _assert_network_description(description, model, X, regression=True)


def test_network_execution_uses_quantized_parameters():
    X = np.random.default_rng(4).integers(-1, 2, size=(40, 2)).astype(float)
    y = 1 + X[:, 0] - 2 * X[:, 1]
    model = MLPRegressor(hidden_layer_sizes=(3, 2), activation="relu",
                         solver="lbfgs", random_state=3, max_iter=2000).fit(X, y)
    rounded = deepcopy(model)
    rounded.coefs_ = [np.vectorize(quantize)(weights) for weights in model.coefs_]
    rounded.intercepts_ = [np.vectorize(quantize)(bias) for bias in model.intercepts_]
    np.testing.assert_allclose(execute(describe_model(model).ast, X),
                               rounded.predict(X), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("activation", ["identity", "relu", "logistic", "tanh"])
@pytest.mark.parametrize("n_classes", [2, 3, 4])
def test_network_classification_iteration_contract(activation, n_classes):
    X, y = make_blobs(n_samples=60, n_features=2, centers=n_classes,
                      cluster_std=0.05, random_state=10)
    model = MLPClassifier(hidden_layer_sizes=(4,), activation=activation,
                          solver="lbfgs", random_state=3, max_iter=2000).fit(X, y)
    description = describe_model(model)
    _assert_network_description(description, model, X, regression=False)


def _assert_network_description(description, model, X, *, regression):
    payload = description.to_dict()
    for token in ("dense(", "relu(", "sigmoid(", "tanh(", "activation(", "classify("):
        assert token not in description.canonical
    for name in ("dense", "activation", "relu", "sigmoid", "tanh", "classify"):
        assert f'"type": "{name}"' not in json.dumps(payload)
    nodes = tuple(_nodes(description.ast))
    assert any(isinstance(node, Sum) for node in nodes)
    assert any(isinstance(node, Comprehension) for node in nodes)
    assert render(parse(description.canonical)) == description.canonical
    restored = ModelDescription.from_dict(json.loads(json.dumps(payload)))
    assert restored.to_dict() == payload
    predictions = execute(description.ast, X)
    np.testing.assert_array_equal(predictions, execute(description.canonical, X))
    np.testing.assert_array_equal(predictions, execute(restored.ast, X))
    if regression:
        np.testing.assert_allclose(predictions, model.predict(X), rtol=0.02, atol=0.03)
    else:
        np.testing.assert_array_equal(predictions, model.predict(X))


def test_network_metric_integration_uses_bounded_program():
    X = np.tile([[-1., 0.], [0., -1.], [0., 1.], [1., 0.]], (20, 1))
    y = 2 * X[:, 0] + X[:, 1]
    model = MLPRegressor(hidden_layer_sizes=(2,), activation="identity",
                         solver="lbfgs", random_state=2, max_iter=1000).fit(X, y)
    description = describe_model(model)
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
