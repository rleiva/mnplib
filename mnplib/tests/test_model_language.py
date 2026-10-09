"""Contract tests for canonical model semantics and safe execution."""

import json
import pickle
from dataclasses import FrozenInstanceError
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

from mnplib import describe_model, ModelDescription, Surfeit, Nescience, Mismodel
from mnplib.models import sklearn_model_artifacts
from mnplib.models.language import (
    ModelNode, Constant, Label, Feature, Negate, Binary, Conditional,
    Vector, Element, Dense, Classify, normalize, render, parse, execute,
    node_to_dict, node_from_dict,
)
from mnplib.automl import CandidateEvaluator

@pytest.mark.parametrize("value,text", [
    (0, "0.00e+00"), (-0.0, "0.00e+00"), (1, "1.00e+00"),
    (-49.7, "-4.97e+01"), (49.736, "4.97e+01"), (49.756, "4.98e+01"),
    (0.001234, "1.23e-03"), (9.996, "1.00e+01"),
    (1e100, "1.00e+100"), (1e-100, "1.00e-100"),
    (np.nextafter(0.0, 1.0), "4.94e-324"),
])
def test_numeric_contract(value, text):
    node = Constant(value)
    assert render(node) == text
    assert render(parse(text)) == text
    assert node_from_dict(node_to_dict(node)) == normalize(node)
    np.testing.assert_array_equal(execute(node, [[0], [1]]), [float(text)] * 2)

@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_parameters_are_rejected(value):
    with pytest.raises(ValueError, match="finite"):
        render(Constant(value))

@pytest.mark.parametrize("node,expected", [
    (Binary("*", Constant(1), Feature(0)), Feature(0)),
    (Binary("*", Feature(0), Constant(1)), Feature(0)),
    (Binary("*", Constant(0), Feature(0)), Constant(0)),
    (Binary("*", Feature(0), Constant(0)), Constant(0)),
    (Binary("+", Feature(0), Constant(0)), Feature(0)),
    (Binary("+", Constant(0), Feature(0)), Feature(0)),
    (Binary("-", Feature(0), Constant(0)), Feature(0)),
    (Negate(Negate(Feature(0))), Feature(0)),
    (Binary("+", Feature(0), Constant(-2)), Binary("-", Feature(0), Constant(2))),
])
def test_local_normalization_rules(node, expected):
    assert normalize(node) == expected
    assert normalize(normalize(node)) == expected

@pytest.mark.parametrize("text", [
    "x2+x0+x1", "x0+(x1+x2)", "x0-(x1-x2)", "x0*(x1/x2)",
    "(x0**x1)**x2", "x0**x1**x2", "(-x0)**x1", "-x0**x1",
    "x0 if x1<x2 else x1 if x0>=x2 else x2",
    "(x0 if x1==x2 else x1) if x0>x2 else x2",
    "[x0,x1/x2]", 'label("\\"class A\\"")',
])
def test_text_ast_json_roundtrips_preserve_grouping(text):
    node = parse(text)
    canonical = render(node)
    assert parse(canonical) == node
    assert node_from_dict(node_to_dict(node)) == node
    assert render(parse(canonical)) == canonical

@pytest.mark.parametrize("text", [
    "import os", '__import__("os")', "x0.__class__", "lambda x: x",
    "[x for x in x0]", "while True: pass", "exec(x0)", "eval(x0)",
    "arbitrary_function(x0)", "x0=1", "x0[0]", "open(x0)", "sin(x0)",
    'dense([x0],[[1]],[0],"arbitrary")',
])
def test_grammar_rejects_undeclared_constructs(text):
    with pytest.raises(ValueError):
        parse(text)

def test_unknown_nodes_operators_and_indices_raise():
    with pytest.raises(TypeError, match="Unsupported model node"):
        execute(ModelNode(), [[0]])
    with pytest.raises(ValueError, match="Unsupported operator"):
        execute(Binary("%", Feature(0), Constant(2)), [[0]])
    with pytest.raises(ValueError, match="out of bounds"):
        execute(Feature(2), [[0]])
    with pytest.raises(ValueError, match="non-negative integers"):
        render(Feature(-1))
    with pytest.raises(ValueError, match="Unsupported semantic node"):
        node_from_dict({"type": "import", "module": "os"})

def test_conditional_evaluation_is_lazy_and_threshold_equality_is_inclusive():
    node = Conditional(Binary("<=", Feature(0), Constant(0)),
                       Constant(2), Binary("/", Constant(1), Feature(0)))
    with np.errstate(divide="raise"):
        np.testing.assert_array_equal(execute(node, [[0], [1], [2]]), [2, 1, 0.5])

def test_tree_threshold_and_nonordinal_labels():
    X = np.array([[0], [1], [2], [3]])
    model = DecisionTreeClassifier(max_depth=1).fit(X, ["z", "z", "a", "a"])
    description = describe_model(model, feature_names=["measurement"])
    points = [[0], [1.5], [3]]
    np.testing.assert_array_equal(execute(description.ast, points), model.predict(points))
    assert description.to_dict()["ast"]["condition"]["left"]["name"] == "measurement"

@pytest.mark.parametrize("labels", [
    np.array([123456, 123457]), np.array(["a", 'b"\\c']), np.array([False, True]),
])
def test_categorical_labels_are_not_quantized(labels):
    model = DecisionTreeClassifier().fit([[0], [1]], labels)
    description = describe_model(model)
    np.testing.assert_array_equal(execute(description.canonical, [[0], [1]]), labels)

@pytest.mark.parametrize("model", [
    LogisticRegression(max_iter=1000), LinearSVC(random_state=42, max_iter=10000),
    GaussianNB(), DecisionTreeClassifier(max_depth=3, random_state=42),
    MLPClassifier(hidden_layer_sizes=(3,), solver="lbfgs", max_iter=1000, random_state=42),
])
def test_classification_execution_matches_fitted_estimator(model):
    X, y = load_iris(return_X_y=True)
    model.fit(X, y)
    description = describe_model(model)
    np.testing.assert_array_equal(execute(description.ast, X), model.predict(X))
    np.testing.assert_array_equal(execute(description.canonical, X), execute(description.ast, X))

@pytest.mark.parametrize("activation", ["identity", "relu", "logistic", "tanh"])
def test_network_activations_have_executable_semantics(activation):
    X = np.random.default_rng(5).normal(size=(100, 2))
    y = (X[:, 0] > 0).astype(int)
    model = MLPClassifier(activation=activation, hidden_layer_sizes=(2,),
                          solver="lbfgs", random_state=1, max_iter=2000).fit(X, y)
    description = describe_model(model)
    np.testing.assert_array_equal(execute(description.ast, X), model.predict(X))

@pytest.mark.parametrize("model", [
    LinearRegression(), LinearSVR(random_state=42, max_iter=10000),
    DecisionTreeRegressor(max_depth=2, random_state=42),
    MLPRegressor(hidden_layer_sizes=(3,), solver="lbfgs", random_state=42, max_iter=1000),
])
def test_regression_execution_quantization_tolerance(model):
    X = np.random.default_rng(4).integers(-2, 3, size=(100, 3)).astype(float)
    y = 1 + 2 * X[:, 0] - 3 * X[:, 2]
    model.fit(X, y)
    description = describe_model(model)
    np.testing.assert_allclose(execute(description.ast, X), model.predict(X), rtol=0.02, atol=0.03)
    np.testing.assert_array_equal(execute(description.ast, X), execute(description.canonical, X))

@pytest.mark.parametrize("model", [
    DecisionTreeRegressor(max_depth=2, random_state=42),
    MLPRegressor(hidden_layer_sizes=(2,), solver="lbfgs", random_state=42, max_iter=1000),
])
def test_multioutput_regression(model):
    X = np.random.default_rng(4).integers(-1, 2, size=(80, 2)).astype(float)
    model.fit(X, np.column_stack([X[:, 0], 2 * X[:, 1]]))
    description = describe_model(model)
    np.testing.assert_allclose(execute(description.ast, X), model.predict(X), rtol=0.02, atol=0.03)


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("n_outputs", [1, 2])
def test_linear_regression_requires_a_one_dimensional_target(fit_intercept, n_outputs):
    X = np.random.default_rng(4).normal(size=(40, 2))
    model = LinearRegression(fit_intercept=fit_intercept).fit(X, X[:, :n_outputs])
    with pytest.raises(ValueError, match="one-dimensional target"):
        describe_model(model)


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("model_type", [LinearRegression, LinearSVR])
def test_single_target_linear_regression_intercepts(model_type, fit_intercept):
    X = np.tile([[-1., 0.], [0., -1.], [0., 1.], [1., 0.]], (20, 1))
    y = 2 * X[:, 0] - 0.5 * X[:, 1] + float(fit_intercept)
    model = model_type(fit_intercept=fit_intercept).fit(X, y)
    description = describe_model(model)
    assert not isinstance(description.ast, Vector)
    predictions = execute(description.ast, X)
    assert predictions.shape == y.shape
    np.testing.assert_allclose(predictions, model.predict(X), rtol=0.02, atol=0.03)

def test_intercept_only_and_single_node_models():
    X = np.zeros((5, 2))
    for model, y in [(LinearRegression(), np.full(5, 2.0)),
                     (DecisionTreeClassifier(), np.full(5, "only"))]:
        model.fit(X, y)
        description = describe_model(model)
        assert description.metadata["features_used"] == ()
        np.testing.assert_array_equal(execute(description.ast, X), y)

def test_dataframe_names_mapping_and_json_schema():
    X = pd.DataFrame({"a": [0, 1, 2, 3], "b": [2, -2, 1, -1]})
    model = LinearRegression().fit(X, 1 + 2 * X.a - 3 * X.b)
    description = describe_model(model, feature_indices=[3, 1])
    assert description.feature_names == ("a", "b")
    original = np.zeros((len(X), 4))
    original[:, [3, 1]] = X
    np.testing.assert_allclose(execute(description.ast, original), model.predict(X))
    assert "a" not in description.canonical
    payload = json.loads(json.dumps(description.to_dict(), allow_nan=False))
    assert payload["schema"] == "mnplib-model"
    assert payload["schema_version"] == 1
    assert payload["ast"]["type"] == "subtract"
    assert ModelDescription.from_dict(payload).ast == description.ast
    assert payload["metadata"]["features_used"] == [3, 1]
    assert "predictions" not in payload
    assert describe_model(model, feature_indices=[3, 1]).canonical == description.canonical
    with pytest.raises(FrozenInstanceError):
        description.ast = Constant(0)
    with pytest.raises(TypeError):
        description.metadata["color"] = "red"

def test_quantized_ast_has_no_hidden_parameter_precision():
    node = Binary("*", Constant(1.23456), Feature(0))
    description = ModelDescription("linear", node, (0,), ("x0",))
    assert description.ast.left.value == 1.23
    assert description.to_dict()["ast"]["left"]["value"] == 1.23
    assert execute(description.ast, [[10]])[0] == pytest.approx(12.3)

def test_metric_and_preprocessing_integration():
    X = np.random.default_rng(10).integers(0, 3, size=(200, 3)).astype(float)
    y = X[:, 2] - X[:, 0]
    selected = [2, 0]
    scaler = StandardScaler().fit(X[:, selected])
    scaled = scaler.transform(X[:, selected])
    model = LinearRegression().fit(scaled, y)
    metric = Nescience().fit(X, y)
    result = CandidateEvaluator(X=X, y=y, nescience=metric,
                               feature_names=["a", "b", "c"]).evaluate(
        name="scaled", family="linear", model=model, feature_indices=selected,
        X_adapter=scaled, input_transformer=scaler,
    )
    description = result.artifacts.description
    np.testing.assert_allclose(execute(description.ast, X), model.predict(scaled), atol=0.01)
    surfeit = Surfeit().fit(X, y)
    direct = describe_model(model, feature_indices=selected)
    assert surfeit.surfeit_model(model, X=scaled, feature_indices=selected) == surfeit.surfeit_string(direct.canonical)
    artifacts = sklearn_model_artifacts(model, scaled, feature_indices=selected)
    assert artifacts.model_string == direct.canonical
    assert result.components["surfeit"] == metric.mismodel_.surfeit_.surfeit_string(description.canonical)
    mismodel = Mismodel().fit(scaled, y)
    assert mismodel.mismodel_model(model) == pytest.approx(
        mismodel.mismodel(predictions=model.predict(scaled), model_string=describe_model(model).canonical)
    )


def test_categorical_score_ties_choose_the_first_label():
    node = Classify(Vector((Constant(1), Constant(1))), (Label("first"), Label("second")))
    np.testing.assert_array_equal(execute(node, [[0], [1]]), ["first", "first"])


def test_description_pickle_preserves_immutable_semantics():
    description = ModelDescription("test", Feature(0, "a"), (0,), ("a",),
                                   {"features_used": [0]})
    restored = pickle.loads(pickle.dumps(description))
    assert restored.to_dict() == description.to_dict()
    with pytest.raises(TypeError):
        restored.metadata["features_used"] = []


@pytest.mark.parametrize("change", [
    {"schema_version": 2}, {"schema_version": True},
    {"schema": "other"}, {"canonical": "x1"},
    {"ast": {"type": "arbitrary_function"}},
    {"feature_indices": [-1]},
])
def test_invalid_description_payload_is_rejected(change):
    payload = ModelDescription("test", Feature(0), (0,), ("a",)).to_dict()
    payload.update(change)
    with pytest.raises(ValueError):
        ModelDescription.from_dict(payload)


@pytest.mark.parametrize("indices", [[0.5], [-1], [True], [0, 0]])
def test_invalid_estimator_coordinate_mapping_is_rejected(indices):
    model = LinearRegression().fit([[0], [1]], [0, 1])
    with pytest.raises(ValueError, match="feature_indices"):
        describe_model(model, feature_indices=indices)


def test_multioutput_classifier_preserves_labels():
    X = np.arange(12).reshape(-1, 1)
    y = np.column_stack([X[:, 0] > 5, X[:, 0] > 7]).astype(int)
    model = DecisionTreeClassifier().fit(X, y)
    np.testing.assert_array_equal(execute(describe_model(model).ast, X), model.predict(X))


@pytest.mark.parametrize("model", [
    LogisticRegression(), LinearSVC(random_state=2), GaussianNB(),
    DecisionTreeClassifier(random_state=2),
    MLPClassifier(hidden_layer_sizes=(2,), solver="lbfgs", random_state=2, max_iter=1000),
])
def test_classifiers_execute_in_original_feature_coordinates(model):
    X = np.random.default_rng(2).choice([-2.0, 2.0], size=(80, 4))
    selected = [3, 1]
    local = X[:, selected]
    y = np.where(X[:, 3] > 0, "positive", "negative")
    model.fit(local, y)
    description = describe_model(model, feature_indices=selected,
                                 feature_names=["fourth", "second"])
    assert description.feature_indices == (3, 1)
    np.testing.assert_array_equal(execute(description.canonical, X), model.predict(local))
    assert "fourth" not in description.canonical


@pytest.mark.parametrize("model", [DecisionTreeClassifier(), DecisionTreeRegressor()])
@pytest.mark.parametrize("missing_label", [0, 1])
def test_tree_nan_routing_matches_fitted_model(model, missing_label):
    X = np.array([[0], [1], [2], [3], [np.nan], [np.nan]])
    y = np.array([0, 0, 1, 1, missing_label, missing_label])
    model.fit(X, y)
    np.testing.assert_array_equal(execute(describe_model(model).ast, X), model.predict(X))


def test_single_class_network_has_constant_semantics():
    X = np.arange(12).reshape(6, 2)
    model = MLPClassifier(hidden_layer_sizes=(2,), solver="lbfgs",
                          random_state=1).fit(X, ["only"] * 6)
    description = describe_model(model)
    assert description.metadata["features_used"] == ()
    assert description.ast == Label("only")
    np.testing.assert_array_equal(execute(description.ast, X), model.predict(X))


def test_closed_vector_operations_validate_dimensions():
    with pytest.raises(ValueError, match="out of bounds"):
        execute(Element(Vector((Constant(1),)), 1), [[0]])
    with pytest.raises(ValueError, match="match the bias"):
        normalize(Dense(Vector((Feature(0),)), ((Constant(1),),), (Constant(1), Constant(2)), "relu"))
    with pytest.raises(ValueError, match="input dimension"):
        execute(Dense(Vector((Feature(0), Feature(1))), ((Constant(1),),), (Constant(1),), "identity"),
                [[0, 1]])


def test_metrics_share_the_finite_window_forecaster_description():
    from mnplib.timeseries import FixedLinearForecaster
    X = np.tile([[0., 1.], [1., 0.], [1., 1.]], (50, 1))
    y = X.mean(axis=1)
    model = FixedLinearForecaster([0.5, 0.5]).fit(X, y)
    description = describe_model(model)
    metric = Surfeit().fit(X, y)
    assert metric.surfeit_model(model) == metric.surfeit_string(description.canonical)
    nescience = Nescience().fit(X, y)
    assert nescience.nescience_model(model) == pytest.approx(
        nescience.nescience(subset=[0, 1], predictions=model.predict(X),
                            model_string=description.canonical)
    )
