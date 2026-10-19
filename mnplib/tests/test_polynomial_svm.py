"""Explicit polynomial kernels, fitted SVM predictions, and canonical fidelity."""

from copy import deepcopy
from dataclasses import fields
import json

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from sklearn.exceptions import NotFittedError
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.svm import SVC, SVR

from mnplib import ModelDescription, describe_model, Surfeit, Mismodel, Nescience
from mnplib.models import sklearn_model_artifacts
from mnplib.models.language import (
    ModelNode, Constant, Feature, Conditional, Vector, Element, Sum,
    execute, parse, render,
)
from mnplib.models.language.normalize import quantize
from mnplib.models.serializers.svm import PolynomialSVMSerializer, _polynomial_kernel


def _nodes(node):
    if isinstance(node, ModelNode):
        yield node
        for field in fields(node):
            yield from _nodes(getattr(node, field.name))
    elif isinstance(node, tuple):
        for child in node:
            yield from _nodes(child)


def _roundtrip(description, X):
    assert parse(description.canonical) == description.ast
    assert render(parse(description.canonical)) == description.canonical
    payload = json.loads(json.dumps(description.to_dict(), allow_nan=False))
    restored = ModelDescription.from_dict(payload)
    assert restored.to_dict() == payload
    predictions = execute(description.ast, X)
    np.testing.assert_array_equal(predictions, execute(description.canonical, X))
    np.testing.assert_array_equal(predictions, execute(restored.ast, X))
    return predictions


@pytest.mark.parametrize("degree", [2, 3, 4])
def test_polynomial_kernel_is_explicit_dot_product_and_power(degree):
    X = np.array([[1., 2.], [-1., 3.], [0., 0.]])
    support = np.array([[2., -1.]])
    node = _polynomial_kernel(
        (Feature(0), Feature(1)),
        Vector((Vector((Constant(2), Constant(-1))),)), 0,
        gamma=0.5, coef0=1, degree=degree,
    )
    dot = node.left.left.right
    assert isinstance(dot, Sum)
    np.testing.assert_array_equal(execute(dot, X), (X @ support.T)[:, 0])
    np.testing.assert_allclose(execute(node, X),
                               polynomial_kernel(X, support, gamma=0.5, coef0=1, degree=degree)[:, 0])


@pytest.mark.parametrize("degree", [2, 3, 4])
@pytest.mark.parametrize("gamma", ["scale", "auto", 0.5])
def test_polynomial_svr_execution_and_roundtrips(degree, gamma):
    rng = np.random.default_rng(4)
    X = rng.uniform(-1, 1, size=(80, 2))
    y = X[:, 0] ** degree + 0.5 * X[:, 1]
    model = SVR(kernel="poly", degree=degree, gamma=gamma, coef0=1, C=1).fit(X, y)
    description = describe_model(model)
    points = np.concatenate([X, rng.uniform(-1, 1, size=(30, 2))])
    predictions = _roundtrip(description, points)
    np.testing.assert_allclose(predictions, model.predict(points), rtol=0.02, atol=0.03)
    assert model.dual_coef_.shape == (1, len(model.support_))
    assert model.intercept_.shape == (1,)
    assert description.model_type == "SVR"
    assert dict(description.metadata) == {
        "features_used": (0, 1), "kernel": "poly", "degree": degree,
        "n_support_vectors": len(model.support_),
    }
    sums = [node for node in _nodes(description.ast) if isinstance(node, Sum)]
    assert [(node.index, node.size) for node in sums] == [("i", len(model.support_)), ("j", 2)]
    score = description.ast.right
    assert isinstance(score.expression.left, Element)
    assert len(score.expression.left.vector.items) == len(model.support_)
    assert not any(token in description.canonical for token in ("svm(", "kernel(", "exp(", "tanh("))


@pytest.mark.parametrize("degree", [2, 3, 4])
def test_polynomial_svc_nonlinear_classification(degree):
    X = np.tile([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]], (10, 1))
    y = np.where(X[:, 0] * X[:, 1] > 0, "same", "different")
    model = SVC(kernel="poly", degree=degree, gamma=0.5, coef0=1).fit(X, y)
    description = describe_model(model)
    np.testing.assert_array_equal(_roundtrip(description, X), model.predict(X))
    np.testing.assert_array_equal(model.predict(X), y)
    assert isinstance(description.ast, Conditional)
    assert description.ast.condition.operator == ">="
    assert description.model_type == "SVC"
    assert dict(description.metadata) == {
        "features_used": (0, 1), "kernel": "poly", "degree": degree,
        "n_support_vectors": len(model.support_), "n_classes": 2,
    }


@pytest.mark.parametrize("labels", [["z", "a"], ["a", "z"], [7, 2], [False, True]])
def test_binary_svc_sign_ordering_and_zero_boundary(labels):
    model = SVC(kernel="poly", degree=3, gamma=1, C=1).fit([[-1.], [1.]], labels)
    X = np.array([[-1.], [-0.01], [0.], [0.01], [1.]])
    description = describe_model(model)
    score = description.ast.condition.left
    np.testing.assert_allclose(execute(score, X), model.decision_function(X), atol=1e-15)
    assert model.decision_function([[0.]])[0] == 0
    assert model.predict([[0.]])[0] == model.classes_[1]
    np.testing.assert_array_equal(_roundtrip(description, X), model.predict(X))


@pytest.mark.parametrize("model_type", [SVC, SVR])
@pytest.mark.parametrize("sparse", [False, True])
def test_polynomial_svm_sparse_parameters_and_feature_mapping(model_type, sparse):
    X = np.tile([[-1., 0.], [1., 0.]], (20, 1))
    y = (X[:, 0] > 0).astype(int) if model_type is SVC else X[:, 0]
    source = csr_matrix(X) if sparse else pd.DataFrame(X, columns=["active", "zero"])
    model = model_type(kernel="poly", degree=3, gamma=1).fit(source, y)
    description = describe_model(model, feature_indices=[3, 1], feature_names=["active", "zero"])
    assert description.feature_indices == (3, 1)
    assert description.feature_names == ("active", "zero")
    assert description.metadata["features_used"] == (3,)
    artifacts = sklearn_model_artifacts(model, source, feature_indices=[3, 1])
    assert artifacts.subset == [3]
    original = np.zeros((len(X), 4))
    original[:, 3] = X[:, 0]
    original[:, 1] = np.arange(len(X))
    actual = _roundtrip(description, original)
    if model_type is SVC:
        np.testing.assert_array_equal(actual, model.predict(source))
    else:
        np.testing.assert_allclose(actual, model.predict(source), rtol=0.02, atol=0.03)


@pytest.mark.parametrize("model_type", [SVC, SVR])
@pytest.mark.parametrize("parameters", [{"degree": 0}, {"gamma": 0}])
def test_constant_polynomial_kernel_has_no_effective_features(model_type, parameters):
    X = np.tile([[-1., 1.], [1., -1.]], (10, 1))
    y = (X[:, 0] > 0).astype(int) if model_type is SVC else X[:, 0]
    model = model_type(kernel="poly", **parameters).fit(X, y)
    assert PolynomialSVMSerializer().subset(model) == []
    points = np.array([[100., -100.], [-5., 42.]])
    np.testing.assert_allclose(_roundtrip(describe_model(model), points), model.predict(points))


def test_polynomial_svr_with_no_support_vectors():
    X = np.array([[-1.], [0.], [1.]])
    model = SVR(kernel="poly", epsilon=10).fit(X, [1., 1., 1.])
    assert model.support_vectors_.shape == (0, 1)
    description = describe_model(model)
    assert description.metadata["features_used"] == ()
    assert description.metadata["n_support_vectors"] == 0
    np.testing.assert_array_equal(_roundtrip(description, X), model.predict(X))


@pytest.mark.parametrize("model_type", [SVC, SVR])
@pytest.mark.parametrize("kernel,message", [
    ("rbf", "RBF.*require exp.*not part"),
    ("sigmoid", "Sigmoid.*require tanh.*not part"),
    ("linear", "Use LinearSVC or LinearSVR"),
    ("precomputed", "Unsupported SVM kernel 'precomputed'"),
    (lambda X, Y: X @ Y.T, "Callable SVM kernels"),
])
def test_unsupported_svm_kernels_fail_clearly(model_type, kernel, message):
    X = np.array([[-1., 0.], [0., -1.], [0., 1.], [1., 0.]])
    y = [0, 0, 1, 1]
    source = X @ X.T if kernel == "precomputed" else X
    model = model_type(kernel=kernel).fit(source, y)
    with pytest.raises(ValueError, match=message):
        describe_model(model)
    with pytest.raises(ValueError, match=message):
        sklearn_model_artifacts(model, source)


def test_multiclass_polynomial_svc_is_explicitly_rejected():
    model = SVC(kernel="poly").fit([[-2.], [-1.], [0.], [1.], [2.], [3.]], [0, 0, 1, 1, 2, 2])
    with pytest.raises(ValueError, match="binary classification only"):
        describe_model(model)


@pytest.mark.parametrize("model_type", [SVC, SVR])
def test_polynomial_svm_requires_fitting(model_type):
    with pytest.raises(NotFittedError):
        describe_model(model_type(kernel="poly"))


def test_polynomial_degree_must_have_exact_canonical_representation():
    model = SVR(kernel="poly").fit([[-1.], [0.], [1.]], [-1., 0., 1.])
    model.degree = 1001
    with pytest.raises(ValueError, match="degree must be exactly representable"):
        describe_model(model)


def test_polynomial_svm_uses_only_canonical_parameter_precision():
    X = np.random.default_rng(4).uniform(-1, 1, size=(80, 2))
    model = SVR(kernel="poly", degree=3, gamma=0.56789, coef0=1.23456).fit(X, X[:, 0] ** 3)
    rounded = deepcopy(model)
    q = np.vectorize(quantize)
    rounded.support_vectors_ = q(model.support_vectors_)
    rounded._dual_coef_ = rounded.dual_coef_ = q(model.dual_coef_)
    rounded._intercept_ = rounded.intercept_ = q(model.intercept_)
    rounded._gamma, rounded.coef0 = quantize(model._gamma), quantize(model.coef0)
    np.testing.assert_allclose(_roundtrip(describe_model(model), X), rounded.predict(X),
                               rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("model_type", [SVC, SVR])
def test_polynomial_svm_metrics_share_the_canonical_description(model_type):
    X = np.tile([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]], (20, 1))
    y = (X[:, 0] * X[:, 1] > 0).astype(int) if model_type is SVC else X[:, 0] * X[:, 1]
    model = model_type(kernel="poly", degree=2, coef0=1).fit(X, y)
    description = describe_model(model)
    artifacts = sklearn_model_artifacts(model, X)
    assert artifacts.model_string == description.canonical
    surfeit = Surfeit().fit(X, y)
    assert surfeit.surfeit_model(model) == surfeit.surfeit_string(description.canonical)
    mismodel = Mismodel().fit(X, y)
    assert mismodel.mismodel_model(model) == pytest.approx(mismodel.mismodel(
        predictions=model.predict(X), model_string=description.canonical))
    nescience = Nescience().fit(X, y)
    assert nescience.nescience_model(model) == pytest.approx(nescience.nescience(
        **artifacts.to_nescience_kwargs()))
