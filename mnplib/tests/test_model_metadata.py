"""Compact semantic hints independent of prediction and metric evaluation."""

from dataclasses import replace
import json

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from mnplib import ModelDescription, describe_model, Surfeit, Mismodel, Nescience
from mnplib.automl import CandidateEvaluator
from mnplib.models import sklearn_model_artifacts
from mnplib.models.language import execute
from mnplib.models.serializers.base import SklearnSerializer
from mnplib.models.serializers import time_series
from mnplib.timeseries import FixedLinearForecaster


@pytest.fixture(params=[
    "linear", "logistic", "gaussian", "tree_classifier", "tree_regressor",
    "linear_svc", "linear_svr", "mlp_classifier", "mlp_regressor", "forecaster",
])
def fitted_case(request):
    X = np.tile([[-2., -1.], [-2., 1.], [0., -1.], [0., 1.], [2., -1.], [2., 1.]], (15, 1))
    classes = np.tile([0, 0, 1, 1, 2, 2], 15)
    regression = 1 + 2 * X[:, 0] + 0.5 * X[:, 1]
    models = {
        "linear": (LinearRegression(), regression),
        "logistic": (LogisticRegression(max_iter=1000), classes),
        "gaussian": (GaussianNB(), classes),
        "tree_classifier": (DecisionTreeClassifier(max_depth=2, random_state=2), classes),
        "tree_regressor": (DecisionTreeRegressor(max_depth=2, random_state=2), regression),
        "linear_svc": (LinearSVC(random_state=2, max_iter=10000), classes),
        "linear_svr": (LinearSVR(random_state=2, max_iter=10000), regression),
        "mlp_classifier": (MLPClassifier(hidden_layer_sizes=(3, 2), activation="tanh",
                                         solver="lbfgs", random_state=2, max_iter=1000), classes),
        "mlp_regressor": (MLPRegressor(hidden_layer_sizes=(3, 2), activation="relu",
                                       solver="lbfgs", random_state=2, max_iter=1000), regression),
        "forecaster": (FixedLinearForecaster([0.5, 0.]), regression),
    }
    model, y = models[request.param]
    return request.param, model.fit(X, y), X, y


def _artifacts(model, X, **kwargs):
    factory = (time_series.time_series_model_artifacts if isinstance(model, FixedLinearForecaster)
               else sklearn_model_artifacts)
    return factory(model, X, **kwargs)


def test_family_metadata_is_small_and_json_compatible(fitted_case):
    name, model, X, _ = fitted_case
    description = describe_model(model)
    payload = json.loads(json.dumps(description.to_dict(), allow_nan=False))
    metadata = payload["metadata"]
    expected = {"features_used": _artifacts(model, X).subset}
    if name in {"linear", "logistic", "linear_svc", "linear_svr"}:
        expected["n_terms"] = int(np.count_nonzero(model.coef_))
    if name in {"logistic", "gaussian", "linear_svc", "mlp_classifier"}:
        expected["n_classes"] = int(len(model.classes_))
    if name == "gaussian":
        expected["distribution"] = "gaussian"
    if name.startswith("tree_"):
        expected.update(depth=model.get_depth(), n_nodes=int(model.tree_.node_count),
                        n_leaves=model.get_n_leaves())
    if name.startswith("linear_sv"):
        expected["kernel"] = "linear"
        assert "n_support_vectors" not in metadata
    if name.startswith("mlp_"):
        expected.update(hidden_layer_sizes=[3, 2],
                        activations=[model.activation, model.activation, model.out_activation_])
    if name == "forecaster":
        expected["n_terms"] = 1
    assert metadata == expected
    assert payload["model_type"] == type(model).__name__
    assert "model_type" not in metadata
    assert "features_used" not in payload
    assert ModelDescription.from_dict(payload).to_dict() == payload


@pytest.mark.parametrize("metadata", [
    {},
    {"features_used": [999], "activation": "unknown", "activations": ["identity"],
     "kernel": "rbf", "distribution": "categorical", "n_classes": 100},
])
def test_metadata_is_not_an_execution_or_scoring_input(fitted_case, metadata, monkeypatch):
    _, model, X, y = fitted_case
    description = describe_model(model)
    artifacts = _artifacts(model, X)
    ast_predictions = execute(description.ast, X)
    estimator_predictions = model.predict(X)
    surfeit = Surfeit().fit(X, y)
    mismodel = Mismodel().fit(X, y)
    nescience = Nescience().fit(X, y)
    scores = (surfeit.surfeit_model(model), mismodel.mismodel_model(model),
              nescience.nescience_model(model))

    changed = replace(description, metadata=metadata)
    restored = ModelDescription.from_dict(json.loads(json.dumps(changed.to_dict())))
    assert restored.metadata == changed.metadata
    assert restored.ast == description.ast
    assert restored.canonical == description.canonical
    np.testing.assert_array_equal(execute(restored.ast, X), ast_predictions)
    np.testing.assert_array_equal(execute(restored.canonical, X), ast_predictions)
    np.testing.assert_array_equal(model.predict(X), estimator_predictions)

    def altered_description(*args, **kwargs):
        return changed

    if isinstance(model, FixedLinearForecaster):
        monkeypatch.setattr(time_series, "fixed_model_description", altered_description)
    else:
        monkeypatch.setattr(SklearnSerializer, "description", altered_description)
    altered_artifacts = _artifacts(model, X)
    assert altered_artifacts.description.metadata == changed.metadata
    assert altered_artifacts.subset == artifacts.subset
    assert altered_artifacts.model_string == artifacts.model_string
    np.testing.assert_array_equal(altered_artifacts.predictions, artifacts.predictions)
    assert surfeit.surfeit_model(model) == scores[0]
    assert mismodel.mismodel_model(model) == scores[1]
    assert nescience.nescience_model(model) == scores[2]


@pytest.mark.parametrize("model_type", [LinearRegression, FixedLinearForecaster])
def test_feature_mapping_is_independent_of_metadata(model_type, monkeypatch):
    X = np.tile([[-1., 0.], [0., -1.], [0., 1.], [1., 0.]], (20, 1))
    y = X[:, 0]
    model = (FixedLinearForecaster([1., 0.]) if model_type is FixedLinearForecaster
             else LinearRegression()).fit(X, y)
    selected = [3, 1]
    description = describe_model(model, feature_indices=selected)
    assert description.metadata["features_used"] == (3,)
    changed = replace(description, metadata={"features_used": [1]})

    def altered_description(*args, **kwargs):
        return changed

    if model_type is FixedLinearForecaster:
        monkeypatch.setattr(time_series, "fixed_model_description", altered_description)
    else:
        monkeypatch.setattr(SklearnSerializer, "description", altered_description)
    artifacts = _artifacts(model, X, feature_indices=selected)
    assert artifacts.subset == [3]
    original = np.zeros((len(X), 4))
    original[:, selected] = X
    metric = Nescience().fit(original, y)
    assert metric.nescience_model(model, X=X, feature_indices=selected) == pytest.approx(
        metric.nescience(**artifacts.to_nescience_kwargs()))


@pytest.mark.parametrize("hidden", [(), 2, (2, 3)])
@pytest.mark.parametrize("model_type", [MLPClassifier, MLPRegressor])
def test_network_metadata_describes_fitted_activation_structure(hidden, model_type):
    X = np.tile([[-1., 0.], [0., -1.], [0., 1.], [1., 0.]], (10, 1))
    y = (X[:, 0] > 0).astype(int) if model_type is MLPClassifier else X[:, 0]
    model = model_type(hidden_layer_sizes=hidden, activation="logistic", solver="lbfgs",
                       random_state=2, max_iter=1000).fit(X, y)
    expected = [] if hidden == () else [2] if hidden == 2 else [2, 3]
    metadata = describe_model(model).to_dict()["metadata"]
    assert metadata["hidden_layer_sizes"] == expected
    output_activation = "logistic" if model_type is MLPClassifier else "identity"
    assert metadata["activations"] == ["logistic"] * len(expected) + [output_activation]
    assert "n_outputs" not in metadata
    assert "n_layers" not in metadata


def test_automl_keeps_search_parameters_separate_from_model_hints(monkeypatch):
    X = np.tile([[-2., -1.], [-2., 1.], [2., -1.], [2., 1.]], (20, 1))
    y = (X[:, 0] > 0).astype(int)
    model = LogisticRegression(C=0.25, solver="lbfgs", max_iter=200).fit(X, y)
    metric = Nescience().fit(X, y)
    evaluator = CandidateEvaluator(X=X, y=y, nescience=metric, feature_names=["a", "b"])
    parameters = {"C": 0.25, "solver": "lbfgs", "max_iter": 200}
    result = evaluator.evaluate(name="candidate", family="logistic", model=model,
                                hyperparameters=parameters)
    metadata = result.artifacts.description.metadata
    assert not metadata.keys() & parameters.keys()
    assert "candidate" not in metadata
    assert result.hyperparameters == parameters
    changed = replace(result.artifacts.description, metadata={"features_used": [], "n_classes": 99})
    monkeypatch.setattr(SklearnSerializer, "description", lambda *args, **kwargs: changed)
    repeated = evaluator.evaluate(name="candidate", family="logistic", model=model,
                                  hyperparameters=parameters)
    assert repeated.nescience == result.nescience
    assert repeated.components == result.components
    assert repeated.artifacts.subset == result.artifacts.subset
    assert repeated.hyperparameters == result.hyperparameters
