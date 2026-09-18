"""Fitted scikit-learn model export for classification and regression."""

from io import BytesIO
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted

from mnplib.automl.wrappers import SelectedFeaturesEstimator, export_sklearn_model
from mnplib.classifier import NescienceClassifier
from mnplib.regressor import NescienceRegressor


MODEL_CASES = [
    (NescienceClassifier, "decision_tree", DecisionTreeClassifier),
    (NescienceClassifier, "logistic_regression", LogisticRegression),
    (NescienceClassifier, "naive_bayes", GaussianNB),
    (NescienceClassifier, "linear_svc", LinearSVC),
    (NescienceClassifier, "mlp", MLPClassifier),
    (NescienceRegressor, "decision_tree", DecisionTreeRegressor),
    (NescienceRegressor, "linear_regression", LinearRegression),
    (NescienceRegressor, "linear_svr", LinearSVR),
    (NescienceRegressor, "mlp", MLPRegressor),
]


def _estimator(model):
    if isinstance(model.model_, SelectedFeaturesEstimator):
        return model.model_.estimator
    return model.model_


def _model_input(model, X):
    if isinstance(model.model_, SelectedFeaturesEstimator):
        return model.model_._select(X)
    return np.asarray(X)


@pytest.fixture(scope="module", params=MODEL_CASES)
def fitted_automl(request):
    automl_class, family, model_class = request.param
    rng = np.random.default_rng(42)
    values = rng.normal(size=(120, 4)) * [1, 5, 0.2, 2] + [0, 10, 3, -5]
    X = pd.DataFrame(values, columns=["a", "b", "c", "d"])
    y = values[:, 1] - 10 + values[:, 3] + 5
    if automl_class is NescienceClassifier:
        y = (y > 0).astype(int)
    model = automl_class(
        models=[family],
        n_bins=2,
        max_feature_prefixes=3,
        random_state=42,
        search_options={"mlp": {"max_candidates": 1, "max_iter": 10}},
    ).fit(X.iloc[:100], y[:100])
    return model, X.iloc[100:], y[100:], model_class


def test_get_model_returns_independent_fitted_estimator(fitted_automl):
    model, X, y, model_class = fitted_automl
    exported = model.get_model()

    assert type(exported) is model_class
    assert exported is not _estimator(model)
    check_is_fitted(exported)
    X_input = _model_input(model, X)
    np.testing.assert_allclose(exported.predict(X_input), model.predict(X))
    assert exported.score(X_input, y) == pytest.approx(model.score(X, y))

    predictions = model.predict(X)
    exported.fit(X_input, np.roll(y, 1))
    np.testing.assert_allclose(model.predict(X), predictions)


def test_pipeline_predictions_scores_and_methods_match(fitted_automl):
    model, X, y, model_class = fitted_automl
    pipeline = model.get_model(as_pipeline=True)

    assert isinstance(pipeline, Pipeline)
    assert type(pipeline.named_steps["estimator"]) is model_class
    assert pipeline.named_steps["estimator"] is not _estimator(model)
    assert pipeline.n_features_in_ == model.n_features_in_
    check_is_fitted(pipeline)
    for inputs in (X, X.to_numpy(), X.to_numpy().tolist()):
        np.testing.assert_allclose(pipeline.predict(inputs), model.predict(X))
        assert pipeline.score(inputs, y) == pytest.approx(model.score(X, y))

    expected_input = _model_input(model, X)
    np.testing.assert_allclose(pipeline[:-1].transform(X), expected_input)
    if hasattr(_estimator(model), "predict_proba"):
        np.testing.assert_allclose(pipeline.predict_proba(X), model.predict_proba(X))
        np.testing.assert_array_equal(pipeline.classes_, model.classes_)
    else:
        assert not hasattr(pipeline, "predict_proba")
    if hasattr(_estimator(model), "decision_function"):
        np.testing.assert_allclose(
            pipeline.decision_function(X.to_numpy()),
            model.model_.decision_function(X.to_numpy()),
        )


def test_pipeline_rejects_wrong_input_dimensions(fitted_automl):
    model, X, _, _ = fitted_automl
    pipeline = model.get_model(as_pipeline=True)
    with pytest.raises(ValueError, match="features"):
        pipeline.predict(X.to_numpy()[:, :-1])
    with pytest.raises(ValueError, match="features"):
        pipeline.predict(np.column_stack([X, np.ones(len(X))]))


def test_pipeline_preprocessing_is_an_independent_fitted_copy(fitted_automl):
    model, X, _, _ = fitted_automl
    pipeline = model.get_model(as_pipeline=True)
    if getattr(model.model_, "transformer", None) is None:
        assert "preprocessing" not in pipeline.named_steps
        return

    scaler = pipeline.named_steps["preprocessing"]
    assert scaler is not model.model_.transformer
    np.testing.assert_array_equal(scaler.mean_, model.model_.transformer.mean_)
    np.testing.assert_array_equal(scaler.scale_, model.model_.transformer.scale_)
    original_mean = model.model_.transformer.mean_.copy()
    predictions = model.predict(X)
    scaler.mean_[:] = 1000
    np.testing.assert_array_equal(model.model_.transformer.mean_, original_mean)
    np.testing.assert_allclose(model.predict(X), predictions)


def test_export_does_not_refit_estimator_or_preprocessing(fitted_automl, monkeypatch):
    model, X, _, model_class = fitted_automl

    def reject_fit(*args, **kwargs):
        raise AssertionError("Export must preserve trained parameters.")

    monkeypatch.setattr(model_class, "fit", reject_fit)
    if getattr(model.model_, "transformer", None) is not None:
        monkeypatch.setattr(type(model.model_.transformer), "fit", reject_fit)
    model.get_model()
    pipeline = model.get_model(as_pipeline=True)
    np.testing.assert_allclose(pipeline.predict(X.to_numpy()), model.predict(X))


@pytest.mark.parametrize("as_pipeline", [False, True])
def test_export_joblib_round_trip(fitted_automl, as_pipeline):
    model, X, _, _ = fitted_automl
    exported = model.get_model(as_pipeline=as_pipeline)
    buffer = BytesIO()
    joblib.dump(exported, buffer)
    buffer.seek(0)
    restored = joblib.load(buffer)
    inputs = X.to_numpy() if as_pipeline else _model_input(model, X)
    np.testing.assert_allclose(restored.predict(inputs), model.predict(X))


@pytest.mark.parametrize("automl_class", [NescienceClassifier, NescienceRegressor])
@pytest.mark.parametrize("as_pipeline", [False, True])
def test_get_model_requires_fit(automl_class, as_pipeline):
    with pytest.raises(NotFittedError):
        automl_class().get_model(as_pipeline=as_pipeline)


@pytest.mark.parametrize("as_pipeline", [None, "yes", 1])
def test_get_model_requires_boolean_option(fitted_automl, as_pipeline):
    model, _, _, _ = fitted_automl
    with pytest.raises(ValueError, match="as_pipeline must be a boolean"):
        model.get_model(as_pipeline=as_pipeline)


@pytest.mark.parametrize("model_class", [DecisionTreeClassifier, DecisionTreeRegressor])
def test_pipeline_preserves_input_order_and_unused_columns(model_class):
    rng = np.random.default_rng(5)
    X = rng.normal(size=(80, 5))
    selected = (3, 1, 4)
    y = (X[:, 3] > 0).astype(int)
    estimator = model_class(max_depth=1, random_state=0).fit(X[:, selected], y)
    assert np.count_nonzero(estimator.feature_importances_) == 1
    wrapped = SelectedFeaturesEstimator(estimator, selected, n_features_in=5)

    pipeline = export_sklearn_model(wrapped, as_pipeline=True)
    np.testing.assert_array_equal(pipeline[:-1].transform(X), X[:, selected])
    np.testing.assert_array_equal(pipeline.predict(X), wrapped.predict(X))
    assert pipeline.named_steps["estimator"].n_features_in_ == 3


def test_pipeline_can_be_refitted_independently(fitted_automl):
    model, X, y, _ = fitted_automl
    predictions = model.predict(X)
    pipeline = model.get_model(as_pipeline=True)
    pipeline.fit(X.to_numpy(), y)
    assert pipeline.predict(X.to_numpy()).shape == y.shape
    np.testing.assert_allclose(model.predict(X), predictions)


def test_pipeline_loads_without_mnplib(fitted_automl, tmp_path):
    model, X, _, _ = fitted_automl
    joblib.dump(model.get_model(as_pipeline=True), tmp_path / "model.joblib")
    np.savez(tmp_path / "data.npz", X=X.to_numpy(), predictions=model.predict(X))
    script = """
import importlib.abc
import sys

class RejectMnplib(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mnplib" or fullname.startswith("mnplib."):
            raise ImportError("mnplib is not available in this prediction process")
        return None

sys.meta_path.insert(0, RejectMnplib())
import joblib
import numpy as np

pipeline = joblib.load("model.joblib")
data = np.load("data.npz")
np.testing.assert_allclose(pipeline.predict(data["X"]), data["predictions"])
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
