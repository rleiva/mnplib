import math
import warnings
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

from mnplib import Mismodel
from mnplib.inaccuracy import Inaccuracy
from mnplib.mismodel import mismodel
from mnplib.models import sklearn_model_artifacts
from mnplib.nescience import Nescience
from mnplib.reporting import format_analysis
from mnplib.surfeit import Surfeit


@pytest.mark.parametrize('inaccuracy,surfeit', [(0, 0), (0.3, 0.4), (1, 1)])
def test_mismodel_is_root_mean_square(inaccuracy, surfeit):
    assert mismodel(inaccuracy=inaccuracy, surfeit=surfeit) == pytest.approx(
        math.sqrt((inaccuracy ** 2 + surfeit ** 2) / 2)
    )


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_components_propagate_nan(value):
    assert math.isnan(mismodel(inaccuracy=value, surfeit=0.1))
    assert math.isnan(mismodel(inaccuracy=0.1, surfeit=value))


def test_negative_components_are_invalid():
    with pytest.raises(ValueError, match='nonnegative'):
        mismodel(inaccuracy=-0.1, surfeit=0.1)


@pytest.fixture
def data():
    X = np.tile([[0, 1], [0, 0], [1, 1], [1, 0]], (8, 1))
    return X, X[:, 0].copy()


DESCRIPTION = "def predict(x):\n    return x[0]\n"


@pytest.mark.parametrize("a,b,expected", [
    (0, 0, 0), (1, 1, 1), (0, 1, 1 / math.sqrt(2)),
    (0.6, 0.8, math.sqrt(0.5)), (0.2, 0.9, math.sqrt(0.425)),
    *[(t, t, t) for t in (0.0, 0.1, 0.5, 0.9, 1.0)],
])
def test_scalar_interfaces_use_normalized_symmetric_rms(a, b, expected):
    for first, second in ((a, b), (b, a)):
        value = mismodel(inaccuracy=first, surfeit=second)
        assert isinstance(value, float)
        assert 0 <= value <= 1
        assert value == pytest.approx(expected)
        assert Mismodel.aggregate_components(inaccuracy=first, surfeit=second) == value
        assert Mismodel().aggregate_components(inaccuracy=first, surfeit=second) == value


@pytest.mark.parametrize("y_type", ["auto", "numeric", "categorical"])
def test_fit_and_explicit_artifacts_match_independent_metrics(data, y_type):
    X, y = data
    predictions = y.copy()
    predictions[::5] = 1 - predictions[::5]
    expected = {
        "inaccuracy": Inaccuracy(y_type=y_type).fit_y(y).inaccuracy_predictions(predictions),
        "surfeit": Surfeit(y_type=y_type).fit_y(y).surfeit_string(DESCRIPTION),
    }
    expected_rms = math.sqrt(sum(value**2 for value in expected.values()) / 2)
    for with_X in (False, True):
        metric = Mismodel(y_type=y_type)
        assert (metric.fit(X, y) if with_X else metric.fit_y(y)) is metric
        check_is_fitted(metric)
        assert metric.n_samples_in_ == len(y)
        np.testing.assert_array_equal(metric.y_, y)
        args = dict(predictions=predictions, model_string=DESCRIPTION)
        components = metric.components(**args)
        assert set(components) == {"inaccuracy", "surfeit"}
        assert components == pytest.approx(expected)
        assert all(isinstance(value, float) for value in components.values())
        assert metric.mismodel(**args) == pytest.approx(expected_rms)
        assert metric.analysis(**args) == pytest.approx({"mismodel": expected_rms, **expected})


@pytest.mark.parametrize("method", ["components", "mismodel", "analysis",
                                    "mismodel_model", "model_analysis"])
def test_evaluation_requires_successful_fitting(method):
    metric = Mismodel()
    with pytest.raises(NotFittedError):
        if method.endswith("_model") or method == "model_analysis":
            getattr(metric, method)(LinearRegression())
        else:
            getattr(metric, method)(predictions=[0, 1], model_string=DESCRIPTION)


@pytest.mark.parametrize("method", ["components", "mismodel", "analysis"])
def test_each_evaluation_computes_each_component_once(data, monkeypatch, method):
    _, y = data
    metric = Mismodel().fit_y(y)
    inaccuracy = Mock(return_value=0.6)
    surfeit = Mock(return_value=0.8)
    monkeypatch.setattr(metric.inaccuracy_, "inaccuracy_predictions", inaccuracy)
    monkeypatch.setattr(metric.surfeit_, "surfeit_string", surfeit)
    result = getattr(metric, method)(predictions=y, model_string=DESCRIPTION)
    inaccuracy.assert_called_once_with(y)
    surfeit.assert_called_once_with(DESCRIPTION)
    if method == "analysis":
        assert result == pytest.approx({"inaccuracy": 0.6, "surfeit": 0.8,
                                       "mismodel": math.sqrt(0.5)})


@pytest.mark.parametrize("component", ["inaccuracy", "surfeit"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_evaluated_nonfinite_components_propagate(data, monkeypatch, component, value):
    _, y = data
    metric = Mismodel().fit_y(y)
    estimator, method = ((metric.inaccuracy_, "inaccuracy_predictions")
                         if component == "inaccuracy" else (metric.surfeit_, "surfeit_string"))
    monkeypatch.setattr(estimator, method, lambda arg: value)
    report = metric.analysis(predictions=y, model_string=DESCRIPTION)
    assert np.isnan(report["mismodel"])
    assert np.isnan(metric.mismodel(predictions=y, model_string=DESCRIPTION))
    assert np.isnan(Mismodel.aggregate_components(inaccuracy=report["inaccuracy"],
                                                 surfeit=report["surfeit"]))


@pytest.mark.parametrize("predictions,description,error", [
    ([0], DESCRIPTION, ValueError),
    ([[0, 1], [1, 0]], DESCRIPTION, ValueError),
    ([np.nan] * 32, DESCRIPTION, ValueError),
    ([0] * 32, "", ValueError),
    ([0] * 32, None, TypeError),
])
def test_invalid_artifacts_raise_validation_errors(data, predictions, description, error):
    metric = Mismodel().fit_y(data[1])
    with pytest.raises(error):
        metric.analysis(predictions=predictions, model_string=description)


def test_refitting_replaces_components_target_and_feature_state(data):
    X, y = data
    metric = Mismodel().fit(pd.DataFrame(X, columns=["a", "b"]), y)
    components = (metric.inaccuracy_, metric.surfeit_)
    model = DecisionTreeClassifier(random_state=1).fit(X, y)
    metric.fit_y(y[:8])
    assert metric.X_ is None
    for name in ("_model_X_", "n_features_in_", "feature_names_in_"):
        assert not hasattr(metric, name)
    assert metric.inaccuracy_ is not components[0]
    assert metric.surfeit_ is not components[1]
    assert len(metric.y_) == metric.n_samples_in_ == 8
    with pytest.raises(ValueError, match="Provide X"):
        metric.mismodel_model(model)
    assert np.isfinite(metric.mismodel_model(model, X=X[:8]))
    metric.fit(X[:, :1], 1 - y)
    assert metric.n_features_in_ == 1
    np.testing.assert_array_equal(metric.y_, 1 - y)
    np.testing.assert_array_equal(metric.X_, X[:, :1])


@pytest.mark.parametrize("fit_method", ["fit", "fit_y"])
@pytest.mark.parametrize("failure", ["validation", "component"])
def test_failed_refit_invalidates_fitted_state(data, monkeypatch, fit_method, failure):
    X, y = data
    metric = Mismodel().fit(X, y)
    if failure == "component":
        monkeypatch.setattr(Surfeit, "fit_y", Mock(side_effect=ValueError("component failure")))
    target = [] if failure == "validation" else y
    with pytest.raises(ValueError):
        metric.fit(X, target) if fit_method == "fit" else metric.fit_y(target)
    with pytest.raises(NotFittedError):
        metric.components(predictions=y, model_string=DESCRIPTION)
    assert metric.X_ is None
    assert not hasattr(metric, "y_")
    assert not hasattr(metric, "inaccuracy_")
    assert not hasattr(metric, "surfeit_")
    assert not hasattr(metric, "_model_X_")


def test_parameter_inspection_set_params_and_clone(data):
    metric = Mismodel().fit_y(data[1])
    assert metric.get_params() == {"y_type": "auto"}
    assert metric.set_params(y_type="categorical") is metric
    copied = clone(metric)
    assert copied.get_params() == {"y_type": "categorical"}
    assert not copied.__sklearn_is_fitted__()
    with pytest.raises(ValueError, match="y_type"):
        Mismodel(y_type="invalid")
    metric.set_params(y_type="invalid")
    with pytest.raises(ValueError, match="y_type"):
        metric.fit_y(data[1])
    assert not metric.__sklearn_is_fitted__()


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("stored", [False, True])
@pytest.mark.parametrize("dataframe", [False, True])
def test_model_methods_use_canonical_artifacts(task, stored, dataframe, monkeypatch):
    X = np.random.default_rng(42).normal(size=(60, 3))
    y = X[:, 0] - 2 * X[:, 2]
    if task == "classification":
        y = (y > 0).astype(int)
        model = DecisionTreeClassifier(max_depth=2, random_state=1)
    else:
        model = LinearRegression()
    if dataframe:
        X = pd.DataFrame(X, columns=["a", "b", "c"])
    model.fit(X, y)
    artifacts = sklearn_model_artifacts(model, X)
    metric = Mismodel().fit(X, y) if stored else Mismodel().fit_y(y)
    expected = metric.analysis(predictions=artifacts.predictions,
                               model_string=artifacts.model_string)
    predict = Mock(wraps=model.predict)
    monkeypatch.setattr(model, "predict", predict)
    kwargs = {} if stored else {"X": X}
    report = metric.model_analysis(model, **kwargs)
    predict.assert_called_once()
    assert report == {**expected, "model_type": artifacts.model_type,
                      "model_string": artifacts.model_string}
    predict.reset_mock()
    assert metric.mismodel_model(model, **kwargs) == expected["mismodel"]
    predict.assert_called_once()
    if dataframe and stored:
        pd.testing.assert_frame_equal(predict.call_args.args[0], X)
        assert metric.feature_names_in_.tolist() == list(X.columns)


def test_feature_subset_mapping_and_explicit_names(data):
    X, y = data
    X = pd.DataFrame(X, columns=["signal", "noise"])
    selected = [1, 0]
    local = X.iloc[:, selected]
    model = LinearRegression().fit(local, y)
    artifacts = sklearn_model_artifacts(model, local, feature_indices=selected,
                                        feature_names=["noise", "signal"])
    for metric, kwargs in ((Mismodel().fit(X, y), {}),
                           (Mismodel().fit_y(y), {"X": local})):
        report = metric.model_analysis(model, feature_indices=selected,
                                       feature_names=["noise", "signal"], **kwargs)
        assert report["model_string"] == artifacts.model_string
        assert report["mismodel"] == metric.mismodel(
            predictions=artifacts.predictions, model_string=artifacts.model_string)


def test_evaluation_matrices_and_metadata_are_validated(data):
    X, y = data
    model = LinearRegression().fit(X[:, :1], y)
    metric = Mismodel().fit(X, y)
    with pytest.raises(ValueError, match="same evaluation rows"):
        metric.mismodel_model(model, X=X[:-1])
    with pytest.raises(ValueError, match="feature_indices is required"):
        metric.mismodel_model(model, X=X[:, :1])
    with pytest.raises(ValueError, match="unique"):
        metric.mismodel_model(model, feature_indices=[0, 0])
    with pytest.raises(ValueError, match="feature_names"):
        metric.mismodel_model(model, feature_indices=[0], feature_names=["a", "b"])
    with pytest.raises(ValueError, match="two-dimensional"):
        metric.mismodel_model(model, X=X[:, 0])


def test_model_errors_are_not_suppressed(data):
    X, y = data
    metric = Mismodel().fit(X, y)
    with pytest.raises(NotFittedError):
        metric.mismodel_model(LinearRegression())
    unsupported = RandomForestRegressor(n_estimators=1, random_state=1).fit(X, y)
    with pytest.raises(ValueError, match="Unsupported.*RandomForestRegressor"):
        metric.mismodel_model(unsupported)


def test_fit_preserves_mixed_dataframe_and_copies_inputs(data):
    X, y = data
    frame = pd.DataFrame({"number": X[:, 0], "category": ["a", "b"] * 16})
    target = y.copy()
    metric = Mismodel().fit(frame, target)
    pd.testing.assert_frame_equal(metric._model_X_, frame)
    frame.iloc[0, 0] = 99
    target[:] = 99
    assert metric._model_X_.iloc[0, 0] == X[0, 0]
    np.testing.assert_array_equal(metric.y_, y)


@pytest.mark.parametrize("fit_kind", ["target", "features", "nescience"])
def test_each_component_is_fitted_once(data, monkeypatch, fit_kind):
    counts = {"inaccuracy": 0, "surfeit": 0}
    for name, cls in (("inaccuracy", Inaccuracy), ("surfeit", Surfeit)):
        original = cls.fit_y
        def fit(self, y, original=original, name=name):
            counts[name] += 1
            return original(self, y)
        monkeypatch.setattr(cls, "fit_y", fit)
    X, y = data
    if fit_kind == "nescience":
        metric = Nescience().fit(X, y)
        assert isinstance(metric.mismodel_, Mismodel)
        assert "inaccuracy_" not in vars(metric)
        assert "surfeit_" not in vars(metric)
        assert metric.mismodel_.X_ is None
    elif fit_kind == "features":
        Mismodel().fit(X, y)
    else:
        Mismodel().fit_y(y)
    assert counts == {"inaccuracy": 1, "surfeit": 1}


def test_nescience_weight_do_not_change_mismodel(data, monkeypatch):
    X, y = data
    model = DecisionTreeClassifier(random_state=1).fit(X, y)
    standalone = Mismodel().fit(X, y).model_analysis(model)
    for weight in (0.5, 0.4, 1):
        metric = Nescience(weight=weight).fit(X, y)
        components = Mock(wraps=metric.mismodel_.components)
        monkeypatch.setattr(metric.mismodel_, "components", components)
        report = metric.model_analysis(model)
        components.assert_called_once()
        for name in ("inaccuracy", "surfeit", "mismodel"):
            assert report[name] == standalone[name]


def test_sparse_miscoding_does_not_invalidate_mismodel_or_duplicate_warnings():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 20))
    y = rng.normal(size=30)
    model = LinearRegression().fit(X, y)
    metric = Nescience().fit(X, y)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = metric.model_analysis(model)
        standalone = Mismodel().fit(X, y).model_analysis(model)
    assert not caught
    assert np.isnan(report["nescience"])
    assert report["is_reliable"] is False
    for name in ("inaccuracy", "surfeit", "mismodel"):
        assert np.isfinite(report[name])
        assert report[name] == standalone[name]
    with pytest.warns(RuntimeWarning, match="joint_distribution_too_sparse") as caught:
        assert np.isnan(metric.nescience_model(model))
    assert len(caught) == 1


def test_plain_text_report_includes_both_components(data):
    report = Mismodel().fit_y(data[1]).analysis(
        predictions=data[1], model_string=DESCRIPTION)
    formatted = format_analysis(report)
    assert formatted.startswith("Mismodel Analysis")
    for label in ("Mismodel", "Inaccuracy", "Surfeit"):
        assert label in formatted
