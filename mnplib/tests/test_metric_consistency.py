"""Shared metric contracts for inputs, diagnostics, and functional calls."""

import inspect
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from mnplib import (
    Inaccuracy, Miscoding, Nescience, Surfeit,
    NescienceClassifier, NescienceRegressor, TimeSeries, AnomalyDetector,
)
from mnplib import inaccuracy, miscoding, nescience, surfeit
from mnplib.utils import _auto_n_bins, empirical_distribution_vector


CLASSES = (Miscoding, Inaccuracy, Surfeit, Nescience)


@pytest.fixture
def data():
    X = np.tile([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], (30, 1))
    return X, X[:, 0]


@pytest.mark.parametrize("cls", CLASSES + (
    NescienceClassifier, NescienceRegressor, TimeSeries, AnomalyDetector,
))
def test_high_level_configuration_exposes_only_estimator_options(cls):
    assert "n_bins" not in inspect.signature(cls).parameters
    assert "n_bins" not in cls().get_params(deep=True)
    with pytest.raises(TypeError, match="n_bins"):
        cls(n_bins=3)
    with pytest.raises(ValueError, match="n_bins"):
        cls().set_params(n_bins=3)


@pytest.mark.parametrize("n_samples", [30, 100, 500])
@pytest.mark.parametrize("y_type", ["numeric", "categorical"])
def test_component_target_code_lengths_share_vector_policy(n_samples, y_type):
    rng = np.random.default_rng(14)
    X = rng.normal(size=(n_samples, 3))
    y = rng.normal(size=n_samples) if y_type == "numeric" else rng.integers(0, 3, n_samples)
    expected = empirical_distribution_vector(y, numeric=y_type == "numeric").code_length
    metric = Nescience(y_type=y_type).fit(X, y)
    assert metric.miscoding_.target_code_length_ == expected
    assert metric.inaccuracy_.len_y_ == expected
    assert metric.surfeit_.len_y_ == expected
    assert Inaccuracy(y_type=y_type).fit_y(y).len_y_ == expected
    assert Surfeit(y_type=y_type).fit_y(y).len_y_ == expected


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("target", [np.ones((2, 2)), np.ones((4, 1)), np.ones((1, 4)), 1])
def test_multidimensional_and_scalar_targets_are_rejected(cls, target):
    with pytest.raises(ValueError, match="y must be a one-dimensional array"):
        cls().fit(np.ones((4, 2)), target)


@pytest.mark.parametrize("cls", (Inaccuracy, Surfeit))
@pytest.mark.parametrize("y_type", ["numeric", "categorical"])
def test_target_only_fitting_uses_shared_validation(cls, y_type):
    with pytest.raises(ValueError, match="one-dimensional"):
        cls(y_type=y_type).fit_y(np.ones((4, 1)))


@pytest.mark.parametrize("cls", CLASSES)
def test_dataframe_column_labels_are_preserved(cls, data):
    X, y = data
    columns = [10, 20]
    metric = cls().fit(pd.DataFrame(X, columns=columns), y)
    assert metric.feature_names_in_.tolist() == columns


@pytest.mark.parametrize("cls", CLASSES)
def test_model_analysis_has_the_same_input_signature(cls):
    parameters = inspect.signature(cls.model_analysis).parameters
    assert list(parameters) == ["self", "model", "X", "feature_names", "feature_indices"]
    assert all(parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
               for name in ("X", "feature_names", "feature_indices"))


@pytest.mark.parametrize("module", [miscoding, inaccuracy, surfeit, nescience])
def test_functional_helpers_expose_keyword_evaluation_data_and_explicit_options(module):
    for name, function in vars(module).items():
        if name.startswith("_") or not inspect.isfunction(function) or function.__module__ != module.__name__:
            continue
        parameters = inspect.signature(function).parameters
        assert "n_bins" not in parameters
        assert not any(param.kind is inspect.Parameter.VAR_KEYWORD for param in parameters.values())
        for data_name in ("X", "y"):
            if data_name in parameters:
                assert parameters[data_name].kind is inspect.Parameter.KEYWORD_ONLY


def test_miscoding_model_analysis_uses_effective_features_and_original_coordinates(data):
    X, y = data
    frame = pd.DataFrame(np.column_stack([X, X[:, 1]]), columns=["a", "b", "c"])
    selected = [2, 0]
    local = frame.iloc[:, selected]
    model = DecisionTreeRegressor(max_depth=1, random_state=0).fit(local, y)
    metric = Miscoding().fit(frame, y)
    report = metric.model_analysis(model, feature_indices=selected)
    explicit = metric.model_analysis(model, X=local, feature_indices=selected)
    functional = miscoding.model_analysis(model, X=frame, y=y, feature_indices=selected)
    expected = metric.subset_analysis([0])

    for result in (report, explicit, functional):
        assert result.keys() == expected.keys()
        for key in expected:
            np.testing.assert_equal(result[key], expected[key])
        assert result["miscoding"] == metric.miscoding_model(model, feature_indices=selected)


def test_nescience_model_analysis_has_flat_components(data):
    X, y = data
    model = LinearRegression().fit(X, y)
    metric = Nescience().fit(X, y)
    report = metric.model_analysis(model)
    assert "components" not in report
    assert all(name in report for name in metric.component_names_)
    assert report["nescience"] == metric.nescience_model(model)
    assert report["nescience"] == metric.aggregate_components(
        **{name: report[name] for name in metric.component_names_})
    functional = nescience.model_analysis(model, X=X, y=y)
    for key in report:
        np.testing.assert_equal(functional[key], report[key])


@pytest.mark.parametrize("numeric_X,numeric_y", [(False, False), (False, True),
                                                (True, False), (True, True)])
def test_subset_bin_diagnostics_reflect_applied_discretization(data, numeric_X, numeric_y):
    X, y = data
    expected = _auto_n_bins(len(y)) if numeric_X or numeric_y else None
    metric = Miscoding(X_type="numeric" if numeric_X else "categorical",
                       y_type="numeric" if numeric_y else "categorical").fit(X, y)
    assert metric.subset_analysis([0])["resolved_n_bins"] == expected
    assert metric.subset_analysis([])["resolved_n_bins"] is None


def test_categorical_bin_diagnostics_and_feature_code_length_units(data):
    X, y = data
    metric = Miscoding(X_type="categorical", y_type="categorical").fit(X, y)
    assert metric.subset_analysis([0])["resolved_n_bins"] is None
    assert Inaccuracy(y_type="categorical").fit_y(y).prediction_analysis(y)["resolved_n_bins"] is None
    features = metric.feature_analysis()
    assert "code_length_bits" in features
    assert features.set_index("feature_index").loc[0, "code_length_bits"] == len(y)


@pytest.mark.parametrize("cls,module,scalar", [(Miscoding, miscoding, "miscoding_model"),
                                              (Nescience, nescience, "nescience_model")])
def test_sparse_model_scoring_warns_but_analysis_is_quiet(cls, module, scalar):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 20))
    y = rng.normal(size=30)
    model = LinearRegression().fit(X, y)
    metric = cls().fit(X, y)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = metric.model_analysis(model)
        functional_report = module.model_analysis(model, X=X, y=y)
    assert not caught
    assert report["is_reliable"] is False
    assert functional_report["is_reliable"] is False
    for call in (lambda: getattr(metric, scalar)(model),
                 lambda: getattr(module, scalar)(model, X=X, y=y)):
        with pytest.warns(RuntimeWarning, match="model_analysis\\(model\\)") as caught:
            assert np.isnan(call())
        assert len(caught) == 1
        assert "joint_distribution_too_sparse" in str(caught[0].message)
