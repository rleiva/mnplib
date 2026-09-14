"""Shared metric contracts for inputs, diagnostics, and functional calls."""

import inspect
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from mnplib import Inaccuracy, Miscoding, Nescience, Surfeit
from mnplib import inaccuracy, miscoding, nescience, surfeit
from mnplib.utils import empirical_distribution


CLASSES = (Miscoding, Inaccuracy, Surfeit, Nescience)


@pytest.fixture
def data():
    X = np.tile([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], (30, 1))
    return X, X[:, 0]


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("bins", [2.9, 3.0, True, np.bool_(False), "3", 1, None, [2]])
def test_bins_require_integer_counts_or_named_policies(cls, bins, data):
    with pytest.raises(ValueError, match="n_bins must be an integer"):
        cls(n_bins=bins)
    metric = cls().set_params(n_bins=bins)
    with pytest.raises(ValueError, match="n_bins must be an integer"):
        metric.fit(*data)


@pytest.mark.parametrize("bins", [2.9, True, "3", [2]])
def test_distribution_bin_validation_matches_metric_validation(bins):
    with pytest.raises(ValueError, match="n_bins must be an integer"):
        empirical_distribution([[0, 1]], numeric=[True], n_bins=bins)


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("bins", [np.int64(3), 3, "auto", "adaptive"])
def test_valid_bin_settings_fit_in_every_metric(cls, bins, data):
    cls(n_bins=bins).fit(*data)


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("target", [np.ones((2, 2)), np.ones((4, 1)), np.ones((1, 4)), 1])
def test_multidimensional_and_scalar_targets_are_rejected(cls, target):
    with pytest.raises(ValueError, match="y must be a one-dimensional array"):
        cls().fit(np.ones((4, 2)), target)


@pytest.mark.parametrize("cls", (Inaccuracy, Surfeit))
def test_target_only_fitting_uses_shared_validation(cls):
    with pytest.raises(ValueError, match="one-dimensional"):
        cls().fit_y(np.ones((4, 1)))
    with pytest.raises(ValueError, match="n_bins"):
        cls().set_params(n_bins=2.9).fit_y([0, 1])


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


@pytest.mark.parametrize("numeric_X,numeric_y,expected", [(False, False, None),
                                                          (False, True, 3),
                                                          (True, False, 3),
                                                          (True, True, 3)])
def test_subset_bin_diagnostics_reflect_applied_discretization(data, numeric_X, numeric_y, expected):
    X, y = data
    metric = Miscoding(X_type="numeric" if numeric_X else "categorical",
                       y_type="numeric" if numeric_y else "categorical", n_bins=3).fit(X, y)
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
