"""Feature-label resolution across metrics, estimators, and model adapters."""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from sklearn.tree import DecisionTreeRegressor

from mnplib import (
    ResidualAnalysis,
    Inaccuracy,
    Miscoding,
    Nescience,
    NescienceClassifier,
    NescienceRegressor,
    Surfeit,
)
from mnplib.automl.wrappers import SelectedFeaturesEstimator
from mnplib.models import inputs, sklearn_model_artifacts
from mnplib.timeseries.lagged import LaggedRepresentationBuilder
from mnplib.utils import _resolve_feature_names


@pytest.mark.parametrize("X", [np.zeros((3, 2)), [[0, 1]] * 3, csr_matrix((3, 2))])
def test_unnamed_inputs_generate_positional_names(X):
    assert _resolve_feature_names(X) == ["x0", "x1"]
    assert _resolve_feature_names(n_features=np.int64(2)) == ["x0", "x1"]
    assert _resolve_feature_names(n_features=0) == []


@pytest.mark.parametrize("columns", [
    ["signal", "noise"], [10, 20], [("sensor", "a"), ("sensor", "b")],
])
def test_dataframe_labels_and_explicit_names_are_preserved(columns):
    frame = pd.DataFrame(np.zeros((3, 2)), columns=columns)
    assert _resolve_feature_names(frame) == columns
    names = _resolve_feature_names(frame, feature_names=["first", "second"])
    assert names == ["first", "second"]
    assert list(frame.columns) == columns


@pytest.mark.parametrize("names", [["a", "b"], np.array(["a", "b"], dtype=object)])
def test_resolved_names_are_independent_of_the_input(names):
    resolved = _resolve_feature_names(feature_names=names, n_features=2)
    names[0] = "changed"
    assert resolved == ["a", "b"]


@pytest.mark.parametrize("names", ["ab", b"ab", 2, np.array([["a", "b"]])])
def test_invalid_name_sequences_are_rejected(names):
    with pytest.raises(ValueError, match="one-dimensional sequence"):
        _resolve_feature_names(n_features=2, feature_names=names)


@pytest.mark.parametrize("count", [-1, 2.5, True, np.bool_(False), "2"])
def test_invalid_feature_counts_are_rejected(count):
    with pytest.raises(ValueError, match="non-negative integer"):
        _resolve_feature_names(n_features=count)


@pytest.mark.parametrize("X", [1, [1, 2], np.zeros((2, 2, 2))])
def test_feature_matrix_must_be_two_dimensional(X):
    with pytest.raises(ValueError, match="two-dimensional"):
        _resolve_feature_names(X)


def test_feature_dimension_is_required_and_must_match_names_and_input():
    with pytest.raises(ValueError, match="Provide X or n_features"):
        _resolve_feature_names()
    with pytest.raises(ValueError, match="number of columns"):
        _resolve_feature_names(np.zeros((3, 2)), n_features=3)
    with pytest.raises(ValueError, match="feature_names must have length 2"):
        _resolve_feature_names(np.zeros((3, 2)), feature_names=["a"])


@pytest.mark.parametrize("cls", [
    Miscoding, Inaccuracy, Surfeit, Nescience,
    NescienceClassifier, NescienceRegressor, ResidualAnalysis,
])
@pytest.mark.parametrize("input_kind", ["array", "list", "strings", "integers", "multiindex"])
def test_fitted_estimators_share_feature_names(cls, input_kind):
    X = np.tile([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], (30, 1))
    y = X[:, 0] + 2 * X[:, 1]
    names = ["x0", "x1"]
    if input_kind == "list":
        X = X.tolist()
    elif input_kind in ("strings", "integers", "multiindex"):
        columns = {
            "strings": ["signal", "noise"],
            "integers": [10, 20],
            "multiindex": pd.MultiIndex.from_tuples([("sensor", "a"), ("sensor", "b")]),
        }[input_kind]
        names = list(columns)
        X = pd.DataFrame(X, columns=columns)

    if cls in (NescienceClassifier, NescienceRegressor):
        metric = cls(models=["decision_tree"], random_state=0).fit(X, y)
    elif cls is ResidualAnalysis:
        metric = cls().fit(X, y, predictions=y)
        assert list(metric.X_frame_.columns) == names
    else:
        metric = cls().fit(X, y)
    assert metric.feature_names_in_.shape == (2,)
    assert metric.feature_names_in_.tolist() == names
    if isinstance(X, pd.DataFrame):
        X.columns = ["changed", "labels"]
        assert metric.feature_names_in_.tolist() == names


def test_selected_feature_wrapper_resolves_and_copies_names():
    names = np.array(["a", "b", "c"], dtype=object)
    wrapper = SelectedFeaturesEstimator(None, [2, 0], n_features_in=3, feature_names=names)
    names[0] = "changed"
    assert wrapper.feature_names_in_.tolist() == ["a", "b", "c"]
    assert wrapper.selected_features == (2, 0)
    unnamed = SelectedFeaturesEstimator(None, [2, 0], n_features_in=3)
    assert unnamed.feature_names_in_.tolist() == ["x0", "x1", "x2"]
    with pytest.raises(ValueError, match="feature_names must have length 3"):
        SelectedFeaturesEstimator(None, [2, 0], n_features_in=3, feature_names=["a"])


@pytest.mark.parametrize("explicit_names", [None, ["third", "first"]])
def test_model_adapter_resolves_names_after_mapping_selected_columns(monkeypatch, explicit_names):
    X = pd.DataFrame(np.tile([[0., 1., 0.], [1., 0., 1.]], (30, 1)), columns=["a", "b", "c"])
    selected = [2, 0]
    y = X["a"].to_numpy()
    model = DecisionTreeRegressor(random_state=0).fit(X.iloc[:, selected], y)
    expected = sklearn_model_artifacts(model, X.iloc[:, selected], feature_indices=selected)
    captured_names = []

    def record_names(model, X, **kwargs):
        captured_names.append(kwargs["feature_names"])
        return sklearn_model_artifacts(model, X, **kwargs)

    monkeypatch.setattr(inputs, "sklearn_model_artifacts", record_names)
    metric = Surfeit().fit(X, y)
    description = metric.model_analysis(model, feature_indices=selected, feature_names=explicit_names)
    assert captured_names == [explicit_names if explicit_names is not None else ["c", "a"]]
    assert description["model_string"] == expected.model_string
    assert description["surfeit"] == metric.surfeit_string(expected.model_string)


@pytest.mark.parametrize("columns", [None, ["temperature", "demand"], [10, 20]])
def test_exogenous_names_are_resolved_before_lag_names_are_generated(columns):
    X = np.zeros((20, 2))
    base_names = ["x0", "x1"]
    if columns is not None:
        X = pd.DataFrame(X, columns=columns)
        base_names = [str(name) for name in columns]
    representation, _, names = LaggedRepresentationBuilder(window_size=2).build(np.arange(20), X)
    assert names == tuple(base_names)
    assert representation.feature_names == (
        "y_lag_1", "y_lag_2",
        f"{base_names[0]}_lag_1", f"{base_names[0]}_lag_2",
        f"{base_names[1]}_lag_1", f"{base_names[1]}_lag_2",
    )
