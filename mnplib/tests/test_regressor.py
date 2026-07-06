"""
Tests for the new minimum-nescience regressor.

These tests target the new ``NescienceRegressor`` implementation, which selects
a regression model by minimizing nescience on the full available representation
``(X, y)``. The class is expected to live in:

    mnplib.regressor.NescienceRegressor

The tests assume that the modular scikit-learn model adapter package is
available under:

    mnplib.models
"""

import numpy as np
import pandas as pd
import pytest

from sklearn.base import clone
from sklearn.datasets import make_regression
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor

from mnplib.models import SerializationConfig
from mnplib.regressor import CandidateResult, NescienceRegressor, Regressor


@pytest.fixture()
def regression_data():
    """Return a small deterministic regression dataset."""
    X, y = make_regression(
        n_samples=120,
        n_features=6,
        n_informative=3,
        noise=8.0,
        random_state=42,
    )
    return X, y


@pytest.fixture()
def regression_dataframe(regression_data):
    """Return the same regression dataset as a pandas DataFrame."""
    X, y = regression_data
    columns = [f"feature_{j}" for j in range(X.shape[1])]
    return pd.DataFrame(X, columns=columns), y


@pytest.fixture()
def small_candidates():
    """Return a compact supported candidate set for fast tests."""
    return [
        ("linear", LinearRegression()),
        ("ridge", Ridge(alpha=1.0)),
        ("lasso", Lasso(alpha=0.01, max_iter=10000)),
        ("elastic_net", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000)),
        ("tree_depth_2", DecisionTreeRegressor(max_depth=2, random_state=42)),
        ("tree_depth_3", DecisionTreeRegressor(max_depth=3, random_state=42)),
    ]


def test_fit_selects_a_candidate_and_sets_fitted_attributes(regression_data, small_candidates):
    X, y = regression_data

    reg = NescienceRegressor(
        candidates=small_candidates,
        n_bins=3,
        random_state=42,
        serialization_config=SerializationConfig(precision=4),
    )
    reg.fit(X, y)

    assert reg.is_fitted_
    assert reg.n_samples_in_ == X.shape[0]
    assert reg.n_features_in_ == X.shape[1]
    assert reg.best_candidate_name_ in dict(small_candidates)
    assert reg.model_ is reg.best_result_.estimator
    assert reg.best_artifacts_ is reg.best_result_.artifacts
    assert isinstance(reg.best_nescience_, float)
    assert reg.best_nescience_ >= 0.0
    assert len(reg.results_) == len(small_candidates)
    assert isinstance(reg.best_result_, CandidateResult)


def test_predict_returns_one_prediction_per_sample(regression_data, small_candidates):
    X, y = regression_data

    reg = NescienceRegressor(candidates=small_candidates, n_bins=3, random_state=42)
    reg.fit(X, y)

    predictions = reg.predict(X[:7])

    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (7,)


def test_score_returns_native_estimator_score(regression_data, small_candidates):
    X, y = regression_data

    reg = NescienceRegressor(candidates=small_candidates, n_bins=3, random_state=42)
    reg.fit(X, y)

    assert reg.score(X, y) == pytest.approx(reg.model_.score(X, y))


def test_nescience_score_and_components_match_best_result(regression_data, small_candidates):
    X, y = regression_data

    reg = NescienceRegressor(candidates=small_candidates, n_bins=3, random_state=42)
    reg.fit(X, y)

    assert reg.nescience_score() == pytest.approx(reg.best_result_.nescience)

    components = reg.components()

    assert components == reg.best_result_.components
    assert set(components) == {"deficiency", "surplus", "inaccuracy", "surfeit"}
    assert all(isinstance(value, float) for value in components.values())


def test_explain_contains_candidate_and_model_metadata(regression_data, small_candidates):
    X, y = regression_data

    reg = NescienceRegressor(candidates=small_candidates, n_bins=3, random_state=42)
    reg.fit(X, y)

    explanation = reg.explain()

    assert explanation["candidate_name"] == reg.best_candidate_name_
    assert explanation["model_type"] == reg.best_artifacts_.model_type
    assert explanation["model_metadata"] == reg.best_artifacts_.metadata
    assert "components" in explanation
    assert "dominant_component" in explanation


def test_get_model_returns_selected_estimator(regression_data, small_candidates):
    X, y = regression_data

    reg = NescienceRegressor(candidates=small_candidates, n_bins=3, random_state=42)
    reg.fit(X, y)

    assert reg.get_model() is reg.model_


def test_model_string_returns_canonical_string(regression_data, small_candidates):
    X, y = regression_data

    reg = NescienceRegressor(
        candidates=small_candidates,
        n_bins=3,
        random_state=42,
        serialization_config=SerializationConfig(precision=4),
    )
    reg.fit(X, y)

    model_string = reg.model_string()

    assert isinstance(model_string, str)
    assert model_string.startswith("SCHEMA canonical_nescience_model_v1")
    assert "MODEL " in model_string
    assert "TASK regression" in model_string
    assert "RULE" in model_string


def test_results_dataframe_has_expected_columns_and_is_sorted(regression_data, small_candidates):
    X, y = regression_data

    reg = NescienceRegressor(candidates=small_candidates, n_bins=3, random_state=42)
    reg.fit(X, y)

    df = reg.results_dataframe()

    expected_columns = {
        "candidate",
        "model_type",
        "nescience",
        "estimator_score",
        "n_features_in_use",
        "description_length",
        "deficiency",
        "surplus",
        "inaccuracy",
        "surfeit",
    }

    assert expected_columns.issubset(df.columns)
    assert len(df) == len(small_candidates)
    assert df["nescience"].is_monotonic_increasing
    assert df.iloc[0]["candidate"] == reg.best_candidate_name_


def test_dataframe_feature_names_are_preserved(regression_dataframe):
    X, y = regression_dataframe

    reg = NescienceRegressor(
        candidates=[("linear", LinearRegression())],
        n_bins=3,
        serialization_config=SerializationConfig(precision=4),
    )
    reg.fit(X, y)

    assert list(reg.feature_names_in_) == list(X.columns)
    assert "feature_" in reg.model_string()


def test_default_candidates_fit_successfully(regression_data):
    X, y = regression_data

    reg = NescienceRegressor(
        candidates="default",
        n_bins=3,
        random_state=42,
        serialization_config=SerializationConfig(precision=4),
    )
    reg.fit(X, y)

    assert reg.is_fitted_
    assert len(reg.results_) > 1
    assert reg.best_nescience_ >= 0.0
    assert reg.best_candidate_name_ in reg.results_dataframe()["candidate"].tolist()


def test_none_candidates_use_default_candidates(regression_data):
    X, y = regression_data

    reg = NescienceRegressor(
        candidates=None,
        n_bins=3,
        random_state=42,
        serialization_config=SerializationConfig(precision=4),
    )
    reg.fit(X, y)

    assert reg.is_fitted_
    assert len(reg.results_) > 1


def test_candidates_can_be_mapping(regression_data):
    X, y = regression_data

    candidates = {
        "linear": LinearRegression(),
        "tree": DecisionTreeRegressor(max_depth=2, random_state=42),
    }

    reg = NescienceRegressor(candidates=candidates, n_bins=3)
    reg.fit(X, y)

    assert {result.name for result in reg.results_} == set(candidates)


def test_candidates_can_be_plain_estimator_sequence(regression_data):
    X, y = regression_data

    candidates = [
        LinearRegression(),
        DecisionTreeRegressor(max_depth=2, random_state=42),
    ]

    reg = NescienceRegressor(candidates=candidates, n_bins=3)
    reg.fit(X, y)

    assert len(reg.results_) == 2
    assert reg.results_[0].name.startswith("LinearRegression_")
    assert reg.results_[1].name.startswith("DecisionTreeRegressor_")


def test_empty_candidates_raise_value_error(regression_data):
    X, y = regression_data

    reg = NescienceRegressor(candidates=[], n_bins=3)

    with pytest.raises(ValueError, match="candidate"):
        reg.fit(X, y)


def test_unsupported_candidate_raises_not_implemented_error(regression_data):
    X, y = regression_data

    reg = NescienceRegressor(
        candidates=[("dummy", DummyRegressor())],
        n_bins=3,
    )

    with pytest.raises(NotImplementedError):
        reg.fit(X, y)


def test_invalid_serialization_config_raises_type_error(regression_data):
    X, y = regression_data

    reg = NescienceRegressor(
        candidates=[("linear", LinearRegression())],
        n_bins=3,
        serialization_config="not-a-config",
    )

    with pytest.raises(TypeError, match="serialization_config"):
        reg.fit(X, y)


@pytest.mark.parametrize(
    "method_name,args",
    [
        ("predict", (np.zeros((3, 2)),)),
        ("score", (np.zeros((3, 2)), np.zeros(3))),
        ("nescience_score", ()),
        ("components", ()),
        ("explain", ()),
        ("get_model", ()),
        ("results_dataframe", ()),
        ("model_string", ()),
    ],
)
def test_unfitted_methods_raise_not_fitted_error(method_name, args):
    reg = NescienceRegressor(candidates=[("linear", LinearRegression())])

    with pytest.raises(NotFittedError):
        getattr(reg, method_name)(*args)


def test_backward_compatible_regressor_alias(regression_data):
    X, y = regression_data

    reg = Regressor(
        candidates=[("linear", LinearRegression())],
        n_bins=3,
    )
    reg.fit(X, y)

    assert isinstance(reg, NescienceRegressor)


def test_sklearn_clone_supports_estimator_parameters(small_candidates):
    reg = NescienceRegressor(
        candidates=small_candidates,
        n_bins=3,
        random_state=42,
        verbose=0,
    )

    cloned = clone(reg)

    assert isinstance(cloned, NescienceRegressor)
    assert cloned.n_bins == 3
    assert cloned.random_state == 42
    assert cloned.verbose == 0


def test_verbose_fit_prints_candidate_results(regression_data, capsys):
    X, y = regression_data

    reg = NescienceRegressor(
        candidates=[("linear", LinearRegression())],
        n_bins=3,
        verbose=1,
    )
    reg.fit(X, y)

    captured = capsys.readouterr()

    assert "linear: nescience=" in captured.out
    assert "estimator_score=" in captured.out


def test_custom_weights_and_aggregation_are_accepted(regression_data):
    X, y = regression_data

    reg = NescienceRegressor(
        candidates=[
            ("linear", LinearRegression()),
            ("tree_depth_2", DecisionTreeRegressor(max_depth=2, random_state=42)),
        ],
        aggregation="arithmetic",
        weights={
            "deficiency": 1.0,
            "surplus": 1.0,
            "inaccuracy": 2.0,
            "surfeit": 1.0,
        },
        n_bins=3,
    )
    reg.fit(X, y)

    assert reg.best_nescience_ >= 0.0
    assert reg.nescience_.aggregation == "arithmetic"
