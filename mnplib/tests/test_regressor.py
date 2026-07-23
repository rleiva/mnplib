"""
Tests for the public NescienceRegressor API.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from sklearn.base import clone
from sklearn.datasets import make_regression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from mnplib.models import SerializationConfig
from mnplib.regressor import CandidateResult, NescienceRegressor, Regressor


FAST_MLP = {"max_candidates": 1, "max_iter": 5, "initial_features": 1}


@pytest.fixture()
def regression_data():
    return make_regression(
        n_samples=80,
        n_features=5,
        n_informative=3,
        noise=3.0,
        random_state=42,
    )


def test_fit_selects_minimum_nescience_candidate(regression_data):
    X, y = regression_data

    reg = NescienceRegressor(
        n_bins=3,
        random_state=42,
        serialization_config=SerializationConfig(precision=4),
        mlp_search_options=FAST_MLP,
    ).fit(X, y)

    assert reg.is_fitted_
    assert reg.n_samples_in_ == X.shape[0]
    assert reg.n_features_in_ == X.shape[1]
    assert isinstance(reg.best_result_, CandidateResult)
    assert reg.model_ is reg.best_result_.model
    assert reg.best_nescience_ == pytest.approx(
        min(result.nescience for result in reg.results_)
    )
    assert set(reg.results_dataframe()["candidate_source"]) == {"internal"}


def test_predict_score_components_explain_and_model_string(regression_data):
    X, y = regression_data

    reg = NescienceRegressor(
        n_bins=3,
        random_state=42,
        serialization_config=SerializationConfig(precision=4),
        mlp_search_options=FAST_MLP,
    ).fit(X, y)

    assert reg.predict(X[:7]).shape == (7,)
    assert reg.score(X, y) == pytest.approx(reg.model_.score(X, y))
    assert reg.nescience_score() == pytest.approx(reg.best_result_.nescience)
    assert set(reg.components()) == {
        "deficiency",
        "surplus",
        "inaccuracy",
        "surfeit",
    }
    assert reg.explain()["candidate_name"] == reg.best_candidate_name_
    assert reg.get_model() is reg.model_
    assert "SCHEMA canonical_nescience_model_v1" in reg.model_string()


def test_results_dataframe_has_expected_columns(regression_data):
    X, y = regression_data

    reg = NescienceRegressor(
        n_bins=3,
        random_state=42,
        mlp_search_options=FAST_MLP,
    ).fit(X, y)
    df = reg.results_dataframe()

    assert {
        "candidate",
        "candidate_source",
        "family",
        "model_type",
        "nescience",
        "estimator_score",
        "native_estimator_score",
        "selected_features",
        "n_features_used",
        "description_length",
        "deficiency",
        "surplus",
        "inaccuracy",
        "surfeit",
    }.issubset(df.columns)
    assert df["nescience"].is_monotonic_increasing


def test_explicit_candidates_are_accepted_as_comparison_candidates(regression_data):
    X, y = regression_data

    candidates = {
        "linear_explicit": LinearRegression(),
        "tree_explicit": DecisionTreeRegressor(max_depth=2, random_state=42),
    }
    reg = NescienceRegressor(
        candidates=candidates,
        n_bins=3,
        random_state=42,
        mlp_search_options=FAST_MLP,
    ).fit(X, y)

    df = reg.results_dataframe()
    assert set(candidates).issubset(set(df["candidate"]))
    assert {"internal", "explicit"}.issubset(set(df["candidate_source"]))


def test_explicit_ensemble_candidate_is_rejected_by_static_adapter(
    regression_data,
):
    X, y = regression_data

    with pytest.raises(
        ValueError,
        match="Unsupported scikit-learn model type RandomForestRegressor",
    ):
        NescienceRegressor(
            candidates={
                "explicit_forest": RandomForestRegressor(
                    n_estimators=3,
                    max_depth=2,
                    random_state=42,
                )
            },
            n_bins=3,
            random_state=42,
            mlp_search_options=FAST_MLP,
        ).fit(X, y)


def test_plain_estimator_sequence_gets_generated_names(regression_data):
    X, y = regression_data

    reg = NescienceRegressor(
        candidates=[LinearRegression()],
        n_bins=3,
        random_state=42,
        mlp_search_options=FAST_MLP,
    ).fit(X, y)

    assert any(
        result.name.startswith("LinearRegression_")
        for result in reg.results_
        if result.metadata["candidate_source"] == "explicit"
    )


def test_empty_candidate_collection_raises_value_error(regression_data):
    X, y = regression_data

    with pytest.raises(ValueError, match="explicit candidate"):
        NescienceRegressor(
            candidates=[],
            n_bins=3,
            mlp_search_options=FAST_MLP,
        ).fit(X, y)


def test_profile_string_candidates_raise_value_error(regression_data):
    X, y = regression_data

    with pytest.raises(ValueError, match="profile strings"):
        NescienceRegressor(
            candidates="standard",
            n_bins=3,
            mlp_search_options=FAST_MLP,
        ).fit(X, y)


def test_unsupported_explicit_candidate_raises_clear_value_error(
    regression_data,
):
    X, y = regression_data

    with pytest.raises(
        ValueError,
        match="Unsupported scikit-learn model type DummyRegressor",
    ):
        NescienceRegressor(
            candidates=[("dummy", DummyRegressor())],
            n_bins=3,
            mlp_search_options=FAST_MLP,
        ).fit(X, y)


def test_dataframe_feature_names_are_preserved(regression_data):
    X, y = regression_data
    X_df = pd.DataFrame(X, columns=[f"feature_{j}" for j in range(X.shape[1])])

    reg = NescienceRegressor(
        n_bins=3,
        serialization_config=SerializationConfig(precision=4),
        mlp_search_options=FAST_MLP,
    ).fit(X_df, y)

    assert list(reg.feature_names_in_) == list(X_df.columns)
    assert "feature_" in reg.model_string()


def test_no_weights_parameter_and_sklearn_clone_support():
    assert "weights" not in inspect.signature(NescienceRegressor).parameters

    reg = NescienceRegressor(
        n_bins=3,
        random_state=42,
        verbose=0,
        mlp_search_options=FAST_MLP,
    )
    cloned = clone(reg)

    assert isinstance(cloned, NescienceRegressor)
    assert cloned.n_bins == 3
    assert cloned.random_state == 42


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
    reg = NescienceRegressor(mlp_search_options=FAST_MLP)

    with pytest.raises(NotFittedError):
        getattr(reg, method_name)(*args)


def test_backward_compatible_regressor_alias(regression_data):
    X, y = regression_data

    reg = Regressor(
        n_bins=3,
        mlp_search_options=FAST_MLP,
    ).fit(X, y)

    assert isinstance(reg, NescienceRegressor)
