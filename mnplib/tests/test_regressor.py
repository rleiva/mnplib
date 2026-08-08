"""
Tests for the public NescienceRegressor API.
"""

from __future__ import annotations

import inspect
import re

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

from mnplib.regressor import CandidateResult, NescienceRegressor, Regressor


FAST_MLP = {"max_candidates": 1, "max_iter": 5, "initial_features": 1}

COMMON_RESULT_COLUMNS = {
    "candidate",
    "family",
    "model_type",
    "hyperparameters",
    "nescience",
    "native_estimator_score",
    "selected_features",
    "n_selected_features",
    "description_length",
    "deficiency",
    "surplus",
    "inaccuracy",
    "surfeit",
}


@pytest.fixture()
def regression_data():
    return make_regression(
        n_samples=80,
        n_features=5,
        n_informative=3,
        noise=3.0,
        random_state=42,
    )


def _assert_common_result_frame(df):
    assert COMMON_RESULT_COLUMNS.issubset(df.columns)
    assert "metadata" not in df.columns
    assert "candidate_source" not in df.columns
    assert "support_level" not in df.columns
    assert "searched_hyperparameters" not in df.columns
    assert "n_input_features" not in df.columns
    assert "n_features_in_use" not in df.columns
    assert "n_features_used" not in df.columns
    assert df["hyperparameters"].map(
        lambda value: isinstance(value, dict)
    ).all()
    assert (
        df["n_selected_features"]
        == df["selected_features"].map(len)
    ).all()


def test_fit_selects_minimum_nescience_candidate(regression_data):
    X, y = regression_data

    reg = NescienceRegressor(
        n_bins=3,
        random_state=42,
        mlp_search_options=FAST_MLP,
    ).fit(X, y)

    assert reg.is_fitted_
    assert reg.n_samples_in_ == X.shape[0]
    assert reg.n_features_in_ == X.shape[1]
    assert isinstance(reg.best_result_, CandidateResult)
    assert reg.model_ is reg.best_result_.model
    assert not hasattr(reg.best_result_, "metadata")
    assert reg.best_nescience_ == pytest.approx(
        min(result.nescience for result in reg.results_)
    )
    assert "candidate_source" not in reg.results_dataframe().columns


def test_predict_score_components_explain_and_model_string(regression_data):
    X, y = regression_data

    reg = NescienceRegressor(
        n_bins=3,
        random_state=42,
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
    explanation = reg.explain()
    assert explanation["candidate_name"] == reg.best_candidate_name_
    assert "hyperparameters" in explanation
    assert "metadata" not in explanation
    assert "model_metadata" not in explanation
    assert "n_input_features" not in explanation
    assert "n_features_in_use" not in explanation
    assert reg.get_model() is reg.model_
    model_string = reg.model_string()

    assert model_string.strip()
    assert "SCHEMA" not in model_string
    assert re.search(
        r"def predict\(x\):|^if\s+|^\s*y\s*=|^P StandardScaler$",
        model_string,
        re.MULTILINE,
    )


def test_results_dataframe_has_expected_columns(regression_data):
    X, y = regression_data

    reg = NescienceRegressor(
        n_bins=3,
        random_state=42,
        mlp_search_options=FAST_MLP,
    ).fit(X, y)
    df = reg.results_dataframe()

    _assert_common_result_frame(df)
    for forbidden in [
        "hidden_layer_sizes",
        "var_smoothing",
        "ccp_alpha",
        "solver",
        "alpha",
        "converged",
    ]:
        assert forbidden not in df.columns
    assert df["nescience"].is_monotonic_increasing
    assert df.iloc[0]["candidate"] == reg.best_candidate_name_


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
    assert "candidate_source" not in df.columns


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
        mlp_search_options=FAST_MLP,
    ).fit(X_df, y)

    assert list(reg.feature_names_in_) == list(X_df.columns)
    assert "feature_" not in reg.model_string()
    assert not hasattr(reg.best_artifacts_, "metadata")


def test_no_weights_parameter_and_sklearn_clone_support():
    assert "weights" not in inspect.signature(NescienceRegressor).parameters
    assert "serialization_config" not in inspect.signature(NescienceRegressor).parameters

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


def test_serialization_config_parameter_is_not_accepted(regression_data):
    X, y = regression_data

    with pytest.raises(TypeError, match="serialization_config"):
        NescienceRegressor(
            serialization_config=object(),
        ).fit(X, y)


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
