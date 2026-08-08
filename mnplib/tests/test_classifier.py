"""
Tests for the public NescienceClassifier API.
"""

from __future__ import annotations

import inspect
import pathlib
import re

import numpy as np
import pandas as pd
import pytest

from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from mnplib.classifier import (
    SUPPORTED_MODELS,
    CandidateResult,
    Classifier,
    NescienceClassifier,
)


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

EXPECTED_DESCRIPTION_KEYS = {
    "candidate",
    "family",
    "model_type",
    "model_string",
    "model_length",
    "model_compressed_length",
    "surfeit",
}


@pytest.fixture()
def binary_classification_data():
    return make_classification(
        n_samples=80,
        n_features=5,
        n_informative=3,
        n_redundant=0,
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
    assert "model_string" not in df.columns
    assert df["hyperparameters"].map(
        lambda value: isinstance(value, dict)
    ).all()
    assert (
        df["n_selected_features"]
        == df["selected_features"].map(len)
    ).all()


def _result_by_name(estimator, candidate_name):
    for result in estimator.results_:
        if result.name == candidate_name:
            return result

    raise AssertionError(f"Missing candidate result {candidate_name!r}.")


def _non_best_result(estimator):
    for result in estimator.results_:
        if result.name != estimator.best_candidate_name_:
            return result

    pytest.skip("Need more than one evaluated candidate.")


def _assert_candidate_model_description(description, result):
    assert set(description) == EXPECTED_DESCRIPTION_KEYS
    assert description["candidate"] == result.name
    assert description["family"] == result.family
    assert description["model_type"] == result.artifacts.model_type
    assert description["model_string"] == result.artifacts.model_string
    assert isinstance(description["model_string"], str)
    assert description["model_string"].strip()
    assert description["model_length"] == len(
        description["model_string"].encode("utf-8")
    )
    assert isinstance(description["model_compressed_length"], int)
    assert description["model_compressed_length"] > 0
    assert description["surfeit"] == pytest.approx(result.components["surfeit"])


def test_default_behavior_uses_all_supported_internal_model_families(
    binary_classification_data,
):
    X, y = binary_classification_data

    clf = NescienceClassifier(
        n_bins=3,
        random_state=42,
        mlp_search_options=FAST_MLP,
    ).fit(X, y)

    assert clf.model_names_ == SUPPORTED_MODELS
    assert [searcher.family for searcher in clf.searchers_] == [
        "decision_tree_classifier",
        "logistic_regression",
        "linear_svc",
        "naive_bayes",
        "mlp_classifier",
    ]
    assert "candidate_source" not in clf.results_dataframe().columns


def test_selecting_only_decision_tree_runs_only_tree_searcher(
    binary_classification_data,
):
    X, y = binary_classification_data

    clf = NescienceClassifier(
        models=["decision_tree"],
        n_bins=3,
        random_state=42,
    ).fit(X, y)

    assert clf.model_names_ == ("decision_tree",)
    assert [searcher.family for searcher in clf.searchers_] == [
        "decision_tree_classifier"
    ]
    assert {result.family for result in clf.results_} == {
        "decision_tree_classifier"
    }


def test_selected_models_preserve_user_order(binary_classification_data):
    X, y = binary_classification_data

    clf = NescienceClassifier(
        models=["decision_tree", "logistic_regression"],
        n_bins=3,
        random_state=42,
    ).fit(X, y)

    assert clf.model_names_ == ("decision_tree", "logistic_regression")
    assert [searcher.family for searcher in clf.searchers_] == [
        "decision_tree_classifier",
        "logistic_regression",
    ]
    assert {result.family for result in clf.results_} == {
        "decision_tree_classifier",
        "logistic_regression",
    }


def test_invalid_model_name_raises_clear_value_error(binary_classification_data):
    X, y = binary_classification_data

    with pytest.raises(ValueError, match="random_forest"):
        NescienceClassifier(
            models=["random_forest"],
            n_bins=3,
        ).fit(X, y)


def test_arbitrary_candidate_mapping_is_rejected(binary_classification_data):
    X, y = binary_classification_data

    with pytest.raises(ValueError, match="does not accept arbitrary candidate"):
        NescienceClassifier(
            candidates={
                "my_model": DecisionTreeClassifier(max_depth=2, random_state=42)
            },
            n_bins=3,
        ).fit(X, y)


def test_arbitrary_candidate_sequence_is_rejected(binary_classification_data):
    X, y = binary_classification_data

    with pytest.raises(ValueError, match="does not accept arbitrary candidate"):
        NescienceClassifier(
            candidates=[DecisionTreeClassifier(max_depth=2, random_state=42)],
            n_bins=3,
        ).fit(X, y)


def test_fit_selects_minimum_nescience_candidate(binary_classification_data):
    X, y = binary_classification_data

    clf = NescienceClassifier(
        n_bins=3,
        random_state=42,
        mlp_search_options=FAST_MLP,
    ).fit(X, y)

    assert clf.is_fitted_
    assert clf.n_samples_in_ == X.shape[0]
    assert clf.n_features_in_ == X.shape[1]
    assert isinstance(clf.best_result_, CandidateResult)
    assert clf.model_ is clf.best_result_.model
    assert not hasattr(clf.best_result_, "metadata")
    assert clf.best_nescience_ == pytest.approx(
        min(result.nescience for result in clf.results_)
    )
    assert np.array_equal(clf.classes_, clf.model_.classes_)


def test_predict_predict_proba_and_score(binary_classification_data):
    X, y = binary_classification_data

    clf = NescienceClassifier(
        n_bins=3,
        random_state=42,
        mlp_search_options=FAST_MLP,
    ).fit(X, y)

    predictions = clf.predict(X[:9])
    assert predictions.shape == (9,)
    assert set(predictions).issubset(set(clf.classes_))
    assert clf.score(X, y) == pytest.approx(clf.model_.score(X, y))
    assert clf.score(X, y) == pytest.approx(accuracy_score(y, clf.predict(X)))

    if hasattr(clf.model_, "predict_proba"):
        probabilities = clf.predict_proba(X[:11])
        assert probabilities.shape == (11, len(clf.classes_))
        assert np.allclose(probabilities.sum(axis=1), 1.0)
    else:
        with pytest.raises(AttributeError):
            clf.predict_proba(X[:11])


def test_nescience_components_explain_model_string_and_get_model(
    binary_classification_data,
):
    X, y = binary_classification_data

    clf = NescienceClassifier(
        n_bins=3,
        random_state=42,
        mlp_search_options=FAST_MLP,
    ).fit(X, y)

    assert clf.nescience_score() == pytest.approx(clf.best_result_.nescience)
    assert set(clf.components()) == {
        "deficiency",
        "surplus",
        "inaccuracy",
        "surfeit",
    }
    explanation = clf.explain()
    assert explanation["candidate_name"] == clf.best_candidate_name_
    assert "candidate_source" not in explanation
    assert "hyperparameters" in explanation
    assert "metadata" not in explanation
    assert "model_metadata" not in explanation
    assert "n_input_features" not in explanation
    assert "n_features_in_use" not in explanation
    assert clf.get_model() is clf.model_
    model_string = clf.model_string()

    assert model_string.strip()
    assert "SCHEMA" not in model_string
    assert re.search(
        r"def predict\(x\):|^if\s+|^P StandardScaler$",
        model_string,
        re.MULTILINE,
    )


def test_candidate_model_description_for_best_and_named_candidate(
    binary_classification_data,
):
    X, y = binary_classification_data

    clf = NescienceClassifier(
        models=["decision_tree", "logistic_regression"],
        n_bins=3,
        random_state=42,
    ).fit(X, y)

    best_description = clf.candidate_model_description()
    _assert_candidate_model_description(best_description, clf.best_result_)
    assert clf.model_string() == best_description["model_string"]

    named_result = _non_best_result(clf)
    named_description = clf.candidate_model_description(named_result.name)
    _assert_candidate_model_description(named_description, named_result)

    assert _result_by_name(clf, named_description["candidate"]) is named_result
    assert "model_string" not in clf.results_dataframe().columns

    with pytest.raises(KeyError, match="missing_candidate"):
        clf.candidate_model_description("missing_candidate")


def test_results_dataframe_has_expected_columns(binary_classification_data):
    X, y = binary_classification_data

    clf = NescienceClassifier(
        n_bins=3,
        random_state=42,
        mlp_search_options=FAST_MLP,
    ).fit(X, y)
    df = clf.results_dataframe()

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
    assert df.iloc[0]["candidate"] == clf.best_candidate_name_


def test_dataframe_feature_names_are_preserved(binary_classification_data):
    X, y = binary_classification_data
    X_df = pd.DataFrame(X, columns=[f"feature_{j}" for j in range(X.shape[1])])

    clf = NescienceClassifier(
        models=["decision_tree"],
        n_bins=3,
    ).fit(X_df, y)

    assert list(clf.feature_names_in_) == list(X_df.columns)
    assert "feature_" not in clf.model_string()
    assert not hasattr(clf.best_artifacts_, "metadata")


def test_no_weights_parameter_and_sklearn_clone_support(binary_classification_data):
    assert "weights" not in inspect.signature(NescienceClassifier).parameters
    assert "serialization_config" not in inspect.signature(NescienceClassifier).parameters

    clf = NescienceClassifier(
        n_bins=3,
        random_state=42,
        verbose=0,
        mlp_search_options=FAST_MLP,
    )
    cloned = clone(clf)

    assert isinstance(cloned, NescienceClassifier)
    assert cloned.n_bins == 3
    assert cloned.random_state == 42


def test_serialization_config_parameter_is_not_accepted(binary_classification_data):
    X, y = binary_classification_data

    with pytest.raises(TypeError, match="serialization_config"):
        NescienceClassifier(
            serialization_config=object(),
        ).fit(X, y)


def test_classifier_does_not_use_clone_for_external_candidates():
    source = pathlib.Path(
        "mnplib/classifier.py"
    ).read_text(encoding="utf-8")

    assert "clone(" not in source
    assert "_fit_explicit_candidates" not in source
    assert "_resolve_candidates" not in source


@pytest.mark.parametrize(
    "method_name,args",
    [
        ("predict", (np.zeros((3, 2)),)),
        ("predict_proba", (np.zeros((3, 2)),)),
        ("score", (np.zeros((3, 2)), np.zeros(3))),
        ("nescience_score", ()),
        ("components", ()),
        ("explain", ()),
        ("get_model", ()),
        ("results_dataframe", ()),
        ("model_string", ()),
        ("candidate_model_description", ()),
    ],
)
def test_unfitted_methods_raise_not_fitted_error(method_name, args):
    clf = NescienceClassifier(mlp_search_options=FAST_MLP)

    with pytest.raises(NotFittedError):
        getattr(clf, method_name)(*args)


def test_backward_compatible_classifier_alias(binary_classification_data):
    X, y = binary_classification_data

    clf = Classifier(
        n_bins=3,
        mlp_search_options=FAST_MLP,
    ).fit(X, y)

    assert isinstance(clf, NescienceClassifier)
