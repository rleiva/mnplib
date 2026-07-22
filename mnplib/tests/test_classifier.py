"""
Tests for the public NescienceClassifier API.
"""

from __future__ import annotations

import inspect
import pathlib

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
from mnplib.models import SerializationConfig


FAST_MLP = {"max_candidates": 1, "max_iter": 5, "initial_features": 1}


@pytest.fixture()
def binary_classification_data():
    return make_classification(
        n_samples=80,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )


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
    assert set(clf.results_dataframe()["candidate_source"]) == {"internal"}


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
        serialization_config=SerializationConfig(precision=4),
        mlp_search_options=FAST_MLP,
    ).fit(X, y)

    assert clf.is_fitted_
    assert clf.n_samples_in_ == X.shape[0]
    assert clf.n_features_in_ == X.shape[1]
    assert isinstance(clf.best_result_, CandidateResult)
    assert clf.model_ is clf.best_result_.model
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
        serialization_config=SerializationConfig(precision=4),
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
    assert clf.get_model() is clf.model_
    assert "SCHEMA canonical_nescience_model_v1" in clf.model_string()


def test_results_dataframe_has_expected_columns(binary_classification_data):
    X, y = binary_classification_data

    clf = NescienceClassifier(
        n_bins=3,
        random_state=42,
        mlp_search_options=FAST_MLP,
    ).fit(X, y)
    df = clf.results_dataframe()

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
    assert set(df["candidate_source"]) == {"internal"}


def test_dataframe_feature_names_are_preserved(binary_classification_data):
    X, y = binary_classification_data
    X_df = pd.DataFrame(X, columns=[f"feature_{j}" for j in range(X.shape[1])])

    clf = NescienceClassifier(
        models=["decision_tree"],
        n_bins=3,
        serialization_config=SerializationConfig(precision=4),
    ).fit(X_df, y)

    assert list(clf.feature_names_in_) == list(X_df.columns)
    assert "feature_" in clf.model_string()


def test_no_weights_parameter_and_sklearn_clone_support(binary_classification_data):
    assert "weights" not in inspect.signature(NescienceClassifier).parameters

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
