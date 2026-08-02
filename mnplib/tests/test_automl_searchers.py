"""
Tests for nescience-guided AutoML searchers.

The tests in this module focus on the internal search policy used by the
AutoML estimators. They intentionally avoid conventional train/test validation
checks, because the current AutoML design evaluates fitted candidates through
the explicit nescience artifact workflow.
"""

from __future__ import annotations

import inspect
import pathlib

import pandas as pd
import pytest

from sklearn.datasets import make_classification, make_regression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier

from mnplib.automl import CandidateEvaluator
from mnplib.automl.searchers import (
    DecisionTreePruningSearcher,
    SearchContext,
)
from mnplib.classifier import SUPPORTED_MODELS, NescienceClassifier
from mnplib.nescience import Nescience
from mnplib.regressor import NescienceRegressor


FAST_MLP = {
    "max_candidates": 1,
    "max_iter": 5,
    "initial_features": 1,
}


def _classification_context(X, y):
    """
    Build the minimal search context needed by classification searchers.
    """
    metric = Nescience(
        X_type="numeric",
        y_type="categorical",
        n_bins=3,
    ).fit(X, y)

    feature_names = [f"x{i}" for i in range(X.shape[1])]

    evaluator = CandidateEvaluator(
        X=X,
        y=y,
        nescience=metric,
        feature_names=feature_names,
    )

    return SearchContext(
        X=X,
        y=y,
        feature_names=feature_names,
        evaluator=evaluator,
        task="classification",
        random_state=42,
    )


def _repo_root() -> pathlib.Path:
    """
    Locate the repository root from the test file location.
    """
    current = pathlib.Path(__file__).resolve()

    for parent in (current.parent, *current.parents):
        if (parent / "mnplib").is_dir():
            return parent

    raise RuntimeError("Could not locate the repository root.")


def _base_estimator(model):
    """
    Return the underlying sklearn estimator when a wrapper is used.
    """
    return getattr(model, "estimator", model)


def test_classifier_default_uses_all_supported_internal_model_families():
    X, y = make_classification(
        n_samples=40,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=42,
    )

    clf = NescienceClassifier(
        n_bins=3,
        random_state=42,
        mlp_search_options=FAST_MLP,
    ).fit(X, y)

    assert clf.model_names_ == SUPPORTED_MODELS


def test_weights_parameter_is_removed_from_automl_constructors():
    assert "weights" not in inspect.signature(NescienceClassifier).parameters
    assert "weights" not in inspect.signature(NescienceRegressor).parameters


def test_internal_classifier_searchers_are_nescience_guided_and_include_mlp():
    clf = NescienceClassifier(mlp_search_options=FAST_MLP)

    assert [searcher.family for searcher in clf._resolve_searchers()] == [
        "decision_tree_classifier",
        "logistic_regression",
        "linear_svc",
        "naive_bayes",
        "mlp_classifier",
    ]


def test_internal_regressor_searchers_are_nescience_guided_and_include_mlp():
    reg = NescienceRegressor(mlp_search_options=FAST_MLP)

    assert [searcher.family for searcher in reg._resolve_searchers()] == [
        "linear_regression",
        "decision_tree_regressor",
        "linear_svr",
        "mlp_regressor",
    ]


def test_no_ensembles_in_internal_automl_search():
    clf_families = {
        searcher.family
        for searcher in NescienceClassifier(
            mlp_search_options=FAST_MLP,
        )._resolve_searchers()
    }
    reg_families = {
        searcher.family
        for searcher in NescienceRegressor(
            mlp_search_options=FAST_MLP,
        )._resolve_searchers()
    }

    forbidden_fragments = {
        "forest",
        "extra",
        "boost",
        "bagging",
        "adaboost",
        "voting",
        "stacking",
    }

    assert not any(
        fragment in family
        for family in clf_families | reg_families
        for fragment in forbidden_fragments
    )


def test_classifier_can_select_decision_tree_only():
    clf = NescienceClassifier(models=["decision_tree"])

    assert [searcher.family for searcher in clf._resolve_searchers()] == [
        "decision_tree_classifier",
    ]


def test_classifier_can_select_decision_tree_and_logistic_regression():
    clf = NescienceClassifier(
        models=["decision_tree", "logistic_regression"],
    )

    assert [searcher.family for searcher in clf._resolve_searchers()] == [
        "decision_tree_classifier",
        "logistic_regression",
    ]


def test_classifier_rejects_single_model_string():
    with pytest.raises((TypeError, ValueError), match="models"):
        NescienceClassifier(models="decision_tree")._resolve_searchers()


def test_classifier_invalid_model_name_raises_value_error():
    with pytest.raises(ValueError, match="random_forest"):
        NescienceClassifier(models=["random_forest"])._resolve_searchers()


def test_classifier_candidate_mapping_is_rejected():
    with pytest.raises(ValueError, match="arbitrary candidate"):
        NescienceClassifier(
            candidates={
                "tree": DecisionTreeClassifier(max_depth=2, random_state=42),
            },
        )._resolve_searchers()


def test_classifier_candidate_sequence_is_rejected():
    with pytest.raises(ValueError, match="arbitrary candidate"):
        NescienceClassifier(
            candidates=[DecisionTreeClassifier(max_depth=2, random_state=42)],
        )._resolve_searchers()


def test_regressor_candidate_mapping_is_accepted_for_comparison():
    X, y = make_regression(
        n_samples=50,
        n_features=4,
        n_informative=2,
        random_state=42,
    )

    reg = NescienceRegressor(
        candidates={
            "linear_svr": LinearSVR(max_iter=10000, random_state=42),
        },
        n_bins=3,
        random_state=42,
        mlp_search_options=FAST_MLP,
    ).fit(X, y)

    assert "linear_svr" in set(reg.results_dataframe()["candidate"])
    assert {"internal", "explicit"}.issubset(
        set(reg.results_dataframe()["candidate_source"])
    )


def test_decision_tree_pruning_path_search_skips_duplicate_structures(monkeypatch):
    X, y = make_classification(
        n_samples=60,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=42,
    )

    searcher = DecisionTreePruningSearcher(
        DecisionTreeClassifier,
        random_state=42,
    )
    monkeypatch.setattr(searcher, "_unique_alphas", lambda alphas: [0.0, 0.0])

    report = searcher.search(_classification_context(X, y))

    assert len(report.results) == 1
    assert report.diagnostics
    assert report.diagnostics[0]["reason"] == "duplicate_tree_structure"


def test_linear_regression_feature_prefix_search_evaluates_all_prefixes():
    X, y = make_regression(
        n_samples=70,
        n_features=5,
        n_informative=3,
        noise=0.2,
        random_state=42,
    )

    reg = NescienceRegressor(
        n_bins=3,
        random_state=42,
        mlp_search_options=FAST_MLP,
    ).fit(X, y)

    linear_results = [
        result
        for result in reg.results_
        if result.family == "linear_regression"
    ]

    assert [result.metadata["n_features_used"] for result in linear_results] == [
        1,
        2,
        3,
        4,
        5,
    ]
    assert all("feature_order" in result.metadata for result in linear_results)
    assert all("selected_feature_indices" in result.metadata for result in linear_results)


def test_logistic_regression_feature_prefix_search_evaluates_all_prefixes():
    X, y = make_classification(
        n_samples=70,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=42,
    )

    clf = NescienceClassifier(
        models=["logistic_regression"],
        n_bins=3,
        random_state=42,
    ).fit(X, y)

    logistic_results = [
        result
        for result in clf.results_
        if result.family == "logistic_regression"
    ]

    assert [result.metadata["n_features_used"] for result in logistic_results] == [
        1,
        2,
        3,
        4,
    ]
    assert all("feature_order" in result.metadata for result in logistic_results)
    assert all("selected_feature_indices" in result.metadata for result in logistic_results)
    assert all(
        result.artifacts.model_string.startswith("def predict(x):")
        for result in logistic_results
    )


def test_linear_svm_searchers_remain_internal_candidates():
    Xc, yc = make_classification(
        n_samples=60,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=42,
    )
    clf = NescienceClassifier(
        models=["linear_svc"],
        n_bins=3,
        random_state=42,
    ).fit(Xc, yc)

    Xr, yr = make_regression(
        n_samples=60,
        n_features=4,
        n_informative=2,
        noise=0.1,
        random_state=42,
    )
    reg = NescienceRegressor(
        n_bins=3,
        random_state=42,
        mlp_search_options=FAST_MLP,
    ).fit(Xr, yr)

    svc_results = [
        result
        for result in clf.results_
        if result.family == "linear_svc"
    ]
    svr_results = [
        result
        for result in reg.results_
        if result.family == "linear_svr"
    ]

    assert svc_results
    assert svr_results
    assert all(isinstance(_base_estimator(result.model), LinearSVC) for result in svc_results)
    assert all(isinstance(_base_estimator(result.model), LinearSVR) for result in svr_results)
    assert all(
        result.artifacts.model_string.startswith("def predict(x):")
        for result in svc_results + svr_results
    )


def test_naive_bayes_uses_gaussian_feature_prefixes_only():
    X, y = make_classification(
        n_samples=60,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=42,
    )

    clf = NescienceClassifier(
        models=["naive_bayes"],
        n_bins=3,
        random_state=42,
    ).fit(X, y)

    nb_results = [
        result
        for result in clf.results_
        if result.family == "naive_bayes"
    ]

    assert [result.metadata["n_features_used"] for result in nb_results] == [
        1,
        2,
        3,
        4,
    ]
    assert all(result.metadata["variant"] == "GaussianNB" for result in nb_results)
    assert all(isinstance(_base_estimator(result.model), GaussianNB) for result in nb_results)
    assert all(
        result.artifacts.model_string.startswith("def predict(x):")
        for result in nb_results
    )


def test_mlp_search_is_internal_bounded_and_serializes_executable_predictor():
    X, y = make_classification(
        n_samples=45,
        n_features=3,
        n_informative=2,
        n_redundant=0,
        random_state=42,
    )

    clf = NescienceClassifier(
        models=["mlp"],
        n_bins=3,
        random_state=42,
        mlp_search_options=FAST_MLP,
    ).fit(X, y)

    mlp_results = [
        result
        for result in clf.results_
        if result.family == "mlp_classifier"
    ]

    assert len(mlp_results) == 1
    assert "hidden_layer_sizes" in mlp_results[0].metadata

    model_string = mlp_results[0].artifacts.model_string

    assert "def predict(x):" in model_string
    assert "W=" in model_string
    assert "B=" in model_string
    assert "for l in range(len(W)):" in model_string
    assert "argmax" not in model_string
    assert model_string.startswith("P StandardScaler\n")


def test_classifier_and_regressor_public_workflows_and_results_columns():
    Xc, yc = make_classification(
        n_samples=60,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=42,
    )
    clf = NescienceClassifier(
        models=["decision_tree", "logistic_regression"],
        n_bins=3,
        random_state=42,
    ).fit(pd.DataFrame(Xc, columns=list("abcd")), yc)

    Xr, yr = make_regression(
        n_samples=60,
        n_features=4,
        n_informative=2,
        noise=0.1,
        random_state=42,
    )
    reg = NescienceRegressor(
        n_bins=3,
        random_state=42,
        mlp_search_options=FAST_MLP,
    ).fit(Xr, yr)

    for estimator, X in [(clf, Xc), (reg, Xr)]:
        assert estimator.predict(X[:5]).shape == (5,)
        assert estimator.nescience_score() >= 0.0
        assert set(estimator.components()) == {
            "deficiency",
            "surplus",
            "inaccuracy",
            "surfeit",
        }
        assert estimator.explain()["candidate_name"] == estimator.best_candidate_name_
        assert estimator.get_model() is estimator.model_

        df = estimator.results_dataframe()

        assert {
            "candidate",
            "candidate_source",
            "family",
            "searched_hyperparameters",
            "nescience",
            "deficiency",
            "surplus",
            "inaccuracy",
            "surfeit",
            "native_estimator_score",
            "selected_features",
            "n_features_used",
            "description_length",
            "support_level",
        }.issubset(df.columns)
        assert df["nescience"].is_monotonic_increasing
        assert df.iloc[0]["candidate"] == estimator.best_candidate_name_
        assert estimator.best_nescience_ == pytest.approx(df.iloc[0]["nescience"])


def test_explicit_artifact_workflow_is_preserved(monkeypatch):
    X, y = make_classification(
        n_samples=50,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=42,
    )

    import mnplib.automl.evaluator as evaluator_module

    calls = []
    original = evaluator_module.sklearn_model_artifacts

    def spy(model, X_adapter, *, feature_names=None, feature_indices=None):
        artifacts = original(
            model,
            X_adapter,
            feature_names=feature_names,
            feature_indices=feature_indices,
        )
        calls.append(artifacts.to_nescience_kwargs())
        return artifacts

    monkeypatch.setattr(evaluator_module, "sklearn_model_artifacts", spy)

    NescienceClassifier(
        models=["decision_tree"],
        n_bins=3,
    ).fit(X, y)

    assert calls
    assert all(
        set(call) == {"subset", "predictions", "model_string"}
        for call in calls
    )


def test_no_train_test_split_or_cross_validation_inside_automl_source():
    root = _repo_root()

    sources = "\n".join(
        (root / path).read_text(encoding="utf-8")
        for path in [
            "mnplib/classifier.py",
            "mnplib/regressor.py",
            "mnplib/automl/searchers/logistic.py",
            "mnplib/automl/searchers/linear_models.py",
            "mnplib/automl/searchers/neural_network.py",
        ]
        if (root / path).exists()
    )

    assert "train_test_split" not in sources
    assert "cross_val" not in sources
    assert "KFold" not in sources
