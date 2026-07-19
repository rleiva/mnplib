"""
Tests for model-family-specific AutoML search.
"""

from __future__ import annotations

import pathlib

import pandas as pd

from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, MultinomialNB
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier

from mnplib.automl import CandidateEvaluator
from mnplib.automl.searchers import (
    DecisionTreePruningSearcher,
    SearchContext,
)
from mnplib.classifier import NescienceClassifier
from mnplib.models import SerializationConfig, sklearn_model_artifacts
from mnplib.nescience import Nescience
from mnplib.regressor import NescienceRegressor


def _classification_context(X, y):
    metric = Nescience(X_type="numeric", y_type="categorical", n_bins=3).fit(X, y)
    evaluator = CandidateEvaluator(
        X=X,
        y=y,
        nescience=metric,
        feature_names=[f"x{i}" for i in range(X.shape[1])],
        serialization_config=SerializationConfig(precision=4),
    )
    return SearchContext(
        X=X,
        y=y,
        feature_names=[f"x{i}" for i in range(X.shape[1])],
        evaluator=evaluator,
        task="classification",
        random_state=42,
    )


def test_decision_tree_pruning_path_search_evaluates_distinct_alphas():
    X, y = make_classification(
        n_samples=80,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )

    clf = NescienceClassifier(
        candidates="compact",
        n_bins=3,
        random_state=42,
        serialization_config=SerializationConfig(precision=4),
    ).fit(X, y)

    tree_results = [
        result
        for result in clf.results_
        if result.family == "decision_tree_classifier"
    ]

    assert len(tree_results) >= 1
    assert all("ccp_alpha" in result.metadata for result in tree_results)
    assert min(tree_results, key=lambda result: result.nescience).model is not None


def test_decision_tree_duplicate_structures_are_skipped(monkeypatch):
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
        candidates="compact",
        n_bins=3,
        random_state=42,
        serialization_config=SerializationConfig(precision=4),
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


def test_logistic_feature_prefix_search_and_l2_fallback(monkeypatch):
    X, y = make_classification(
        n_samples=70,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=42,
    )

    from mnplib.automl.searchers.logistic import LogisticRegressionPrefixSearcher

    def fail_unregularized(self, X_selected, y_selected):
        return None, {}, True

    monkeypatch.setattr(
        LogisticRegressionPrefixSearcher,
        "_fit_unregularized",
        fail_unregularized,
    )

    clf = NescienceClassifier(
        candidates="compact",
        n_bins=3,
        random_state=42,
        logistic_fallback_C_values=(1.0,),
    ).fit(X, y)

    logistic_results = [
        result
        for result in clf.results_
        if result.family == "logistic_regression"
    ]

    assert logistic_results
    assert all(
        result.metadata["used_stability_fallback"]
        for result in logistic_results
    )


def test_linear_svm_searchers_are_in_standard_profiles():
    Xc, yc = make_classification(
        n_samples=60,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=42,
    )
    clf = NescienceClassifier(
        candidates="standard",
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
        candidates="standard",
        n_bins=3,
        random_state=42,
    ).fit(Xr, yr)

    assert any(isinstance(result.model, LinearSVC) for result in clf.results_)
    assert any(isinstance(result.model, LinearSVR) for result in reg.results_)


def test_naive_bayes_compatibility_skips_incompatible_variants():
    X, y = make_classification(
        n_samples=60,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=42,
    )

    clf = NescienceClassifier(
        candidates="standard",
        n_bins=3,
        random_state=42,
    ).fit(X, y)

    skipped = {
        diagnostic["family"]
        for diagnostic in clf.diagnostics_
        if diagnostic.get("searcher_family") == "naive_bayes"
    }

    assert "multinomial_nb" in skipped
    assert "bernoulli_nb" in skipped
    assert "categorical_nb" in skipped
    assert not any(
        isinstance(result.model, (MultinomialNB, BernoulliNB, CategoricalNB))
        for result in clf.results_
    )


def test_optional_mlp_extended_search_records_evaluated_candidates():
    X, y = make_classification(
        n_samples=45,
        n_features=3,
        n_informative=2,
        n_redundant=0,
        random_state=42,
    )

    clf = NescienceClassifier(
        candidates="extended",
        n_bins=3,
        random_state=42,
        mlp_search_options={
            "max_candidates": 2,
            "max_iter": 5,
            "initial_features": 1,
            "max_features": 2,
        },
    ).fit(X, y)

    mlp_results = [
        result
        for result in clf.results_
        if result.family == "mlp_classifier"
    ]

    assert mlp_results
    assert all("hidden_layer_sizes" in result.metadata for result in mlp_results)
    assert "PREPROCESSOR StandardScaler" in mlp_results[0].artifacts.model_string


def test_candidate_profiles_have_expected_family_boundaries():
    compact = NescienceClassifier(candidates="compact")._resolve_searchers()
    standard = NescienceClassifier(candidates="standard")._resolve_searchers()
    extended = NescienceClassifier(
        candidates="extended",
        mlp_search_options={"max_candidates": 1},
    )._resolve_searchers()

    compact_families = {searcher.family for searcher in compact}
    standard_families = {searcher.family for searcher in standard}
    extended_families = {searcher.family for searcher in extended}

    assert compact_families == {
        "logistic_regression",
        "decision_tree_classifier",
    }
    assert "linear_svc" in standard_families
    assert "naive_bayes" in standard_families
    assert "mlp_classifier" in extended_families


def test_classifier_and_regressor_public_workflows_and_results_columns():
    Xc, yc = make_classification(
        n_samples=60,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=42,
    )
    clf = NescienceClassifier(
        candidates="compact",
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
        candidates="compact",
        n_bins=3,
        random_state=42,
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
            "family",
            "searched_hyperparameters",
            "nescience",
            "deficiency",
            "surplus",
            "inaccuracy",
            "surfeit",
            "native_estimator_score",
            "n_features_used",
            "description_length",
            "support_level",
        }.issubset(df.columns)
        assert df["nescience"].is_monotonic_increasing


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

    def spy(model, X_adapter, *, feature_names=None, config=None):
        artifacts = original(
            model,
            X_adapter,
            feature_names=feature_names,
            config=config,
        )
        calls.append(artifacts.to_nescience_kwargs())
        return artifacts

    monkeypatch.setattr(evaluator_module, "sklearn_model_artifacts", spy)

    NescienceClassifier(
        candidates=[("tree", DecisionTreeClassifier(max_depth=2, random_state=42))],
        n_bins=3,
    ).fit(X, y)

    assert calls
    assert set(calls[0]) == {"subset", "predictions", "model_string"}


def test_model_adapter_api_still_supports_existing_functions():
    X, y = make_classification(
        n_samples=50,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=42,
    )
    model = LogisticRegression(max_iter=1000).fit(X, y)

    artifacts = sklearn_model_artifacts(model, X)

    assert artifacts.to_nescience_kwargs()["subset"]
    assert artifacts.model_type == "LogisticRegression"


def test_removed_and_obsolete_parameters_are_not_in_automl_source():
    root = pathlib.Path(__file__).resolve().parents[2]
    sources = "\n".join(
        (root / path).read_text(encoding="utf-8")
        for path in [
            "mnplib/classifier.py",
            "mnplib/regressor.py",
            "mnplib/automl/searchers/logistic.py",
            "mnplib/automl/searchers/linear_models.py",
        ]
    )

    assert 'max_features="auto"' not in sources
    assert "Ridge(" not in sources
    assert "Lasso(" not in sources
    assert "ElasticNet(" not in sources
