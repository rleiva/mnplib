"""Nescience-guided polynomial candidates and prediction-fidelity diagnostics."""

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.svm import SVC, SVR

from mnplib import NescienceClassifier, NescienceRegressor, Nescience, describe_model
from mnplib.automl import CandidateEvaluator
from mnplib.automl.searchers import PolynomialSVCSearcher, PolynomialSVRSearcher, SearchContext
from mnplib.automl.searchers import polynomial_svm
from mnplib.models.language import execute


@pytest.fixture(params=["classification", "regression"])
def context(request):
    X = np.tile([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]], (20, 1))
    y = (X[:, 0] * X[:, 1] > 0).astype(int) if request.param == "classification" else X[:, 0] * X[:, 1]
    metric = Nescience(y_type="categorical" if request.param == "classification" else "numeric").fit(X, y)
    evaluator = CandidateEvaluator(X=X, y=y, nescience=metric, feature_names=["a", "b"])
    return SearchContext(X=X, y=y, feature_names=["a", "b"], evaluator=evaluator,
                         task=request.param, max_feature_prefixes=2)


def _searcher(context, **options):
    cls = PolynomialSVCSearcher if context.task == "classification" else PolynomialSVRSearcher
    return cls(**options)


def test_polynomial_search_grid_uses_nescience_and_explicit_descriptions(context, monkeypatch):
    calls = []
    evaluate = context.evaluator.evaluate

    def record(**kwargs):
        calls.append(kwargs)
        return evaluate(**kwargs)

    monkeypatch.setattr(context.evaluator, "evaluate", record)
    report = _searcher(context).search(context)
    assert len(report.results) == len(calls) == 9
    assert not report.diagnostics
    assert {(result.hyperparameters["C"], result.hyperparameters["degree"])
            for result in report.results} == {(C, degree) for C in (0.1, 1., 10.) for degree in (2, 3, 4)}
    assert len({result.model.selected_features for result in report.results}) == 1
    for result in report.results:
        estimator = result.model.estimator
        assert type(estimator) is (SVC if context.task == "classification" else SVR)
        assert estimator.kernel == "poly"
        assert estimator.gamma == "scale" and estimator.coef0 == 1
        assert set(result.hyperparameters) == {"C", "degree", "max_iter", "tol"}
        assert np.isfinite(result.nescience)
        assert result.nescience == pytest.approx(context.evaluator.nescience.nescience(
            **result.artifacts.to_nescience_kwargs()))
        assert result.components["surfeit"] == context.evaluator.nescience.mismodel_.surfeit_.surfeit_string(
            result.artifacts.description.canonical)
        expected = result.model.predict(context.X)
        actual = execute(result.artifacts.description.ast, context.X)
        if context.task == "classification":
            np.testing.assert_array_equal(actual, expected)
        else:
            np.testing.assert_allclose(actual, expected, rtol=0.02, atol=0.03)


@pytest.mark.parametrize("estimator_type,family", [
    (NescienceClassifier, "polynomial_svc"), (NescienceRegressor, "polynomial_svr"),
])
def test_public_polynomial_search_respects_prefix_limit_and_exports(estimator_type, family):
    X = np.tile([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]], (20, 1))
    y = (X[:, 0] > 0).astype(int) if estimator_type is NescienceClassifier else X[:, 0]
    estimator = estimator_type(models=[family], max_feature_prefixes=1, search_options={
        family: {"C_values": [0.1, 1.], "degree_values": [2, 3]},
    }).fit(X, y)
    expected = tuple(estimator.nescience_.miscoding_.rank_features(max_features=1))
    assert all(result.model.selected_features == expected for result in estimator.results_)
    table = estimator.results_dataframe()
    assert set(table.family) == {family}
    assert table.nescience.is_monotonic_increasing
    assert table.iloc[0].candidate == estimator.best_candidate_name_
    assert describe_model(estimator).canonical == estimator.best_artifacts_.model_string
    pipeline = estimator.get_model(as_pipeline=True)
    np.testing.assert_array_equal(pipeline.predict(X), estimator.predict(X))


def test_polynomial_search_reports_failed_fits_and_continues(context):
    report = _searcher(context, C_values=(-1, 1), degree_values=(2,)).search(context)
    assert len(report.results) == 1
    failure, = report.diagnostics
    assert failure["reason"] == "fit_failed"
    assert failure["C"] == -1 and failure["degree"] == 2
    assert "error" in failure


def test_polynomial_search_records_nonconvergence(context):
    report = _searcher(context, C_values=(0.1,), degree_values=(2,), max_iter=1).search(context)
    assert any(item["reason"] == "polynomial_svm_convergence_warning" for item in report.diagnostics)


def test_polynomial_search_deduplicates_candidate_parameters(context):
    report = _searcher(context, C_values=(1, 1), degree_values=(2, 2)).search(context)
    assert len(report.results) == 1


def test_polynomial_search_handles_empty_feature_order(context, monkeypatch):
    monkeypatch.setattr(polynomial_svm, "miscoding_feature_order", lambda *args, **kwargs: ([], {}))
    report = _searcher(context).search(context)
    assert not report.results
    assert report.diagnostics[0]["reason"] == "empty_feature_order"


def test_multiclass_search_skips_polynomial_svc_with_diagnostic():
    X = np.tile([[-1.], [0.], [1.]], (20, 1))
    y = np.tile([0, 1, 2], 20)
    estimator = NescienceClassifier(models=["polynomial_svc", "naive_bayes"]).fit(X, y)
    assert all(result.family == "naive_bayes" for result in estimator.results_)
    assert estimator.diagnostics_[0]["reason"] == "unsupported_multiclass"
    assert "binary classification only" in estimator.diagnostics_[0]["error"]
    with pytest.raises(ValueError, match="No candidate"):
        NescienceClassifier(models=["polynomial_svc"]).fit(X, y)


def test_polynomial_search_excludes_canonical_prediction_mismatches(context, monkeypatch):
    def altered_predictions(*args):
        actual = execute(*args)
        return 1 - actual if context.task == "classification" else actual + 100

    monkeypatch.setattr(polynomial_svm, "execute", altered_predictions)
    report = _searcher(context, C_values=(1,), degree_values=(2,)).search(context)
    assert not report.results
    diagnostic, = report.diagnostics
    assert diagnostic["reason"] == "description_fidelity_failed"
    assert diagnostic["n_mismatched_predictions"] == len(context.X)


def test_polynomial_search_reports_description_errors(context, monkeypatch):
    def fail(*args):
        raise ValueError("Invalid model expression")

    monkeypatch.setattr(polynomial_svm, "execute", fail)
    report = _searcher(context, C_values=(1,), degree_values=(2,)).search(context)
    assert not report.results
    assert report.diagnostics[0]["reason"] == "description_failed"


def test_polynomial_svr_rejects_excessive_quantization_drift():
    X, y = make_regression(n_samples=80, n_features=4, random_state=42)
    estimator = NescienceRegressor(models=["polynomial_svr"], search_options={
        "polynomial_svr": {"C_values": [10.], "degree_values": [3]},
    })
    with pytest.raises(ValueError, match="No candidate"):
        estimator.fit(X, y)
    assert not estimator.results_
    assert any(item["reason"] == "description_fidelity_failed" for item in estimator.diagnostics_)


@pytest.mark.parametrize("estimator_type,family", [
    (NescienceClassifier, "polynomial_svc"), (NescienceRegressor, "polynomial_svr"),
])
@pytest.mark.parametrize("option", ["kernel", "gamma", "coef0"])
def test_polynomial_search_rejects_fixed_kernel_options(estimator_type, family, option):
    estimator = estimator_type(models=[family], search_options={family: {option: "other"}})
    with pytest.raises(ValueError, match=option):
        estimator._resolve_searchers()
