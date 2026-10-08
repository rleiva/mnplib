"""Reliability describes statistical support without suppressing estimates."""

from dataclasses import replace
import warnings

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

import mnplib.miscoding as miscoding_module
import mnplib.nescience as nescience_module
from mnplib import ResidualAnalysis
from mnplib.automl import CandidateEvaluator
from mnplib.automl.results import candidate_results_dataframe
from mnplib.classifier import NescienceClassifier
from mnplib.inaccuracy import Inaccuracy
from mnplib.miscoding import Miscoding
from mnplib.mismodel import Mismodel
from mnplib.nescience import Nescience
from mnplib.regressor import NescienceRegressor
from mnplib.reporting import format_analysis
from mnplib.surfeit import Surfeit
from mnplib.timeseries import TimeSeries
from mnplib.utils import empirical_distribution_array


@pytest.mark.parametrize("categorical", [False, True])
@pytest.mark.parametrize("reliable", [False, True])
def test_empirical_scores_match_counts_with_reliability_diagnostics(categorical, reliable):
    rng = np.random.default_rng(1)
    X = rng.integers(0, 2, size=(500, 2)) if reliable else rng.normal(size=(30, 20))
    y = (2 * X[:, 0] + X[:, 1] if reliable else rng.integers(0, 3, size=len(X)))
    if categorical:
        X = X.astype(str)
    metric = Miscoding(X_type="categorical" if categorical else "numeric",
                       y_type="categorical").fit(X, y)
    selected = list(range(X.shape[1]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        report = metric.subset_analysis(selected)
        bins = report["resolved_n_bins"] or "auto"
        features = empirical_distribution_array(X, numeric=not categorical, n_bins=bins)
        target = empirical_distribution_array(y[:, None], numeric=False, n_bins=bins)
        joint = empirical_distribution_array(
            np.column_stack([X, y]), numeric=[not categorical] * X.shape[1] + [False],
            n_bins=bins,
        )
    deficiency = np.clip((joint.code_length - features.code_length) / target.code_length, 0, 1)
    surplus = np.clip((joint.code_length - target.code_length) / features.code_length, 0, 1)
    assert report["deficiency"] == pytest.approx(deficiency)
    assert report["surplus"] == pytest.approx(surplus)
    assert report["miscoding"] == pytest.approx(np.sqrt((deficiency**2 + surplus**2) / 2))
    assert report["is_reliable"] is reliable
    assert report["n_observed_joint_states"] == joint.n_states
    for name in ("deficiency", "surplus", "miscoding"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = getattr(metric, name + "_subset")(selected)
        assert value == pytest.approx(report[name])
        assert len(caught) == (0 if reliable else 1)


@pytest.mark.parametrize("quantity", ["deficiency", "surplus", "miscoding"])
@pytest.mark.parametrize("level", ["feature", "subset"])
@pytest.mark.parametrize("functional", [False, True])
def test_sparse_convenience_interfaces_warn_once_at_caller(quantity, level, functional):
    x = np.arange(20).astype(str)
    X = np.column_stack([x, x])
    metric = Miscoding().fit(X, x)
    argument = None if level == "feature" else [0, 1]
    method = quantity + "_" + level
    with pytest.warns(RuntimeWarning, match="sparsely populated") as caught:
        value = (getattr(miscoding_module, method)(argument, X=X, y=x)
                 if functional else getattr(metric, method)(argument))
    assert np.isfinite(value).all()
    assert len(caught) == 1
    assert caught[0].filename == __file__
    for field in ("n_samples=20", "n_observed_joint_states=20",
                  "mean_joint_occupancy=1", "singleton_fraction=1"):
        assert field in str(caught[0].message)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        table = metric.feature_analysis()
        assert not table["is_reliable"].any()
        assert "False" in format_analysis(table)
        metric.subset_analysis([0, 1])


@pytest.mark.parametrize("method", ["rank_features", "select_features"])
@pytest.mark.parametrize("functional", [False, True])
def test_search_uses_sparse_candidates_and_detailed_paths_are_quiet(method, functional):
    x = np.arange(20).astype(str)
    X = np.column_stack([x, x])
    metric = Miscoding().fit(X, x)
    call = getattr(miscoding_module, method) if functional else getattr(metric, method)
    kwargs = {"X": X, "y": x} if functional else {}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        report = call(return_details=True, **kwargs)
    assert not report["path"].empty
    assert not report["path"]["is_reliable"].any()
    assert (report["path"]["miscoding"] == 0).all()
    with pytest.warns(RuntimeWarning, match="sparsely populated") as caught:
        result = call(**kwargs)
    assert len(caught) == 1
    assert caught[0].filename == __file__
    expected = report["feature_order"] if method == "rank_features" else report["mask"]
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("weight", [0, 0.3, 1])
def test_sparse_model_components_aggregate_normally_without_duplicate_warnings(weight):
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(30, 20)), rng.normal(size=30)
    model = LinearRegression().fit(X, y)
    metric = Nescience(weight=weight).fit(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        report = metric.model_analysis(model)
        fitted = Mismodel().fit(X, y).model_analysis(model)
        candidate = CandidateEvaluator(X=X, y=y, nescience=metric,
                                       feature_names=metric.feature_names_in_).evaluate(
            name="linear", family="linear_regression", model=model,
        )
    assert not report["is_reliable"]
    assert np.isfinite(report["miscoding"])
    assert report["mismodel"] == pytest.approx(fitted["mismodel"])
    expected = np.sqrt(weight * report["miscoding"]**2 + (1 - weight) * report["mismodel"]**2)
    assert candidate.nescience == pytest.approx(expected)
    for call in (
        lambda: metric.nescience_model(model),
        lambda: metric.nescience(**candidate.artifacts.to_nescience_kwargs()),
        lambda: nescience_module.nescience_model(model, X=X, y=y, weight=weight),
        lambda: nescience_module.nescience(X=X, y=y, weight=weight,
                                         **candidate.artifacts.to_nescience_kwargs()),
    ):
        with pytest.warns(RuntimeWarning, match="sparsely populated") as caught:
            assert call() == pytest.approx(expected)
        assert len(caught) == 1
        assert caught[0].filename == __file__
    ranked = candidate_results_dataframe([
        replace(candidate, name="missing", nescience=np.nan),
        replace(candidate, name="reliable", nescience=0.9, subset_diagnostics={"is_reliable": True}),
        replace(candidate, name="sparse", nescience=0.1),
    ])
    assert ranked["candidate"].tolist() == ["sparse", "reliable", "missing"]


@pytest.mark.parametrize("kind", ["classifier", "regressor", "timeseries"])
def test_automl_rejects_unavailable_scores_and_keeps_diagnostics_quiet(monkeypatch, kind):
    rng = np.random.default_rng(123)
    X = rng.normal(size=(30, 4))
    y = rng.normal(size=30)
    if kind == "classifier":
        metric = NescienceClassifier(models=["linear_svc"], random_state=1)
        args = (X, (y > 0).astype(int))
    elif kind == "regressor":
        metric = NescienceRegressor(models=["linear_regression"], random_state=1)
        args = (X, y)
    else:
        metric = TimeSeries(models=["moving_average"], window_size=4)
        args = (y,)
    search = metric._fit_searchers

    def controlled_search():
        search()
        candidate = metric.results_[0]
        diagnostics = dict(candidate.subset_diagnostics, is_reliable=False,
                           failure_reason="joint_distribution_too_sparse")
        metric.results_ = [
            replace(candidate, name="missing", nescience=np.nan),
            replace(candidate, name="finite", nescience=0.2, subset_diagnostics=diagnostics),
        ]

    monkeypatch.setattr(metric, "_fit_searchers", controlled_search)
    with pytest.warns(RuntimeWarning, match="sparsely populated") as caught:
        metric.fit(*args)
    assert len(caught) == 1
    assert metric.best_result_.name == "finite"
    with pytest.warns(RuntimeWarning, match="sparsely populated") as caught:
        assert metric.nescience() == 0.2
    assert len(caught) == 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        metric.analysis()
        table = metric.results_dataframe()
    assert table.iloc[0]["candidate"] == "finite"
    assert np.isnan(table.iloc[-1]["nescience"])

    def unavailable_search():
        controlled_search()
        metric.results_ = [replace(result, nescience=np.nan) for result in metric.results_]

    monkeypatch.setattr(metric, "_fit_searchers", unavailable_search)
    with pytest.raises(ValueError, match="finite nescience"):
        metric.fit(*args)


def test_sparse_residual_explanations_do_not_change_anomaly_detection():
    y = np.arange(20).astype(str)
    predictions = np.roll(y, 1)
    X = np.column_stack([y, y])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        metric = ResidualAnalysis(task="classification").fit(X, y, predictions=predictions)
        report = metric.feature_analysis()
        table = report["feature_analysis"]
        metric.analysis()
        metric.results_dataframe()
    np.testing.assert_array_equal(metric.anomalies(), np.arange(20))
    assert not table["is_reliable"].any()
    assert np.isfinite(table["miscoding"]).all()
    assert not report["subset_analysis"]["is_reliable"]
    assert np.isfinite(report["subset_analysis"]["miscoding"])


def test_sparse_primitive_diagnostics_and_invalid_inputs():
    y = np.arange(20).astype(str)
    predictions = np.roll(y, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        inaccuracy = Inaccuracy(y_type="categorical").fit_y(y)
        report = inaccuracy.prediction_analysis(predictions)
        assert report["n_observed_joint_states"] == 20
        assert np.isfinite(inaccuracy.inaccuracy_predictions(predictions))
        assert np.isfinite(Surfeit().fit_y(y).surfeit_string("def model(x):\n    return 1\n"))
        assert np.isfinite(Mismodel().fit_y(y).mismodel(
            predictions=predictions, model_string="def model(x):\n    return 1\n"))
    with pytest.raises(ValueError):
        inaccuracy.inaccuracy_predictions(predictions[:-1])
    with pytest.raises(ValueError):
        Surfeit().fit_y(y).surfeit_string("")
    with pytest.raises(ValueError):
        Miscoding().fit(y[:, None], y).miscoding_subset([2])
    with pytest.raises(ValueError):
        empirical_distribution_array([[np.nan]], numeric=True)


@pytest.mark.parametrize("weight", [0, 0.5, 1])
def test_unavailable_component_propagates_nan_without_a_reliability_warning(monkeypatch, weight):
    x = np.tile([0., 1.], 50)
    X = x[:, None]
    model = LinearRegression().fit(X, x)
    metric = Nescience(weight=weight).fit(X, x)
    monkeypatch.setattr(metric.mismodel_.inaccuracy_, "inaccuracy_predictions", lambda _: np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        report = metric.model_analysis(model)
        assert report["is_reliable"]
        assert np.isfinite(report["miscoding"])
        assert np.isnan(report["mismodel"])
        assert np.isnan(metric.nescience_model(model))


@pytest.mark.parametrize("method", ["select_features", "rank_features"])
def test_feature_search_stops_when_no_finite_candidate_remains(monkeypatch, method):
    x = np.tile([0, 1], 50)
    metric = Miscoding().fit(x[:, None], x)
    extensions = metric._candidate_extensions

    def unavailable_extensions(*args):
        candidates = extensions(*args)
        candidates[["deficiency", "surplus", "miscoding", "miscoding_improvement"]] = np.nan
        return candidates

    monkeypatch.setattr(metric, "_candidate_extensions", unavailable_extensions)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        report = getattr(metric, method)(return_details=True)
    assert report["path"].empty


def test_lag_analysis_retains_sparse_empirical_estimates_quietly():
    y = np.random.default_rng(123).normal(size=10)
    with pytest.warns(RuntimeWarning, match="sparsely populated"):
        metric = TimeSeries(window_size=5, models=["moving_average"]).fit(y)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        report = metric.lag_analysis(max_lag=8)
    assert not report["is_reliable"].all()
    assert np.isfinite(report[["deficiency", "surplus", "miscoding"]]).all().all()
    assert (report.loc[~report["is_reliable"], "failure_reason"] == "joint_distribution_too_sparse").all()
    assert np.isnan(metric.fitted_values()[:5]).all()
