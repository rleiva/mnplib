"""Plain-text presentation of metric analysis reports."""

from copy import deepcopy
from types import MappingProxyType
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from mnplib import Inaccuracy, Miscoding, Nescience, Surfeit
from mnplib import NescienceClassifier, NescienceRegressor, TimeSeries
from mnplib.reporting import format_analysis


def test_inaccuracy_summary_formats_scores_percentages_and_bits(capsys):
    report = {
        "n_samples": 150,
        "y_type": "categorical",
        "resolved_n_bins": None,
        "inaccuracy": 0.09909685339379763,
        "accuracy": 0.9733333333333334,
        "target_code_length_bits": 237.74437510817341,
        "prediction_code_length_bits": 237.6289287076936,
        "joint_code_length_bits": 261.1886481929883,
        "mean_joint_occupancy": 30.0,
    }
    result = format_analysis(report)
    assert result == (
        "Inaccuracy Analysis\n"
        "========================================\n"
        "Samples                              150\n"
        "Target type                  categorical\n"
        "\n"
        "Inaccuracy                        0.0991\n"
        "Accuracy                          97.33%\n"
        "\n"
        "Code lengths (bits)\n"
        "----------------------------------------\n"
        "Target                           237.744\n"
        "Predictions                      237.629\n"
        "Joint                            261.189\n"
        "========================================"
    )
    assert "Numeric bins" not in result
    assert "occupancy" not in result
    assert capsys.readouterr().out == ""


@pytest.fixture
def fitted_model():
    X = pd.DataFrame(
        np.tile([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], (30, 1)),
        columns=["signal", "noise"],
    )
    y = X["signal"].to_numpy()
    return X, y, DecisionTreeRegressor(max_depth=1, random_state=42).fit(X, y)


@pytest.mark.parametrize("cls", [Miscoding, Inaccuracy, Surfeit, Nescience])
def test_fitted_model_reports_are_formatted_without_mutation(cls, fitted_model):
    X, y, model = fitted_model
    report = cls(n_bins=3).fit(X, y).model_analysis(model)
    original = deepcopy(report)
    with warnings.catch_warnings(record=True) as caught:
        result = format_analysis(MappingProxyType(report))
    assert not caught
    assert result.startswith(f"{cls.__name__} Analysis\n")
    assert result == format_analysis(dict(reversed(list(report.items()))))
    assert max(map(len, result.splitlines())) <= 78
    assert report.keys() == original.keys()
    for key in report:
        np.testing.assert_equal(report[key], original[key])
    assert "model_string" not in result
    assert "def predict" not in result
    assert "redundancy_weights" not in result


def test_numeric_inaccuracy_shows_errors_in_target_units():
    report = Inaccuracy(y_type="numeric", n_bins=3).fit_y([0, 1, 2]).prediction_analysis([0, 2, 4])
    text = format_analysis(report)
    assert "MAE" in text and "1.0000" in text
    assert "RMSE" in text and f"{report['rmse']:.4f}" in text
    assert "Numeric bins" in text
    assert "Accuracy" not in text and "%" not in text
    assert "Subset reliability" not in text


def test_surfeit_summary_uses_corrected_code_lengths_in_bits():
    report = Surfeit().fit_y([0, 1] * 20).description_analysis("x = 1\n" * 40)
    text = format_analysis(report)
    for label, key in [
        ("Model", "model_code_length_bits"),
        ("Compressed", "compressed_code_length_bits"),
        ("Effective compressed", "effective_compressed_code_length_bits"),
        ("Target", "target_code_length_bits"),
        ("Reference", "reference_code_length_bits"),
    ]:
        row = next(line for line in text.splitlines() if line.startswith(label + "  "))
        assert row.endswith(f"{report[key]:.3f}")
    assert "Reference source" in text
    assert "Compression ratio" in text
    assert "Code lengths (bits)" in text
    assert "overfitting" not in text


def test_nescience_includes_components_and_aggregation(fitted_model):
    X, y, model = fitted_model
    report = Nescience(aggregation="arithmetic", weights={"surfeit": 0.5}).fit(X, y).model_analysis(model)
    text = format_analysis(report)
    for name in Nescience.component_names_:
        assert name.title() in text
        assert f"{report[name]:.4f}" in text
        assert f"{name}={report['weights'][name]:.4g}" in text
    assert "Aggregation" in text and "arithmetic" in text
    assert "Miscoding Analysis" not in text
    assert "Subset reliability" in text and "Reliable" in text


@pytest.mark.parametrize("cls", [Miscoding, Nescience])
def test_unreliable_model_reports_always_include_failure_reason(cls):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 20))
    y = rng.normal(size=30)
    model = LinearRegression().fit(X, y)
    report = cls().fit(X, y).model_analysis(model)
    text = format_analysis(report)
    assert report["is_reliable"] is False
    assert "Unreliable" in text and "Failure reason" in text
    assert "joint_distribution_too_sparse" in text
    assert "NaN" in text


def test_empty_subset_is_explicit_and_reliable(fitted_model):
    X, y, _ = fitted_model
    report = Miscoding().fit(X, y).subset_analysis([])
    text = format_analysis(report)
    assert "(none)" in text and "Reliable" in text
    assert "Failure reason" not in text
    assert "Numeric bins" not in text


def test_features_use_names_and_summarize_large_subsets(fitted_model):
    X, y, model = fitted_model
    report = Miscoding().fit(X, y).model_analysis(model)
    text = format_analysis(report)
    assert "Selected features" in text and "signal" in text
    assert "noise" not in text
    report["selected_feature_names"] = [f"feature_{i}" for i in range(20)]
    report["n_selected_features"] = 20
    text = format_analysis(report)
    assert "feature_0" in text and "feature_7" in text
    assert "feature_8" not in text and "(+12 more)" in text
    assert max(map(len, text.splitlines())) <= 78


def test_feature_indices_are_used_when_names_are_absent():
    text = format_analysis({"surfeit": 0.5, "selected_features": np.array([2, 0])})
    assert "Selected features" in text and "2, 0" in text


def test_long_labels_are_wrapped_and_embedded_newlines_normalized():
    report = {"miscoding": 0.5, "selected_feature_names": ["x" * 120, "a\nb"]}
    text = format_analysis(report)
    assert max(map(len, text.splitlines())) <= 78
    assert "a b" in text
    assert text.count("x") == 120


@pytest.mark.parametrize("value,expected", [(np.nan, "NaN"), (np.inf, "inf"), (-np.inf, "-inf")])
def test_nonfinite_scores_are_preserved(value, expected):
    text = format_analysis({"inaccuracy": value})
    assert text.splitlines()[2].endswith(expected)


def test_unreliable_report_without_reason_is_not_silently_accepted():
    text = format_analysis({"miscoding": np.nan, "is_reliable": np.bool_(False)})
    assert "Unreliable" in text and "Not provided" in text


@pytest.mark.parametrize("report", [None, [], "report", pd.DataFrame()])
def test_non_mapping_inputs_raise_clear_error(report):
    with pytest.raises(TypeError, match="analysis mapping"):
        format_analysis(report)


@pytest.mark.parametrize("report", [{}, {"accuracy": 0.9}, {"model_string": "x = 1"},
                                   {"inaccuracy": None}])
def test_reports_without_metric_keys_raise_clear_error(report):
    with pytest.raises(ValueError, match="must contain"):
        format_analysis(report)


@pytest.fixture(params=["classification", "regression", "forecasting"])
def automl_estimator(request, fitted_model):
    X, y, _ = fitted_model
    if request.param == "classification":
        return NescienceClassifier(models=["decision_tree"], n_bins=3, random_state=42).fit(X, y)
    if request.param == "regression":
        return NescienceRegressor(models=["linear_regression"], n_bins=3, random_state=42).fit(X, y)
    series = np.sin(np.arange(120) / 5) + np.arange(120) / 100
    return TimeSeries(window_size=3, models=["moving_average"], n_bins=3).fit(series)


def test_automl_summary_matches_the_selected_candidate(automl_estimator):
    estimator = automl_estimator
    report = estimator.explain()
    original = deepcopy(report)
    table = estimator.results_dataframe()
    first = table.iloc[0]
    text = format_analysis(MappingProxyType(report))
    assert report["candidate"] == first["candidate"]
    assert report["native_estimator_score"] == pytest.approx(first["native_estimator_score"])
    assert report["nescience"] == pytest.approx(first["nescience"])
    assert report["is_reliable"] is True
    assert report["model_type"] in text and report["family"] in text
    assert "Selected candidate" in text and "Evaluated samples" in text
    assert "Candidate evaluation" in text and "Subset reliability" in text
    assert "Method" in text and "Weights" in text
    for key in Nescience.component_names_:
        assert key.title() in text and f"{report[key]:.4f}" in text
    for key, value in report["hyperparameters"].items():
        assert key in text and str(value) in text
    assert max(map(len, text.splitlines())) <= 78
    assert "model_string" not in text and "redundancy_weights" not in text
    for key in original:
        np.testing.assert_equal(report[key], original[key])
    pd.testing.assert_frame_equal(table, estimator.results_dataframe())


def test_automl_scores_have_task_specific_labels_and_context(automl_estimator):
    report = automl_estimator.explain()
    text = format_analysis(report)
    if report["task"] == "classification":
        assert text.startswith("Auto-Classification Analysis\n")
        assert "Accuracy" in text and f"{report['native_estimator_score']:.2%}" in text
        assert "Training data" in text
    elif report["task"] == "regression":
        assert text.startswith("Auto-Regression Analysis\n")
        assert "R-squared" in text and f"{report['native_estimator_score']:.4f}" in text
        assert "Training data" in text
    else:
        assert text.startswith("Auto-Time-Series Analysis\n")
        assert "Lagged training data" in text and "R-squared" in text
        assert "Lag window" in text and "Selected lags" in text
        assert report["n_samples"] == len(automl_estimator.y_supervised_)
        assert report["n_samples"] != len(automl_estimator.y_)
        for lag in report["selected_lags"]:
            assert lag["feature_name"] in text


def test_explain_and_formatting_do_not_rescore_the_model(automl_estimator, monkeypatch):
    def unexpected_call(*args, **kwargs):
        raise AssertionError("Reporting must use the saved candidate evaluation.")

    monkeypatch.setattr(type(automl_estimator.model_), "score", unexpected_call)
    monkeypatch.setattr(type(automl_estimator.model_), "predict", unexpected_call)
    monkeypatch.setattr(type(automl_estimator), "score", unexpected_call)
    if isinstance(automl_estimator, TimeSeries):
        monkeypatch.setattr(TimeSeries, "forecast", unexpected_call)
    format_analysis(automl_estimator.explain())


@pytest.mark.parametrize("score", [-0.35, 0.0, np.nan, np.inf])
def test_candidate_r_squared_is_not_clipped_or_formatted_as_percentage(score):
    report = {
        "task": "regression", "candidate": "linear", "nescience": 0.4,
        "native_estimator_score": score, "evaluation_context": "training",
    }
    text = format_analysis(report)
    expected = "NaN" if np.isnan(score) else format(score, ".4f")
    assert "R-squared" in text and expected in text
    assert "%" not in text


def test_automl_formatting_uses_only_reported_hyperparameters():
    report = {
        "task": "regression", "candidate": "linear", "nescience": 0.4,
        "hyperparameters": {"alpha": 0.25, "layer_sizes": (4, 2), "max_depth": None},
    }
    text = format_analysis(report)
    assert "Hyperparameters" in text
    assert "alpha" in text and "0.25" in text
    assert "layer_sizes" in text and "(4, 2)" in text
    assert "max_depth" in text and "None" in text
    assert "fit_intercept" not in text and "random_state" not in text
    assert "Candidate evaluation" not in text
    report["hyperparameters"] = dict(reversed(list(report["hyperparameters"].items())))
    assert format_analysis(report) == text


def test_automl_long_candidate_and_parameter_names_wrap():
    report = {
        "task": "regression", "candidate": "candidate_" * 20, "nescience": 0.4,
        "hyperparameters": {"parameter_" * 10: "value_" * 20},
    }
    assert max(map(len, format_analysis(report).splitlines())) <= 78


def test_automl_unknown_evaluation_context_is_not_reported_as_training():
    report = {"task": "classification", "candidate": "tree", "nescience": 0.4,
              "native_estimator_score": 0.9}
    text = format_analysis(report)
    assert "Not specified" in text and "Training data" not in text


def test_time_series_summary_keeps_exogenous_lags_distinct():
    report = {
        "task": "forecasting", "candidate": "autoregressive", "nescience": 0.4,
        "selected_lags": [{"feature_name": "y_lag_1", "source": "target", "lag": 1},
                          {"feature_name": "temperature_lag_1", "source": "exogenous", "lag": 1}],
        "selected_feature_names": ["y_lag_1", "temperature_lag_1"],
        "n_selected_features": 2,
    }
    text = format_analysis(report)
    assert "Selected lags" in text and "Selected features" not in text
    assert text.count("y_lag_1") == 1 and text.count("temperature_lag_1") == 1
