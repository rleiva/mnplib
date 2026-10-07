"""Empirical calculation details use the same contexts as miscoding scores."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.tree import DecisionTreeClassifier

import mnplib.miscoding as miscoding_module
from mnplib.miscoding import Miscoding
from mnplib.reporting import format_analysis
from mnplib.utils import _resolve_bins, empirical_distribution_array


CODE_FIELDS = {
    "code_length_bits", "target_code_length_bits", "joint_code_length_bits",
    "target_conditional_code_length_bits", "feature_conditional_code_length_bits",
}
DEBUG_FIELDS = CODE_FIELDS | {
    "target_n_bins", "n_observed_feature_states", "n_observed_target_states",
}
JOINT_FIELDS = {
    "resolved_n_bins", "n_samples", "n_observed_joint_states", "mean_joint_occupancy",
    "n_singleton_joint_states", "singleton_fraction", "is_reliable", "failure_reason",
}


@pytest.fixture
def data():
    X = pd.DataFrame(np.tile([[0, 0], [0, 1], [1, 0], [1, 1]], (80, 1)),
                     columns=["first", "second"])
    X["constant"] = 1
    return X, 2 * X["first"].to_numpy() + X["second"].to_numpy()


def assert_report_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        np.testing.assert_equal(actual[key], value)


@pytest.mark.parametrize("kind", ["numeric", "categorical", "mixed"])
@pytest.mark.parametrize("y_type", ["numeric", "categorical"])
def test_debug_code_lengths_match_empirical_distributions(kind, y_type):
    x = np.linspace(0, 1, 240)
    X = pd.DataFrame({"a": x, "b": np.floor(x * 3), "constant": np.ones(len(x))})
    if kind != "numeric":
        for column in (X.columns if kind == "categorical" else ["b"]):
            X[column] = np.where(X[column] > 0.5, "high", "low")
    y = x if y_type == "numeric" else np.where(x > 0.5, "yes", "no")
    metric = Miscoding(y_type=y_type, debug=True).fit(X, y)
    table = metric.feature_analysis().set_index("feature_index")
    assert DEBUG_FIELDS | JOINT_FIELDS <= set(table.columns)

    for selected in ([0], [1], [2], [0, 1], [0, 1, 2]):
        report = metric.subset_analysis(selected)
        bins = _resolve_bins("adaptive", len(X), subset_size=len(selected))
        numeric = [metric.X_isnumeric_[j] for j in selected]
        features = empirical_distribution_array(X.iloc[:, selected], numeric=numeric, n_bins=bins)
        target = empirical_distribution_array(np.asarray(y)[:, None], numeric=[metric.y_isnumeric_], n_bins=bins)
        joint = empirical_distribution_array(
            np.column_stack([X.iloc[:, selected].to_numpy(dtype=object), y]),
            numeric=[*numeric, metric.y_isnumeric_], n_bins=bins,
        )
        assert report["code_length_bits"] == pytest.approx(features.code_length)
        assert report["target_code_length_bits"] == pytest.approx(target.code_length)
        assert report["joint_code_length_bits"] == pytest.approx(joint.code_length)
        assert report["target_conditional_code_length_bits"] == pytest.approx(joint.code_length - features.code_length)
        assert report["feature_conditional_code_length_bits"] == pytest.approx(joint.code_length - target.code_length)
        assert report["n_observed_feature_states"] == features.n_states
        assert report["n_observed_target_states"] == target.n_states
        assert report["n_observed_joint_states"] == joint.n_states
        assert report["n_samples"] == len(X)
        assert report["target_n_bins"] == (bins if metric.y_isnumeric_ else None)
        assert report["resolved_n_bins"] == (bins if any(numeric) or metric.y_isnumeric_ else None)
        if len(selected) == 1:
            row = table.loc[selected[0]]
            for key in DEBUG_FIELDS | JOINT_FIELDS:
                expected = report[key]
                if expected is None and key in ("target_n_bins", "resolved_n_bins"):
                    assert pd.isna(row[key])
                else:
                    np.testing.assert_equal(row[key], expected)


def test_debug_features_keep_scores_names_sorting_and_scalar_results(data, capsys):
    ordinary = Miscoding().fit(*data)
    debug = Miscoding(debug=True).fit(*data)
    expected = ordinary.feature_analysis()
    actual = debug.feature_analysis()
    pd.testing.assert_frame_equal(actual[expected.columns], expected)
    assert "target_code_length_bits" not in expected
    assert actual["miscoding"].is_monotonic_increasing
    for name in ("deficiency_feature", "surplus_feature", "miscoding_feature"):
        np.testing.assert_equal(getattr(debug, name)(), getattr(ordinary, name)())
        for feature in (0, "second"):
            assert getattr(debug, name)(feature) == getattr(ordinary, name)(feature)
    for name in ("deficiency_subset", "surplus_subset", "miscoding_subset"):
        assert getattr(debug, name)([0, 1]) == getattr(ordinary, name)([0, 1])
    np.testing.assert_array_equal(debug.pairwise_miscoding_, ordinary.pairwise_miscoding_)
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


def test_balanced_binary_features_have_exact_code_lengths_in_bits(data):
    X, y = data
    table = Miscoding(X_type="categorical", debug=True).fit(X, y).feature_analysis()
    for _, row in table.iterrows():
        k_x = 0 if row["feature_name"] == "constant" else len(X)
        assert row["code_length_bits"] == k_x
        assert row["target_code_length_bits"] == 2 * len(X)
        assert row["joint_code_length_bits"] == 2 * len(X)
        assert row["target_conditional_code_length_bits"] == 2 * len(X) - k_x
        assert row["feature_conditional_code_length_bits"] == 0


def test_target_debug_length_uses_the_subsets_resolved_bins():
    x = np.linspace(0, 1, 240)
    metric = Miscoding(y_type="numeric", debug=True).fit(np.column_stack([x, x, x]), x)
    feature = metric.subset_analysis([0])
    subset = metric.subset_analysis([0, 1, 2])
    assert subset["resolved_n_bins"] < feature["resolved_n_bins"]
    assert subset["target_code_length_bits"] < feature["target_code_length_bits"]
    assert feature["target_code_length_bits"] == metric.target_code_length_
    assert subset["target_n_bins"] == subset["resolved_n_bins"]


def test_sparse_subset_exposes_raw_values_without_finite_scores():
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(30, 20)), rng.normal(size=30)
    ordinary = Miscoding().fit(X, y)
    debug = Miscoding(debug=True).fit(X, y)
    selected = list(range(20))
    report = debug.subset_analysis(selected)
    for key, value in ordinary.subset_analysis(selected).items():
        np.testing.assert_equal(report[key], value)
    assert report["is_reliable"] is False
    assert report["failure_reason"] == "joint_distribution_too_sparse"
    for key in CODE_FIELDS:
        assert np.isfinite(report[key])
    for method in (debug.deficiency_subset, debug.surplus_subset, debug.miscoding_subset):
        assert np.isnan(method(selected))
    assert debug._pairwise_miscoding_matrix_ is None


def test_sparse_feature_report_keeps_feature_scores_numeric():
    x = np.arange(15).astype(str)
    metric = Miscoding(debug=True).fit(x[:, None], x)
    row = metric.feature_analysis().iloc[0]
    assert not row["is_reliable"]
    assert row["failure_reason"] == "joint_distribution_too_sparse"
    assert row["n_observed_joint_states"] == 15
    assert row["n_singleton_joint_states"] == 15
    assert row["singleton_fraction"] == 1
    for key in ("deficiency", "surplus", "miscoding"):
        assert row[key] == 0
        assert np.isnan(metric.subset_analysis([0])[key])


@pytest.mark.parametrize("constant", [False, True])
@pytest.mark.parametrize("y_type", ["numeric", "categorical"])
def test_empty_subset_debug_uses_defined_code_lengths(constant, y_type):
    X = np.arange(40).reshape(-1, 1)
    y = np.zeros(40) if constant else X[:, 0]
    metric = Miscoding(y_type=y_type, debug=True).fit(X, y)
    report = metric.subset_analysis([])
    assert report["is_reliable"] and report["failure_reason"] is None
    assert report["code_length_bits"] == 0
    assert report["feature_conditional_code_length_bits"] == 0
    assert report["target_conditional_code_length_bits"] == metric.target_code_length_
    assert report["joint_code_length_bits"] == report["target_code_length_bits"] == metric.target_code_length_
    assert report["n_observed_feature_states"] == 1
    assert report["deficiency"] == (0 if constant else 1)
    assert report["surplus"] == 0
    for key in ("resolved_n_bins", "n_observed_joint_states", "mean_joint_occupancy",
                "n_singleton_joint_states", "singleton_fraction"):
        assert report[key] is None
    assert report["target_n_bins"] == (_resolve_bins("auto", len(y)) if y_type == "numeric" else None)


@pytest.mark.parametrize("method", ["rank_features", "select_features"])
@pytest.mark.parametrize("max_features", [0, 3])
def test_debug_search_reports_preserve_decisions_and_enrich_paths(data, method, max_features):
    options = dict(return_details=True, include_pairwise_miscoding=False, max_features=max_features)
    expected = getattr(Miscoding().fit(*data), method)(**options)
    metric = Miscoding(debug=True).fit(*data)
    actual = getattr(metric, method)(**options)
    key = "feature_order" if method == "rank_features" else "selected_features"
    assert actual[key] == expected[key]
    assert DEBUG_FIELDS | JOINT_FIELDS <= set(actual["path"].columns)
    pd.testing.assert_frame_equal(actual["path"][expected["path"].columns], expected["path"])
    for _, row in actual["path"].iterrows():
        report = metric.subset_analysis(list(row["selected_features"]))
        for field in DEBUG_FIELDS | JOINT_FIELDS:
            np.testing.assert_equal(row[field], report[field])
    if method == "select_features":
        assert_report_equal(actual["subset"], metric.subset_analysis(actual[key]))
    assert metric._pairwise_miscoding_matrix_ is None


@pytest.mark.parametrize("method", ["rank_features", "select_features"])
def test_debug_search_stops_when_every_candidate_is_unreliable(method):
    x = np.arange(15).astype(str)
    metric = Miscoding(debug=True).fit(np.column_stack([x, x]), x)
    current = metric._empirical_subset_measures([])
    candidates = metric._candidate_extensions([], current)
    assert DEBUG_FIELDS | JOINT_FIELDS <= set(candidates.columns)
    assert not candidates["is_reliable"].any()
    assert candidates["miscoding"].isna().all()
    assert np.isfinite(candidates[list(CODE_FIELDS)]).all().all()
    result = getattr(metric, method)(return_details=True, include_pairwise_miscoding=False)
    assert result["path"].empty
    assert DEBUG_FIELDS <= set(result["path"].columns)
    key = "feature_order" if method == "rank_features" else "selected_features"
    assert result[key] == []
    exhausted = metric._candidate_extensions([0, 1], current)
    assert exhausted.empty
    assert DEBUG_FIELDS <= set(exhausted.columns)


def test_model_and_functional_reports_propagate_debug_details(data):
    X, y = data
    metric = Miscoding(debug=True).fit(X, y)
    model = DecisionTreeClassifier(random_state=0).fit(X, y)
    report = metric.model_analysis(model)
    assert_report_equal(report, metric.subset_analysis(report["selected_features"]))
    assert_report_equal(report, miscoding_module.model_analysis(model, X=X, y=y, debug=True))
    assert_report_equal(metric.subset_analysis([0, 1]),
                        miscoding_module.subset_analysis([0, 1], X=X, y=y, debug=True))
    pd.testing.assert_frame_equal(metric.feature_analysis(),
                                  miscoding_module.feature_analysis(X=X, y=y, debug=True))
    for method in ("rank_features", "select_features"):
        options = dict(return_details=True, include_pairwise_miscoding=False)
        direct = getattr(miscoding_module, method)(X=X, y=y, debug=True, **options)
        expected = getattr(metric, method)(**options)
        pd.testing.assert_frame_equal(direct["path"], expected["path"])
        pd.testing.assert_frame_equal(direct["features"], expected["features"])


def test_debug_reports_reuse_cache_without_pairwise_work(data, monkeypatch):
    metric = Miscoding(debug=True).fit(*data)
    expected = metric.subset_analysis([0, 1])
    cache = metric._empirical_cache_.copy()

    def unexpected(*args, **kwargs):
        raise AssertionError("Cached debug reports must not compute new distributions.")

    monkeypatch.setattr(miscoding_module, "empirical_distribution_array", unexpected)
    monkeypatch.setattr(metric, "_compute_pairwise_miscoding_matrix", unexpected)
    metric.feature_analysis()
    assert_report_equal(metric.subset_analysis([0, 1]), expected)
    metric.subset_analysis([1, 0])
    assert metric._empirical_cache_ == cache
    assert metric._pairwise_miscoding_matrix_ is None


def test_debug_supports_cloning_toggling_and_refitting(data):
    metric = Miscoding(debug=True).fit(*data)
    assert clone(metric).get_params() == metric.get_params()
    report = metric.feature_analysis()
    metric.set_params(debug=False)
    assert "target_code_length_bits" not in metric.feature_analysis()
    metric.set_params(debug=True)
    pd.testing.assert_frame_equal(metric.feature_analysis(), report)
    X = pd.DataFrame({"replacement": np.zeros(20)})
    metric.fit(X, np.zeros(20))
    table = metric.feature_analysis()
    assert table["feature_name"].tolist() == ["replacement"]
    assert table["n_samples"].tolist() == [20]
    for key in CODE_FIELDS:
        assert table.iloc[0][key] == 0


@pytest.mark.parametrize("debug", [None, 0, 1, "true", [], np.array([True])])
def test_debug_requires_a_boolean_at_construction_and_fit(data, debug):
    with pytest.raises(TypeError, match="debug must be a boolean"):
        Miscoding(debug=debug)
    metric = Miscoding().fit(*data).set_params(debug=debug)
    with pytest.raises(TypeError, match="debug must be a boolean"):
        metric.fit(*data)
    assert not metric.__sklearn_is_fitted__()


def test_debug_accepts_numpy_boolean_and_requires_fit():
    metric = Miscoding(debug=np.bool_(True))
    with pytest.raises(NotFittedError):
        metric.feature_analysis()
    with pytest.raises(NotFittedError):
        metric.subset_analysis([])


def test_formatted_debug_reports_include_code_lengths_and_bins(data):
    metric = Miscoding(debug=True).fit(*data)
    table = metric.feature_analysis()
    original = table.copy(deep=True)
    feature_text = format_analysis(table)
    subset_text = format_analysis(metric.subset_analysis([0, 1]))
    for text in (feature_text, subset_text):
        for label in ("K(X)", "K(Y)", "K(X, Y)", "K(Y | X)", "K(X | Y)",
                      "Numeric bins", "Feature states", "Target states"):
            assert label in text
        assert "bits" in text
    pd.testing.assert_frame_equal(table, original)
    assert "K(Y | X)" not in format_analysis(Miscoding().fit(*data).feature_analysis())
