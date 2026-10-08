"""
Tests for fitted-model metrics, explicit artifacts, and nescience aggregation.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from sklearn.exceptions import NotFittedError
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier

from mnplib.automl import CandidateEvaluator
from mnplib.nescience import Nescience, nescience, nescience_components, nescience_model


def make_simple_data():
    """Return a dataset where the first feature perfectly represents y."""
    X = np.array(
        [
            [0, 1],
            [0, 0],
            [1, 1],
            [1, 0],
            [0, 1],
            [1, 0],
            [0, 0],
            [1, 1],
        ]
    )
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    return X, y


def make_model_string():
    """Return a compact model description string."""
    return "def model(x):\n    return int(x[0] > 0)\n"


def fitted_metric():
    """Return a fitted categorical Nescience estimator and its toy data."""
    X, y = make_simple_data()
    metric = Nescience(X_type="categorical", y_type="categorical").fit(X, y)
    return metric, X, y


def test_constructor_defaults():
    metric = Nescience()

    assert metric.X_type == "auto"
    assert metric.y_type == "auto"
    assert metric.weight == 0.5
    assert set(metric.get_params()) == {"X_type", "y_type", "weight"}


def test_default_model_nescience_is_finite_for_iris_tree():
    X, y = load_iris(return_X_y=True)
    model = DecisionTreeClassifier(min_samples_leaf=5, random_state=42).fit(X, y)
    metric = Nescience().fit(X, y)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = metric.nescience_model(model)
        details = metric.model_analysis(model)
        functional = nescience_model(model, X=X, y=y)

    assert not caught
    assert np.isfinite(value)
    assert value == pytest.approx(details["nescience"])
    assert functional == pytest.approx(value)
    assert details["selected_features"] == [0, 2, 3]
    assert details["resolved_n_bins"] == 5
    assert details["is_reliable"] is True
    assert details["n_observed_joint_states"] == 29


@pytest.mark.parametrize("functional", [False, True])
def test_model_nescience_warns_with_sparse_joint_diagnostics(functional):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 20))
    y = rng.normal(size=30)
    model = LinearRegression().fit(X, y)
    metric = Nescience().fit(X, y)

    with pytest.warns(RuntimeWarning, match="sparsely populated") as caught:
        value = (nescience_model(model, X=X, y=y) if functional
                 else metric.nescience_model(model))

    assert np.isfinite(value)
    assert len(caught) == 1
    message = str(caught[0].message)
    for field in ("n_samples=30", "n_observed_joint_states=30", "resolved_n_bins=2",
                  "mean_joint_occupancy=", "singleton_fraction=",
                  "analysis report"):
        assert field in message


def test_sparse_adaptive_model_nescience_warns_and_returns_value():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 20))
    y = rng.normal(size=30)
    model = LinearRegression().fit(X, y)
    metric = Nescience().fit(X, y)

    with pytest.warns(RuntimeWarning, match="resolved_n_bins=2"):
        assert np.isfinite(metric.nescience_model(model))

    details = metric.model_analysis(model)
    assert details["is_reliable"] is False
    assert details["failure_reason"] == "joint_distribution_too_sparse"
    assert all(np.isfinite(details[key]) for key in ("deficiency", "surplus", "miscoding"))


def test_sparse_diagnostics_and_candidate_evaluation_are_quiet():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 20))
    y = rng.normal(size=30)
    model = LinearRegression().fit(X, y)
    metric = Nescience().fit(X, y)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        details = metric.model_analysis(model)
        result = CandidateEvaluator(X=X, y=y, nescience=metric,
                                    feature_names=metric.feature_names_in_).evaluate(
            name="linear", family="linear_regression", model=model)


    assert not caught
    assert np.isfinite(details["nescience"])
    assert result.nescience == pytest.approx(details["nescience"])
    with pytest.warns(RuntimeWarning, match="sparsely populated") as caught:
        primitive = metric.nescience(**result.artifacts.to_nescience_kwargs())
    assert len(caught) == 1
    assert primitive == pytest.approx(result.nescience)
    assert result.is_reliable is False
    assert result.subset_diagnostics["resolved_n_bins"] == 2


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"X_type": "mixed"}, "X_type"),
        ({"X_type": "invalid"}, "X_type"),
        ({"y_type": "invalid"}, "y_type"),
    ],
)
def test_constructor_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Nescience(**kwargs)


def test_fit_sets_attributes_and_component_estimators():
    metric, X, y = fitted_metric()

    assert metric.is_fitted_ is True
    assert metric.n_samples_in_ == X.shape[0]
    assert metric.n_features_in_ == X.shape[1]
    assert metric.weight == 0.5
    assert hasattr(metric, "miscoding_")
    assert hasattr(metric, "mismodel_")
    assert hasattr(metric.mismodel_, "inaccuracy_")
    assert hasattr(metric.mismodel_, "surfeit_")


def test_fit_preserves_dataframe_feature_names_through_miscoding():
    X, y = make_simple_data()
    X_df = pd.DataFrame(X, columns=["signal", "noise"])

    metric = Nescience(X_type="auto", y_type="categorical").fit(X_df, y)

    assert list(metric.miscoding_.feature_names_in_) == ["signal", "noise"]


def test_components_returns_four_scalar_values():
    metric, _, y = fitted_metric()

    values = metric.components(
        subset=[0],
        predictions=y.copy(),
        model_string=make_model_string(),
    )

    assert set(values) == {"deficiency", "surplus", "inaccuracy", "surfeit"}

    for value in values.values():
        assert isinstance(value, float)
        assert 0.0 <= value <= 1.0


def test_components_for_perfect_feature_and_predictions():
    metric, _, y = fitted_metric()

    values = metric.components(
        subset=[0],
        predictions=y.copy(),
        model_string=make_model_string(),
    )

    assert values["deficiency"] == pytest.approx(0.0)
    assert values["surplus"] == pytest.approx(0.0)
    assert values["inaccuracy"] == pytest.approx(0.0)
    assert 0.0 <= values["surfeit"] <= 1.0


def test_nescience_matches_aggregate_components():
    metric, _, y = fitted_metric()

    values = metric.components(
        subset=[0],
        predictions=y.copy(),
        model_string=make_model_string(),
    )

    assert metric.nescience(
        subset=[0],
        predictions=y.copy(),
        model_string=make_model_string(),
    ) == pytest.approx(metric.aggregate_components(
        miscoding=np.sqrt((values["deficiency"]**2 + values["surplus"]**2) / 2),
        mismodel=np.sqrt((values["inaccuracy"]**2 + values["surfeit"]**2) / 2)))


def test_analysis_and_scalar_nescience_agree():
    metric, _, y = fitted_metric()

    value = metric.nescience(
        subset=[0],
        predictions=y.copy(),
        model_string=make_model_string(),
    )

    assert metric.analysis(
        subset=[0],
        predictions=y.copy(),
        model_string=make_model_string(),
    )["nescience"] == pytest.approx(value)


def test_analysis_returns_numerical_report():
    metric, _, y = fitted_metric()
    report = metric.analysis(subset=[0], predictions=y.copy(), model_string=make_model_string())
    assert set(report) == set(metric.miscoding_.subset_analysis([0])) | {
        "nescience", "weight", "mismodel", *metric.component_names_,
    }
    assert report["mismodel"] == pytest.approx(
        np.sqrt((report["inaccuracy"] ** 2 + report["surfeit"] ** 2) / 2)
    )
    assert report["nescience"] == pytest.approx(metric.aggregate_components(
        miscoding=report["miscoding"], mismodel=report["mismodel"]
    ))


def test_functional_nescience_score_matches_estimator():
    X, y = make_simple_data()

    direct = nescience(
        X=X,
        y=y,
        subset=[0],
        predictions=y.copy(),
        model_string=make_model_string(),
        X_type="categorical",
        y_type="categorical",
    )

    metric = Nescience(X_type="categorical", y_type="categorical").fit(X, y)
    via_estimator = metric.nescience(
        subset=[0],
        predictions=y.copy(),
        model_string=make_model_string(),
    )

    assert direct == pytest.approx(via_estimator)


def test_functional_nescience_components_matches_estimator():
    X, y = make_simple_data()

    direct = nescience_components(
        X=X,
        y=y,
        subset=[0],
        predictions=y.copy(),
        model_string=make_model_string(),
        X_type="categorical",
        y_type="categorical",
    )

    metric = Nescience(X_type="categorical", y_type="categorical").fit(X, y)
    via_estimator = metric.components(
        subset=[0],
        predictions=y.copy(),
        model_string=make_model_string(),
    )

    assert direct == pytest.approx(via_estimator)


def test_components_accepts_binary_mask_subset():
    metric, _, y = fitted_metric()

    from_indices = metric.components(
        subset=[0],
        predictions=y.copy(),
        model_string=make_model_string(),
    )
    from_mask = metric.components(
        subset=[True, False],
        predictions=y.copy(),
        model_string=make_model_string(),
    )

    assert from_mask == pytest.approx(from_indices)


def test_invalid_subset_is_rejected():
    metric, X, y = fitted_metric()

    with pytest.raises(ValueError, match="selected indices"):
        metric.components(
            subset=[X.shape[1]],
            predictions=y.copy(),
            model_string=make_model_string(),
        )

    with pytest.raises(ValueError, match="duplicate"):
        metric.components(
            subset=[0, 0, 0],
            predictions=y.copy(),
            model_string=make_model_string(),
        )


def test_invalid_predictions_are_rejected():
    metric, _, y = fitted_metric()

    with pytest.raises(ValueError, match="same number of samples"):
        metric.components(
            subset=[0],
            predictions=y[:2],
            model_string=make_model_string(),
        )


def test_invalid_model_string_is_rejected():
    metric, _, y = fitted_metric()

    with pytest.raises(TypeError, match="model_string"):
        metric.components(
            subset=[0],
            predictions=y.copy(),
            model_string=123,
        )

    with pytest.raises(ValueError, match="must not be empty"):
        metric.components(
            subset=[0],
            predictions=y.copy(),
            model_string="",
        )


@pytest.mark.parametrize(
    "method_name",
    ["components", "nescience", "analysis"],
)
def test_methods_requiring_fit_raise_not_fitted_error(method_name):
    metric = Nescience()

    kwargs = {
        "subset": [0],
        "predictions": [0, 1, 0, 1],
        "model_string": make_model_string(),
    }

    with pytest.raises(NotFittedError):
        getattr(metric, method_name)(**kwargs)
