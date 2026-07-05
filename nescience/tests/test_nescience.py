"""
Tests for the simplified Nescience class.

These tests target the explicit-input API:

    - Nescience(...).fit(X, y)
    - components(subset=..., predictions=..., model_string=...)
    - nescience(subset=..., predictions=..., model_string=...)
    - explain(subset=..., predictions=..., model_string=...)
    - score(subset=..., predictions=..., model_string=...)
    - aggregate_components(...)
    - nescience_score(...)
    - nescience_components(...)

The class deliberately does not inspect fitted model objects. The caller must
provide the selected feature subset, prediction vector, and model description
string explicitly.
"""

import math

import numpy as np
import pandas as pd
import pytest

from sklearn.exceptions import NotFittedError

from nescience.nescience import Nescience, nescience_score, nescience_components


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
    assert metric.aggregation == "euclidean"
    assert metric.weights is None
    assert metric.n_bins == "auto"
    assert metric.threshold_fraction == pytest.approx(0.01)
    assert metric.surplus_penalty == pytest.approx(1.0)
    assert metric.zlib_level == 9
    assert metric.zlib_overhead == 6


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"X_type": "mixed"}, "X_type"),
        ({"X_type": "invalid"}, "X_type"),
        ({"y_type": "invalid"}, "y_type"),
        ({"aggregation": "invalid"}, "aggregation"),
        ({"threshold_fraction": -0.1}, "threshold_fraction"),
        ({"surplus_penalty": -0.1}, "surplus_penalty"),
        ({"zlib_level": -1}, "zlib_level"),
        ({"zlib_level": 10}, "zlib_level"),
        ({"zlib_overhead": -1}, "zlib_overhead"),
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
    assert metric.weights_.shape == (4,)
    assert hasattr(metric, "miscoding_")
    assert hasattr(metric, "inaccuracy_")
    assert hasattr(metric, "surfeit_")


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
    ) == pytest.approx(metric.aggregate_components(**values))


def test_score_is_one_minus_nescience():
    metric, _, y = fitted_metric()

    value = metric.nescience(
        subset=[0],
        predictions=y.copy(),
        model_string=make_model_string(),
    )

    assert metric.score(
        subset=[0],
        predictions=y.copy(),
        model_string=make_model_string(),
    ) == pytest.approx(1.0 - value)


def test_explain_returns_expected_keys():
    metric, _, y = fitted_metric()

    explanation = metric.explain(
        subset=[0],
        predictions=y.copy(),
        model_string=make_model_string(),
    )

    assert set(explanation.keys()) == {
        "nescience",
        "aggregation",
        "weights",
        "components",
        "dominant_component",
        "profile",
        "profile_explanation",
        "recommendation",
    }
    assert explanation["dominant_component"] in metric.component_names_
    assert set(explanation["components"]) == set(metric.component_names_)
    assert isinstance(explanation["recommendation"], str)
    assert len(explanation["recommendation"]) > 0


def test_functional_nescience_score_matches_estimator():
    X, y = make_simple_data()

    direct = nescience_score(
        X,
        y,
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
        X,
        y,
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
        subset=[1, 0],
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

    # [0, 0] would be a valid two-feature binary mask, so use length 3.
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
    ["components", "nescience", "explain", "score"],
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


@pytest.mark.parametrize(
    "aggregation",
    [
        "euclidean",
        "arithmetic",
        "geometric",
        "harmonic",
        "maximum",
        "addition",
        "product",
    ],
)
def test_all_aggregation_modes_return_float(aggregation):
    metric = Nescience(aggregation=aggregation)

    value = metric.aggregate_components(
        deficiency=0.2,
        surplus=0.3,
        inaccuracy=0.4,
        surfeit=0.5,
    )

    assert isinstance(value, float)
    assert value >= 0.0


def test_euclidean_aggregation_formula():
    metric = Nescience(aggregation="euclidean")

    value = metric.aggregate_components(
        deficiency=0.2,
        surplus=0.3,
        inaccuracy=0.4,
        surfeit=0.5,
    )

    expected = math.sqrt((0.2**2 + 0.3**2 + 0.4**2 + 0.5**2) / 4.0)
    assert value == pytest.approx(expected)


def test_arithmetic_aggregation_formula():
    metric = Nescience(aggregation="arithmetic")

    value = metric.aggregate_components(
        deficiency=0.2,
        surplus=0.3,
        inaccuracy=0.4,
        surfeit=0.5,
    )

    assert value == pytest.approx((0.2 + 0.3 + 0.4 + 0.5) / 4.0)


def test_maximum_aggregation_formula():
    metric = Nescience(aggregation="maximum")

    value = metric.aggregate_components(
        deficiency=0.2,
        surplus=0.3,
        inaccuracy=0.4,
        surfeit=0.5,
    )

    assert value == pytest.approx(0.5)


def test_addition_aggregation_formula():
    metric = Nescience(aggregation="addition")

    value = metric.aggregate_components(
        deficiency=0.2,
        surplus=0.3,
        inaccuracy=0.4,
        surfeit=0.5,
    )

    assert value == pytest.approx(1.4)


def test_product_aggregation_formula():
    metric = Nescience(aggregation="product")

    value = metric.aggregate_components(
        deficiency=0.2,
        surplus=0.3,
        inaccuracy=0.4,
        surfeit=0.5,
    )

    assert value == pytest.approx(0.2 * 0.3 * 0.4 * 0.5)


def test_geometric_and_harmonic_return_zero_when_any_active_component_is_zero():
    geometric = Nescience(aggregation="geometric")
    harmonic = Nescience(aggregation="harmonic")

    kwargs = {
        "deficiency": 0.0,
        "surplus": 0.3,
        "inaccuracy": 0.4,
        "surfeit": 0.5,
    }

    assert geometric.aggregate_components(**kwargs) == pytest.approx(0.0)
    assert harmonic.aggregate_components(**kwargs) == pytest.approx(0.0)


def test_weights_sequence_changes_aggregation():
    metric = Nescience(
        aggregation="arithmetic",
        weights=[1.0, 0.0, 0.0, 0.0],
    )

    value = metric.aggregate_components(
        deficiency=0.2,
        surplus=0.3,
        inaccuracy=0.4,
        surfeit=0.5,
    )

    assert value == pytest.approx(0.2)


def test_weights_mapping_defaults_missing_keys_to_one():
    metric = Nescience(
        aggregation="arithmetic",
        weights={"deficiency": 2.0},
    )

    weights = metric._resolve_weights()

    assert weights.tolist() == pytest.approx([2.0, 1.0, 1.0, 1.0])


def test_maximum_ignores_zero_weighted_components():
    metric = Nescience(
        aggregation="maximum",
        weights=[1.0, 0.0, 0.0, 0.0],
    )

    value = metric.aggregate_components(
        deficiency=0.2,
        surplus=0.9,
        inaccuracy=0.8,
        surfeit=0.7,
    )

    assert value == pytest.approx(0.2)


@pytest.mark.parametrize(
    "weights, message",
    [
        ([1.0, 2.0, 3.0], "sequence of four"),
        ([1.0, -1.0, 1.0, 1.0], "non-negative"),
        ([0.0, 0.0, 0.0, 0.0], "positive"),
        ([1.0, float("nan"), 1.0, 1.0], "finite"),
        ("bad", "mapping or a sequence"),
        ({"unknown": 1.0}, "Unknown weight keys"),
    ],
)
def test_invalid_weights_are_rejected(weights, message):
    metric = Nescience(weights=weights)

    with pytest.raises(ValueError, match=message):
        metric._resolve_weights()


def test_invalid_weights_are_rejected_during_fit():
    X, y = make_simple_data()

    with pytest.raises(ValueError, match="positive"):
        Nescience(weights=[0.0, 0.0, 0.0, 0.0]).fit(X, y)


def test_profile_low_nescience_model():
    metric = Nescience()

    profile, explanation = metric._profile_from_components(
        {
            "deficiency": 0.1,
            "surplus": 0.1,
            "inaccuracy": 0.1,
            "surfeit": 0.1,
        }
    )

    assert profile == "low_nescience_model"
    assert isinstance(explanation, str)


@pytest.mark.parametrize(
    "dominant",
    ["deficiency", "surplus", "inaccuracy", "surfeit"],
)
def test_recommendations_for_each_dominant_component(dominant):
    metric = Nescience()
    components = {
        "deficiency": 0.1,
        "surplus": 0.1,
        "inaccuracy": 0.1,
        "surfeit": 0.1,
    }
    components[dominant] = 0.9

    recommendation = metric._recommendation_from_dominant_component(
        dominant,
        components,
    )

    assert dominant in recommendation
    assert isinstance(recommendation, str)
    assert len(recommendation) > 0
