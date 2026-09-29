"""
Tests for prediction-vector and fitted-model inaccuracy computation.
"""

import numpy as np
import pandas as pd
import pytest

from sklearn.tree       import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.exceptions import NotFittedError
from sklearn.datasets   import load_breast_cancer
from sklearn.utils.validation import check_is_fitted

from mnplib.inaccuracy import Inaccuracy, inaccuracy_predictions
import mnplib.inaccuracy as inaccuracy_module
from mnplib.utils import empirical_distribution_array, empirical_distribution_vector


def test_constructor_defaults():
    metric = Inaccuracy()

    assert metric.y_type == "auto"


def test_constructor_rejects_invalid_y_type():
    with pytest.raises(ValueError, match="Valid options for 'y_type'"):
        Inaccuracy(y_type="invalid")


@pytest.mark.parametrize("method", ["fit", "fit_y"])
@pytest.mark.parametrize("fitted", [False, True])
@pytest.mark.parametrize("failure", ["empty", "shape", "nonfinite", "parameter"])
def test_failed_fitting_invalidates_prediction_diagnostics(method, fitted, failure):
    X = np.arange(8).reshape(4, 2)
    y = np.array([0., 0., 1., 1.])
    metric = Inaccuracy(y_type="numeric")
    if fitted:
        metric.fit(X, y)
    bad_y = {
        "empty": [], "shape": y[:, None], "nonfinite": [0., 1., np.nan, 1.],
        "parameter": y,
    }[failure]
    if failure == "parameter":
        metric.set_params(y_type="invalid")

    with pytest.raises(ValueError):
        getattr(metric, method)(X, bad_y) if method == "fit" else metric.fit_y(bad_y)
    with pytest.raises(NotFittedError):
        check_is_fitted(metric)
    with pytest.raises(NotFittedError):
        metric.prediction_analysis(y)
    with pytest.raises(NotFittedError):
        metric.model_analysis(object(), X=X)

    metric.set_params(y_type="numeric").fit(X, y)
    assert metric.inaccuracy_predictions(y) == pytest.approx(0.0)


@pytest.mark.parametrize("as_dataframe", [False, True])
def test_fitted_evaluation_data_are_independent_of_input_mutation(as_dataframe):
    X = np.array([[0.], [0.], [1.], [1.]])
    y = np.array([0, 0, 1, 1])
    if as_dataframe:
        X = pd.DataFrame(X, columns=["signal"])
        y = pd.Series(y)
    model = DecisionTreeClassifier(random_state=0).fit(X, y)
    metric = Inaccuracy().fit(X, y)
    expected = metric.model_analysis(model)
    if as_dataframe:
        X.iloc[:, :] = 0
        X.columns = ["changed"]
        y.iloc[:] = 0
    else:
        X[:] = 0
        y[:] = 0

    assert metric.model_analysis(model) == expected
    np.testing.assert_array_equal(metric.y_, [0, 0, 1, 1])
    np.testing.assert_array_equal(metric.X_.ravel(), [0, 0, 1, 1])
    if as_dataframe:
        assert list(metric.feature_names_in_) == list(metric._model_X_.columns) == ["signal"]


def test_target_only_fit_keeps_an_independent_snapshot():
    y = np.array([0, 0, 1, 1])
    predictions = np.array([0, 1, 1, 1])
    metric = Inaccuracy().fit_y(y)
    expected = metric.prediction_analysis(predictions)
    y[:] = 0

    assert metric.prediction_analysis(predictions) == expected
    np.testing.assert_array_equal(metric.y_, [0, 0, 1, 1])


@pytest.mark.parametrize("prediction,expected", [("1.0", 0.), ("2.0", 1.)])
def test_constant_numeric_strings_have_valid_inaccuracy(prediction, expected):
    metric = Inaccuracy(y_type="numeric").fit_y(["1.0"] * 4)
    assert metric.inaccuracy_predictions([prediction] * 4) == pytest.approx(expected)


def test_marginal_and_joint_distributions_share_target_bin_count(monkeypatch):
    expected_bins = 9
    y = np.linspace(0, 1, 100)
    predictions = np.roll(y, 7)
    calls = []

    def record_vector(x, *, numeric, n_bins):
        calls.append((1, n_bins))
        return empirical_distribution_vector(x, numeric=numeric, n_bins=n_bins)

    def record_array(X, *, numeric, n_bins):
        calls.append((X.shape[1], n_bins))
        return empirical_distribution_array(X, numeric=numeric, n_bins=n_bins)

    monkeypatch.setattr(inaccuracy_module, "empirical_distribution_vector", record_vector)
    monkeypatch.setattr(inaccuracy_module, "empirical_distribution_array", record_array)
    metric = Inaccuracy(y_type="numeric").fit_y(y)
    report = metric.prediction_analysis(predictions)
    assert calls == [(1, expected_bins), (1, expected_bins), (2, expected_bins)]
    assert report["resolved_n_bins"] == expected_bins
    for key, columns in [("target_code_length_bits", [y]),
                         ("prediction_code_length_bits", [predictions]),
                         ("joint_code_length_bits", [predictions, y])]:
        summary = empirical_distribution_array(np.column_stack(columns), n_bins=expected_bins)
        assert report[key] == summary.code_length


def test_fit_classification_sets_fitted_attributes():
    X = np.array([[0.0], [0.1], [1.0], [1.1]])
    y = np.array([0, 0, 1, 1])

    metric = Inaccuracy().fit(X, y)

    assert metric.is_fitted_ is True
    assert metric.n_samples_in_ == 4
    assert metric.n_features_in_ == 1
    assert metric.y_isnumeric_ is False
    assert metric.len_y_ >= 0.0


def test_fit_regression_sets_numeric_target():
    X = np.array([[0.0], [0.1], [1.0], [1.1]])
    y = np.array([1.0, 1.1, 2.0, 2.1])

    metric = Inaccuracy().fit(X, y)

    assert metric.is_fitted_ is True
    assert metric.y_isnumeric_ is True
    assert metric.len_y_ >= 0.0


def test_fit_y_allows_prediction_only_usage():
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 0, 1, 1])

    metric = Inaccuracy().fit_y(y)

    assert metric.X_ is None
    assert metric.is_fitted_ is True
    assert metric.inaccuracy_predictions(pred) == pytest.approx(0.0)


def test_perfect_classification_predictions_have_zero_inaccuracy():
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 0, 1, 1])

    metric = Inaccuracy().fit_y(y)

    assert metric.inaccuracy_predictions(pred) == pytest.approx(0.0)


def test_perfect_regression_predictions_have_zero_inaccuracy():
    y = np.array([1.0, 1.1, 2.0, 2.1])
    pred = y.copy()

    metric = Inaccuracy(y_type="numeric").fit_y(y)

    assert metric.inaccuracy_predictions(pred) == pytest.approx(0.0)


def test_constant_equal_targets_and_predictions_have_zero_inaccuracy():
    y = np.array([1, 1, 1, 1])
    pred = np.array([1, 1, 1, 1])

    metric = Inaccuracy().fit_y(y)

    assert metric.len_y_ == pytest.approx(0.0)
    assert metric.inaccuracy_predictions(pred) == pytest.approx(0.0)


def test_constant_different_targets_and_predictions_have_unit_inaccuracy():
    y = np.array([1, 1, 1, 1])
    pred = np.array([0, 0, 0, 0])

    metric = Inaccuracy().fit_y(y)

    assert metric.len_y_ == pytest.approx(0.0)
    assert metric.inaccuracy_predictions(pred) == pytest.approx(1.0)


def test_inaccuracy_predictions_returns_value_between_zero_and_one():
    y = np.array([0, 0, 1, 1, 0, 1])
    pred = np.array([0, 1, 1, 0, 0, 1])

    metric = Inaccuracy().fit_y(y)
    value = metric.inaccuracy_predictions(pred)

    assert isinstance(value, float)
    assert 0.0 <= value <= 1.0


def test_inaccuracy_model_with_classifier():
    X = np.array([[0.0], [0.1], [1.0], [1.1], [0.2], [1.2]])
    y = np.array([0, 0, 1, 1, 0, 1])

    model = DecisionTreeClassifier(random_state=0).fit(X, y)
    metric = Inaccuracy().fit(X, y)

    value = metric.inaccuracy_model(model)

    assert isinstance(value, float)
    assert 0.0 <= value <= 1.0


def test_inaccuracy_model_with_regressor():
    X = np.array([[0.0], [0.1], [1.0], [1.1], [0.2], [1.2]])
    y = np.array([1.0, 1.1, 2.0, 2.1, 1.2, 2.2])

    model = DecisionTreeRegressor(random_state=0).fit(X, y)
    metric = Inaccuracy().fit(X, y)

    value = metric.inaccuracy_model(model)

    assert isinstance(value, float)
    assert 0.0 <= value <= 1.0


def test_model_and_prediction_inaccuracy_agree():
    X = np.array([[0.0], [0.1], [1.0], [1.1]])
    y = np.array([0, 0, 1, 1])

    model = DecisionTreeClassifier(random_state=0).fit(X, y)
    metric = Inaccuracy().fit(X, y)

    assert metric.inaccuracy_predictions(model.predict(X)) == pytest.approx(metric.inaccuracy_model(model))


def test_inaccuracy_score_matches_estimator_usage():
    y = np.array([0, 0, 1, 1, 0, 1])
    pred = np.array([0, 1, 1, 0, 0, 1])

    direct = inaccuracy_predictions(pred, y=y)

    metric = Inaccuracy().fit_y(y)
    via_estimator = metric.inaccuracy_predictions(pred)

    assert direct == pytest.approx(via_estimator)


def test_methods_requiring_fit_raise_not_fitted_error():
    metric = Inaccuracy()

    with pytest.raises(NotFittedError):
        metric.inaccuracy_predictions([0, 1, 1, 0])


def test_prediction_length_mismatch_raises_value_error():
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 1])

    metric = Inaccuracy().fit_y(y)

    with pytest.raises(ValueError, match="same number of samples"):
        metric.inaccuracy_predictions(pred)


def test_2d_predictions_raise_value_error():
    y = np.array([0, 0, 1, 1])
    pred = np.array([[0], [0], [1], [1]])

    metric = Inaccuracy().fit_y(y)

    with pytest.raises(ValueError, match="one-dimensional"):
        metric.inaccuracy_predictions(pred)


def test_empty_target_raises_value_error():
    metric = Inaccuracy()

    with pytest.raises(ValueError, match="must not be empty"):
        metric.fit_y([])


def test_fit_y_then_inaccuracy_model_raises_value_error():
    y = np.array([0, 0, 1, 1])
    metric = Inaccuracy().fit_y(y)

    model = DecisionTreeClassifier(random_state=0)

    with pytest.raises(ValueError, match="Provide X"):
        metric.inaccuracy_model(model)


def test_model_without_predict_raises_type_error():
    X = np.array([[0.0], [0.1], [1.0], [1.1]])
    y = np.array([0, 0, 1, 1])

    metric = Inaccuracy().fit(X, y)

    with pytest.raises(TypeError, match="predict"):
        metric.inaccuracy_model(object())


def test_manual_y_type_numeric_overrides_auto_detection():
    y = np.array([0, 1, 2, 3])

    metric = Inaccuracy(y_type="numeric").fit_y(y)

    assert metric.y_isnumeric_ is True


def test_manual_y_type_categorical_overrides_auto_detection():
    y = np.array([0.0, 1.0, 2.0, 3.0])

    metric = Inaccuracy(y_type="categorical").fit_y(y)

    assert metric.y_isnumeric_ is False


def test_detailed_prediction_api_is_not_present():
    metric = Inaccuracy()

    assert not hasattr(metric, "inaccuracy_predictions_detailed")

# No error in list
def test_no_error_list():

    y = [0, 1, 2, 3] * 25
    X = [[0, 1]] * 100

    y_hat = y.copy()

    inacc = Inaccuracy()
    inacc.fit(X, y)
    inaccuracy = inacc.inaccuracy_predictions(y_hat)

    assert inaccuracy == 0

# No error in model
def test_no_error_model():

    X, y = load_breast_cancer(return_X_y=True)

    tree = DecisionTreeClassifier()
    tree.fit(X, y)

    inacc = Inaccuracy()
    inacc.fit(X, y)
    inaccuracy = inacc.inaccuracy_model(tree)

    assert inaccuracy == 0

# One error in list
def test_one_error_list():

    y = [0, 1, 2, 3] * 25
    X = [[0, 1]] * 100

    y_hat = y.copy()
    y_hat[0] = 4

    inacc = Inaccuracy()
    inacc.fit(X, y)
    inaccuracy = inacc.inaccuracy_predictions(y_hat)

    assert inaccuracy > 0

# One error in model
def test_one_error_model():

    X, y = load_breast_cancer(return_X_y=True)

    tree = DecisionTreeClassifier()
    tree.fit(X, y)

    y[0] = 1 - y[0]
    inacc = Inaccuracy()
    inacc.fit(X, y)
    inaccuracy = inacc.inaccuracy_model(tree)

    assert inaccuracy > 0

# All errors in list
def test_all_errors_list():

    y = [0, 1, 2, 3] * 25
    X = [[0, 1]] * 100

    y_hat = [4] * 100

    inacc = Inaccuracy()
    inacc.fit(X, y)
    inaccuracy = inacc.inaccuracy_predictions(y_hat)

    assert inaccuracy > 0

# All errors in model
def test_all_errors_model():

    X, y = load_breast_cancer(return_X_y=True)

    tree = DecisionTreeClassifier()
    tree.fit(X, y)

    y = [2] * len(y)
    inacc = Inaccuracy()
    inacc.fit(X, y)
    inaccuracy = inacc.inaccuracy_model(tree)

    assert inaccuracy == 1
