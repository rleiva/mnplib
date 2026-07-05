"""
Tests for the Inaccuracy class.

These tests target the simplified Inaccuracy API:

    - Inaccuracy(y_type="auto", n_bins="auto")
    - fit(X, y)
    - fit_y(y)
    - inaccuracy_model(model)
    - inaccuracy_predictions(predictions)
    - score(model)
    - inaccuracy_score(y_true, y_pred)

"""

import numpy as np
import pytest

from sklearn.tree       import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.exceptions import NotFittedError
from sklearn.datasets   import load_breast_cancer

from nescience.inaccuracy import Inaccuracy, inaccuracy_score


def test_constructor_defaults():
    metric = Inaccuracy()

    assert metric.y_type == "auto"
    assert metric.n_bins == "auto"


def test_constructor_rejects_invalid_y_type():
    with pytest.raises(ValueError, match="Valid options for 'y_type'"):
        Inaccuracy(y_type="invalid")


def test_fit_classification_sets_fitted_attributes():
    X = np.array([[0.0], [0.1], [1.0], [1.1]])
    y = np.array([0, 0, 1, 1])

    metric = Inaccuracy(n_bins=2).fit(X, y)

    assert metric.is_fitted_ is True
    assert metric.n_samples_in_ == 4
    assert metric.n_features_in_ == 1
    assert metric.y_isnumeric_ is False
    assert metric.len_y_ >= 0.0


def test_fit_regression_sets_numeric_target():
    X = np.array([[0.0], [0.1], [1.0], [1.1]])
    y = np.array([1.0, 1.1, 2.0, 2.1])

    metric = Inaccuracy(n_bins=2).fit(X, y)

    assert metric.is_fitted_ is True
    assert metric.y_isnumeric_ is True
    assert metric.len_y_ >= 0.0


def test_fit_y_allows_prediction_only_usage():
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 0, 1, 1])

    metric = Inaccuracy(n_bins=2).fit_y(y)

    assert metric.X_ is None
    assert metric.is_fitted_ is True
    assert metric.inaccuracy_predictions(pred) == pytest.approx(0.0)


def test_perfect_classification_predictions_have_zero_inaccuracy():
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 0, 1, 1])

    metric = Inaccuracy(n_bins=2).fit_y(y)

    assert metric.inaccuracy_predictions(pred) == pytest.approx(0.0)


def test_perfect_regression_predictions_have_zero_inaccuracy():
    y = np.array([1.0, 1.1, 2.0, 2.1])
    pred = y.copy()

    metric = Inaccuracy(y_type="numeric", n_bins=2).fit_y(y)

    assert metric.inaccuracy_predictions(pred) == pytest.approx(0.0)


def test_constant_equal_targets_and_predictions_have_zero_inaccuracy():
    y = np.array([1, 1, 1, 1])
    pred = np.array([1, 1, 1, 1])

    metric = Inaccuracy(n_bins=2).fit_y(y)

    assert metric.len_y_ == pytest.approx(0.0)
    assert metric.inaccuracy_predictions(pred) == pytest.approx(0.0)


def test_constant_different_targets_and_predictions_have_unit_inaccuracy():
    y = np.array([1, 1, 1, 1])
    pred = np.array([0, 0, 0, 0])

    metric = Inaccuracy(n_bins=2).fit_y(y)

    assert metric.len_y_ == pytest.approx(0.0)
    assert metric.inaccuracy_predictions(pred) == pytest.approx(1.0)


def test_inaccuracy_predictions_returns_value_between_zero_and_one():
    y = np.array([0, 0, 1, 1, 0, 1])
    pred = np.array([0, 1, 1, 0, 0, 1])

    metric = Inaccuracy(n_bins=2).fit_y(y)
    value = metric.inaccuracy_predictions(pred)

    assert isinstance(value, float)
    assert 0.0 <= value <= 1.0


def test_inaccuracy_model_with_classifier():
    X = np.array([[0.0], [0.1], [1.0], [1.1], [0.2], [1.2]])
    y = np.array([0, 0, 1, 1, 0, 1])

    model = DecisionTreeClassifier(random_state=0).fit(X, y)
    metric = Inaccuracy(n_bins=2).fit(X, y)

    value = metric.inaccuracy_model(model)

    assert isinstance(value, float)
    assert 0.0 <= value <= 1.0


def test_inaccuracy_model_with_regressor():
    X = np.array([[0.0], [0.1], [1.0], [1.1], [0.2], [1.2]])
    y = np.array([1.0, 1.1, 2.0, 2.1, 1.2, 2.2])

    model = DecisionTreeRegressor(random_state=0).fit(X, y)
    metric = Inaccuracy(n_bins=2).fit(X, y)

    value = metric.inaccuracy_model(model)

    assert isinstance(value, float)
    assert 0.0 <= value <= 1.0


def test_score_is_one_minus_model_inaccuracy():
    X = np.array([[0.0], [0.1], [1.0], [1.1]])
    y = np.array([0, 0, 1, 1])

    model = DecisionTreeClassifier(random_state=0).fit(X, y)
    metric = Inaccuracy(n_bins=2).fit(X, y)

    assert metric.score(model) == pytest.approx(1.0 - metric.inaccuracy_model(model))


def test_inaccuracy_score_matches_estimator_usage():
    y = np.array([0, 0, 1, 1, 0, 1])
    pred = np.array([0, 1, 1, 0, 0, 1])

    direct = inaccuracy_score(y, pred, n_bins=2)

    metric = Inaccuracy(n_bins=2).fit_y(y)
    via_estimator = metric.inaccuracy_predictions(pred)

    assert direct == pytest.approx(via_estimator)


def test_methods_requiring_fit_raise_not_fitted_error():
    metric = Inaccuracy(n_bins=2)

    with pytest.raises(NotFittedError):
        metric.inaccuracy_predictions([0, 1, 1, 0])


def test_prediction_length_mismatch_raises_value_error():
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 1])

    metric = Inaccuracy(n_bins=2).fit_y(y)

    with pytest.raises(ValueError, match="same number of samples"):
        metric.inaccuracy_predictions(pred)


def test_2d_predictions_raise_value_error():
    y = np.array([0, 0, 1, 1])
    pred = np.array([[0], [0], [1], [1]])

    metric = Inaccuracy(n_bins=2).fit_y(y)

    with pytest.raises(ValueError, match="one-dimensional"):
        metric.inaccuracy_predictions(pred)


def test_empty_target_raises_value_error():
    metric = Inaccuracy(n_bins=2)

    with pytest.raises(ValueError, match="must not be empty"):
        metric.fit_y([])


def test_fit_y_then_inaccuracy_model_raises_value_error():
    y = np.array([0, 0, 1, 1])
    metric = Inaccuracy(n_bins=2).fit_y(y)

    model = DecisionTreeClassifier(random_state=0)

    with pytest.raises(ValueError, match="no feature matrix is available"):
        metric.inaccuracy_model(model)


def test_model_without_predict_raises_type_error():
    X = np.array([[0.0], [0.1], [1.0], [1.1]])
    y = np.array([0, 0, 1, 1])

    metric = Inaccuracy(n_bins=2).fit(X, y)

    with pytest.raises(TypeError, match="predict"):
        metric.inaccuracy_model(object())


def test_manual_y_type_numeric_overrides_auto_detection():
    y = np.array([0, 1, 2, 3])

    metric = Inaccuracy(y_type="numeric", n_bins=2).fit_y(y)

    assert metric.y_isnumeric_ is True


def test_manual_y_type_categorical_overrides_auto_detection():
    y = np.array([0.0, 1.0, 2.0, 3.0])

    metric = Inaccuracy(y_type="categorical", n_bins=2).fit_y(y)

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
