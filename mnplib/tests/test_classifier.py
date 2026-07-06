"""
Tests for the new minimum-nescience classifier.

These tests target the new ``NescienceClassifier`` implementation, which selects
a classification model by minimizing nescience on the full available
representation ``(X, y)``. The class is expected to live in:

    mnplib.classifier.NescienceClassifier

The tests assume that the modular scikit-learn model adapter package is
available under:

    mnplib.models
"""

import numpy as np
import pandas as pd
import pytest

from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from mnplib.classifier import CandidateResult, Classifier, NescienceClassifier
from mnplib.models import SerializationConfig

@pytest.fixture()
def binary_classification_data():
    """Return a deterministic binary classification dataset."""
    X, y = make_classification(
        n_samples=140,
        n_features=6,
        n_informative=3,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        class_sep=1.2,
        random_state=42,
    )
    return X, y


@pytest.fixture()
def multiclass_classification_data():
    """Return a deterministic multiclass classification dataset."""
    X, y = make_classification(
        n_samples=160,
        n_features=7,
        n_informative=4,
        n_redundant=0,
        n_repeated=0,
        n_classes=3,
        n_clusters_per_class=1,
        class_sep=1.3,
        random_state=42,
    )
    return X, y


@pytest.fixture()
def classification_dataframe(binary_classification_data):
    """Return the binary dataset as a pandas DataFrame."""
    X, y = binary_classification_data
    columns = [f"feature_{j}" for j in range(X.shape[1])]
    return pd.DataFrame(X, columns=columns), y


@pytest.fixture()
def small_candidates():
    """Return a compact supported candidate set for fast tests."""
    return [
        (
            "logistic_C_1",
            LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        ),
        (
            "logistic_C_10",
            LogisticRegression(C=10.0, max_iter=1000, random_state=42),
        ),
        (
            "tree_depth_2",
            DecisionTreeClassifier(max_depth=2, random_state=42),
        ),
        (
            "tree_depth_3",
            DecisionTreeClassifier(max_depth=3, random_state=42),
        ),
    ]


def test_fit_selects_a_candidate_and_sets_fitted_attributes(
    binary_classification_data,
    small_candidates,
):
    X, y = binary_classification_data

    clf = NescienceClassifier(
        candidates=small_candidates,
        n_bins=3,
        random_state=42,
        serialization_config=SerializationConfig(precision=4),
    )
    clf.fit(X, y)

    assert clf.is_fitted_
    assert clf.n_samples_in_ == X.shape[0]
    assert clf.n_features_in_ == X.shape[1]
    assert clf.best_candidate_name_ in dict(small_candidates)
    assert clf.model_ is clf.best_result_.estimator
    assert clf.best_artifacts_ is clf.best_result_.artifacts
    assert isinstance(clf.best_nescience_, float)
    assert clf.best_nescience_ >= 0.0
    assert len(clf.results_) == len(small_candidates)
    assert isinstance(clf.best_result_, CandidateResult)
    assert np.array_equal(clf.classes_, clf.model_.classes_)


def test_predict_returns_one_label_per_sample(binary_classification_data, small_candidates):
    X, y = binary_classification_data

    clf = NescienceClassifier(candidates=small_candidates, n_bins=3, random_state=42)
    clf.fit(X, y)

    predictions = clf.predict(X[:9])

    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (9,)
    assert set(predictions).issubset(set(clf.classes_))


def test_predict_proba_returns_probability_matrix(binary_classification_data, small_candidates):
    X, y = binary_classification_data

    clf = NescienceClassifier(candidates=small_candidates, n_bins=3, random_state=42)
    clf.fit(X, y)

    probabilities = clf.predict_proba(X[:11])

    assert isinstance(probabilities, np.ndarray)
    assert probabilities.shape == (11, len(clf.classes_))
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_score_returns_native_accuracy(binary_classification_data, small_candidates):
    X, y = binary_classification_data

    clf = NescienceClassifier(candidates=small_candidates, n_bins=3, random_state=42)
    clf.fit(X, y)

    assert clf.score(X, y) == pytest.approx(clf.model_.score(X, y))
    assert clf.score(X, y) == pytest.approx(accuracy_score(y, clf.predict(X)))


def test_nescience_score_and_components_match_best_result(
    binary_classification_data,
    small_candidates,
):
    X, y = binary_classification_data

    clf = NescienceClassifier(candidates=small_candidates, n_bins=3, random_state=42)
    clf.fit(X, y)

    assert clf.nescience_score() == pytest.approx(clf.best_result_.nescience)

    components = clf.components()

    assert components == clf.best_result_.components
    assert set(components) == {"deficiency", "surplus", "inaccuracy", "surfeit"}
    assert all(isinstance(value, float) for value in components.values())


def test_explain_contains_candidate_and_model_metadata(
    binary_classification_data,
    small_candidates,
):
    X, y = binary_classification_data

    clf = NescienceClassifier(candidates=small_candidates, n_bins=3, random_state=42)
    clf.fit(X, y)

    explanation = clf.explain()

    assert explanation["candidate_name"] == clf.best_candidate_name_
    assert explanation["model_type"] == clf.best_artifacts_.model_type
    assert explanation["model_metadata"] == clf.best_artifacts_.metadata
    assert "components" in explanation
    assert "dominant_component" in explanation


def test_get_model_returns_selected_estimator(binary_classification_data, small_candidates):
    X, y = binary_classification_data

    clf = NescienceClassifier(candidates=small_candidates, n_bins=3, random_state=42)
    clf.fit(X, y)

    assert clf.get_model() is clf.model_


def test_model_string_returns_canonical_string(binary_classification_data, small_candidates):
    X, y = binary_classification_data

    clf = NescienceClassifier(
        candidates=small_candidates,
        n_bins=3,
        random_state=42,
        serialization_config=SerializationConfig(precision=4),
    )
    clf.fit(X, y)

    model_string = clf.model_string()

    assert isinstance(model_string, str)
    assert model_string.startswith("SCHEMA canonical_nescience_model_v1")
    assert "MODEL " in model_string
    assert "TASK classification" in model_string
    assert "RULE" in model_string


def test_results_dataframe_has_expected_columns_and_is_sorted(
    binary_classification_data,
    small_candidates,
):
    X, y = binary_classification_data

    clf = NescienceClassifier(candidates=small_candidates, n_bins=3, random_state=42)
    clf.fit(X, y)

    df = clf.results_dataframe()

    expected_columns = {
        "candidate",
        "model_type",
        "nescience",
        "estimator_score",
        "n_features_in_use",
        "description_length",
        "deficiency",
        "surplus",
        "inaccuracy",
        "surfeit",
    }

    assert expected_columns.issubset(df.columns)
    assert len(df) == len(small_candidates)
    assert df["nescience"].is_monotonic_increasing
    assert df.iloc[0]["candidate"] == clf.best_candidate_name_


def test_dataframe_feature_names_are_preserved(classification_dataframe):
    X, y = classification_dataframe

    clf = NescienceClassifier(
        candidates=[
            (
                "logistic",
                LogisticRegression(max_iter=1000, random_state=42),
            )
        ],
        n_bins=3,
        serialization_config=SerializationConfig(precision=4),
    )
    clf.fit(X, y)

    assert list(clf.feature_names_in_) == list(X.columns)
    assert "feature_" in clf.model_string()


def test_default_candidates_fit_successfully(binary_classification_data):
    X, y = binary_classification_data

    clf = NescienceClassifier(
        candidates="default",
        n_bins=3,
        random_state=42,
        serialization_config=SerializationConfig(precision=4),
    )
    clf.fit(X, y)

    assert clf.is_fitted_
    assert len(clf.results_) > 1
    assert clf.best_nescience_ >= 0.0
    assert clf.best_candidate_name_ in clf.results_dataframe()["candidate"].tolist()


def test_none_candidates_use_default_candidates(binary_classification_data):
    X, y = binary_classification_data

    clf = NescienceClassifier(
        candidates=None,
        n_bins=3,
        random_state=42,
        serialization_config=SerializationConfig(precision=4),
    )
    clf.fit(X, y)

    assert clf.is_fitted_
    assert len(clf.results_) > 1


def test_candidates_can_be_mapping(binary_classification_data):
    X, y = binary_classification_data

    candidates = {
        "logistic": LogisticRegression(max_iter=1000, random_state=42),
        "tree": DecisionTreeClassifier(max_depth=2, random_state=42),
    }

    clf = NescienceClassifier(candidates=candidates, n_bins=3)
    clf.fit(X, y)

    assert {result.name for result in clf.results_} == set(candidates)


def test_candidates_can_be_plain_estimator_sequence(binary_classification_data):
    X, y = binary_classification_data

    candidates = [
        LogisticRegression(max_iter=1000, random_state=42),
        DecisionTreeClassifier(max_depth=2, random_state=42),
    ]

    clf = NescienceClassifier(candidates=candidates, n_bins=3)
    clf.fit(X, y)

    assert len(clf.results_) == 2
    assert clf.results_[0].name.startswith("LogisticRegression_")
    assert clf.results_[1].name.startswith("DecisionTreeClassifier_")


def test_empty_candidates_raise_value_error(binary_classification_data):
    X, y = binary_classification_data

    clf = NescienceClassifier(candidates=[], n_bins=3)

    with pytest.raises(ValueError, match="candidate"):
        clf.fit(X, y)


def test_unsupported_candidate_raises_not_implemented_error(binary_classification_data):
    X, y = binary_classification_data

    clf = NescienceClassifier(
        candidates=[("dummy", DummyClassifier(strategy="most_frequent"))],
        n_bins=3,
    )

    with pytest.raises(NotImplementedError):
        clf.fit(X, y)


def test_invalid_serialization_config_raises_type_error(binary_classification_data):
    X, y = binary_classification_data

    clf = NescienceClassifier(
        candidates=[
            (
                "logistic",
                LogisticRegression(max_iter=1000, random_state=42),
            )
        ],
        n_bins=3,
        serialization_config="not-a-config",
    )

    with pytest.raises(TypeError, match="serialization_config"):
        clf.fit(X, y)


@pytest.mark.parametrize(
    "method_name,args",
    [
        ("predict", (np.zeros((3, 2)),)),
        ("predict_proba", (np.zeros((3, 2)),)),
        ("score", (np.zeros((3, 2)), np.zeros(3))),
        ("nescience_score", ()),
        ("components", ()),
        ("explain", ()),
        ("get_model", ()),
        ("results_dataframe", ()),
        ("model_string", ()),
    ],
)
def test_unfitted_methods_raise_not_fitted_error(method_name, args):
    clf = NescienceClassifier(
        candidates=[
            (
                "logistic",
                LogisticRegression(max_iter=1000, random_state=42),
            )
        ]
    )

    with pytest.raises(NotFittedError):
        getattr(clf, method_name)(*args)


def test_backward_compatible_classifier_alias(binary_classification_data):
    X, y = binary_classification_data

    clf = Classifier(
        candidates=[
            (
                "logistic",
                LogisticRegression(max_iter=1000, random_state=42),
            )
        ],
        n_bins=3,
    )
    clf.fit(X, y)

    assert isinstance(clf, NescienceClassifier)


def test_sklearn_clone_supports_estimator_parameters(small_candidates):
    clf = NescienceClassifier(
        candidates=small_candidates,
        n_bins=3,
        random_state=42,
        verbose=0,
    )

    cloned = clone(clf)

    assert isinstance(cloned, NescienceClassifier)
    assert cloned.n_bins == 3
    assert cloned.random_state == 42
    assert cloned.verbose == 0


def test_verbose_fit_prints_candidate_results(binary_classification_data, capsys):
    X, y = binary_classification_data

    clf = NescienceClassifier(
        candidates=[
            (
                "logistic",
                LogisticRegression(max_iter=1000, random_state=42),
            )
        ],
        n_bins=3,
        verbose=1,
    )
    clf.fit(X, y)

    captured = capsys.readouterr()

    assert "logistic: nescience=" in captured.out
    assert "estimator_score=" in captured.out


def test_custom_weights_and_aggregation_are_accepted(binary_classification_data):
    X, y = binary_classification_data

    clf = NescienceClassifier(
        candidates=[
            (
                "logistic",
                LogisticRegression(max_iter=1000, random_state=42),
            ),
            (
                "tree_depth_2",
                DecisionTreeClassifier(max_depth=2, random_state=42),
            ),
        ],
        aggregation="arithmetic",
        weights={
            "deficiency": 1.0,
            "surplus": 1.0,
            "inaccuracy": 2.0,
            "surfeit": 1.0,
        },
        n_bins=3,
    )
    clf.fit(X, y)

    assert clf.best_nescience_ >= 0.0
    assert clf.nescience_.aggregation == "arithmetic"


def test_multiclass_classification_is_supported(multiclass_classification_data):
    X, y = multiclass_classification_data

    clf = NescienceClassifier(
        candidates=[
            (
                "logistic_multiclass",
                LogisticRegression(max_iter=1000, random_state=42),
            ),
            (
                "tree_depth_3",
                DecisionTreeClassifier(max_depth=3, random_state=42),
            ),
        ],
        n_bins=3,
        serialization_config=SerializationConfig(precision=4),
    )
    clf.fit(X, y)

    predictions = clf.predict(X[:13])
    probabilities = clf.predict_proba(X[:13])

    assert len(clf.classes_) == 3
    assert predictions.shape == (13,)
    assert set(predictions).issubset(set(clf.classes_))
    assert probabilities.shape == (13, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert "TASK classification" in clf.model_string()


def test_string_labels_are_preserved(binary_classification_data):
    X, y_numeric = binary_classification_data
    y = np.array([f"class_{value}" for value in y_numeric])

    clf = NescienceClassifier(
        candidates=[
            (
                "tree_depth_3",
                DecisionTreeClassifier(max_depth=3, random_state=42),
            )
        ],
        n_bins=3,
        serialization_config=SerializationConfig(precision=4),
    )
    clf.fit(X, y)

    predictions = clf.predict(X[:15])

    assert set(clf.classes_) == set(y)
    assert set(predictions).issubset(set(y))
    assert "class_" in clf.model_string()


def test_predict_proba_matches_selected_model(binary_classification_data):
    X, y = binary_classification_data

    clf = NescienceClassifier(
        candidates=[
            (
                "logistic",
                LogisticRegression(max_iter=1000, random_state=42),
            )
        ],
        n_bins=3,
    )
    clf.fit(X, y)

    assert np.allclose(clf.predict_proba(X[:10]), clf.model_.predict_proba(X[:10]))


def test_results_dataframe_description_length_matches_model_string(
    binary_classification_data,
):
    X, y = binary_classification_data

    clf = NescienceClassifier(
        candidates=[
            (
                "logistic",
                LogisticRegression(max_iter=1000, random_state=42),
            )
        ],
        n_bins=3,
        serialization_config=SerializationConfig(precision=4),
    )
    clf.fit(X, y)

    df = clf.results_dataframe()

    assert df.iloc[0]["description_length"] == len(
        clf.model_string().encode("utf-8")
    )
