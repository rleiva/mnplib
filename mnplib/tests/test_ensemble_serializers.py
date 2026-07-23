"""
Tests that tree ensembles stay outside the fixed adapter dispatch.
"""

from __future__ import annotations

import pytest

from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)

from mnplib.models import sklearn_model_artifacts


@pytest.mark.parametrize(
    "model",
    [
        RandomForestRegressor(n_estimators=3, max_depth=3, random_state=42),
        ExtraTreesRegressor(n_estimators=3, max_depth=3, random_state=42),
        GradientBoostingRegressor(n_estimators=3, max_depth=2, random_state=42),
        HistGradientBoostingRegressor(max_iter=3, max_leaf_nodes=4, random_state=42),
    ],
)
def test_tree_ensemble_regressors_are_not_supported_by_static_adapter(model):
    X, y = make_regression(
        n_samples=60,
        n_features=4,
        n_informative=2,
        noise=0.5,
        random_state=42,
    )
    model.fit(X, y)

    with pytest.raises(
        ValueError,
        match=f"Unsupported scikit-learn model type {type(model).__name__}",
    ):
        sklearn_model_artifacts(model, X)


@pytest.mark.parametrize(
    "model",
    [
        RandomForestClassifier(n_estimators=3, max_depth=3, random_state=42),
        ExtraTreesClassifier(n_estimators=3, max_depth=3, random_state=42),
        GradientBoostingClassifier(n_estimators=3, max_depth=2, random_state=42),
        HistGradientBoostingClassifier(max_iter=3, max_leaf_nodes=4, random_state=42),
    ],
)
def test_tree_ensemble_classifiers_are_not_supported_by_static_adapter(model):
    X, y = make_classification(
        n_samples=80,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )
    model.fit(X, y)

    with pytest.raises(
        ValueError,
        match=f"Unsupported scikit-learn model type {type(model).__name__}",
    ):
        sklearn_model_artifacts(model, X)
