"""
Small fitted-estimator wrappers used by feature-prefix searchers.
"""

from __future__ import annotations

import numpy as np

from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils import check_array


class SelectedFeaturesEstimator:
    """
    Fitted estimator facade that accepts the original full feature matrix.
    """

    def __init__(
        self,
        estimator,
        selected_features,
        *,
        n_features_in: int,
        feature_names=None,
        transformer=None,
    ):
        self.estimator = estimator
        self.selected_features = tuple(int(index) for index in selected_features)
        self.n_features_in_ = int(n_features_in)
        self.feature_names_in_ = (
            np.asarray(feature_names, dtype=object)
            if feature_names is not None
            else np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)
        )
        self.transformer = transformer

        if hasattr(estimator, "classes_"):
            self.classes_ = np.asarray(estimator.classes_)
        if hasattr(estimator, "n_outputs_"):
            self.n_outputs_ = estimator.n_outputs_

    def _select(self, X):
        X_checked = check_array(X, dtype=None, ensure_2d=True)
        X_selected = X_checked[:, self.selected_features]

        if self.transformer is not None:
            return self.transformer.transform(X_selected)

        return X_selected

    def predict(self, X):
        return self.estimator.predict(self._select(X))

    def predict_proba(self, X):
        if not hasattr(self.estimator, "predict_proba"):
            raise AttributeError(
                "The selected estimator does not implement predict_proba()."
            )
        return self.estimator.predict_proba(self._select(X))

    def decision_function(self, X):
        if not hasattr(self.estimator, "decision_function"):
            raise AttributeError(
                "The selected estimator does not implement decision_function()."
            )
        return self.estimator.decision_function(self._select(X))

    def score(self, X, y):
        if hasattr(self.estimator, "score"):
            return self.estimator.score(self._select(X), y)

        predictions = self.predict(X)
        if hasattr(self, "classes_"):
            return accuracy_score(y, predictions)
        return r2_score(y, predictions)
