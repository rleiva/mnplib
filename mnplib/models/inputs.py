"""Resolve fitted-model inputs and canonical artifacts for metric evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.utils.validation import check_is_fitted

from .artifacts import ModelArtifacts
from .sklearn import sklearn_model_artifacts


def model_input(metric, model, *, X=None, feature_indices=None, allow_dummy=False):
    """Return estimator input columns, preserving DataFrame labels."""
    check_is_fitted(metric)
    source = X if X is not None else getattr(metric, "_model_X_", metric.X_)
    if source is None:
        if not allow_dummy:
            raise ValueError("Provide X or fit the metric with X and y.")
        n_features = getattr(model, "n_features_in_", None)
        if n_features is None:
            raise ValueError("Cannot infer model input dimension; provide X.")
        return np.zeros((1, int(n_features)))
    if np.ndim(source) != 2:
        raise ValueError("X must be a two-dimensional feature matrix.")
    if feature_indices is not None:
        indices = np.asarray(feature_indices)
        if indices.ndim != 1 or indices.dtype.kind not in "iu":
            raise ValueError("feature_indices must be a sequence of integer indices.")
        if len(set(indices.tolist())) != len(indices) or np.any(indices < 0):
            raise ValueError("feature_indices must contain unique non-negative indices.")
        # Explicit X is in estimator coordinates; stored X is in original coordinates.
        if X is None:
            if np.any(indices >= np.shape(source)[1]):
                raise ValueError("feature_indices exceed the fitted feature dimension.")
            source = source.iloc[:, indices] if hasattr(source, "iloc") else np.asarray(source)[:, indices]
        elif np.shape(source)[1] != len(indices):
            raise ValueError("Explicit X must contain the estimator input columns in order.")
    if not allow_dummy and len(source) != len(metric.y_):
        raise ValueError("X must contain the same evaluation rows as the fitted y.")
    return source


def model_artifacts(metric, model, *, X=None, feature_names=None,
                    feature_indices=None, allow_dummy=False):
    """Describe a fitted estimator using the canonical serializer layer."""
    source = model_input(metric, model, X=X, feature_indices=feature_indices,
                         allow_dummy=allow_dummy)
    if not allow_dummy and feature_indices is None and np.shape(source)[1] != metric.n_features_in_:
        raise ValueError("feature_indices is required when X contains a selected feature subset.")
    if hasattr(model, "best_artifacts_"):
        if feature_indices is not None:
            raise ValueError("AutoML models accept the full original feature representation.")
        artifacts = model.best_artifacts_
        if not hasattr(model, "predict"):
            if allow_dummy:
                return artifacts
            if np.array_equal(source, getattr(model, "X_supervised_", None)):
                return artifacts
            raise ValueError("Forecasting metrics require the fitted lagged evaluation representation.")
        return ModelArtifacts(list(artifacts.subset), np.asarray(model.predict(source)),
                              artifacts.model_string, artifacts.model_type)
    names = feature_names
    if names is None:
        names = getattr(metric, "feature_names_in_", None)
        if names is not None and feature_indices is not None:
            names = np.asarray(names, dtype=object)[list(feature_indices)]
    if names is None:
        names = getattr(source, "columns", None)
    if names is None:
        names = [f"x{i}" for i in range(np.shape(source)[1])]
    return sklearn_model_artifacts(model, source, feature_names=names,
                                   feature_indices=feature_indices)
