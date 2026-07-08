"""
Minimum-nescience regressor.

This module implements a simplified AutoML-style regressor based on the
Minimum Nescience Principle. It is designed for the revised mnplib architecture:

    * ``Miscoding`` computes feature deficiency and surplus.
    * ``Inaccuracy`` computes prediction mismatch.
    * ``Surfeit`` computes redundancy of a canonical model description.
    * ``Nescience`` aggregates the four components.

The class does not split the data into train/test subsets. In the theory of
nescience, the selected model is the one that minimizes nescience with respect
to the available effective representation ``(X, y)``. Model complexity and
excessive descriptive structure are penalized internally through the nescience
components, especially surfeit and surplus.

The class relies on the scikit-learn adapter layer to translate fitted models
into explicit nescience artifacts:

    model + X -> subset, predictions, model_string

Supported models depend on the serializers registered in ``mnplib.models``.
The default candidate set uses only models supported by the current stable
adapter implementation:

    * LinearRegression
    * Ridge
    * Lasso
    * ElasticNet
    * DecisionTreeRegressor

@author:    Rafael Garcia Leiva
@mail:      rgarcialeiva@gmail.com
@copyright: GNU GPLv3
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted

from .nescience import Nescience
from .models import SerializationConfig, sklearn_model_artifacts


XType = Literal["auto", "numeric", "categorical"]
BinSpec = int | Literal["auto"]
Aggregation = Literal[
    "euclidean",
    "arithmetic",
    "geometric",
    "harmonic",
    "maximum",
    "addition",
    "product",
]


@dataclass(frozen=True)
class CandidateResult:
    """
    Result obtained by evaluating one candidate regressor.

    Parameters
    ----------
    name : str
        Candidate name.

    estimator : object
        Fitted scikit-learn regressor.

    nescience : float
        Aggregated nescience value.

    components : dict
        Four scalar nescience components: deficiency, surplus, inaccuracy, and
        surfeit.

    estimator_score : float
        Native estimator score on the full training representation. For
        scikit-learn regressors this is usually :math:`R^2`.

    artifacts : object
        ``ModelArtifacts`` object returned by the scikit-learn adapter.

    metadata : dict
        Optional adapter/model metadata.
    """

    name: str
    estimator: object
    nescience: float
    components: dict[str, float]
    estimator_score: float
    artifacts: object
    metadata: dict[str, Any]


class NescienceRegressor(BaseEstimator, RegressorMixin):
    """
    Select a regression model by minimizing nescience.

    The estimator evaluates a set of candidate scikit-learn regressors on the
    same representation ``(X, y)`` used for fitting. The selected candidate is
    the one with the lowest aggregated nescience.

    Parameters
    ----------
    candidates : "default", mapping, sequence, or None, default="default"
        Candidate regressors to evaluate.

        If ``"default"`` or ``None``, a built-in set of supported regressors is
        used. A mapping should map names to estimators. A sequence may contain
        estimators or ``(name, estimator)`` pairs.

    include_ensembles : bool, default=False
        Whether the built-in default search should include supported ensemble
        regressors such as random forests, extra-trees, and gradient boosting.
        This option is ignored when explicit ``candidates`` are supplied.

    X_type : {"auto", "numeric", "categorical"}, default="numeric"
        Type of the feature matrix used by ``Nescience``.

    aggregation : {"euclidean", "arithmetic", "geometric", "harmonic",
                   "maximum", "addition", "product"}, default="euclidean"
        Aggregation method used by ``Nescience``.

    weights : mapping or sequence of 4 floats, optional
        Component weights passed to ``Nescience``.

    n_bins : int or "auto", default="auto"
        Number of bins used for numeric variables by the underlying metrics.

    threshold_fraction : float, default=0.01
        Passed to the internal ``Nescience`` object and used by its
        ``Miscoding`` component.

    surplus_penalty : float, default=1.0
        Passed to the internal ``Nescience`` object and used by its
        ``Miscoding`` component.

    zlib_level : int, default=9
        Compression level used by ``Surfeit``.

    zlib_overhead : int, default=6
        zlib wrapper overhead subtracted by ``Surfeit``.

    serialization_config : SerializationConfig, optional
        Configuration used by the model serializers. If omitted, the default
        canonical serialization configuration is used.

    random_state : int, RandomState instance, or None, default=None
        Random seed assigned to default candidates that expose a
        ``random_state`` parameter.

    verbose : int, default=0
        Verbosity level. If greater than zero, candidate scores are printed
        during fitting.
    """

    def __init__(
        self,
        candidates="default",
        include_ensembles: bool = False,
        X_type: XType = "numeric",
        aggregation: Aggregation = "euclidean",
        weights: Mapping[str, float] | Sequence[float] | None = None,
        n_bins: BinSpec = "auto",
        threshold_fraction: float = 0.01,
        surplus_penalty: float = 1.0,
        zlib_level: int = 9,
        zlib_overhead: int = 6,
        serialization_config: SerializationConfig | None = None,
        random_state=None,
        verbose: int = 0,
    ):
        self.candidates = candidates
        self.include_ensembles = include_ensembles
        self.X_type = X_type
        self.aggregation = aggregation
        self.weights = weights
        self.n_bins = n_bins
        self.threshold_fraction = threshold_fraction
        self.surplus_penalty = surplus_penalty
        self.zlib_level = zlib_level
        self.zlib_overhead = zlib_overhead
        self.serialization_config = serialization_config
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        """
        Fit all candidate regressors and select the one with minimum nescience.

        Parameters
        ----------
        X : array-like or pandas.DataFrame of shape (n_samples, n_features)
            Input representation.

        y : array-like of shape (n_samples,)
            Numeric target representation.

        Returns
        -------
        self : NescienceRegressor
            Fitted selector.
        """
        feature_names = self._resolve_input_feature_names(X)

        X_checked, y_checked = check_X_y(X, y, dtype=None, ensure_2d=True)

        self.X_ = X_checked
        self.y_ = y_checked
        self.n_samples_in_, self.n_features_in_ = X_checked.shape
        self.feature_names_in_ = np.asarray(feature_names, dtype=object)
        self.serialization_config_ = self._resolve_serialization_config()

        self.nescience_ = Nescience(
            X_type=self.X_type,
            y_type="numeric",
            aggregation=self.aggregation,
            weights=self.weights,
            n_bins=self.n_bins,
            threshold_fraction=self.threshold_fraction,
            surplus_penalty=self.surplus_penalty,
            zlib_level=self.zlib_level,
            zlib_overhead=self.zlib_overhead,
        )
        self.nescience_.fit(self.X_, self.y_)

        self.candidates_ = self._resolve_candidates()
        self.results_ = []

        best_result = None

        for name, estimator in self.candidates_:
            result = self._evaluate_candidate(name, estimator)
            self.results_.append(result)

            if self.verbose:
                print(
                    f"{name}: nescience={result.nescience:.6f}, "
                    f"estimator_score={result.estimator_score:.6f}"
                )

            if best_result is None or result.nescience < best_result.nescience:
                best_result = result

        if best_result is None:
            raise RuntimeError("No candidate regressor was successfully evaluated.")

        self.best_result_ = best_result
        self.model_ = best_result.estimator
        self.best_nescience_ = float(best_result.nescience)
        self.best_components_ = dict(best_result.components)
        self.best_artifacts_ = best_result.artifacts
        self.best_candidate_name_ = best_result.name

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """
        Predict using the selected minimum-nescience regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        numpy.ndarray
            Predictions of the selected regressor.
        """
        check_is_fitted(self)

        X_checked = check_array(X, dtype=None, ensure_2d=True)
        return self.model_.predict(X_checked)

    def score(self, X, y):
        """
        Return the native score of the selected regressor.

        For standard scikit-learn regressors this is usually :math:`R^2`. This
        method follows the scikit-learn ``RegressorMixin`` convention and is
        therefore not the nescience score.
        """
        check_is_fitted(self)

        X_checked = check_array(X, dtype=None, ensure_2d=True)
        return self.model_.score(X_checked, y)

    def nescience_score(self) -> float:
        """
        Return the selected model's nescience value.

        Lower values are better.
        """
        check_is_fitted(self)
        return float(self.best_nescience_)

    def components(self) -> dict[str, float]:
        """
        Return the four nescience components of the selected model.
        """
        check_is_fitted(self)
        return dict(self.best_components_)

    def explain(self) -> dict[str, object]:
        """
        Return the nescience explanation for the selected model.
        """
        check_is_fitted(self)

        explanation = self.nescience_.explain(
            **self.best_artifacts_.to_nescience_kwargs()
        )
        explanation["candidate_name"] = self.best_candidate_name_
        explanation["model_type"] = self.best_artifacts_.model_type
        explanation["model_metadata"] = self.best_artifacts_.metadata

        return explanation

    def get_model(self):
        """
        Return the selected fitted scikit-learn regressor.
        """
        check_is_fitted(self)
        return self.model_

    def results_dataframe(self) -> pd.DataFrame:
        """
        Return a tabular summary of all evaluated candidates.

        Returns
        -------
        pandas.DataFrame
            Candidate results sorted by ascending nescience.
        """
        check_is_fitted(self)

        rows = []
        for result in self.results_:
            row = {
                "candidate": result.name,
                "model_type": result.artifacts.model_type,
                "nescience": result.nescience,
                "estimator_score": result.estimator_score,
                "n_features_in_use": len(result.artifacts.subset),
                "description_length": len(
                    result.artifacts.model_string.encode("utf-8")
                ),
            }
            row.update(result.components)
            rows.append(row)

        return pd.DataFrame(rows).sort_values("nescience").reset_index(drop=True)

    def model_string(self) -> str:
        """
        Return the canonical model string of the selected model.
        """
        check_is_fitted(self)
        return str(self.best_artifacts_.model_string)

    def _evaluate_candidate(self, name: str, estimator) -> CandidateResult:
        """
        Fit and evaluate a candidate regressor on the full representation.
        """
        model = clone(estimator)
        model.fit(self.X_, self.y_)

        artifacts = sklearn_model_artifacts(
            model,
            self.X_,
            feature_names=list(self.feature_names_in_),
            config=self.serialization_config_,
        )

        components = self.nescience_.components(**artifacts.to_nescience_kwargs())
        value = self.nescience_.aggregate_components(**components)

        return CandidateResult(
            name=name,
            estimator=model,
            nescience=float(value),
            components=dict(components),
            estimator_score=float(model.score(self.X_, self.y_)),
            artifacts=artifacts,
            metadata=dict(artifacts.metadata),
        )

    def _resolve_candidates(self) -> list[tuple[str, object]]:
        """
        Resolve user-supplied or default candidate regressors.
        """
        if self.candidates is None or self.candidates == "default":
            return self._default_candidates()

        if isinstance(self.candidates, Mapping):
            return [(str(name), estimator) for name, estimator in self.candidates.items()]

        resolved = []
        for index, item in enumerate(self.candidates):
            if isinstance(item, tuple) and len(item) == 2:
                name, estimator = item
                resolved.append((str(name), estimator))
            else:
                resolved.append((f"{type(item).__name__}_{index}", item))

        if not resolved:
            raise ValueError("At least one candidate regressor must be provided.")

        return resolved

    def _default_candidates(self) -> list[tuple[str, object]]:
        """
        Return the default candidate regressors.

        Only models supported by the stable adapter serializers are included.
        """
        candidates = [
            ("linear", LinearRegression()),
            ("ridge_alpha_0.1", Ridge(alpha=0.1)),
            ("ridge_alpha_1", Ridge(alpha=1.0)),
            ("ridge_alpha_10", Ridge(alpha=10.0)),
            ("lasso_alpha_0.001", Lasso(alpha=0.001, max_iter=10000)),
            ("lasso_alpha_0.01", Lasso(alpha=0.01, max_iter=10000)),
            ("lasso_alpha_0.1", Lasso(alpha=0.1, max_iter=10000)),
            (
                "elastic_net_alpha_0.001_l1_0.5",
                ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000),
            ),
            (
                "elastic_net_alpha_0.01_l1_0.5",
                ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000),
            ),
            (
                "elastic_net_alpha_0.1_l1_0.5",
                ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000),
            ),
            (
                "tree_depth_2",
                DecisionTreeRegressor(max_depth=2, random_state=self.random_state),
            ),
            (
                "tree_depth_3",
                DecisionTreeRegressor(max_depth=3, random_state=self.random_state),
            ),
            (
                "tree_depth_5",
                DecisionTreeRegressor(max_depth=5, random_state=self.random_state),
            ),
            (
                "tree_unrestricted",
                DecisionTreeRegressor(random_state=self.random_state),
            ),
        ]

        if self.include_ensembles:
            candidates.extend(self._default_ensemble_candidates())

        return candidates

    def _default_ensemble_candidates(self) -> list[tuple[str, object]]:
        """
        Return supported ensemble regressors for opt-in default searches.
        """
        return [
            (
                "random_forest_depth_3",
                RandomForestRegressor(
                    n_estimators=25,
                    max_depth=3,
                    random_state=self.random_state,
                ),
            ),
            (
                "extra_trees_depth_3",
                ExtraTreesRegressor(
                    n_estimators=25,
                    max_depth=3,
                    random_state=self.random_state,
                ),
            ),
            (
                "gradient_boosting_depth_2",
                GradientBoostingRegressor(
                    n_estimators=25,
                    max_depth=2,
                    random_state=self.random_state,
                ),
            ),
            (
                "hist_gradient_boosting",
                HistGradientBoostingRegressor(
                    max_iter=25,
                    max_leaf_nodes=15,
                    random_state=self.random_state,
                ),
            ),
        ]

    def _resolve_serialization_config(self) -> SerializationConfig:
        """
        Return the serialization configuration used during fitting.
        """
        if self.serialization_config is None:
            return SerializationConfig()

        if not isinstance(self.serialization_config, SerializationConfig):
            raise TypeError(
                "serialization_config must be an instance of SerializationConfig "
                "or None."
            )

        return self.serialization_config

    @staticmethod
    def _resolve_input_feature_names(X) -> list[str]:
        """
        Resolve feature names before scikit-learn validation strips DataFrames.
        """
        if hasattr(X, "columns"):
            return [str(name) for name in X.columns]

        n_features = int(getattr(X, "shape")[1])
        return [f"x{i}" for i in range(n_features)]


# Backward-compatible alias. Prefer NescienceRegressor in new code.
Regressor = NescienceRegressor
