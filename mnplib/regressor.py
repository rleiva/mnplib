"""
Minimum-nescience regressor with model-family-specific AutoML search.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted

from .automl import CandidateEvaluator, CandidateResult
from .automl.searchers import (
    DecisionTreePruningSearcher,
    LinearRegressionPrefixSearcher,
    LinearSVRSearcher,
    MLPRegressorSearch,
    SearchContext,
)
from .models import SerializationConfig
from .nescience import Nescience


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
CandidateProfile = Literal["default", "compact", "standard", "extended"]


class NescienceRegressor(BaseEstimator, RegressorMixin):
    """
    Select a regressor by minimizing nescience within model families.
    """

    def __init__(
        self,
        candidates: CandidateProfile | Mapping | Sequence | None = "standard",
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
        min_samples_leaf: int = 1,
        alpha_tol: float = 1e-12,
        n_jobs: int | None = None,
        include_neural_networks: bool = False,
        feature_patience: int | None = None,
        mlp_search_options: Mapping[str, object] | None = None,
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
        self.min_samples_leaf = min_samples_leaf
        self.alpha_tol = alpha_tol
        self.n_jobs = n_jobs
        self.include_neural_networks = include_neural_networks
        self.feature_patience = feature_patience
        self.mlp_search_options = mlp_search_options
        self.verbose = verbose

    def fit(self, X, y):
        """
        Search supported regressor families and select minimum nescience.
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

        self.evaluator_ = CandidateEvaluator(
            X=self.X_,
            y=self.y_,
            nescience=self.nescience_,
            feature_names=list(self.feature_names_in_),
            serialization_config=self.serialization_config_,
        )
        self.results_ = []
        self.diagnostics_ = []

        if self._uses_searchers():
            self.searchers_ = self._resolve_searchers()
            self._fit_searchers()
        else:
            self.candidates_ = self._resolve_candidates()
            self._fit_explicit_candidates()

        if not self.results_:
            raise RuntimeError("No candidate regressor was successfully evaluated.")

        best_result = min(self.results_, key=lambda result: result.nescience)
        self.best_result_ = best_result
        self.model_ = best_result.model
        self.best_nescience_ = float(best_result.nescience)
        self.best_components_ = dict(best_result.components)
        self.best_artifacts_ = best_result.artifacts
        self.best_candidate_name_ = best_result.name

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """
        Predict with the selected regressor.
        """
        check_is_fitted(self)
        X_checked = check_array(X, dtype=None, ensure_2d=True)
        return self.model_.predict(X_checked)

    def score(self, X, y):
        """
        Return the native score of the selected regressor.
        """
        check_is_fitted(self)
        X_checked = check_array(X, dtype=None, ensure_2d=True)
        return self.model_.score(X_checked, y)

    def nescience_score(self) -> float:
        """
        Return the selected model's nescience value.
        """
        check_is_fitted(self)
        return float(self.best_nescience_)

    def components(self) -> dict[str, float]:
        """
        Return the selected model's nescience components.
        """
        check_is_fitted(self)
        return dict(self.best_components_)

    def explain(self) -> dict[str, object]:
        """
        Return a nescience explanation for the selected model.
        """
        check_is_fitted(self)

        explanation = self.nescience_.explain(
            **self.best_artifacts_.to_nescience_kwargs()
        )
        explanation["candidate_name"] = self.best_candidate_name_
        explanation["model_type"] = self.best_artifacts_.model_type
        explanation["model_family"] = self.best_result_.family
        explanation["model_metadata"] = self.best_artifacts_.metadata

        return explanation

    def get_model(self):
        """
        Return the selected fitted estimator.
        """
        check_is_fitted(self)
        return self.model_

    def results_dataframe(self) -> pd.DataFrame:
        """
        Return evaluated candidates sorted by ascending nescience.
        """
        check_is_fitted(self)

        rows = [self._result_row(result) for result in self.results_]
        return pd.DataFrame(rows).sort_values("nescience").reset_index(drop=True)

    def model_string(self) -> str:
        """
        Return the selected model's canonical description string.
        """
        check_is_fitted(self)
        return str(self.best_artifacts_.model_string)

    def _fit_searchers(self) -> None:
        context = SearchContext(
            X=self.X_,
            y=self.y_,
            feature_names=list(self.feature_names_in_),
            evaluator=self.evaluator_,
            task="regression",
            random_state=self.random_state,
            verbose=self.verbose,
        )

        for searcher in self.searchers_:
            report = searcher.search(context)
            self.results_.extend(report.results)
            self.diagnostics_.extend(report.diagnostics)
            if self.verbose:
                for result in report.results:
                    self._print_result(result)

    def _fit_explicit_candidates(self) -> None:
        for name, estimator in self.candidates_:
            model = clone(estimator)
            model.fit(self.X_, self.y_)
            result = self.evaluator_.evaluate(
                name=name,
                family=type(model).__name__,
                model=model,
                metadata={"candidate_source": "explicit"},
            )
            self.results_.append(result)
            if self.verbose:
                self._print_result(result)

    def _uses_searchers(self) -> bool:
        if self.candidates is None:
            return True
        return isinstance(self.candidates, str) and self.candidates in {
            "default",
            "compact",
            "standard",
            "extended",
        }

    def _resolve_searchers(self):
        profile = self._candidate_profile()
        searchers = [
            LinearRegressionPrefixSearcher(patience=self.feature_patience),
            DecisionTreePruningSearcher(
                DecisionTreeRegressor,
                min_samples_leaf=self.min_samples_leaf,
                alpha_tol=self.alpha_tol,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            ),
        ]

        if profile in {"standard", "extended"}:
            searchers.append(LinearSVRSearcher(random_state=self.random_state))

        if profile == "extended" or self.include_neural_networks:
            options = {} if self.mlp_search_options is None else dict(self.mlp_search_options)
            options.setdefault("random_state", self.random_state)
            searchers.append(MLPRegressorSearch(**options))

        return searchers

    def _candidate_profile(self) -> str:
        if self.candidates in (None, "default"):
            return "standard"
        if self.candidates in {"compact", "standard", "extended"}:
            return str(self.candidates)
        raise ValueError(
            "candidates must be 'compact', 'standard', 'extended', 'default', "
            "None, or an explicit mapping/sequence of estimators."
        )

    def _resolve_candidates(self) -> list[tuple[str, object]]:
        """
        Resolve explicit user-supplied candidates for backward compatibility.
        """
        if self.candidates is None or self.candidates == "default":
            return self._default_candidates()

        if isinstance(self.candidates, str):
            if self.candidates in {"compact", "standard", "extended"}:
                return self._default_candidates()
            raise ValueError(f"Unknown candidate profile {self.candidates!r}.")

        if isinstance(self.candidates, Mapping):
            return [
                (str(name), estimator)
                for name, estimator in self.candidates.items()
            ]

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
        Return a compact direct-evaluation preview of default families.
        """
        candidates = [
            ("linear_regression", LinearRegression()),
            (
                "decision_tree_pruned_family",
                DecisionTreeRegressor(random_state=self.random_state),
            ),
        ]
        if self.include_ensembles:
            candidates.extend(self._default_ensemble_candidates())
        return candidates

    def _default_ensemble_candidates(self) -> list[tuple[str, object]]:
        """
        Return opt-in ensemble candidates for legacy explicit evaluation.
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

    def _result_row(self, result: CandidateResult) -> dict[str, object]:
        metadata = dict(result.metadata)
        description_length = int(
            metadata.get(
                "description_length",
                len(result.artifacts.model_string.encode("utf-8")),
            )
        )
        row = {
            "candidate": result.name,
            "family": result.family,
            "model_family": result.family,
            "model_type": result.artifacts.model_type,
            "searched_hyperparameters": self._searched_hyperparameters(metadata),
            "nescience": float(result.nescience),
            "native_estimator_score": result.estimator_score,
            "estimator_score": result.estimator_score,
            "n_features_in_use": int(len(result.artifacts.subset)),
            "n_features_used": int(
                metadata.get("n_features_used", len(result.artifacts.subset))
            ),
            "description_length": description_length,
            "model_description_length": description_length,
            "support_level": metadata.get("support_level"),
        }
        row.update(result.components)

        for key, value in metadata.items():
            if key not in row:
                row[key] = value

        return row

    @staticmethod
    def _searched_hyperparameters(metadata: Mapping[str, object]) -> dict[str, object]:
        keys = {
            "ccp_alpha",
            "min_samples_leaf",
            "C",
            "epsilon",
            "alpha",
            "hidden_layer_sizes",
            "activation",
            "max_iter",
            "tol",
        }
        return {
            key: metadata[key]
            for key in sorted(keys)
            if key in metadata
        }

    @staticmethod
    def _print_result(result: CandidateResult) -> None:
        print(
            f"{result.name}: nescience={result.nescience:.6f}, "
            f"estimator_score={result.estimator_score:.6f}"
        )

    def _resolve_serialization_config(self) -> SerializationConfig:
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
        if hasattr(X, "columns"):
            return [str(name) for name in X.columns]

        n_features = int(getattr(X, "shape")[1])
        return [f"x{i}" for i in range(n_features)]


Regressor = NescienceRegressor
