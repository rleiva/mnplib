"""
Minimum-nescience regressor with nescience-guided model construction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin, clone
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


class NescienceRegressor(BaseEstimator, RegressorMixin):
    """
    Construct and select regressors by the minimum-nescience principle.
    """

    def __init__(
        self,
        candidates: Mapping | Sequence | None = None,
        X_type: XType = "numeric",
        aggregation: Aggregation = "euclidean",
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
        feature_patience: int | None = None,
        mlp_search_options: Mapping[str, object] | None = None,
        verbose: int = 0,
    ):
        self.candidates = candidates
        self.X_type = X_type
        self.aggregation = aggregation
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
        self.feature_patience = feature_patience
        self.mlp_search_options = mlp_search_options
        self.verbose = verbose

    def fit(self, X, y):
        """
        Fit internal searchers and optional explicit candidates.
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
        self.candidates_ = self._resolve_candidates()

        self.searchers_ = self._resolve_searchers()
        self._fit_searchers()

        if self.candidates_:
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
        explanation["candidate_source"] = self.best_result_.metadata.get(
            "candidate_source"
        )
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

    def _resolve_searchers(self):
        """
        Return the internal nescience-guided regressor searchers.
        """
        options = {} if self.mlp_search_options is None else dict(self.mlp_search_options)
        options.setdefault("random_state", self.random_state)

        return [
            LinearRegressionPrefixSearcher(patience=self.feature_patience),
            DecisionTreePruningSearcher(
                DecisionTreeRegressor,
                min_samples_leaf=self.min_samples_leaf,
                alpha_tol=self.alpha_tol,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            ),
            LinearSVRSearcher(random_state=self.random_state),
            MLPRegressorSearch(**options),
        ]

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

    def _resolve_candidates(self) -> list[tuple[str, object]]:
        """
        Resolve explicit user-supplied candidates for comparison.
        """
        if self.candidates is None:
            return []

        if isinstance(self.candidates, str):
            raise ValueError(
                "candidates must be an explicit mapping or sequence of "
                "estimators; profile strings are not supported."
            )

        if isinstance(self.candidates, Mapping):
            resolved = [
                (str(name), estimator)
                for name, estimator in self.candidates.items()
            ]
        else:
            resolved = []
            for index, item in enumerate(self.candidates):
                if isinstance(item, tuple) and len(item) == 2:
                    name, estimator = item
                    resolved.append((str(name), estimator))
                else:
                    resolved.append((f"{type(item).__name__}_{index}", item))

        if not resolved:
            raise ValueError("At least one explicit candidate regressor is required.")

        return resolved

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
            "candidate_source": metadata.get("candidate_source"),
            "family": result.family,
            "model_family": result.family,
            "model_type": result.artifacts.model_type,
            "searched_hyperparameters": self._searched_hyperparameters(metadata),
            "nescience": float(result.nescience),
            "native_estimator_score": result.estimator_score,
            "estimator_score": result.estimator_score,
            "selected_features": list(result.artifacts.subset),
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
