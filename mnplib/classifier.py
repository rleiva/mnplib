"""
Minimum-nescience classifier with selectable internal model families.

``NescienceClassifier`` constructs classifiers through a fixed set of
nescience-guided model-family searchers. Users may restrict which supported
families are evaluated, but arbitrary external estimators are intentionally not
accepted by this core estimator.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted

from .automl import CandidateEvaluator, CandidateResult
from .automl.searchers import (
    DecisionTreePruningSearcher,
    LinearSVCSearcher,
    LogisticRegressionPrefixSearcher,
    MLPClassifierSearch,
    NaiveBayesSearcher,
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

# Human readable names for supported models
SUPPORTED_MODELS = (
    "Decision tree",
    "Logistic regression",
    "Linear SVC",
    "Naive Bayes",
    "MLP",
)

class NescienceClassifier(BaseEstimator, ClassifierMixin):
    """
    Construct and select classifiers by the minimum-nescience principle.

    Parameters
    ----------
    models : sequence of str or None, default=None
        Supported internal model families to evaluate. If ``None``, all
        families supported are used. If provided, the order is
        preserved after removing duplicate names. Valid names are
        ``"Decision tree"``, ``"Logistic regression"``, ``"Linear SVC"``,
        ``"Naive Bayes"``, and ``"MLP"``.
    X_type : {"auto", "numeric", "categorical"}, default="numeric"
        Type policy used when fitting the nescience components on the input
        representation.
    aggregation : {"euclidean", "arithmetic", "geometric", "harmonic", \
            "maximum", "addition", "product"}, default="euclidean"
        Aggregation rule used by the underlying ``Nescience`` object.
    n_bins : int or "auto", default="auto"
        Discretization rule for numerical variables where required by the
        nescience components.
    threshold_fraction : float, default=0.01
        Threshold parameter forwarded to the underlying nescience machinery.
    surplus_penalty : float, default=1.0
        Surplus penalty parameter forwarded to the underlying nescience
        machinery.
    serialization_config : SerializationConfig or None, default=None
        Configuration for converting fitted models into canonical descriptions.
    random_state : int, RandomState instance, or None, default=None
        Random state forwarded to stochastic searchers and estimators.
    alpha_tol : float, default=1e-12
        Tolerance used to collapse nearly duplicate cost-complexity pruning
        alphas.
    logistic_max_iter : int, default=1000
        Maximum iterations for logistic-regression candidates.
    logistic_fallback_C_values : sequence, default=(1.0, 10.0, 100.0)
        Small set of L2 fallback strengths used only when unregularized
        logistic regression is numerically problematic.
    feature_patience : int or None, default=None
        Optional patience for prefix-based feature searches.
    mlp_search_options : mapping or None, default=None
        Additional options forwarded to the MLP architecture-growth searcher.
    verbose : int, default=0
        If nonzero, print evaluated candidate summaries during fitting.
    """

    def __init__(
        self,
        models               : Sequence[str] | None = None,
        X_type               : XType = "numeric",
        aggregation          : Aggregation = "euclidean",
        n_bins               : BinSpec = "auto",
        threshold_fraction   : float = 0.01,
        surplus_penalty      : float = 1.0,
        serialization_config : SerializationConfig | None = None,
        random_state         = None,
        alpha_tol            : float = 1e-12,
        logistic_max_iter    : int = 1000,
        logistic_fallback_C_values=(1.0, 10.0, 100.0),
        feature_patience     : int | None = None,
        mlp_search_options   : Mapping[str, object] | None = None,
        verbose              : int = 0,
    ):
        self.models                     = models
        self.X_type                     = X_type
        self.aggregation                = aggregation
        self.n_bins                     = n_bins
        self.threshold_fraction         = threshold_fraction
        self.surplus_penalty            = surplus_penalty
        self.serialization_config       = serialization_config
        self.random_state               = random_state
        self.alpha_tol                  = alpha_tol
        self.logistic_max_iter          = logistic_max_iter
        self.logistic_fallback_C_values = logistic_fallback_C_values
        self.feature_patience           = feature_patience
        self.mlp_search_options         = mlp_search_options
        self.verbose                    = verbose

    def fit(self, X, y):
        """
        Fit selected internal searchers and choose minimum nescience.

        Parameters
        ----------
        X : array-like or pandas.DataFrame of shape (n_samples, n_features)
            Feature matrix. pandas DataFrames preserve column names and allow
            automatic per-column type inference.

        y : array-like of shape (n_samples,)
            Target vector.

        Returns
        -------
        self : NescienceClassifier
            Fitted estimator.        
        """
        model_names = self._resolve_model_names()
        feature_names = self._resolve_input_feature_names(X)
        X_checked, y_checked = check_X_y(X, y, dtype=None, ensure_2d=True)

        self.X_ = X_checked
        self.y_ = y_checked
        self.n_samples_in_, self.n_features_in_ = X_checked.shape
        self.model_names_ = tuple(model_names)
        self.feature_names_in_ = np.asarray(feature_names, dtype=object)
        self.classes_ = np.unique(y_checked)
        self.serialization_config_ = self._resolve_serialization_config()

        self.nescience_ = Nescience(
            X_type             = self.X_type,
            y_type             = "categorical",
            aggregation        = self.aggregation,
            n_bins             = self.n_bins,
            threshold_fraction = self.threshold_fraction,
            surplus_penalty    = self.surplus_penalty
        )
        self.nescience_.fit(self.X_, self.y_)

        self.evaluator_ = CandidateEvaluator(
            X                    = self.X_,
            y                    = self.y_,
            nescience            = self.nescience_,
            feature_names        = list(self.feature_names_in_),
            serialization_config = self.serialization_config_,
        )

        self.results_     = []
        self.diagnostics_ = []
        self.searchers_   = self._resolve_searchers(model_names)
        self._fit_searchers()

        if not self.results_:
            raise RuntimeError("No candidate classifier was successfully evaluated.")

        best_result               = min(self.results_, key=lambda result: result.nescience)
        self.best_result_         = best_result
        self.model_               = best_result.model
        self.best_nescience_      = float(best_result.nescience)
        self.best_components_     = dict(best_result.components)
        self.best_artifacts_      = best_result.artifacts
        self.best_candidate_name_ = best_result.name

        if hasattr(self.model_, "classes_"):
            self.classes_ = np.asarray(self.model_.classes_)

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """
        Predict classes with the selected minimum-nescience classifier.
        """
        check_is_fitted(self)
        X_checked = check_array(X, dtype=None, ensure_2d=True)
        return self.model_.predict(X_checked)

    def predict_proba(self, X):
        """
        Predict class probabilities with the selected classifier.
        """
        check_is_fitted(self)

        if not hasattr(self.model_, "predict_proba"):
            raise AttributeError(
                "The selected classifier does not implement predict_proba()."
            )

        X_checked = check_array(X, dtype=None, ensure_2d=True)
        return self.model_.predict_proba(X_checked)

    def score(self, X, y):
        """
        Return the native score of the selected classifier.
        """
        check_is_fitted(self)
        X_checked, y_checked = check_X_y(X, y, dtype=None, ensure_2d=True)
        return self.model_.score(X_checked, y_checked)

    def nescience_score(self) -> float:
        """
        Return the nescience value of the selected classifier.
        """
        check_is_fitted(self)
        return float(self.best_nescience_)

    def components(self) -> dict[str, float]:
        """
        Return the nescience components of the selected classifier.
        """
        check_is_fitted(self)
        return dict(self.best_components_)

    def explain(self) -> dict[str, object]:
        """
        Return a structured nescience explanation for the selected classifier.
        """
        check_is_fitted(self)

        explanation = self.nescience_.explain(
            **self.best_artifacts_.to_nescience_kwargs()
        )
        explanation["candidate_name"] = self.best_candidate_name_
        explanation["model_type"]     = self.best_artifacts_.model_type
        explanation["model_family"]   = self.best_result_.family
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
        Return all evaluated candidates sorted by ascending nescience.
        """
        check_is_fitted(self)

        rows = [self._result_row(result) for result in self.results_]
        return pd.DataFrame(rows).sort_values("nescience").reset_index(drop=True)

    def model_string(self) -> str:
        """
        Return the canonical description string of the selected classifier.
        """
        check_is_fitted(self)
        return str(self.best_artifacts_.model_string)

    def _resolve_model_names(self) -> list[str]:
        """
        Validate and normalize selected internal model-family names.
        """
        if self.models is None:
            return list(SUPPORTED_MODELS)

        if isinstance(self.models, str):
            raise ValueError(
                "models must be a sequence of supported model-family names, "
                "not a single string. Valid options are: "
                f"{', '.join(SUPPORTED_MODELS)}."
            )

        try:
            requested = list(self.models)
        except TypeError as exc:
            raise ValueError(
                "models must be None or a sequence of supported model-family names."
            ) from exc

        if not requested:
            raise ValueError(
                "At least one NescienceClassifier model family must be selected. "
                f"Valid options are: {', '.join(SUPPORTED_MODELS)}."
            )

        unknown = [name for name in requested if name not in SUPPORTED_MODELS]
        if unknown:
            invalid = ", ".join(repr(name) for name in unknown)
            valid = ", ".join(repr(name) for name in SUPPORTED_MODELS)
            raise ValueError(
                f"Unsupported NescienceClassifier model family: {invalid}. "
                f"Valid options are: {valid}."
            )

        selected: list[str] = []
        seen: set[str] = set()
        for name in requested:
            if name in seen:
                continue
            selected.append(str(name))
            seen.add(str(name))

        return selected

    def _resolve_searchers(self, model_names: Sequence[str] | None = None):
        """
        Build searchers in the selected model-family order.
        """
        names = self._resolve_model_names() if model_names is None else list(model_names)
        return [self._make_searcher(name) for name in names]

    def _make_searcher(self, name: str):
        """
        Instantiate the searcher for one supported model-family name.
        """
        if name == "Decision tree":
            return DecisionTreePruningSearcher(
                DecisionTreeClassifier,
                alpha_tol    = self.alpha_tol,
                random_state = self.random_state,
            )

        if name == "Logistic regression":
            return LogisticRegressionPrefixSearcher(
                max_iter          = self.logistic_max_iter,
                fallback_C_values = self.logistic_fallback_C_values,
                patience          = self.feature_patience,
                random_state      = self.random_state,
            )

        if name == "Linear SVC":
            return LinearSVCSearcher(
                random_state=self.random_state
            )

        if name == "Naive bayes":
            return NaiveBayesSearcher()

        if name == "MLP":
            options = (
                {}
                if self.mlp_search_options is None
                else dict(self.mlp_search_options)
            )
            options.setdefault("random_state", self.random_state)
            return MLPClassifierSearch(**options)

        raise RuntimeError(f"Validated unsupported model family {name!r}.")

    def _fit_searchers(self) -> None:
        """
        Execute selected internal searchers and collect their results.
        """
        context = SearchContext(
            X             = self.X_,
            y             = self.y_,
            feature_names = list(self.feature_names_in_),
            evaluator     = self.evaluator_,
            task          = "classification",
            random_state  = self.random_state,
            verbose       = self.verbose,
        )

        for searcher in self.searchers_:
            report = searcher.search(context)
            self.results_.extend(report.results)
            self.diagnostics_.extend(report.diagnostics)
            if self.verbose:
                for result in report.results:
                    self._print_result(result)

    def _result_row(self, result: CandidateResult) -> dict[str, object]:
        """
        Convert one candidate result into a diagnostics-table row.
        """
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
        """
        Extract common hyperparameters from candidate metadata.
        """
        keys = {
            "ccp_alpha",
            "C",
            "epsilon",
            "alpha",
            "var_smoothing",
            "penalty",
            "solver",
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
        """
        Print a compact progress line for one evaluated candidate.
        """
        print(
            f"{result.name}: nescience={result.nescience:.6f}, "
            f"estimator_score={result.estimator_score:.6f}"
        )

    def _resolve_serialization_config(self) -> SerializationConfig:
        """
        Return a valid serialization configuration for model descriptions.
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
        Return stable feature names for pandas and array-like inputs.
        """
        if hasattr(X, "columns"):
            return [str(name) for name in X.columns]

        n_features = int(getattr(X, "shape")[1])
        return [f"x{i}" for i in range(n_features)]

Classifier = NescienceClassifier
