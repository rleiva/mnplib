"""
Minimum-nescience classifier with selectable internal model families.

``NescienceClassifier`` constructs classifiers through a fixed set of
nescience-guided model-family searchers. Users may restrict which supported
families are evaluated, but arbitrary external estimators are not accepted
by this core estimator.
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
    LogisticRegressionPrefixSearcher,
    NaiveBayesSearcher,
    LinearSVCSearcher,
    MLPClassifierSearch,
    SearchContext,
)
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

SUPPORTED_MODELS = (
    "decision_tree",
    "logistic_regression",
    "linear_svc",
    "naive_bayes",
    "mlp",
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
        ``"decision_tree"``, ``"logistic_regression"``, ``"linear_svc"``,
        ``"naive_bayes"``, and ``"mlp"``.
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
    The canonical model string is fixed by the library because it contributes
    directly to surfeit and therefore to nescience.

    random_state : int, RandomState instance, or None, default=None
        Random state forwarded to stochastic searchers and estimators.
    alpha_tol : float, default=1e-12
        Tolerance used to collapse nearly duplicate cost-complexity pruning
        alphas.
    logistic_max_iter : int, default=1000
        Maximum iterations for logistic-regression candidates.
    mlp_search_options : mapping or None, default=None
        Additional options forwarded to the MLP architecture-growth searcher.
    verbose : int, default=0
        If nonzero, print evaluated candidate summaries during fitting.
    """

    def __init__(
        self,
        models               : Sequence[str] | None = None,
        candidates           = None,
        X_type               : XType = "numeric",
        aggregation          : Aggregation = "euclidean",
        n_bins               : BinSpec = "auto",
        threshold_fraction   : float = 0.01,
        surplus_penalty      : float = 1.0,
        random_state         = None,
        alpha_tol            : float = 1e-12,
        logistic_max_iter    : int = 1000,
        mlp_search_options   : Mapping[str, object] | None = None,
        verbose              : int = 0,
    ):
        self.models                     = models
        self.candidates                 = candidates
        self.X_type                     = X_type
        self.aggregation                = aggregation
        self.n_bins                     = n_bins
        self.threshold_fraction         = threshold_fraction
        self.surplus_penalty            = surplus_penalty
        self.random_state               = random_state
        self.alpha_tol                  = alpha_tol
        self.logistic_max_iter          = logistic_max_iter
        self.mlp_search_options         = mlp_search_options
        self.verbose                    = verbose

        # Addtional private attributes
        # self.X_
        # self.y_ 
        # self.n_samples_in_
        # self.n_features_in_
        # self.model_names_ 
        # self.feature_names_in_ 
        # self.classes_
        # self.nescience_
        # self.evaluator_    # CandidateEvaluator instance
        # self.results_      # Searcher report.results
        # self.diagnostics_  # Searcher report.diagnostics
        # self.searchers_    # List of selected searchers
        # self.best_result_
        # self.model_
        # self.best_nescience_
        # self.best_components_
        # self.best_artifacts_
        # self.best_candidate_name_
        # self.classes_
        # self.is_fitted_

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
        model_names   = self._resolve_model_names()
        feature_names = self._resolve_input_feature_names(X)
        X_checked, y_checked = check_X_y(X, y, dtype=None, ensure_2d=True)

        self.X_                = X_checked
        self.y_                = y_checked
        self.n_samples_in_     = X_checked.shape[0]
        self.n_features_in_    = X_checked.shape[1]
        self.model_names_      = tuple(model_names)
        self.feature_names_in_ = np.asarray(feature_names, dtype=object)
        self.classes_          = np.unique(y_checked)

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
        explanation["hyperparameters"] = dict(self.best_result_.hyperparameters)

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
        if self.candidates is not None:
            raise ValueError(
                "NescienceClassifier does not accept arbitrary candidate "
                "estimators; use the models parameter to select supported "
                "internal model families."
            )

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
        if name == "decision_tree":
            return DecisionTreePruningSearcher(
                DecisionTreeClassifier,
                alpha_tol    = self.alpha_tol,
                random_state = self.random_state,
            )

        if name == "logistic_regression":
            return LogisticRegressionPrefixSearcher(
                max_iter          = self.logistic_max_iter,
                random_state      = self.random_state,
            )

        if name == "linear_svc":
            return LinearSVCSearcher(
                random_state=self.random_state
            )

        if name == "naive_bayes":
            return NaiveBayesSearcher()

        if name == "mlp":
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
        description_length = int(
            len(result.artifacts.model_string.encode("utf-8"))
        )
        n_selected_features = (
            int(result.n_selected_features)
            if result.n_selected_features is not None
            else int(len(result.artifacts.subset))
        )
        row = {
            "candidate"                : result.name,
            "family"                  : result.family,
            "model_type"               : result.artifacts.model_type,
            "hyperparameters"          : dict(result.hyperparameters),
            "nescience"                : float(result.nescience),
            "deficiency"               : float(result.components["deficiency"]),
            "surplus"                  : float(result.components["surplus"]),
            "inaccuracy"               : float(result.components["inaccuracy"]),
            "surfeit"                  : float(result.components["surfeit"]),
            "native_estimator_score"   : result.estimator_score,
            "selected_features"        : list(result.artifacts.subset),
            "n_selected_features"      : n_selected_features,
            "description_length"       : description_length,
        }

        return row

    @staticmethod
    def _print_result(result: CandidateResult) -> None:
        """
        Print a compact progress line for one evaluated candidate.
        """
        print(
            f"{result.name}: nescience={result.nescience:.6f}, "
            f"estimator_score={result.estimator_score:.6f}"
        )

    @staticmethod
    def _resolve_input_feature_names(X) -> list[str]:
        """
        Return stable feature names for pandas and array-like inputs.
        """
        if hasattr(X, "columns"):
            return [str(name) for name in X.columns]

        n_features = int(getattr(X, "shape")[1])
        return [f"X{i}" for i in range(n_features)]


Classifier = NescienceClassifier
