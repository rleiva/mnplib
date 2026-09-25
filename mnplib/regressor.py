"""
Minimum-nescience regressor with selectable internal model families.

``NescienceRegressor`` constructs regressors through a fixed set of
nescience-guided model-family searchers. Users may restrict which supported
families are evaluated, but arbitrary external estimators are not accepted
by this core estimator.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted

from ._types import Aggregation, XType
from .automl import CandidateEvaluator, CandidateResult
from .automl.descriptions import describe_candidate_model
from .automl.configuration import validated_search_options
from .automl.results import candidate_results_dataframe
from .automl.wrappers import export_sklearn_model
from .automl.searchers import (
    DecisionTreePruningSearcher,
    LinearRegressionPrefixSearcher,
    LinearSVRSearcher,
    MLPRegressorSearch,
    SearchContext,
)
from .nescience import Nescience


SUPPORTED_MODELS = (
    "linear_regression",
    "decision_tree",
    "linear_svr",
    "mlp",
)


class NescienceRegressor(RegressorMixin, BaseEstimator):
    """
    Construct and select regressors by the minimum-nescience principle.

    Parameters
    ----------
    models : sequence of str or None, default=None
        Supported internal model families to evaluate. If ``None``, all
        families supported are used. If provided, the order is preserved after
        removing duplicate names. Valid names are ``"linear_regression"``,
        ``"decision_tree"``, ``"linear_svr"``, and ``"mlp"``.
    X_type : {"auto", "numeric", "categorical"}, default="numeric"
        Type policy used when fitting the nescience components on the input
        representation.
    aggregation : {"euclidean", "arithmetic", "geometric", "harmonic", \
            "maximum", "addition", "product"}, default="euclidean"
        Aggregation rule used by the underlying ``Nescience`` object.
    weights : mapping, optional
        Named deficiency, surplus, inaccuracy, and surfeit weights.
    feature_ranking_criterion : {"deficiency", "miscoding"}, default="deficiency"
        Criterion used to rank features for model-family prefix search.
    max_feature_prefixes : int or None, default=None
        Maximum number of ranked feature prefixes evaluated by searchers. If
        ``None``, every feature can appear in the prefix sequence.
    zlib_level : int, default=9
        Compression level forwarded to the surfeit component.
    zlib_overhead : int, default=6
        Estimated zlib wrapper overhead forwarded to the surfeit component.
    random_state : int, RandomState instance, or None, default=None
        Random state forwarded to stochastic searchers and estimators.
    search_options : mapping, optional
        Options keyed by model family, for example
        ``{"linear_regression": {"patience": 2}, "mlp": {"max_candidates": 5}}``.
        Unknown families and option names raise ValueError.
    verbose : int, default=0
        If nonzero, print evaluated candidate summaries during fitting.
    """

    def __init__(
        self,
        models: Sequence[str] | None = None,
        X_type: XType = "numeric",
        aggregation: Aggregation = "euclidean",
        weights: Mapping[str, float] | None = None,
        feature_ranking_criterion: str = "deficiency",
        max_feature_prefixes: int | None = None,
        zlib_level: int = 9,
        zlib_overhead: int = 6,
        random_state=None,
        search_options: Mapping[str, Mapping[str, object]] | None = None,
        verbose: int = 0,
    ):
        self.models = models
        self.X_type = X_type
        self.aggregation = aggregation
        self.weights = weights
        self.feature_ranking_criterion = feature_ranking_criterion
        self.max_feature_prefixes = max_feature_prefixes
        self.zlib_level = zlib_level
        self.zlib_overhead = zlib_overhead
        self.random_state = random_state
        self.search_options = search_options
        self.verbose = verbose

    def fit(self, X, y):
        """
        Fit selected internal searchers and choose minimum nescience.
        """
        model_names          = self._resolve_model_names()
        feature_names        = self._resolve_input_feature_names(X)
        X_checked, y_checked = check_X_y(X, y, dtype=None, ensure_2d=True)

        self.X_ = X_checked
        self.y_ = y_checked
        self.n_samples_in_, self.n_features_in_ = X_checked.shape
        self.model_names_ = tuple(model_names)
        self.feature_names_in_ = np.asarray(feature_names, dtype=object)

        self.nescience_ = Nescience(
            X_type=self.X_type,
            y_type="numeric",
            aggregation=self.aggregation,
            weights=self.weights,
            zlib_level=self.zlib_level,
            zlib_overhead=self.zlib_overhead,
        )
        self.nescience_.fit(self.X_, self.y_)

        self.evaluator_ = CandidateEvaluator(
            X=self.X_,
            y=self.y_,
            nescience=self.nescience_,
            feature_names=list(self.feature_names_in_),
        )
        self.results_     = []
        self.diagnostics_ = []
        self.searchers_   = self._resolve_searchers(model_names)
        self._fit_searchers()

        valid_results = [
            result
            for result in self.results_
            if result.is_reliable and np.isfinite(result.nescience)
        ]
        if not valid_results:
            raise ValueError(
                "No reliable candidate subset could be evaluated with the "
                "available sample size and discretization."
            )

        best_result = min(valid_results, key=lambda result: result.nescience)
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

    def get_model(self, *, as_pipeline: bool = False) -> BaseEstimator:
        """
        Return an independent copy of the selected fitted regressor.

        Parameters
        ----------
        as_pipeline : bool, default=False
            If False, return the scikit-learn estimator itself. Its inputs must
            have the selected columns in training order, with any fitted
            preprocessing already applied. If True, return a fitted
            scikit-learn Pipeline containing column selection, preprocessing,
            and the estimator. The pipeline accepts the full feature matrix
            with columns in their original training order.

        Returns
        -------
        model : sklearn.base.BaseEstimator or sklearn.pipeline.Pipeline
            A fitted copy that can be used without refitting or importing
            mnplib. Changing it does not affect this regressor.
        """
        check_is_fitted(self, "is_fitted_")
        return export_sklearn_model(self.model_, as_pipeline=as_pipeline)

    def nescience(self) -> float:
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

    def analysis(self) -> dict[str, object]:
        """
        Return a numerical nescience analysis for the selected model.

        The native estimator score is R-squared recorded during candidate
        evaluation on the training data, not a held-out score.
        """
        check_is_fitted(self)

        explanation = self.nescience_.analysis(
            **self.best_artifacts_.to_nescience_kwargs()
        )
        explanation["candidate"] = self.best_candidate_name_
        explanation["model_type"] = self.best_artifacts_.model_type
        explanation["family"] = self.best_result_.family
        explanation["hyperparameters"] = dict(self.best_result_.hyperparameters)
        explanation["task"] = "regression"
        explanation["native_estimator_score"] = float(self.best_result_.estimator_score)
        explanation["evaluation_context"] = "training"

        return explanation


    def results_dataframe(self) -> pd.DataFrame:
        """
        Return evaluated candidates sorted by ascending nescience.
        """
        check_is_fitted(self)

        return candidate_results_dataframe(self.results_, self.feature_names_in_)


    def model_description(
        self,
        candidate: str | None = None,
    ) -> dict[str, object]:
        """
        Return model-description diagnostics for an evaluated candidate.
        """
        check_is_fitted(self)
        return describe_candidate_model(
            self.results_,
            candidate,
            best_result=self.best_result_,
            surfeit=self.nescience_.surfeit_,
        )

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
                "At least one NescienceRegressor model family must be selected. "
                f"Valid options are: {', '.join(SUPPORTED_MODELS)}."
            )

        unknown = [name for name in requested if name not in SUPPORTED_MODELS]
        if unknown:
            invalid = ", ".join(repr(name) for name in unknown)
            valid = ", ".join(repr(name) for name in SUPPORTED_MODELS)
            raise ValueError(
                f"Unsupported NescienceRegressor model family: {invalid}. "
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
        self._search_options_ = validated_search_options(self.search_options, {
            "linear_regression": LinearRegressionPrefixSearcher,
            "decision_tree": DecisionTreePruningSearcher,
            "linear_svr": LinearSVRSearcher,
            "mlp": MLPRegressorSearch,
        })
        return [self._make_searcher(name) for name in names]

    def _make_searcher(self, name: str):
        """
        Instantiate the searcher for one supported model-family name.
        """
        options = dict(self._search_options_[name])
        if name != "linear_regression":
            options.setdefault("random_state", self.random_state)
        if name == "linear_regression":
            return LinearRegressionPrefixSearcher(**options)

        if name == "decision_tree":
            return DecisionTreePruningSearcher(
                DecisionTreeRegressor,
                **options,
            )

        if name == "linear_svr":
            return LinearSVRSearcher(**options)

        if name == "mlp":
            return MLPRegressorSearch(**options)

        raise RuntimeError(f"Validated unsupported model family {name!r}.")

    def _fit_searchers(self) -> None:
        """
        Execute selected internal searchers and collect their results.
        """
        context = SearchContext(
            X=self.X_,
            y=self.y_,
            feature_names=list(self.feature_names_in_),
            evaluator=self.evaluator_,
            task="regression",
            random_state=self.random_state,
            verbose=self.verbose,
            feature_ranking_criterion=self.feature_ranking_criterion,
            max_feature_prefixes=self.max_feature_prefixes,
        )

        for searcher in self.searchers_:
            report = searcher.search(context)
            self.results_.extend(report.results)
            self.diagnostics_.extend(report.diagnostics)
            if self.verbose:
                for result in report.results:
                    self._print_result(result)


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
        return [f"x{i}" for i in range(n_features)]
