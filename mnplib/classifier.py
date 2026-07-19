"""
Minimum-nescience classifier with model-family-specific AutoML search.

This module provides :class:`NescienceClassifier`, a scikit-learn-compatible
estimator that selects a fitted classifier by minimizing nescience. The class
coordinates model-family searchers, evaluates each fitted candidate through the
explicit-artifact workflow, and exposes the selected estimator through the usual
``fit``/``predict``/``score`` interface.

The implementation intentionally keeps model-search logic outside this class.
Searchers generate meaningful candidates for each model family, while the shared
candidate evaluator computes the nescience components for every fitted model.

@author:    Rafael Garcia Leiva
@mail:      rgarcialeiva@gmail.com
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
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


# Public type aliases used by the estimator constructor. Keeping them explicit
# improves readability without requiring users to inspect lower-level classes.
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


class NescienceClassifier(BaseEstimator, ClassifierMixin):
    """
    Select a classifier by minimizing nescience within model families.

    The estimator searches one or more classifier families, evaluates each
    fitted candidate with the same nescience criterion, and keeps the candidate
    with the smallest aggregated nescience value.

    Parameters
    ----------
    candidates : {"default", "compact", "standard", "extended"}, mapping, \
            sequence, or None, default="standard"
        Candidate profile or explicit candidate collection.

        When a profile is used, the classifier delegates model generation to
        family-specific searchers:

        * ``"compact"`` evaluates low-surfeit classifier families.
        * ``"standard"`` adds moderate-surfeit families such as linear SVM and
          compatible Naive Bayes variants.
        * ``"extended"`` may include high-capacity families such as MLPs.
        * ``"default"`` and ``None`` map to ``"standard"``.

        A mapping or sequence of estimators is evaluated directly. This mode is
        useful for users who want full control over the candidate estimators.
    X_type : {"auto", "numeric", "categorical"}, default="numeric"
        Representation type passed to the nescience metric.
    aggregation : str, default="euclidean"
        Aggregation rule used to combine nescience components.
    weights : mapping, sequence, or None, default=None
        Optional component weights used by the nescience aggregation rule.
    n_bins : int or "auto", default="auto"
        Number of bins used when numeric quantities need discretization.
    threshold_fraction : float, default=0.01
        Threshold parameter forwarded to the nescience implementation.
    surplus_penalty : float, default=1.0
        Surplus penalty forwarded to the nescience implementation.
    zlib_level : int, default=9
        Compression level used by the surfeit approximation.
    zlib_overhead : int, default=6
        Estimated zlib overhead removed from compressed descriptions.
    serialization_config : SerializationConfig or None, default=None
        Configuration for estimator serialization. If ``None``, a default
        ``SerializationConfig`` is created during fitting.
    random_state : int, RandomState, or None, default=None
        Random state forwarded to candidate searchers and compatible sklearn
        estimators.
    min_samples_leaf : int, default=1
        Minimum number of samples per leaf for decision-tree pruning search.
    alpha_tol : float, default=1e-12
        Tolerance used to merge nearly identical cost-complexity pruning alphas.
    n_jobs : int or None, default=None
        Optional parallelism parameter for searchers that support it.
    include_neural_networks : bool, default=False
        If ``True``, include the MLP architecture searcher even when the profile
        is not ``"extended"``.
    logistic_max_iter : int, default=1000
        Maximum number of iterations for logistic-regression candidates.
    logistic_fallback_C_values : sequence of float, default=(1.0, 10.0, 100.0)
        Small L2-stabilized fallback path used when unregularized logistic
        regression fails or does not converge.
    feature_patience : int or None, default=None
        Optional early-stopping patience for prefix-based feature searches.
    mlp_search_options : mapping or None, default=None
        Extra keyword arguments forwarded to ``MLPClassifierSearch``.
    verbose : int, default=0
        Verbosity level. When nonzero, evaluated candidates are printed as they
        are produced.

    Attributes
    ----------
    model_ : estimator
        Selected fitted classifier.
    best_result_ : CandidateResult
        Structured result for the selected candidate.
    best_nescience_ : float
        Aggregated nescience of the selected candidate.
    best_components_ : dict
        Nescience components of the selected candidate.
    best_artifacts_ : ModelArtifacts-like object
        Explicit artifacts used to evaluate the selected candidate.
    results_ : list of CandidateResult
        Results for all successfully evaluated candidates.
    diagnostics_ : list
        Diagnostics emitted by searchers, such as skipped incompatible models.
    """

    def __init__(
        self,
        candidates: CandidateProfile | Mapping | Sequence | None = "standard",
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
        logistic_max_iter: int = 1000,
        logistic_fallback_C_values=(1.0, 10.0, 100.0),
        feature_patience: int | None = None,
        mlp_search_options: Mapping[str, object] | None = None,
        verbose: int = 0,
    ):
        # Store constructor arguments verbatim to preserve sklearn estimator
        # compatibility with get_params, set_params, cloning, and pipelines.
        self.candidates = candidates
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
        self.logistic_max_iter = logistic_max_iter
        self.logistic_fallback_C_values = logistic_fallback_C_values
        self.feature_patience = feature_patience
        self.mlp_search_options = mlp_search_options
        self.verbose = verbose

    def fit(self, X, y):
        """
        Search supported classifier families and select the minimum-nescience model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input representation.
        y : array-like of shape (n_samples,)
            Categorical target values.

        Returns
        -------
        self : NescienceClassifier
            Fitted estimator.
        """
        # Preserve user-facing feature names before sklearn validation converts
        # pandas inputs into NumPy arrays.
        feature_names = self._resolve_input_feature_names(X)
        X_checked, y_checked = check_X_y(X, y, dtype=None, ensure_2d=True)

        # Store validated training data and sklearn-compatible fitted metadata.
        self.X_ = X_checked
        self.y_ = y_checked
        self.n_samples_in_, self.n_features_in_ = X_checked.shape
        self.feature_names_in_ = np.asarray(feature_names, dtype=object)
        self.classes_ = np.unique(y_checked)
        self.serialization_config_ = self._resolve_serialization_config()

        # Fit the nescience object once on the training representation. Candidate
        # evaluation reuses this fitted metric to ensure all models are compared
        # under the same representation, target, and aggregation settings.
        self.nescience_ = Nescience(
            X_type             = self.X_type,
            y_type             = "categorical",
            aggregation        = self.aggregation,
            weights            = self.weights,
            n_bins             = self.n_bins,
            threshold_fraction = self.threshold_fraction,
            surplus_penalty    = self.surplus_penalty,
            zlib_level         = self.zlib_level,
            zlib_overhead      = self.zlib_overhead,
        )
        self.nescience_.fit(self.X_, self.y_)

        # CandidateEvaluator is the only component that computes nescience for a
        # fitted model. Searchers generate fitted candidates; the evaluator turns
        # them into explicit artifacts and comparable CandidateResult instances.
        self.evaluator_ = CandidateEvaluator(
            X                    = self.X_,
            y                    = self.y_,
            nescience            = self.nescience_,
            feature_names        = list(self.feature_names_in_),
            serialization_config = self.serialization_config_,
        )
        self.results_     = []
        self.diagnostics_ = []

        # Profile-based operation uses model-family searchers. Explicit mappings
        # or estimator sequences are evaluated directly for user-controlled runs.
        if self._uses_searchers():
            self.searchers_ = self._resolve_searchers()
            self._fit_searchers()
        else:
            self.candidates_ = self._resolve_candidates()
            self._fit_explicit_candidates()

        if not self.results_:
            raise RuntimeError("No candidate classifier was successfully evaluated.")

        # Selection is deliberately simple: every evaluated candidate has already
        # been converted to the same nescience scale by the shared evaluator.
        best_result               = min(self.results_, key=lambda result: result.nescience)
        self.best_result_         = best_result
        self.model_               = best_result.model
        self.best_nescience_      = float(best_result.nescience)
        self.best_components_     = dict(best_result.components)
        self.best_artifacts_      = best_result.artifacts
        self.best_candidate_name_ = best_result.name

        # Keep sklearn's classification metadata aligned with the selected model
        # when the estimator exposes its own class ordering.
        if hasattr(self.model_, "classes_"):
            self.classes_ = np.asarray(self.model_.classes_)

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """
        Predict classes with the selected classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input representation.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self)
        X_checked = check_array(X, dtype=None, ensure_2d=True)
        return self.model_.predict(X_checked)

    def predict_proba(self, X):
        """
        Predict class probabilities when the selected classifier supports them.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input representation.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Class-probability estimates.

        Raises
        ------
        AttributeError
            If the selected classifier does not implement ``predict_proba``.
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

        For sklearn classifiers this is usually mean accuracy, although custom
        estimators may define their own scoring semantics.
        """
        check_is_fitted(self)
        X_checked, y_checked = check_X_y(X, y, dtype=None, ensure_2d=True)
        return self.model_.score(X_checked, y_checked)

    def nescience_score(self) -> float:
        """
        Return the aggregated nescience of the selected candidate.
        """
        check_is_fitted(self)
        return float(self.best_nescience_)

    def components(self) -> dict[str, float]:
        """
        Return the nescience components of the selected candidate.
        """
        check_is_fitted(self)
        return dict(self.best_components_)

    def explain(self) -> dict[str, object]:
        """
        Return a structured nescience explanation for the selected model.

        The explanation is recomputed from the explicit artifacts used during
        candidate evaluation and enriched with model-family metadata.
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
        Return the selected fitted sklearn-compatible estimator.
        """
        check_is_fitted(self)
        return self.model_

    def results_dataframe(self) -> pd.DataFrame:
        """
        Return all evaluated candidates sorted by ascending nescience.

        The returned DataFrame is intended for inspection and diagnostics. It
        includes common columns for all model families plus any metadata emitted
        by individual searchers.
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
        """
        Run all resolved model-family searchers.

        Searchers receive a shared context containing data, feature names, the
        evaluator, task type, and reproducibility settings. Each searcher returns
        a report with successful evaluations and optional diagnostics.
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

    def _fit_explicit_candidates(self) -> None:
        """
        Fit and evaluate user-supplied estimator candidates directly.

        This path bypasses family-specific searchers but still uses the same
        evaluator and explicit-artifact workflow as profile-based search.
        """
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
        """
        Return whether ``candidates`` denotes a built-in search profile.
        """
        if self.candidates is None:
            return True
        return isinstance(self.candidates, str) and self.candidates in {
            "default",
            "compact",
            "standard",
            "extended",
        }

    def _resolve_searchers(self):
        """
        Build the searcher list for the selected candidate profile.

        Compact search includes low-surfeit families. Standard search adds
        moderate-surfeit families. Extended search adds high-capacity families
        that are useful for comparison but should not be selected by default.
        """
        profile = self._candidate_profile()
        searchers = [
            LogisticRegressionPrefixSearcher(
                max_iter          = self.logistic_max_iter,
                fallback_C_values = self.logistic_fallback_C_values,
                patience          = self.feature_patience,
                random_state      = self.random_state,
            ),
            DecisionTreePruningSearcher(
                DecisionTreeClassifier,
                min_samples_leaf  = self.min_samples_leaf,
                alpha_tol         = self.alpha_tol,
                n_jobs            = self.n_jobs,
                random_state      = self.random_state,
            ),
        ]

        if profile in {"standard", "extended"}:
            searchers.extend(
                [
                    LinearSVCSearcher(random_state=self.random_state),
                    NaiveBayesSearcher(),
                ]
            )

        if profile == "extended" or self.include_neural_networks:
            options = (
                {} if self.mlp_search_options is None else dict(self.mlp_search_options)
            )
            options.setdefault("random_state", self.random_state)
            searchers.append(MLPClassifierSearch(**options))

        return searchers

    def _candidate_profile(self) -> str:
        """
        Normalize the candidate-profile name.

        ``None`` and ``"default"`` intentionally map to ``"standard"`` so that
        the default profile includes compact and moderate-surfeit families while
        excluding extended high-surfeit models.
        """
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
        Resolve explicit user-supplied candidates for direct evaluation.

        A mapping preserves user-provided names. A sequence may contain either
        ``(name, estimator)`` pairs or bare estimators, in which case stable
        generated names are used.
        """
        if self.candidates is None or self.candidates == "default":
            return self._default_candidates()

        if isinstance(self.candidates, str):
            if self.candidates in {"compact", "standard", "extended"}:
                return self._default_candidates()
            raise ValueError(f"Unknown candidate profile {self.candidates!r}.")

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
            raise ValueError("At least one candidate classifier must be provided.")

        return resolved

    def _default_candidates(self) -> list[tuple[str, object]]:
        """
        Return fallback direct-evaluation candidates.

        This method is only used by the explicit-candidate path. Profile-based
        operation should normally use searchers rather than these fixed models.
        """
        return [
            (
                "logistic_regression",
                LogisticRegression(
                    penalty=None,
                    solver="lbfgs",
                    max_iter=self.logistic_max_iter,
                    random_state=self.random_state,
                ),
            ),
            (
                "decision_tree_pruned_family",
                DecisionTreeClassifier(random_state=self.random_state),
            ),
        ]

    def _result_row(self, result: CandidateResult) -> dict[str, object]:
        """
        Convert a candidate result into one row for ``results_dataframe``.

        The method creates a stable set of common diagnostic columns and then
        appends any additional searcher-specific metadata that is not already
        represented.
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
            "n_features_in_use": int(len(result.artifacts.subset)),
            "n_features_used": int(
                metadata.get("n_features_used", len(result.artifacts.subset))
            ),
            "description_length": description_length,
            "model_description_length": description_length,
            "support_level": metadata.get("support_level"),
        }
        row.update(result.components)

        # Preserve family-specific diagnostics such as ccp_alpha, feature order,
        # convergence flags, and compatibility-skip metadata.
        for key, value in metadata.items():
            if key not in row:
                row[key] = value

        return row

    @staticmethod
    def _searched_hyperparameters(metadata: Mapping[str, object]) -> dict[str, object]:
        """
        Extract common hyperparameter diagnostics from searcher metadata.
        """
        keys = {
            "ccp_alpha",
            "min_samples_leaf",
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
        return {key: metadata[key] for key in sorted(keys) if key in metadata}

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
        Validate or create the serialization configuration.
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
        Return feature names for pandas inputs or generated names otherwise.
        """
        if hasattr(X, "columns"):
            return [str(name) for name in X.columns]

        n_features = int(getattr(X, "shape")[1])
        return [f"x{i}" for i in range(n_features)]


# Backward-compatible public alias.
Classifier = NescienceClassifier