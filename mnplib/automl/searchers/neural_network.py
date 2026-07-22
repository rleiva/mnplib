"""
Optional nescience-guided neural-network architecture search.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np

from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

from mnplib.automl.wrappers import SelectedFeaturesEstimator

from ._feature_order import feature_mask, miscoding_feature_order
from .base import ModelFamilySearcher, SearchContext, search_report


class _BaseMLPSearch(ModelFamilySearcher):
    """
    Shared greedy search for MLP classifiers and regressors.
    """

    estimator_cls = None
    family = "mlp"

    def __init__(
        self,
        *,
        initial_features: int = 2,
        initial_hidden_units: int = 3,
        unit_step: int = 1,
        layer_width: int = 3,
        max_features: int | None = None,
        max_hidden_layers: int = 2,
        max_units_per_layer: int = 6,
        max_candidates: int = 5,
        min_improvement: float = 1e-4,
        patience: int = 1,
        solver: str = "adam",
        activation: str = "relu",
        alpha: float = 0.0001,
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state=None,
    ):
        self.initial_features = int(initial_features)
        self.initial_hidden_units = int(initial_hidden_units)
        self.unit_step = int(unit_step)
        self.layer_width = int(layer_width)
        self.max_features = max_features
        self.max_hidden_layers = int(max_hidden_layers)
        self.max_units_per_layer = int(max_units_per_layer)
        self.max_candidates = int(max_candidates)
        self.min_improvement = float(min_improvement)
        self.patience = int(patience)
        self.solver = solver
        self.activation = activation
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state

    def search(self, context: SearchContext):
        order, _ = miscoding_feature_order(
            context.evaluator.nescience.miscoding_,
            context.X.shape[1],
        )
        max_features = self._resolve_max_features(context.X.shape[1])
        initial_n_features = min(
            max(1, self.initial_features),
            max_features,
            context.X.shape[1],
        )
        initial_hidden = (
            min(max(1, self.initial_hidden_units), self.max_units_per_layer),
        )

        results = []
        diagnostics = []
        seen_states = set()

        initial = self._evaluate_state(
            context,
            order=order,
            n_features_used=initial_n_features,
            hidden_layer_sizes=initial_hidden,
            move="initial",
            diagnostics=diagnostics,
        )
        if initial is None:
            return search_report(self.family, results, diagnostics)

        initial_result, current_state = initial
        results.append(initial_result)
        seen_states.add(current_state)
        current_best = initial_result
        no_improvement_rounds = 0

        while len(results) < self.max_candidates:
            move_results = []
            for move, state in self._moves(current_state, max_features):
                if state in seen_states:
                    continue
                seen_states.add(state)

                evaluated = self._evaluate_state(
                    context,
                    order=order,
                    n_features_used=state[0],
                    hidden_layer_sizes=state[1],
                    move=move,
                    diagnostics=diagnostics,
                )
                if evaluated is None:
                    continue

                result, _ = evaluated
                results.append(result)
                move_results.append((result, state))

                if len(results) >= self.max_candidates:
                    break

            if not move_results:
                diagnostics.append(
                    {
                        "family": self.family,
                        "reason": "no_more_local_moves",
                    }
                )
                break

            best_move, best_state = min(
                move_results,
                key=lambda item: item[0].nescience,
            )
            if best_move.nescience < current_best.nescience - self.min_improvement:
                current_best = best_move
                current_state = best_state
                no_improvement_rounds = 0
            else:
                no_improvement_rounds += 1
                if no_improvement_rounds >= self.patience:
                    diagnostics.append(
                        {
                            "family": self.family,
                            "reason": "no_nescience_improvement",
                            "patience": self.patience,
                        }
                    )
                    break

        return search_report(self.family, results, diagnostics)

    def _evaluate_state(
        self,
        context: SearchContext,
        *,
        order,
        n_features_used: int,
        hidden_layer_sizes,
        move: Literal["initial", "add_feature", "add_layer", "add_unit"],
        diagnostics,
    ):
        selected = tuple(order[:n_features_used])
        state = (int(n_features_used), tuple(int(size) for size in hidden_layer_sizes))
        X_selected = context.X[:, selected]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)
        model = self.estimator_cls(
            hidden_layer_sizes=state[1],
            activation=self.activation,
            solver=self.solver,
            alpha=self.alpha,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )

        try:
            converged = self._fit_with_convergence_flag(model, X_scaled, context.y)
        except Exception as exc:
            diagnostics.append(
                {
                    "family": self.family,
                    "reason": "fit_failed",
                    "move": move,
                    "n_features_used": int(n_features_used),
                    "hidden_layer_sizes": state[1],
                    "error": str(exc),
                }
            )
            return None

        public_model = SelectedFeaturesEstimator(
            model,
            selected,
            n_features_in=context.X.shape[1],
            feature_names=context.feature_names,
            transformer=scaler,
        )
        metadata = {
            "family": self.family,
            "move": move,
            "feature_order": list(order),
            "selected_features": feature_mask(selected, context.X.shape[1]),
            "selected_feature_indices": list(selected),
            "n_features_used": int(n_features_used),
            "feature_names": [
                context.feature_names[index]
                for index in selected
            ],
            "hidden_layer_sizes": state[1],
            "n_hidden_layers": int(len(state[1])),
            "n_hidden_units": int(sum(state[1])),
            "n_parameters": int(self._n_parameters(model)),
            "activation": self.activation,
            "solver": self.solver,
            "alpha": self.alpha,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "converged": bool(converged),
            "n_iter": int(getattr(model, "n_iter_", 0)),
            "scaler": "StandardScaler",
        }
        result = context.evaluator.evaluate(
            name=self._candidate_name(state, move),
            family=self.family,
            model=model,
            feature_indices=selected,
            X_adapter=X_scaled,
            result_model=public_model,
            model_string_prefix=self._scaler_model_string(
                scaler,
                selected,
                context.feature_names,
            ),
            metadata=metadata,
        )
        return result, state

    def _moves(self, state, max_features):
        n_features_used, hidden = state
        hidden = tuple(hidden)

        if n_features_used < max_features:
            yield "add_feature", (int(n_features_used) + 1, hidden)

        if len(hidden) < self.max_hidden_layers:
            yield "add_layer", (
                int(n_features_used),
                hidden + (min(self.layer_width, self.max_units_per_layer),),
            )

        if max(hidden, default=0) < self.max_units_per_layer:
            yield "add_unit", (
                int(n_features_used),
                tuple(
                    min(int(size) + self.unit_step, self.max_units_per_layer)
                    for size in hidden
                ),
            )

    def _resolve_max_features(self, n_features: int) -> int:
        if self.max_features is None:
            return int(n_features)
        return max(1, min(int(self.max_features), int(n_features)))

    def _candidate_name(self, state, move: str) -> str:
        n_features_used, hidden = state
        hidden_text = "x".join(str(size) for size in hidden)
        return (
            f"{self.family}_{move}_features_{n_features_used}_"
            f"hidden_{hidden_text}"
        )

    @staticmethod
    def _fit_with_convergence_flag(model, X, y) -> bool:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X, y)

        return not any(
            issubclass(warning.category, ConvergenceWarning)
            for warning in caught
        )

    @staticmethod
    def _n_parameters(model) -> int:
        total = 0
        for matrix in model.coefs_:
            total += int(np.asarray(matrix).size)
        for vector in model.intercepts_:
            total += int(np.asarray(vector).size)
        return total

    @staticmethod
    def _scaler_model_string(scaler, selected, feature_names) -> str:
        lines = [
            "PREPROCESSOR StandardScaler",
            "PARAMETERS",
            "    with_mean = True",
            "    with_std = True",
        ]
        for local_index, feature_index in enumerate(selected):
            name = feature_names[feature_index]
            mean = float(scaler.mean_[local_index])
            scale = float(scaler.scale_[local_index])
            lines.append(
                f"    {name}: mean = {mean:.6g}, scale = {scale:.6g}"
            )
        return "\n".join(lines) + "\n"


class MLPClassifierSearch(_BaseMLPSearch):
    """
    Extended greedy architecture search for MLPClassifier.
    """

    estimator_cls = MLPClassifier
    family = "mlp_classifier"


class MLPRegressorSearch(_BaseMLPSearch):
    """
    Extended greedy architecture search for MLPRegressor.
    """

    estimator_cls = MLPRegressor
    family = "mlp_regressor"
