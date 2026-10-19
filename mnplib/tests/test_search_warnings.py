"""Warning visibility and optimization diagnostics in candidate fitting."""

import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from mnplib import NescienceClassifier
from mnplib.automl.searchers._fit import fit_with_convergence_flag
from mnplib.automl.searchers.logistic import LogisticRegressionPrefixSearcher


@pytest.mark.parametrize("categories", [
    [], [ConvergenceWarning], [FutureWarning], [RuntimeWarning],
    [ConvergenceWarning, FutureWarning, UserWarning],
])
def test_candidate_fitting_only_records_convergence_warnings(categories):
    class Model:
        def fit(self, X, y):
            for category in categories:
                warnings.warn(category.__name__, category)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        converged = fit_with_convergence_flag(Model(), [[0]], [0])
    assert converged == (ConvergenceWarning not in categories)
    assert [warning.category for warning in caught] == [
        category for category in categories if category is not ConvergenceWarning
    ]
    assert all(warning.filename == __file__ for warning in caught)


def test_fitting_warnings_remain_visible_when_fitting_fails():
    class Model:
        def fit(self, X, y):
            warnings.warn("input diagnostic", UserWarning)
            raise ValueError("cannot fit")

    with pytest.warns(UserWarning, match="input diagnostic"):
        with pytest.raises(ValueError, match="cannot fit"):
            fit_with_convergence_flag(Model(), [[0]], [0])


def test_logistic_candidates_are_unregularized_and_refit_without_warnings():
    X = np.tile([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], (25, 1))
    y = X[:, 0].astype(int)
    searcher = LogisticRegressionPrefixSearcher(random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model, converged = searcher._fit_model(X, y)
        model.fit(X, 1 - y)
    assert converged
    assert model.C == np.inf


@pytest.mark.parametrize("family,options,reason", [
    ("mlp", {"max_iter": 1, "max_candidates": 1}, "mlp_convergence_warning"),
    ("linear_svc", {"max_iter": 1, "tol": 1e-12}, "linear_svc_convergence_warning"),
])
def test_search_reports_nonconverged_candidate_fits(family, options, reason):
    X = np.tile([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], (25, 1))
    y = X[:, 0].astype(int)
    model = NescienceClassifier(
        models=[family], random_state=42, search_options={family: options},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.fit(X, y)
    assert model.results_
    assert any(item["reason"] == reason for item in model.diagnostics_)
