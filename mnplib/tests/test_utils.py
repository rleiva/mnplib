from dataclasses import fields
from inspect import isfunction

import numpy as np
import pandas as pd
import pytest

from mnplib import utils
from mnplib.utils import (
    EmpiricalSummary,
    _code_length_from_counts,
    _resolve_bins,
    discretize_vector,
    empirical_distribution_array,
    empirical_distribution_vector,
)


def _object_array(values):
    """Build a one-dimensional object array without NumPy expanding lists."""
    array = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        array[index] = value
    return array


def test_utils_public_api():
    # Verify that utils exposes only the intended public API.
    assert set(utils.__all__) == {
        "EmpiricalSummary", "discretize_vector",
        "empirical_distribution_vector", "empirical_distribution_array",
    }
    public_functions = {
        name for name, value in vars(utils).items()
        if not name.startswith("_") and isfunction(value)
        and value.__module__ == utils.__name__
    }
    assert public_functions == {
        "discretize_vector", "empirical_distribution_vector", "empirical_distribution_array"
    }


# ---------------------------------------------------------------------------
# Numeric discretization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_bins,subset_size,expected",
    [(2, 1, 2), (np.int64(4), 3, 4), (4, 10, 4),
     ("auto", 1, 9), ("auto", 3, 9), ("adaptive", 1, 9),
     ("adaptive", 3, 4), ("adaptive", 100, 2)],
)
def test_resolve_bins_uses_policy_and_subset_context(n_bins, subset_size, expected):
    # Verify explicit, automatic, and adaptive bin policies resolve to the expected counts.
    resolved = _resolve_bins(n_bins, 100, subset_size=subset_size)
    assert type(resolved) is int
    assert resolved == expected


def test_resolve_bins_defaults_to_one_dimensional_adaptation():
    # Verify adaptive binning defaults to the one-dimensional automatic rule.
    assert _resolve_bins("adaptive", 100) == _resolve_bins("auto", 100)


@pytest.mark.parametrize("n_bins", [3, "auto", "adaptive"])
@pytest.mark.parametrize("n_samples,subset_size,message",
                         [(0, 1, "n_samples"), (-1, 1, "n_samples"),
                          (10, 0, "subset_size"), (10, -1, "subset_size")])
def test_resolve_bins_rejects_nonpositive_dimensions(n_bins, n_samples, subset_size, message):
    # Verify bin resolution rejects non-positive sample counts and subset sizes.
    with pytest.raises(ValueError, match=message):
        _resolve_bins(n_bins, n_samples, subset_size=subset_size)


@pytest.mark.parametrize("n_bins", [0, 1, -2, True, np.bool_(False), 3.0, 2.5,
                                  "3", "invalid", None, [2], np.array([2, 3])])
def test_bin_validation_is_shared_by_standalone_utilities(n_bins):
    # Verify all public discretization utilities reject the same invalid bin specifications.
    x = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError, match="n_bins must be an integer"):
        _resolve_bins(n_bins, 4)
    with pytest.raises(ValueError, match="n_bins must be an integer"):
        discretize_vector(x, n_bins=n_bins)
    for numeric in (True, False):
        with pytest.raises(ValueError, match="n_bins must be an integer"):
            empirical_distribution_vector(x, numeric=numeric, n_bins=n_bins)
        with pytest.raises(ValueError, match="n_bins must be an integer"):
            empirical_distribution_array(x[:, None], numeric=numeric, n_bins=n_bins)


@pytest.mark.parametrize("n_variables,expected_bins", [(1, 9), (2, 5), (3, 4), (5, 3)])
def test_adaptive_distribution_uses_supplied_variable_count(n_variables, expected_bins):
    # Verify adaptive binning reduces the bin count according to the number of variables.
    x = np.linspace(0, 1, 100)
    X = np.column_stack([x + index for index in range(n_variables)])
    adaptive = empirical_distribution_array(X)
    explicit = empirical_distribution_array(X, n_bins=expected_bins)
    np.testing.assert_array_equal(adaptive.states, explicit.states)
    np.testing.assert_array_equal(adaptive.counts, explicit.counts)
    np.testing.assert_array_equal(adaptive.probabilities, explicit.probabilities)
    assert adaptive.code_length == explicit.code_length
    assert adaptive.n_states == expected_bins
    if n_variables > 1:
        auto = empirical_distribution_array(X, n_bins="auto")
        assert adaptive.n_states < auto.n_states
        assert adaptive.code_length < auto.code_length


@pytest.mark.parametrize("n_variables", [1, 2, 3, 5])
@pytest.mark.parametrize("n_bins,expected_bins", [(4, 4), ("auto", 9)])
def test_explicit_and_auto_bin_counts_do_not_depend_on_dimension(
    n_variables, n_bins, expected_bins
):
    # Verify explicit and auto bin counts remain independent of dimensionality.
    x = np.linspace(0, 1, 100)
    summary = empirical_distribution_array(np.column_stack([x] * n_variables), n_bins=n_bins)
    assert summary.n_states == expected_bins
    np.testing.assert_array_equal(summary.states[:, 0], np.arange(expected_bins))


def test_adaptive_mixed_distribution_counts_categorical_and_constant_variables():
    # Verify adaptive joint distributions account for numeric, categorical, and constant variables.
    x = np.linspace(0, 1, 100)
    X = pd.DataFrame({"value": x, "category": np.where(x < 0.5, "low", "high"),
                      "constant": np.ones(100)})
    numeric = [True, False, True]
    adaptive = empirical_distribution_array(X, numeric=numeric)
    explicit = empirical_distribution_array(X, numeric=numeric, n_bins=4)
    np.testing.assert_array_equal(adaptive.states, explicit.states)
    np.testing.assert_array_equal(adaptive.counts, explicit.counts)
    assert len(np.unique(adaptive.states[:, 0])) == 4
    assert set(adaptive.states[:, 1]) == {0, 1}
    assert set(adaptive.states[:, 2]) == {0}


@pytest.mark.parametrize("n_bins", [3, "auto", "adaptive"])
def test_categorical_joint_distribution_is_independent_of_bin_policy(n_bins):
    # Verify binning policy does not affect purely categorical joint distributions.
    X = np.array([["a", "a", "b", "b"], [0, 1, 1, 0], ["x"] * 4], dtype=object).T
    summary = empirical_distribution_array(X, numeric=False, n_bins=n_bins)
    explicit = empirical_distribution_array(X, numeric=[False] * 3, n_bins=2)
    np.testing.assert_array_equal(summary.states, explicit.states)
    np.testing.assert_array_equal(summary.counts, explicit.counts)
    assert summary.code_length == explicit.code_length


def test_discretize_constant_numeric_vector_returns_zero_labels():
    # Verify a constant numeric vector is encoded into a single zero-valued state.
    x = np.array([5.0, 5.0, 5.0, 5.0])

    labels = discretize_vector(x, n_bins="auto")

    assert labels.dtype.kind in {"i", "u"}
    assert np.array_equal(labels, np.zeros(x.size, dtype=int))


def test_discretize_vector_uses_uniform_bins():
    # Verify numeric vectors are discretized using uniform-width bins.
    x = np.linspace(0.0, 1.0, 10)

    labels = discretize_vector(x, n_bins=2)

    assert labels.shape == x.shape
    assert set(np.unique(labels)).issubset({0, 1})
    assert np.array_equal(labels[:5], np.zeros(5, dtype=int))
    assert np.array_equal(labels[5:], np.ones(5, dtype=int))


def test_discretize_vector_auto_uses_floor_cube_root_rule():
    # Verify automatic discretization follows the configured cube-root bin-count rule.
    x = np.arange(10, dtype=float)

    labels = discretize_vector(x, n_bins="auto")

    assert labels.shape == x.shape
    assert set(np.unique(labels)) == {0, 1, 2, 3}


def test_discretize_vector_adaptive_matches_auto_for_one_vector():
    # Verify adaptive and auto discretization agree for a single variable.
    x = np.arange(10, dtype=float)

    assert np.array_equal(
        discretize_vector(x, n_bins="adaptive"),
        discretize_vector(x, n_bins="auto"),
    )


def test_discretize_vector_rejects_one_bin():
    # Verify discretization rejects a bin count smaller than two.
    x = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        discretize_vector(x, n_bins=1)


def test_discretize_vector_rejects_empty_input():
    # Verify discretization rejects an empty input vector.
    with pytest.raises(ValueError):
        discretize_vector([], n_bins="auto")


def test_discretize_vector_rejects_non_numeric_input():
    # Verify numeric discretization rejects non-numeric values.
    with pytest.raises(ValueError):
        discretize_vector(["a", "b", "c"], n_bins=2)


def test_discretize_vector_rejects_missing_numeric_values():
    # Verify numeric discretization rejects missing values.
    with pytest.raises(ValueError):
        discretize_vector([1.0, np.nan, 2.0], n_bins=2)


def test_discretize_vector_rejects_infinite_numeric_values():
    # Verify numeric discretization rejects infinite values.
    with pytest.raises(ValueError):
        discretize_vector([1.0, np.inf, 2.0], n_bins=2)


def test_discretize_vector_rejects_invalid_bin_count():
    # Verify discretization rejects an invalid bin count.
    with pytest.raises(ValueError):
        discretize_vector([1.0, 2.0, 3.0], n_bins=0)


# ---------------------------------------------------------------------------
# Empirical distributions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("make_vector", [np.asarray, list, pd.Series])
def test_vector_distribution_defaults_to_numeric_auto_bins(make_vector):
    # Verify vector distributions default to numeric encoding with automatic binning.
    x = make_vector(np.linspace(0, 1, 100))
    summary = empirical_distribution_vector(x)
    explicit = empirical_distribution_vector(x, n_bins=9)
    assert summary.n_samples == 100
    assert summary.n_states == 9
    np.testing.assert_array_equal(summary.states, explicit.states)
    np.testing.assert_array_equal(summary.counts, explicit.counts)
    assert summary.code_length == explicit.code_length


@pytest.mark.parametrize("make_array", [np.asarray, list, pd.DataFrame])
def test_array_distribution_uses_rows_as_samples_and_columns_as_variables(make_array):
    # Verify array distributions interpret rows as samples and columns as variables.
    X = make_array([[0, 0], [0, 0], [0, 1], [1, 1]])
    summary = empirical_distribution_array(X, numeric=False)
    assert summary.n_samples == 4
    assert summary.n_states == 3
    np.testing.assert_array_equal(summary.states, [[0, 0], [0, 1], [1, 1]])
    np.testing.assert_array_equal(summary.counts, [2, 1, 1])
    assert summary.code_length == pytest.approx(6.0)


@pytest.mark.parametrize("n_bins", [3, "auto", "adaptive"])
@pytest.mark.parametrize("numeric", [True, False, np.bool_(True), np.bool_(False)])
def test_single_column_array_matches_vector(n_bins, numeric):
    # Verify a one-column array produces the same distribution as the equivalent vector.
    x = np.linspace(0, 1, 50)
    vector = empirical_distribution_vector(x, numeric=numeric, n_bins=n_bins)
    array = empirical_distribution_array(x[:, None], numeric=numeric, n_bins=n_bins)
    np.testing.assert_array_equal(vector.states, array.states)
    np.testing.assert_array_equal(vector.counts, array.counts)
    np.testing.assert_array_equal(vector.probabilities, array.probabilities)
    assert vector.code_length == array.code_length
    assert vector.n_samples == array.n_samples
    assert vector.n_states == array.n_states


@pytest.mark.parametrize("numeric", [True, False])
def test_array_distribution_broadcasts_boolean_flags(numeric):
    # Verify a scalar numeric flag is broadcast consistently across array columns.
    X = np.arange(30).reshape(10, 3)
    scalar = empirical_distribution_array(X, numeric=numeric)
    per_column = empirical_distribution_array(X, numeric=np.array([numeric] * 3))
    np.testing.assert_array_equal(scalar.states, per_column.states)
    np.testing.assert_array_equal(scalar.counts, per_column.counts)
    assert scalar.code_length == per_column.code_length


def test_distributions_preserve_distinct_numeric_and_string_categories():
    # Verify categorical encoding keeps numeric values distinct from equal-looking strings.
    x = [1, "1", 1, "1"]
    vector = empirical_distribution_vector(x, numeric=False)
    array = empirical_distribution_array([[value, "a"] for value in x], numeric=False)
    assert vector.n_states == array.n_states == 2
    np.testing.assert_array_equal(vector.counts, [2, 2])
    np.testing.assert_array_equal(array.counts, [2, 2])
    assert vector.code_length == array.code_length == 4.0


def test_array_distribution_preserves_mixed_dataframe_values():
    # Verify mixed DataFrame values are encoded correctly without modifying the input.
    X = pd.DataFrame({"number": [0., 0.1, 0.9, 1.], "category": [1, "1", 1, "1"]})
    original = X.copy(deep=True)
    summary = empirical_distribution_array(X, numeric=[True, False], n_bins=2)
    assert summary.n_states == 4
    assert summary.code_length == 8.0
    pd.testing.assert_frame_equal(X, original)


def test_numeric_joint_and_marginal_lengths_share_explicit_bins():
    # Verify joint code length is compatible with marginals when shared numeric bins are explicit.
    x = np.linspace(0, 1, 100)
    X = np.column_stack([x, x ** 2, np.sin(np.pi * x)])
    joint = empirical_distribution_array(X, n_bins=4)
    for column in X.T:
        marginal = empirical_distribution_vector(column, n_bins=4)
        assert joint.code_length >= marginal.code_length


@pytest.mark.parametrize("form", ["vector", "array"])
def test_empirical_distribution_returns_summary_dataclass(form):
    # Verify distribution functions return a complete EmpiricalSummary object.
    x = np.array([0, 0, 1, 1])

    summary = (empirical_distribution_vector(x, numeric=False) if form == "vector"
               else empirical_distribution_array(x[:, None], numeric=False))

    assert isinstance(summary, EmpiricalSummary)
    assert {field.name for field in fields(summary)} == {
        "states", "counts", "probabilities", "code_length", "n_samples", "n_states"
    }
    assert summary.n_samples == 4
    assert summary.n_states == 2
    assert summary.states.shape == (2, 1)
    assert summary.counts.shape == (2,)
    assert summary.probabilities.shape == (2,)


@pytest.mark.parametrize("n_bins", [3, "auto", "adaptive"])
@pytest.mark.parametrize("columns,numeric", [
    ([[5.0] * 10], [True]),
    ([["a"] * 7 + ["b"] * 2 + ["c"]], [False]),
    ([np.arange(10), ["a", "b"] * 5], [True, False]),
    ([np.arange(10), np.arange(10) ** 2], [True, True]),
])
def test_summary_code_length_matches_count_and_probability_formulas(n_bins, columns, numeric):
    # Verify reported code length agrees with both count-based and probability-based calculations.
    X = np.asarray(columns, dtype=object).T
    summary = empirical_distribution_array(X, numeric=numeric, n_bins=n_bins)
    expected = -float(np.sum(summary.counts * np.log2(summary.probabilities)))
    assert summary.code_length == _code_length_from_counts(summary.counts)
    assert summary.code_length == pytest.approx(expected)


def test_constant_numeric_vector_has_zero_code_length():
    # Verify a constant numeric variable has one state and zero empirical code length.
    x = np.array([5.0, 5.0, 5.0, 5.0])

    summary = empirical_distribution_vector(x)

    assert summary.n_states == 1
    assert summary.code_length == pytest.approx(0.0)


def test_categorical_distribution_counts_probabilities_and_states():
    # Verify categorical distributions report the expected states, counts, probabilities, and code length.
    x = np.array(["a", "a", "b", "c"])

    summary = empirical_distribution_vector(x, numeric=False)

    assert summary.n_samples == 4
    assert summary.n_states == 3
    assert np.array_equal(summary.states, np.array([[0], [1], [2]]))
    assert np.array_equal(summary.counts, np.array([2.0, 1.0, 1.0]))
    assert np.sum(summary.probabilities) == pytest.approx(1.0)
    assert summary.code_length == pytest.approx(6.0)


def test_fair_binary_variable_uses_one_bit_per_observation():
    # Verify a balanced binary variable requires one bit per observation.
    x = np.array([0, 0, 1, 1])

    summary = empirical_distribution_vector(x, numeric=False)

    assert summary.code_length == pytest.approx(4.0)


def test_joint_code_length_is_at_least_each_marginal_code_length():
    # Verify a joint distribution is not shorter than either of its marginal distributions.
    x = np.array([0, 0, 1, 1])
    y = np.array([0, 1, 0, 1])

    kx = empirical_distribution_vector(x, numeric=False).code_length
    ky = empirical_distribution_vector(y, numeric=False).code_length
    kxy = empirical_distribution_array(np.column_stack([x, y]), numeric=False).code_length

    assert kxy + 1e-12 >= kx
    assert kxy + 1e-12 >= ky
    assert kxy == pytest.approx(8.0)


def test_joint_code_length_of_identical_variables_equals_marginal_code_length():
    # Verify duplicating an identical variable does not increase empirical code length.
    x = np.array([0, 0, 1, 1])

    kx = empirical_distribution_vector(x, numeric=False).code_length
    kxx = empirical_distribution_array(np.column_stack([x, x]), numeric=False).code_length

    assert kxx == pytest.approx(kx)


def test_mixed_numeric_and_categorical_distribution():
    # Verify joint distributions correctly combine numeric discretization and categorical encoding.
    x_num = np.array([0.0, 0.1, 0.9, 1.0])
    x_cat = np.array(["a", "a", "b", "b"])

    X = pd.DataFrame({"number": x_num, "category": x_cat})
    summary = empirical_distribution_array(X, numeric=[True, False], n_bins=2)

    assert summary.n_samples == 4
    assert summary.n_states == 2
    assert np.array_equal(summary.counts, np.array([2.0, 2.0]))
    assert summary.code_length == pytest.approx(4.0)


def test_numeric_auto_bins_are_bounded_by_number_of_samples():
    # Verify automatic binning cannot create more observed states than samples.
    x = np.arange(2, dtype=float)

    summary = empirical_distribution_vector(x)

    assert summary.n_samples == 2
    assert summary.n_states <= 2
    assert summary.code_length == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Empirical-distribution validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("x", [1, [[1, 2]], [[1], [2]], np.zeros((2, 2, 2))])
def test_vector_distribution_requires_one_dimension(x):
    # Verify vector distributions reject inputs that are not one-dimensional.
    with pytest.raises(ValueError, match="one-dimensional"):
        empirical_distribution_vector(x)


@pytest.mark.parametrize("X", [1, [1, 2], pd.Series([1, 2]), np.zeros((2, 2, 2))])
def test_array_distribution_requires_two_dimensions(X):
    # Verify array distributions reject inputs that are not two-dimensional.
    with pytest.raises(ValueError):
        empirical_distribution_array(X)


@pytest.mark.parametrize("numeric", [1, "numeric", None, [True]])
def test_vector_distribution_requires_boolean_flag(numeric):
    # Verify vector distributions require a boolean numeric flag.
    with pytest.raises(TypeError, match="numeric must be a boolean"):
        empirical_distribution_vector([0, 1], numeric=numeric)


@pytest.mark.parametrize("numeric", [[1, 0], ["True", "False"], [True, None]])
def test_array_distribution_requires_boolean_flags(numeric):
    # Verify array distributions require boolean numeric flags for all columns.
    with pytest.raises(TypeError, match="booleans"):
        empirical_distribution_array([[0, 1], [1, 0]], numeric=numeric)


@pytest.mark.parametrize("X", [np.empty((0, 2)), np.empty((2, 0)), []])
def test_array_distribution_rejects_empty_dimensions(X):
    # Verify array distributions reject arrays with empty sample or variable dimensions.
    with pytest.raises(ValueError):
        empirical_distribution_array(X)


@pytest.mark.parametrize("numeric", [[], [True], [True] * 3, [[True, False]], None])
def test_array_distribution_rejects_numeric_flag_mismatch(numeric):
    # Verify array distributions reject numeric-flag specifications with the wrong shape or length.
    with pytest.raises(ValueError, match="numeric"):
        empirical_distribution_array([[1, 2], [3, 4]], numeric=numeric)


def test_array_distribution_rejects_ragged_rows():
    # Verify array distributions reject ragged input rows.
    with pytest.raises(ValueError):
        empirical_distribution_array([[1, 2, 3], [1, 2]])


def test_vector_distribution_rejects_empty_input():
    # Verify vector distributions reject empty input.
    with pytest.raises(ValueError):
        empirical_distribution_vector([])


@pytest.mark.parametrize("values,numeric", [
    (["a", "b", "c"], True),
    ([1.0, np.nan, 2.0], True),
    ([1.0, np.inf, 2.0], True),
    (["a", None, "b"], False),
    (["a", np.nan, "b"], False),
])
def test_distributions_reject_invalid_values(values, numeric):
    # Verify vector and array distributions reject invalid numeric or categorical values.
    with pytest.raises(ValueError):
        empirical_distribution_vector(values, numeric=numeric)
    with pytest.raises(ValueError):
        empirical_distribution_array(np.asarray(values, dtype=object)[:, None], numeric=numeric)


def test_distributions_reject_unhashable_categorical_values():
    # Verify categorical distributions reject values that cannot be factorized because they are unhashable.
    x = _object_array([["a"], ["b"], ["a"]])

    with pytest.raises(TypeError, match="unhashable"):
        empirical_distribution_vector(x, numeric=False)
    with pytest.raises(TypeError, match="unhashable"):
        empirical_distribution_array(x[:, None], numeric=False)


# ---------------------------------------------------------------------------
# Counts and code length
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "counts,expected",
    [([2, 2], 4.0), ([2, 0, 2], 4.0), ([4], 0.0), ([2, 1, 1], 6.0)],
)
def test_code_length_from_counts_matches_expected_bits(counts, expected):
    # Verify empirical code length from state counts matches known expected values.
    assert _code_length_from_counts(counts) == pytest.approx(expected)


@pytest.mark.parametrize(
    "counts",
    [
        [],
        [-1, 2],
        [0, 0],
        [1, np.nan],
        [1, np.inf],
        [[1, 2], [3, 4]],
    ],
)
def test_code_length_from_counts_rejects_invalid_counts(counts):
    # Verify code-length calculation rejects malformed, non-finite, negative, or empty counts.
    with pytest.raises(ValueError):
        _code_length_from_counts(counts)
