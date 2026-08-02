import numpy as np
import pytest

from utils import (
    EmpiricalSummary,
    code_length_from_counts,
    discretize_vector,
    empirical_distribution,
    entropy_from_counts,
)


def _object_array(values):
    """Build a one-dimensional object array without NumPy expanding lists."""
    array = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        array[index] = value
    return array


# ---------------------------------------------------------------------------
# Numeric discretization
# ---------------------------------------------------------------------------


def test_discretize_constant_numeric_vector_returns_zero_labels():
    x = np.array([5.0, 5.0, 5.0, 5.0])

    labels = discretize_vector(x, n_bins="auto")

    assert labels.dtype.kind in {"i", "u"}
    assert np.array_equal(labels, np.zeros(x.size, dtype=int))


def test_discretize_vector_uses_uniform_bins():
    x = np.linspace(0.0, 1.0, 10)

    labels = discretize_vector(x, n_bins=2)

    assert labels.shape == x.shape
    assert set(np.unique(labels)).issubset({0, 1})
    assert np.array_equal(labels[:5], np.zeros(5, dtype=int))
    assert np.array_equal(labels[5:], np.ones(5, dtype=int))


def test_discretize_vector_auto_uses_rice_rule():
    x = np.arange(8, dtype=float)

    labels = discretize_vector(x, n_bins="auto")

    # Rice's rule gives ceil(2 * 8**(1/3)) = 4 bins.
    assert labels.shape == x.shape
    assert set(np.unique(labels)) == {0, 1, 2, 3}


def test_discretize_vector_one_bin_returns_zero_labels():
    x = np.array([1.0, 2.0, 3.0])

    labels = discretize_vector(x, n_bins=1)

    assert np.array_equal(labels, np.zeros(x.size, dtype=int))


def test_discretize_vector_rejects_empty_input():
    with pytest.raises(ValueError):
        discretize_vector([], n_bins="auto")


def test_discretize_vector_rejects_non_numeric_input():
    with pytest.raises(ValueError):
        discretize_vector(["a", "b", "c"], n_bins=2)


def test_discretize_vector_rejects_missing_numeric_values():
    with pytest.raises(ValueError):
        discretize_vector([1.0, np.nan, 2.0], n_bins=2)


def test_discretize_vector_rejects_infinite_numeric_values():
    with pytest.raises(ValueError):
        discretize_vector([1.0, np.inf, 2.0], n_bins=2)


def test_discretize_vector_rejects_invalid_bin_count():
    with pytest.raises(ValueError):
        discretize_vector([1.0, 2.0, 3.0], n_bins=0)


# ---------------------------------------------------------------------------
# Empirical distributions
# ---------------------------------------------------------------------------


def test_empirical_distribution_returns_summary_dataclass():
    x = np.array([0, 0, 1, 1])

    summary = empirical_distribution([x], [False])

    assert isinstance(summary, EmpiricalSummary)
    assert summary.n_samples == 4
    assert summary.n_states == 2
    assert summary.states.shape == (2, 1)
    assert summary.counts.shape == (2,)
    assert summary.probabilities.shape == (2,)


def test_constant_numeric_vector_has_zero_entropy_and_code_length():
    x = np.array([5.0, 5.0, 5.0, 5.0])

    summary = empirical_distribution([x], [True])

    assert summary.n_states == 1
    assert summary.entropy == pytest.approx(0.0)
    assert summary.code_length == pytest.approx(0.0)


def test_categorical_distribution_counts_probabilities_and_states():
    x = np.array(["a", "a", "b", "c"])

    summary = empirical_distribution([x], [False])

    assert summary.n_samples == 4
    assert summary.n_states == 3
    assert np.array_equal(summary.states, np.array([[0], [1], [2]]))
    assert np.array_equal(summary.counts, np.array([2.0, 1.0, 1.0]))
    assert np.sum(summary.probabilities) == pytest.approx(1.0)
    assert summary.entropy == pytest.approx(1.5)
    assert summary.code_length == pytest.approx(6.0)


def test_fair_binary_variable_has_one_bit_entropy():
    x = np.array([0, 0, 1, 1])

    summary = empirical_distribution([x], [False])

    assert summary.entropy == pytest.approx(1.0)
    assert summary.code_length == pytest.approx(4.0)


def test_joint_entropy_is_at_least_each_marginal_entropy():
    x = np.array([0, 0, 1, 1])
    y = np.array([0, 1, 0, 1])

    hx = empirical_distribution([x], [False]).entropy
    hy = empirical_distribution([y], [False]).entropy
    hxy = empirical_distribution([x, y], [False, False]).entropy

    assert hxy + 1e-12 >= hx
    assert hxy + 1e-12 >= hy
    assert hxy == pytest.approx(2.0)


def test_joint_entropy_of_identical_variables_equals_marginal_entropy():
    x = np.array([0, 0, 1, 1])

    hx = empirical_distribution([x], [False]).entropy
    hxx = empirical_distribution([x, x], [False, False]).entropy

    assert hxx == pytest.approx(hx)


def test_mixed_numeric_and_categorical_distribution():
    x_num = np.array([0.0, 0.1, 0.9, 1.0])
    x_cat = np.array(["a", "a", "b", "b"])

    summary = empirical_distribution([x_num, x_cat], [True, False], n_bins=2)

    assert summary.n_samples == 4
    assert summary.n_states == 2
    assert np.array_equal(summary.counts, np.array([2.0, 2.0]))
    assert summary.entropy == pytest.approx(1.0)
    assert summary.code_length == pytest.approx(4.0)


def test_numeric_auto_bins_are_bounded_by_number_of_samples():
    x = np.arange(2, dtype=float)

    summary = empirical_distribution([x], [True], n_bins="auto")

    assert summary.n_samples == 2
    assert summary.n_states <= 2
    assert summary.entropy == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Empirical-distribution validation
# ---------------------------------------------------------------------------


def test_empirical_distribution_rejects_empty_column_sequence():
    with pytest.raises(ValueError):
        empirical_distribution([], [])


def test_empirical_distribution_rejects_numeric_flag_mismatch():
    with pytest.raises(ValueError):
        empirical_distribution([[1, 2, 3]], [True, False])


def test_empirical_distribution_rejects_inconsistent_lengths():
    with pytest.raises(ValueError):
        empirical_distribution([[1, 2, 3], [1, 2]], [True, True])


def test_empirical_distribution_rejects_empty_variable():
    with pytest.raises(ValueError):
        empirical_distribution([[]], [True])


def test_empirical_distribution_rejects_non_numeric_values_marked_numeric():
    with pytest.raises(ValueError):
        empirical_distribution([["a", "b", "c"]], [True])


def test_empirical_distribution_rejects_missing_numeric_values():
    x = np.array([1.0, np.nan, 2.0])

    with pytest.raises(ValueError):
        empirical_distribution([x], [True])


def test_empirical_distribution_rejects_missing_categorical_values():
    x = np.array(["a", None, "b"], dtype=object)

    with pytest.raises(ValueError):
        empirical_distribution([x], [False])


def test_empirical_distribution_rejects_unhashable_categorical_values():
    x = _object_array([["a"], ["b"], ["a"]])

    with pytest.raises(TypeError, match="unhashable"):
        empirical_distribution([x], [False])


# ---------------------------------------------------------------------------
# Counts, entropy, and code length
# ---------------------------------------------------------------------------


def test_entropy_from_counts_for_fair_binary_counts():
    counts = np.array([2, 2])

    entropy = entropy_from_counts(counts)

    assert entropy == pytest.approx(1.0)


def test_entropy_from_counts_ignores_zero_counts():
    counts = np.array([2, 0, 2])

    entropy = entropy_from_counts(counts)

    assert entropy == pytest.approx(1.0)


def test_code_length_from_counts_matches_entropy_times_sample_count():
    counts = np.array([2, 2])

    entropy = entropy_from_counts(counts)
    code_length = code_length_from_counts(counts)

    assert code_length == pytest.approx(np.sum(counts) * entropy)


def test_code_length_from_counts_ignores_zero_counts():
    counts = np.array([2, 0, 2])

    code_length = code_length_from_counts(counts)

    assert code_length == pytest.approx(4.0)


@pytest.mark.parametrize(
    "counts",
    [
        [],
        [-1, 2],
        [0, 0],
        [1, np.nan],
        [[1, 2], [3, 4]],
    ],
)
def test_entropy_from_counts_rejects_invalid_counts(counts):
    with pytest.raises(ValueError):
        entropy_from_counts(counts)


@pytest.mark.parametrize(
    "counts",
    [
        [],
        [-1, 2],
        [0, 0],
        [1, np.nan],
        [[1, 2], [3, 4]],
    ],
)
def test_code_length_from_counts_rejects_invalid_counts(counts):
    with pytest.raises(ValueError):
        code_length_from_counts(counts)