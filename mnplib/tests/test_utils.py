import numpy as np
import pytest

from utils import (
    EmpiricalEncoder,
    code_length_from_counts,
    discretize_vector,
    empirical_code_length,
    empirical_distribution,
    empirical_entropy,
    entropy_from_counts,
    optimal_number_of_bins,
)


def test_constant_numeric_vector_has_zero_entropy():
    x = np.array([5.0, 5.0, 5.0, 5.0])

    summary = empirical_distribution([x], [True])

    assert summary.n_states == 1
    assert summary.entropy == pytest.approx(0.0)
    assert summary.code_length == pytest.approx(0.0)


def test_empirical_distribution_counts_probabilities_and_states():
    x = np.array(["a", "a", "b", "c"])

    summary = empirical_distribution([x], [False])

    assert summary.n_samples == 4
    assert summary.n_states == 3
    assert np.sum(summary.counts) == pytest.approx(4)
    assert np.sum(summary.probabilities) == pytest.approx(1.0)


def test_entropy_for_fair_binary_variable_is_one_bit():
    x = np.array([0, 0, 1, 1])

    h = empirical_entropy([x], [False], base=2)

    assert h == pytest.approx(1.0)


def test_code_length_is_n_times_entropy():
    x = np.array([0, 0, 1, 1])

    h = empirical_entropy([x], [False], base=2)
    l = empirical_code_length([x], [False], base=2)

    assert l == pytest.approx(len(x) * h)


def test_per_sample_code_length_equals_entropy():
    x = np.array([0, 0, 1, 1])

    h = empirical_entropy([x], [False], base=2)
    l = empirical_code_length([x], [False], base=2, per_sample=True)

    assert l == pytest.approx(h)


def test_normalized_entropy_is_between_zero_and_one():
    x = np.array([0, 0, 1, 1, 2, 2])

    h = empirical_entropy([x], [False], normalized=True)

    assert 0.0 <= h <= 1.0


def test_joint_entropy_is_at_least_marginal_entropy():
    x = np.array([0, 0, 1, 1])
    y = np.array([0, 1, 0, 1])

    hx = empirical_entropy([x], [False])
    hy = empirical_entropy([y], [False])
    hxy = empirical_entropy([x, y], [False, False])

    assert hxy + 1e-12 >= hx
    assert hxy + 1e-12 >= hy


def test_joint_entropy_of_identical_variables_equals_marginal_entropy():
    x = np.array([0, 0, 1, 1])

    hx = empirical_entropy([x], [False])
    hxx = empirical_entropy([x, x], [False, False])

    assert hxx == pytest.approx(hx)


def test_uniform_discretization_without_sklearn_dependency():
    x = np.linspace(0.0, 1.0, 10)

    z = discretize_vector(x, n_bins=2, strategy="uniform")

    assert z.shape == x.shape
    assert set(np.unique(z)).issubset({0, 1})


def test_quantile_discretization():
    x = np.arange(100)

    z = discretize_vector(x, n_bins=4, strategy="quantile")

    assert z.shape == x.shape
    assert len(np.unique(z)) == 4


def test_encoder_reuses_numeric_bin_edges():
    y_train = np.array([0.0, 1.0, 2.0, 3.0])
    y_pred = np.array([0.2, 1.2, 2.2, 3.2])

    encoder = EmpiricalEncoder([True], n_bins=2, strategy="uniform")
    encoder.fit([y_train])

    encoded_y = encoder.transform([y_train])[:, 0]
    encoded_pred = encoder.transform([y_pred])[:, 0]

    assert encoded_y.shape == y_train.shape
    assert encoded_pred.shape == y_pred.shape
    assert np.array_equal(encoded_y, np.array([0, 0, 1, 1]))
    assert np.array_equal(encoded_pred, np.array([0, 0, 1, 1]))


def test_unseen_category_raises_by_default():
    train = np.array(["a", "b", "a"])
    test = np.array(["a", "c"])

    encoder = EmpiricalEncoder([False])
    encoder.fit([train])

    with pytest.raises(ValueError):
        encoder.transform([test])


def test_unseen_category_can_use_encoded_value():
    train = np.array(["a", "b", "a"])
    test = np.array(["a", "c"])

    encoder = EmpiricalEncoder(
        [False],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    encoder.fit([train])

    encoded = encoder.transform([test])[:, 0]

    assert encoded[1] == -1


def test_missing_numeric_values_raise():
    x = np.array([1.0, np.nan, 2.0])

    with pytest.raises(ValueError):
        empirical_entropy([x], [True])


def test_missing_categorical_values_raise():
    x = np.array(["a", None, "b"], dtype=object)

    with pytest.raises(ValueError):
        empirical_entropy([x], [False])


def test_miller_madow_entropy_is_not_smaller_than_plugin():
    counts = np.array([10, 10, 10])

    h_plugin = entropy_from_counts(counts, correction="none")
    h_mm = entropy_from_counts(counts, correction="miller_madow")

    assert h_mm >= h_plugin


def test_dirichlet_entropy_runs_and_is_non_negative():
    counts = np.array([10, 0, 5])

    h = entropy_from_counts(
        counts,
        correction="dirichlet",
        alpha=0.5,
        alphabet_size=3,
    )

    assert h >= 0.0


def test_code_length_from_counts_matches_entropy_times_n():
    counts = np.array([2, 2])

    h = entropy_from_counts(counts)
    l = code_length_from_counts(counts)

    assert l == pytest.approx(np.sum(counts) * h)


def test_base_parameter_changes_units():
    counts = np.array([1, 1])

    h_bits = entropy_from_counts(counts, base=2)
    h_nats = entropy_from_counts(counts, base=np.e)

    assert h_bits == pytest.approx(1.0)
    assert h_nats == pytest.approx(np.log(2))


def test_invalid_base_raises():
    counts = np.array([1, 1])

    with pytest.raises(ValueError):
        entropy_from_counts(counts, base=1)


def test_occupancy_based_bins_are_positive():
    bins = optimal_number_of_bins(
        n_samples=100,
        n_numeric=2,
        n_categorical_states=1,
        min_samples_per_cell=5,
    )

    assert bins >= 1


def test_empirical_summary_contains_expected_fields():
    x = np.array([0, 0, 1, 1])

    summary = empirical_distribution([x], [False])

    assert summary.states.shape[0] == summary.n_states
    assert summary.counts.shape[0] == summary.n_states
    assert summary.probabilities.shape[0] == summary.n_states
    assert summary.entropy == pytest.approx(1.0)
    assert summary.code_length == pytest.approx(4.0)


def test_encoder_with_mixed_numeric_and_categorical_variables():
    x_num = np.array([0.0, 0.1, 0.9, 1.0])
    x_cat = np.array(["a", "a", "b", "b"])

    summary = empirical_distribution(
        [x_num, x_cat],
        [True, False],
        n_bins=2,
        strategy="uniform",
    )

    assert summary.n_samples == 4
    assert summary.n_states >= 2
    assert summary.entropy >= 0.0