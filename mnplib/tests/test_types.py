"""Shared configuration types and annotation resolution."""

from typing import Literal, get_args, get_type_hints

import pytest

from mnplib import (
    AnomalyDetector,
    Inaccuracy,
    Miscoding,
    Nescience,
    NescienceClassifier,
    NescienceRegressor,
    Surfeit,
    TimeSeries,
)
from mnplib._types import Aggregation, BinSpec, ResolvedTask, Task, XType, YType
from mnplib.anomalies import AnomalyKind
from mnplib.automl.searchers.base import SearchContext
from mnplib.miscoding import RankingCriterion
from mnplib.models.serializers.base import SklearnSerializer
from mnplib.models.serializers.linear import LinearModelSerializer, LogisticRegressionSerializer
from mnplib.models.serializers.naive_bayes import NaiveBayesSerializer
from mnplib.models.serializers.neural_network import MLPSerializer
from mnplib.models.serializers.svm import LinearSVMSerializer
from mnplib.models.serializers.tree import DecisionTreeSerializer
from mnplib.timeseries.lagged import LaggedRepresentationBuilder, WindowSize
from mnplib.timeseries.estimator import ModelName
from mnplib.utils import (
    discretize_vector,
    empirical_distribution_array,
    empirical_distribution_vector,
)


def test_shared_aliases_describe_supported_configuration_values():
    assert get_args(BinSpec) == (int, Literal["auto", "adaptive"])
    assert get_args(XType) == ("auto", "numeric", "categorical")
    assert YType is XType
    assert get_args(Aggregation) == (
        "euclidean", "arithmetic", "geometric", "harmonic", "maximum", "addition", "product"
    )
    assert get_args(ResolvedTask) == ("classification", "regression")
    assert get_args(Task) == ("auto", "classification", "regression")


@pytest.mark.parametrize("cls", [
    AnomalyDetector, Inaccuracy, Miscoding, Nescience,
    NescienceClassifier, NescienceRegressor, Surfeit, TimeSeries,
])
def test_estimator_configuration_annotations_use_shared_types(cls):
    hints = get_type_hints(cls.__init__)
    assert hints["n_bins"] == BinSpec
    for parameter, alias in [("X_type", XType), ("y_type", YType),
                             ("aggregation", Aggregation), ("task", Task)]:
        if parameter in hints:
            assert hints[parameter] == alias


@pytest.mark.parametrize("function", [
    discretize_vector, empirical_distribution_vector, empirical_distribution_array,
])
def test_discretization_annotations_use_shared_bin_spec(function):
    assert get_type_hints(function)["n_bins"] == BinSpec


def test_configured_and_resolved_task_annotations_are_distinct():
    assert get_type_hints(AnomalyDetector.__init__)["task"] == Task
    assert get_type_hints(AnomalyDetector._resolve_task)["return"] == ResolvedTask
    assert get_type_hints(SearchContext)["task"] == ResolvedTask


@pytest.mark.parametrize("serializer", [
    SklearnSerializer, LinearModelSerializer, LogisticRegressionSerializer,
    NaiveBayesSerializer, MLPSerializer, LinearSVMSerializer, DecisionTreeSerializer,
])
def test_serializer_task_annotations_are_resolved(serializer):
    assert get_type_hints(serializer.task)["return"] == ResolvedTask


def test_timeseries_window_annotations_share_the_lagged_representation_type():
    assert get_type_hints(TimeSeries.__init__)["window_size"] == WindowSize
    assert get_type_hints(LaggedRepresentationBuilder.__init__)["window_size"] == WindowSize


@pytest.mark.parametrize("cls,attribute,alias", [
    (Miscoding, "_VALID_X_TYPES", XType),
    (Miscoding, "_VALID_Y_TYPES", YType),
    (Miscoding, "_VALID_RANKING_CRITERIA", RankingCriterion),
    (Nescience, "_VALID_X_TYPES", XType),
    (Nescience, "_VALID_Y_TYPES", YType),
    (Nescience, "_VALID_AGGREGATIONS", Aggregation),
    (Inaccuracy, "_VALID_Y_TYPES", YType),
    (Surfeit, "_VALID_Y_TYPES", YType),
    (AnomalyDetector, "_VALID_TASKS", Task),
    (AnomalyDetector, "_VALID_X_TYPES", XType),
    (AnomalyDetector, "_VALID_KINDS", AnomalyKind),
    (TimeSeries, "_VALID_X_TYPES", XType),
    (TimeSeries, "_VALID_MODELS", ModelName),
])
def test_runtime_validation_choices_match_literal_aliases(cls, attribute, alias):
    assert getattr(cls, attribute) == get_args(alias)


@pytest.mark.parametrize("cls,parameter,alias", [
    (Miscoding, "X_type", XType),
    (Miscoding, "y_type", YType),
    (Nescience, "X_type", XType),
    (Nescience, "y_type", YType),
    (Nescience, "aggregation", Aggregation),
    (Inaccuracy, "y_type", YType),
    (Surfeit, "y_type", YType),
])
def test_metric_validation_accepts_literal_choices_and_reports_invalid_values(cls, parameter, alias):
    choices = get_args(alias)
    for value in choices:
        metric = cls(**{parameter: value})
        assert getattr(metric, parameter) == value
    with pytest.raises(ValueError) as error:
        cls(**{parameter: "invalid"})
    received = "'invalid'." if cls is Miscoding else f"{parameter}='invalid' instead."
    assert str(error.value) == (
        f"Valid options for '{parameter}' are {choices}. Got {received}"
    )


@pytest.mark.parametrize("cls,parameter,alias", [
    (AnomalyDetector, "X_type", XType),
    (AnomalyDetector, "task", Task),
    (TimeSeries, "X_type", XType),
])
def test_deferred_validation_uses_literal_choices(cls, parameter, alias):
    choices = get_args(alias)
    for value in choices:
        cls(**{parameter: value})._validate_configuration()
    with pytest.raises(ValueError) as error:
        cls(**{parameter: "invalid"})._validate_configuration()
    assert str(error.value) == f"Valid options for {parameter} are {choices}. Got 'invalid'."


@pytest.mark.parametrize("task", get_args(ResolvedTask))
def test_explicit_anomaly_task_does_not_require_inference(task):
    assert AnomalyDetector(task=task)._resolve_task([0, 1]) == task


@pytest.mark.parametrize("family", get_args(ModelName))
def test_timeseries_configuration_accepts_declared_model_families(family):
    TimeSeries(models=[family])._validate_configuration()
