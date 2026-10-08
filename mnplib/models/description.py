"""Public, immutable computational descriptions independent of evaluation data."""

from dataclasses import dataclass, field
from collections.abc import Mapping
from types import MappingProxyType
import math

from .language import ModelNode, normalize, render, node_to_dict, node_from_dict
from .language.normalize import _index


def _freeze(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Metadata keys must be strings.")
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise TypeError("Metadata must contain finite JSON-compatible values.")


def _json_value(value):
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


@dataclass(frozen=True)
class ModelDescription:
    """Canonical semantics and computational metadata, without predictions."""

    model_type: str
    ast: ModelNode
    feature_indices: tuple[int, ...] = ()
    feature_names: tuple[str | None, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)
    canonical: str = field(init=False)

    def __post_init__(self):
        ast = normalize(self.ast)
        if not isinstance(self.model_type, str) or not self.model_type:
            raise ValueError("model_type must be a non-empty string.")
        indices = tuple(_index(index) for index in self.feature_indices)
        names = tuple(self.feature_names)
        if any(name is not None and not isinstance(name, str) for name in names):
            raise ValueError("Feature names must be strings or None.")
        if len(indices) != len(names):
            raise ValueError("Feature indices and names must have equal lengths.")
        if len(set(indices)) != len(indices):
            raise ValueError("Feature indices must be unique.")
        object.__setattr__(self, "ast", ast)
        object.__setattr__(self, "canonical", render(ast))
        object.__setattr__(self, "feature_indices", indices)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "metadata", _freeze(self.metadata))

    def __reduce__(self):
        return (type(self).from_dict, (self.to_dict(),))

    def to_dict(self):
        """Export schema version 1 using stable semantic node names."""
        return {
            "schema": "mnplib-model", "schema_version": 1,
            "model_type": self.model_type, "ast": node_to_dict(self.ast),
            "canonical": self.canonical,
            "feature_indices": list(self.feature_indices),
            "feature_names": list(self.feature_names),
            "metadata": _json_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data):
        """Import the supported semantic schema and validate its canonical text."""
        if (not isinstance(data, dict) or data.get("schema") != "mnplib-model"
                or type(data.get("schema_version")) is not int or data["schema_version"] != 1):
            raise ValueError("Unsupported model-description schema or version.")
        result = cls(data["model_type"], node_from_dict(data["ast"]),
                     tuple(data.get("feature_indices", ())),
                     tuple(data.get("feature_names", ())), data.get("metadata", {}))
        if "canonical" in data and data["canonical"] != result.canonical:
            raise ValueError("Canonical text does not match the semantic AST.")
        return result


def describe_model(model, *, feature_names=None, feature_indices=None):
    """Describe a supported fitted estimator without evaluating predictions."""
    from ..timeseries.models import FixedLinearForecaster, StatsmodelsForecastModel
    if isinstance(model, StatsmodelsForecastModel):
        raise ValueError("Stateful ARIMA and state-space descriptions are not supported "
                         "by model-language schema version 1.")
    if isinstance(model, FixedLinearForecaster):
        from .serializers.time_series import fixed_model_description
        return fixed_model_description(model, feature_names=feature_names,
                                       feature_indices=feature_indices)
    if hasattr(model, "best_artifacts_"):
        if feature_names is not None or feature_indices is not None:
            raise ValueError("AutoML descriptions already carry their fitted feature mapping.")
        return model.best_artifacts_.description
    from .sklearn import _find_serializer
    return _find_serializer(model).description(
        model, feature_names=feature_names, feature_indices=feature_indices,
    )
