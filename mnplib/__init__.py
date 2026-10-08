__version__ = "2.0.0b1"

from .miscoding import Miscoding
from .inaccuracy import Inaccuracy
from .surfeit import Surfeit
from .mismodel import Mismodel
from .nescience import Nescience
from .classifier import NescienceClassifier
from .regressor import NescienceRegressor
from .timeseries import TimeSeries
from .residuals import ResidualAnalysis
from .models import ModelDescription, describe_model

__all__ = [
    "Miscoding", "Inaccuracy", "Surfeit", "Mismodel", "Nescience",
    "NescienceClassifier", "NescienceRegressor", "TimeSeries", "ResidualAnalysis",
    "ModelDescription", "describe_model",
]
