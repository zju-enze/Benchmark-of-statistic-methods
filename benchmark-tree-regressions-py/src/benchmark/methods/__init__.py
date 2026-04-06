"""Tree-based regression method wrappers."""

from .base import BasePredictor
from .mean import MeanPredictor
from .xgboost import XGBoostPredictor
from .catboost import CatBoostPredictor
from .random_forest import RandomForestPredictor
from .bart import BARTPredictor
from .mars import MARSPredictor
from .xbart import XBARTPredictor

__all__ = [
    "BasePredictor",
    "MeanPredictor",
    "XGBoostPredictor",
    "CatBoostPredictor",
    "RandomForestPredictor",
    "BARTPredictor",
    "MARSPredictor",
    "XBARTPredictor",
]
