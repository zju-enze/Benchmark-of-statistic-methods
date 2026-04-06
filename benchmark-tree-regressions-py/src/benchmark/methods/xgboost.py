"""XGBoost predictor wrapper."""

import numpy as np
from numpy.typing import NDArray

from .base import BasePredictor


class XGBoostPredictor(BasePredictor):
    """XGBoost regression wrapper."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        early_stopping_rounds: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.early_stopping_rounds = early_stopping_rounds
        self._model = None

    def fit(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> "XGBoostPredictor":
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost is required. Install with: pip install xgboost")

        self._model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            verbosity=0,
            early_stopping_rounds=self.early_stopping_rounds,
        )
        self._model.fit(X, y)
        return self

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._model.predict(X)
