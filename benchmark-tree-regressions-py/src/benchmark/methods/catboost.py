"""CatBoost predictor wrapper."""

import numpy as np
from numpy.typing import NDArray

from .base import BasePredictor


class CatBoostPredictor(BasePredictor):
    """CatBoost regression wrapper."""

    def __init__(
        self,
        iterations: int = 100,
        depth: int = 6,
        learning_rate: float = 0.1,
    ) -> None:
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self._model = None

    def fit(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> "CatBoostPredictor":
        try:
            from catboost import CatBoostRegressor
        except ImportError:
            raise ImportError("catboost is required. Install with: pip install catboost")

        self._model = CatBoostRegressor(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            verbose=False,
        )
        self._model.fit(X, y)
        return self

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._model.predict(X)
