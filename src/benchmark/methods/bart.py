"""BART-like predictor using sklearn GradientBoosting as approximation."""

import numpy as np
from numpy.typing import NDArray

from .base import BasePredictor


class BARTPredictor(BasePredictor):
    """
    BART-like predictor using HistGradientBoostingRegressor as approximation.

    Note: This is not a true BART implementation. For exact BART behavior,
    use the R version. This provides a similar tree-based ensemble approach.
    """

    def __init__(
        self,
        max_iter: int = 100,
        max_depth: int | None = 5,
        learning_rate: float = 0.1,
        n_estimators: int = 200,
        burnin: int = 100,
        n_samples: int = 1000,
    ) -> None:
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.burnin = burnin
        self.n_samples = n_samples
        self._model = None

    def fit(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> "BARTPredictor":
        from sklearn.ensemble import HistGradientBoostingRegressor

        self._model = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            early_stopping=False,
        )
        self._model.fit(X, y)
        return self

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._model.predict(X)
