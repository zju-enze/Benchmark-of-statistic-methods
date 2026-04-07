"""Mean baseline predictor."""

import numpy as np
from numpy.typing import NDArray

from .base import BasePredictor


class MeanPredictor(BasePredictor):
    """Simple baseline that predicts the mean of training values."""

    def __init__(self) -> None:
        self._mean: float = 0.0

    def fit(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> "MeanPredictor":
        self._mean = np.mean(y)
        return self

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        return np.full(X.shape[0], self._mean)
