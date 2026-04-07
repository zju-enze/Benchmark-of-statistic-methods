"""Random Forest predictor wrapper."""

import numpy as np
from numpy.typing import NDArray

from .base import BasePredictor


class RandomForestPredictor(BasePredictor):
    """Scikit-learn Random Forest regression wrapper."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self._model = None

    def fit(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> "RandomForestPredictor":
        from sklearn.ensemble import RandomForestRegressor

        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=-1,
        )
        self._model.fit(X, y)
        return self

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._model.predict(X)
