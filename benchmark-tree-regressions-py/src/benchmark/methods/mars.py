"""MARS (Multivariate Adaptive Regression Splines) predictor wrapper."""

import numpy as np
from numpy.typing import NDArray

from .base import BasePredictor


class MARSPredictor(BasePredictor):
    """
    MARS regression wrapper using py-earth.

    Note: Requires the py-earth package.
    """

    def __init__(
        self,
        degree: int = 1,
        nprune: int | None = None,
        penalty: float = 3.0,
    ) -> None:
        self.degree = degree
        self.nprune = nprune
        self.penalty = penalty
        self._model = None

    def fit(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> "MARSPredictor":
        try:
            from pyearth import Earth
        except ImportError:
            raise ImportError("py-earth is required. Install with: pip install py-earth")

        self._model = Earth(
            max_degree=self.degree,
            max_terms=self.nprune,
            penalty=self.penalty,
            verbose=False,
        )
        self._model.fit(X, y)
        return self

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._model.predict(X)
