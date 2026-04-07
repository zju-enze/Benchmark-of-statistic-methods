"""XBART predictor wrapper."""

import numpy as np
from numpy.typing import NDArray

from .base import BasePredictor


class XBARTPredictor(BasePredictor):
    """
    XBART (Accelerated BART) predictor wrapper.

    Note: Requires the pyxbart package.
    """

    def __init__(
        self,
        num_trees: int = 100,
        num_sweeps: int = 40,
        burnin: int = 15,
        max_depth: int | None = None,
        n_classes: int = 1,
    ) -> None:
        self.num_trees = num_trees
        self.num_sweeps = num_sweeps
        self.burnin = burnin
        self.max_depth = max_depth
        self.n_classes = n_classes
        self._model = None

    def fit(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> "XBARTPredictor":
        try:
            import pyxbart
        except ImportError:
            raise ImportError("pyxbart is required. Install with: pip install pyxbart")

        # Convert to float64 if needed
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        # XBART uses a different interface
        self._model = pyxbart.XBART(
            n_trees=self.num_trees,
            num_sweeps=self.num_sweeps,
            burnin=self.burnin,
            max_depth=self.max_depth,
        )
        self._model.fit(X, y)
        return self

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X = np.asarray(X, dtype=np.float64)
        return self._model.predict(X)
