"""Base class for all regression predictors."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray


class BasePredictor(ABC):
    """Abstract base class for all regression predictors."""

    @abstractmethod
    def fit(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> "BasePredictor":
        """
        Fit the predictor on training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training features.
        y : ndarray of shape (n_samples,)
            Training target values.

        Returns
        -------
        self : BasePredictor
            Returns self.
        """
        pass

    @abstractmethod
    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Predict on new data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Features to predict on.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        pass

    @property
    def name(self) -> str:
        """Return the predictor name."""
        return self.__class__.__name__
