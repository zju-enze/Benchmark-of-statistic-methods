"""Cross-validation framework."""

import time
from typing import Callable, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import KFold

from ..methods.base import BasePredictor


def cross_validate(
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    predictor_factory: Callable[[], BasePredictor],
    n_folds: int = 5,
    seed: int = 1234,
) -> Tuple[float, float]:
    """
    Evaluate a predictor using K-fold cross-validation.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Target vector.
    predictor_factory : callable
        Function that returns a new unfitted predictor instance.
    n_folds : int
        Number of cross-validation folds.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    cverr : float
        Mean squared error across all folds.
    runtime : float
        Total runtime in seconds.
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    errors = []
    start_time = time.perf_counter()

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = predictor_factory()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        errors.append(np.mean((y_pred - y_test) ** 2))

    total_time = time.perf_counter() - start_time
    cverr = float(np.mean(errors))
    runtime = float(total_time)

    return cverr, runtime
