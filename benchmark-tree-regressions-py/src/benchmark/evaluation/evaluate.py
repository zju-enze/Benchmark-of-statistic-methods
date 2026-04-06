"""Single method evaluation."""

from typing import Callable, Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from .cross_validation import cross_validate
from ..methods.base import BasePredictor


def evaluate(
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    predictor_factory: Callable[[], BasePredictor],
    n_folds: int = 5,
    seed: int = 1234,
) -> Dict[str, float]:
    """
    Evaluate a single method on a single dataset.

    Parameters
    ----------
    X : ndarray
        Feature matrix.
    y : ndarray
        Target vector.
    predictor_factory : callable
        Returns a new predictor instance.
    n_folds : int
        Number of CV folds.
    seed : int
        Random seed.

    Returns
    -------
    results : dict
        Dictionary with 'cverr' and 'runtime' keys.
    """
    cverr, runtime = cross_validate(X, y, predictor_factory, n_folds, seed)
    return {"cverr": cverr, "runtime": runtime}
