"""Full benchmark runner with parallelization support."""

from itertools import product
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.typing import NDArray

from .evaluate import evaluate
from ..methods.base import BasePredictor
from ..datasets.synthetic import SYNTHETIC_FUNCTIONS


def _get_synthetic_data(
    data_name: str,
    n: int,
    p: int,
    structure: str,
    sigma: float = 1.0,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Generate synthetic data by name."""
    func = SYNTHETIC_FUNCTIONS.get(data_name)
    if func is None:
        raise ValueError(f"Unknown synthetic data: {data_name}")
    return func(n=n, p=p, structure=structure, sigma=sigma)


def run_benchmark_synthetic(
    data_names: List[str],
    method_factories: Dict[str, Callable[[], BasePredictor]],
    structures: List[str] = ["indep", "ar1", "ar1+", "factor"],
    ns: List[int] = [100, 200, 500, 1000],
    ps: List[int] = [20, 50, 100, 200],
    n_folds: int = 5,
    n_jobs: int = -1,
    seed: int = 1234,
) -> pd.DataFrame:
    """
    Run benchmark on synthetic datasets.

    Parameters
    ----------
    data_names : list of str
        Names of synthetic data generators (e.g., ["sim_friedman", "sim_linear"]).
    method_factories : dict
        Dictionary mapping method names to factory functions.
    structures : list of str
        Covariance structures to test.
    ns : list of int
        Sample sizes to test.
    ps : list of int
        Dimensions to test.
    n_folds : int
        Number of CV folds.
    n_jobs : int
        Number of parallel jobs (-1 for all cores).
    seed : int
        Random seed.

    Returns
    -------
    results : DataFrame
        Results with columns: data_model, structure, n, p, method, cverr, runtime.
    """
    tasks = []
    for data_name, structure, n, p in product(data_names, structures, ns, ps):
        for method_name, factory in method_factories.items():
            tasks.append((data_name, structure, n, p, method_name, factory))

    def evaluate_single(data_name, structure, n, p, method_name, factory):
        try:
            X, y = _get_synthetic_data(data_name, n, p, structure)
            result = evaluate(X, y, factory, n_folds=n_folds, seed=seed)
            return {
                "data_model": data_name,
                "structure": structure,
                "n": n,
                "p": p,
                "method": method_name,
                "cverr": result["cverr"],
                "runtime": result["runtime"],
            }
        except Exception as e:
            return {
                "data_model": data_name,
                "structure": structure,
                "n": n,
                "p": p,
                "method": method_name,
                "cverr": np.nan,
                "runtime": np.nan,
                "error": str(e),
            }

    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_single)(*task) for task in tasks
    )

    return pd.DataFrame(results)


def run_benchmark_real(
    data_loaders: Dict[str, Callable[[], Tuple[NDArray[np.floating], NDArray[np.floating]]]],
    method_factories: Dict[str, Callable[[], BasePredictor]],
    n_folds: int = 5,
    n_jobs: int = -1,
    seed: int = 1234,
) -> pd.DataFrame:
    """
    Run benchmark on real datasets.

    Parameters
    ----------
    data_loaders : dict
        Dictionary mapping dataset names to loader functions.
    method_factories : dict
        Dictionary mapping method names to factory functions.
    n_folds : int
        Number of CV folds.
    n_jobs : int
        Number of parallel jobs (-1 for all cores).
    seed : int
        Random seed.

    Returns
    -------
    results : DataFrame
        Results with columns: data_model, method, cverr, runtime.
    """
    tasks = []
    for data_name, loader in data_loaders.items():
        for method_name, factory in method_factories.items():
            tasks.append((data_name, loader, method_name, factory))

    def evaluate_single(data_name, loader, method_name, factory):
        try:
            X, y = loader()
            result = evaluate(X, y, factory, n_folds=n_folds, seed=seed)
            return {
                "data_model": data_name,
                "method": method_name,
                "cverr": result["cverr"],
                "runtime": result["runtime"],
            }
        except Exception as e:
            return {
                "data_model": data_name,
                "method": method_name,
                "cverr": np.nan,
                "runtime": np.nan,
                "error": str(e),
            }

    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_single)(*task) for task in tasks
    )

    return pd.DataFrame(results)
