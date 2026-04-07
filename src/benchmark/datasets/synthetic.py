"""Synthetic data generators matching the R implementation."""

import numpy as np
from numpy.typing import NDArray
from typing import Literal, Tuple


def gen_x(
    n: int = 500,
    p: int = 200,
    structure: Literal["indep", "ar1", "ar1+", "factor"] = "indep",
    rho: float = 0.9,
    rho1: float = 0.5,
    rho2: float = 0.2,
) -> NDArray[np.floating]:
    """
    Generate feature matrix X with various covariance structures.

    Parameters
    ----------
    n : int
        Number of samples.
    p : int
        Number of features.
    structure : str
        Covariance structure: "indep", "ar1", "ar1+", or "factor".
    rho : float
        Correlation parameter for AR(1).
    rho1, rho2 : float
        Correlation parameters for AR(1)+.

    Returns
    -------
    X : ndarray of shape (n, p)
        Generated feature matrix.
    """
    rng = np.random.default_rng()

    if structure == "indep":
        X = rng.standard_normal((n, p))
    elif structure == "ar1":
        # Build AR(1) covariance matrix: Sigma[j,k] = rho^|j-k|
        idx = np.arange(p)
        cov = rho ** np.abs(idx[:, None] - idx[None, :])
        X = rng.multivariate_normal(mean=np.zeros(p), cov=cov, size=n)
    elif structure == "ar1+":
        # AR(1) plus additional correlation
        idx = np.arange(p)
        cov = rho1 ** np.abs(idx[:, None] - idx[None, :]) + rho2 * (idx[:, None] != idx[None, :])
        X = rng.multivariate_normal(mean=np.zeros(p), cov=cov, size=n)
    elif structure == "factor":
        # Factor model: requires p to be divisible by 5
        if p % 5 != 0:
            raise ValueError(f"p={p} must be divisible by 5 for factor structure")
        k = p // 5
        # Generate latent factors F (k x n)
        Fmat = rng.standard_normal((k, n))
        # Generate random loading matrix B (5k x k) with block diagonal structure
        rawB_list = []
        for _ in range(5):
            rawB_list.append(np.eye(k))
        rawB = np.vstack(rawB_list)  # shape (5k, k)
        # Sample rows of B
        B = rawB[rng.integers(0, len(rawB), size=len(rawB)), :]  # shape (5k, k)
        # Compute X = t(B %*% Fmat) + noise = Fmat.T @ B.T + noise
        # B is (5k, k), Fmat is (k, n), B @ Fmat is (5k, n), t() is (n, 5k)
        X = (B @ Fmat).T + rng.standard_normal((n, p)) * 0.1 * np.sqrt(k)
        X = (X - X.mean(axis=0)) / X.std(axis=0)
    else:
        import warnings
        warnings.warn(f"Unsupported structure '{structure}', using independent.")
        X = rng.standard_normal((n, p))

    return X


def sim_friedman(
    n: int = 500,
    p: int = 200,
    sigma: float = 1.0,
    structure: Literal["indep", "ar1", "ar1+", "factor"] = "indep",
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Friedman simulation model.

    y = 10*sin(pi*x1*x2) + 20*(x3-0.5)^2 + 10*x4 + 5*x5 + noise

    Reference: Linero (2018)
    """
    X = gen_x(n, p, structure=structure)
    y0 = (
        10 * np.sin(np.pi * X[:, 0] * X[:, 1])
        + 20 * (X[:, 2] - 0.5) ** 2
        + 10 * X[:, 3]
        + 5 * X[:, 4]
    )
    rng = np.random.default_rng()
    y = y0 + rng.standard_normal(n) * sigma
    return X, y


def sim_checkerboard(
    n: int = 500,
    p: int = 200,
    sigma: float = 1.0,
    structure: Literal["indep", "ar1", "ar1+", "factor"] = "indep",
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Checkerboard simulation model.

    y = 2*x1*x2 + 2*x3*x4 + noise

    Reference: Linero (2018)
    """
    X = gen_x(n, p, structure=structure)
    y0 = 2 * X[:, 0] * X[:, 1] + 2 * X[:, 2] * X[:, 3]
    rng = np.random.default_rng()
    y = y0 + rng.standard_normal(n) * sigma
    return X, y


def sim_linear(
    n: int = 500,
    p: int = 200,
    sigma: float = 1.0,
    structure: Literal["indep", "ar1", "ar1+", "factor"] = "indep",
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Linear simulation model.

    y = 2*x1 + 2*x2 + 4*x3 + noise

    Reference: Linero (2018)
    """
    X = gen_x(n, p, structure=structure)
    y0 = 2 * X[:, 0] + 2 * X[:, 1] + 4 * X[:, 2]
    rng = np.random.default_rng()
    y = y0 + rng.standard_normal(n) * sigma
    return X, y


def sim_max(
    n: int = 500,
    p: int = 200,
    sigma: float = 1.0,
    structure: Literal["indep", "ar1", "ar1+", "factor"] = "indep",
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Max simulation model.

    y = max(x1, x2, x3) + noise

    Reference: He and Hahn (2023)
    """
    X = gen_x(n, p, structure=structure)
    y0 = np.maximum.reduce([X[:, 0], X[:, 1], X[:, 2]])
    rng = np.random.default_rng()
    y = y0 + rng.standard_normal(n) * sigma
    return X, y


def sim_single_index(
    n: int = 500,
    p: int = 200,
    sigma: float = 1.0,
    structure: Literal["indep", "ar1", "ar1+", "factor"] = "indep",
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Single Index simulation model.

    y = 10*sqrt(sum((xi - gamma_i)^2)) + sin(5*a) + noise
    where gamma = -1.5 + (0:9)/3 and a = sum((xi - gamma_i)^2) for i=1:10

    Reference: He and Hahn (2023)
    """
    X = gen_x(n, p, structure=structure)
    gamma = -1.5 + np.arange(10) / 3.0
    a = np.sum((X[:, :10] - gamma) ** 2, axis=1)
    y0 = 10 * np.sqrt(a) + np.sin(5 * a)
    rng = np.random.default_rng()
    y = y0 + rng.standard_normal(n) * sigma
    return X, y


# Mapping from name to function for easy access
SYNTHETIC_FUNCTIONS = {
    "sim_friedman": sim_friedman,
    "sim_checkerboard": sim_checkerboard,
    "sim_linear": sim_linear,
    "sim_max": sim_max,
    "sim_single_index": sim_single_index,
}
