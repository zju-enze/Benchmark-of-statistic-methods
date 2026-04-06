"""Tests for synthetic data generators."""

import numpy as np
from benchmark.datasets.synthetic import (
    gen_x,
    sim_friedman,
    sim_checkerboard,
    sim_linear,
    sim_max,
    sim_single_index,
)


def test_gen_x_indep():
    """Test independent X generation."""
    X = gen_x(n=100, p=20, structure="indep")
    assert X.shape == (100, 20)
    assert np.abs(X.mean()) < 1.0  # Should be close to 0
    assert np.abs(X.std() - 1.0) < 0.2  # Should be close to 1


def test_gen_x_ar1():
    """Test AR(1) X generation."""
    X = gen_x(n=100, p=20, structure="ar1", rho=0.9)
    assert X.shape == (100, 20)


def test_gen_x_ar1_plus():
    """Test AR(1)+ X generation."""
    X = gen_x(n=100, p=20, structure="ar1+", rho1=0.5, rho2=0.2)
    assert X.shape == (100, 20)


def test_gen_x_factor():
    """Test factor X generation."""
    X = gen_x(n=100, p=20, structure="factor")
    assert X.shape == (100, 20)


def test_gen_x_factor_invalid_p():
    """Test factor X generation with invalid p."""
    try:
        gen_x(n=100, p=21, structure="factor")
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def test_sim_friedman():
    """Test Friedman simulation."""
    X, y = sim_friedman(n=100, p=20)
    assert X.shape == (100, 20)
    assert y.shape == (100,)


def test_sim_checkerboard():
    """Test Checkerboard simulation."""
    X, y = sim_checkerboard(n=100, p=20)
    assert X.shape == (100, 20)
    assert y.shape == (100,)


def test_sim_linear():
    """Test Linear simulation."""
    X, y = sim_linear(n=100, p=20)
    assert X.shape == (100, 20)
    assert y.shape == (100,)


def test_sim_max():
    """Test Max simulation."""
    X, y = sim_max(n=100, p=20)
    assert X.shape == (100, 20)
    assert y.shape == (100,)


def test_sim_single_index():
    """Test Single Index simulation."""
    X, y = sim_single_index(n=100, p=20)
    assert X.shape == (100, 20)
    assert y.shape == (100,)


if __name__ == "__main__":
    test_gen_x_indep()
    test_gen_x_ar1()
    test_gen_x_ar1_plus()
    test_gen_x_factor()
    test_gen_x_factor_invalid_p()
    test_sim_friedman()
    test_sim_checkerboard()
    test_sim_linear()
    test_sim_max()
    test_sim_single_index()
    print("All tests passed!")
