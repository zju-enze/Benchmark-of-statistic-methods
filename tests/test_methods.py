"""Tests for method wrappers."""

import numpy as np
from benchmark.methods import MeanPredictor, RandomForestPredictor


def create_test_data(n=100, p=20):
    """Create test data."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, p))
    y = rng.standard_normal(n)
    return X, y


def test_mean_predictor():
    """Test MeanPredictor."""
    X, y = create_test_data()
    model = MeanPredictor()
    model.fit(X, y)

    pred = model.predict(X)
    assert pred.shape == (100,)
    assert np.allclose(pred, np.mean(y))


def test_random_forest():
    """Test RandomForestPredictor."""
    X, y = create_test_data()
    model = RandomForestPredictor(n_estimators=10)
    model.fit(X, y)

    pred = model.predict(X)
    assert pred.shape == (100,)


if __name__ == "__main__":
    test_mean_predictor()
    test_random_forest()
    print("All tests passed!")
