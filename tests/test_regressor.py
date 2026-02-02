"""Tests for ERBFRegressor."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_friedman1

from erbf import ERBFRegressor


def test_fit_predict_synthetic_1d():
    """sin(x) regression, R2 > 0.9."""
    rng = np.random.RandomState(42)
    X = rng.uniform(-3, 3, (300, 1))
    y = np.sin(X[:, 0]) + rng.normal(0, 0.05, 300)
    model = ERBFRegressor(n_rbf=20, random_state=42)
    model.fit(X, y)
    r2 = model.score(X, y)
    assert r2 > 0.9, f"R2={r2:.3f} too low on sin(x)"


def test_fit_predict_synthetic_nd():
    """friedman1 (5d), R2 > 0.5."""
    X, y = make_friedman1(n_samples=500, n_features=5, random_state=42)
    model = ERBFRegressor(n_rbf=40, random_state=42)
    model.fit(X, y)
    r2 = model.score(X, y)
    assert r2 > 0.5, f"R2={r2:.3f} too low on friedman1"


def test_auto_n_rbf():
    """n_rbf='auto' resolves to int."""
    X, y = make_friedman1(n_samples=200, n_features=5, random_state=42)
    model = ERBFRegressor(n_rbf='auto', random_state=42)
    model.fit(X, y)
    assert isinstance(model.centres_.shape[0], (int, np.integer))


@pytest.mark.parametrize("method", ['kmeans', 'lipschitz', 'random'])
def test_center_init_strategies(method):
    rng = np.random.RandomState(0)
    X = rng.randn(200, 3)
    y = X[:, 0] + rng.normal(0, 0.1, 200)
    model = ERBFRegressor(n_rbf=10, center_init=method, width_optim=None, random_state=0)
    model.fit(X, y)
    assert model.centres_.shape == (10, 3)


@pytest.mark.parametrize("method", ['local_ridge', 'local_variance', 'uniform'])
def test_width_init_strategies(method):
    rng = np.random.RandomState(0)
    X = rng.randn(200, 3)
    y = X[:, 0] + rng.normal(0, 0.1, 200)
    model = ERBFRegressor(n_rbf=10, width_init=method, width_optim=None, random_state=0)
    model.fit(X, y)
    assert model.widths_.shape == (10, 3)


def test_sklearn_clone():
    model = ERBFRegressor(n_rbf=30, alpha=0.5)
    cloned = clone(model)
    assert cloned.n_rbf == 30
    assert cloned.alpha == 0.5


def test_get_params_set_params():
    model = ERBFRegressor(n_rbf=30)
    params = model.get_params()
    assert params['n_rbf'] == 30
    model.set_params(n_rbf=50)
    assert model.n_rbf == 50


def test_standardize_flag():
    rng = np.random.RandomState(42)
    X = rng.randn(100, 2) * 100  # large scale
    y = X[:, 0] + rng.normal(0, 1, 100)
    for flag in [True, False]:
        model = ERBFRegressor(n_rbf=10, standardize=flag, width_optim=None, random_state=42)
        model.fit(X, y)
        pred = model.predict(X)
        assert pred.shape == (100,)
